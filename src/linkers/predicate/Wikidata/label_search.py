import os
from collections import OrderedDict

import requests

from src.linkers.base import BasePredicateLinker, LinkingInput, LinkingOutput
from src.utils.retry import call_with_retry


ENDPOINT = os.environ.get("ENDPOINT_URL", "http://localhost:7001/sparql")
HEADERS = {
    "User-Agent": "simple-label-linker/0.1",
    "Accept": "application/sparql-results+json",
}
PREFIXES = """
PREFIX wd:       <http://www.wikidata.org/entity/>
PREFIX wdt:      <http://www.wikidata.org/prop/direct/>
PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
"""

# score values
_SCORE_EXACT    = 0   # exact match
_SCORE_PREFIX   = 1   # prefix match
_SCORE_CONTAINS = 2   # containing match

# --------------------------------------------
# Cache

class BoundedCache(OrderedDict):
    def __init__(self, maxsize=2000):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        if key not in self and len(self) >= self.maxsize:
            self.popitem(last=False)  # evict oldest
        super().__setitem__(key, value)


class Linker(BasePredicateLinker):
    """
    Wikidata predicate linker: direct rdfs:label substring/prefix/exact
    match against every wikibase:directClaim property, scored by match
    tightness. This is the Wikidata replacement for the Freebase
    passthrough predicate linker (which just echoes fbp:-tagged tokens
    straight through with confidence 1.0, since ChatKBQA's Freebase
    predictions already emit canonical relation paths) -- Wikidata
    predictions instead emit human-readable property text that has to be
    resolved to a PID via an actual lookup, hence the SPARQL query.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self._cache: BoundedCache = BoundedCache(maxsize=125)

    def get_params(self) -> dict:
        return {"k": self.k}

    # --------------------------------------------
    # SPARQL

    def _sparql(self, query: str) -> list[dict]:
        if query in self._cache:
            return self._cache[query]
        full_query = PREFIXES + query.strip()

        def _do_request():
            resp = requests.get(
                ENDPOINT,
                headers=HEADERS,
                params={"query": full_query},
                timeout=60,
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            return resp

        resp = call_with_retry(
            _do_request,
            retries=2,
            base_delay=1.0,
            backoff=2.0,
            exceptions=(requests.RequestException,),
        )
        if resp is None:
            return []

        results = []
        for b in resp.json()["results"]["bindings"]:
            pid   = b["p"]["value"].split("/")[-1]
            label = b.get("pLabel", {}).get("value", pid)
            score = int(b.get("score", {}).get("value", _SCORE_CONTAINS))
            results.append({"id": pid, "label": label, "score": score})

        self._cache[query] = results
        return results

    # --------------------------------------------
    # Label search

    def _search(self, label: str) -> list[dict]:
        mention = label.replace("_", " ").lower().replace('"', "")

        query = f"""
        SELECT DISTINCT ?p ?pLabel ?score WHERE {{
          ?prop wikibase:directClaim ?p ;
                rdfs:label ?pLabel .
          FILTER(LANG(?pLabel) = "en")
          FILTER(CONTAINS(LCASE(?pLabel), "{mention}"))
          BIND(
            IF(LCASE(?pLabel) = "{mention}",        {_SCORE_EXACT},
            IF(STRSTARTS(LCASE(?pLabel), "{mention}"), {_SCORE_PREFIX},
                                                      {_SCORE_CONTAINS}))
            AS ?score
          )
        }}
        ORDER BY ?score STRLEN(?pLabel)
        LIMIT {self.k * 3}
        """
        candidates = self._sparql(query)

        seen: set[str] = set()
        deduped = []
        for c in candidates:
            if c["id"] not in seen:
                seen.add(c["id"])
                deduped.append(c)
        return deduped[: self.k]

    # --------------------------------------------
    # Main

    def link(self, inp: LinkingInput, entity_map: dict[str, str]) -> LinkingOutput:
        resolved: dict[str, str] = {}
        candidates_map: dict[str, list[tuple[str, float]]] = {}
        failed: list[str] = []
        debug: dict = {}

        for label in inp.labels:
            candidates = self._search(label)

            # convert [0, 1, 2] scores to [0...1] scores
            def _to_conf(score: int) -> float:
                return round(1.0 - score / 3.0, 4)

            topk = [(c["id"], _to_conf(c["score"])) for c in candidates]
            candidates_map[label] = topk
            debug[label] = candidates

            if topk:
                resolved[label] = topk[0][0]
            else:
                failed.append(label)

        return LinkingOutput(
            label_map=resolved,
            candidates=candidates_map,
            failed=failed,
            debug=debug,
        )