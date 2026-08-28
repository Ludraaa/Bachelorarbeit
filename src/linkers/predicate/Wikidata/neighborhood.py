import os
import time
from collections import OrderedDict

import requests
from sentence_transformers import SentenceTransformer, util

from src.linkers.base import BasePredicateLinker, LinkingInput, LinkingOutput
from src.utils.retry import call_with_retry


ENDPOINT = os.environ.get(
    "ENDPOINT_URL",
    "http://localhost:7001/sparql",
)

HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "ChatKBQA-repro/1.0",
}

PREFIXES = """PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd:   <http://www.wikidata.org/entity/>
PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
"""

_WD_NS  = "http://www.wikidata.org/entity/"
_WDT_NS = "http://www.wikidata.org/prop/direct/"


class BoundedCache(OrderedDict):

    def __init__(self, maxsize: int = 200):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        if key not in self and len(self) >= self.maxsize:
            self.popitem(last=False)
        super().__setitem__(key, value)


class Linker(BasePredicateLinker):
    """
    Wikidata predicate linker replicating the ChatKBQA 2-hop neighbourhood
    expansion fallback (try_relation in the original codebase), ported
    from the Freebase neighborhood linker.

    Two things differ from the Freebase version, both structural rather
    than cosmetic:

    1. Relation identity. Freebase relations (domain.type.property) are
       both the identifier AND rough natural-language text, so the
       Freebase version scores raw relation-URI local names directly
       against the extracted question labels. Wikidata PIDs (P17, P31,
       ...) carry no lexical content -- scoring "P17" against "country"
       would be meaningless. So candidate PIDs are additionally resolved
       to their rdfs:label (fetched from the wd:Pxxx entity URI, per
       Wikidata.normalize() in src/kb/wikidata.py -- the label does NOT
       live on the wdt:Pxxx direct-claim URI used to traverse the graph)
       before scoring, and results are mapped back to the PID afterwards.

    2. Self-candidate seeding. The Freebase version additionally seeds the
       candidate pool with the raw extracted labels themselves (in case
       the model already predicted the correct dot-path relation
       verbatim, so it's not lost when 2-hop expansion misses it). That
       relies on Freebase relation names already being valid, meaningful
       identifiers on their own. Wikidata's extracted labels are free-text
       property guesses, not PIDs -- seeding them into the pool would let
       raw English text end up in predicate_map, which substitute()
       expects to contain a real PID. This step is dropped rather than
       ported.

    QLever/the target SPARQL endpoint performs only cheap structural
    filters; the original string-based predicate filters are applied
    locally in Python after retrieval, same as the Freebase version.
    """

    def __init__(
        self,
        ke: int = 15,
        te: float = 0.01,
        limit: int = 10000,
        timeout: int = 300,
        language: str = "en",
        entity_cache_size: int = 200,
        pool_cache_size: int = 500,
        score_cache_size: int = 500,
        label_cache_size: int = 2000,
    ):
        self.ke = ke
        self.te = te
        self.limit = limit
        self.timeout = timeout
        self.language = language

        self.model_name = "princeton-nlp/unsup-simcse-roberta-large"
        self.model = SentenceTransformer(self.model_name)

        self._entity_cache: BoundedCache = BoundedCache(maxsize=entity_cache_size)
        self._pool_cache: BoundedCache = BoundedCache(maxsize=pool_cache_size)
        self._score_cache: BoundedCache = BoundedCache(maxsize=score_cache_size)
        self._label_cache: BoundedCache = BoundedCache(maxsize=label_cache_size)

        self._stats = {
            "total_fetches": 0,
            "entity_cache_hits": 0,
            "entity_cache_misses": 0,
            "pool_cache_hits": 0,
            "pool_cache_misses": 0,
            "score_cache_hits": 0,
            "score_cache_misses": 0,
            "label_cache_hits": 0,
            "label_cache_misses": 0,
            "query_attempts": {"q1": 0, "q2": 0, "q3": 0, "q4": 0},
            "query_successes": {"q1": 0, "q2": 0, "q3": 0, "q4": 0},
            "query_failures": {"q1": 0, "q2": 0, "q3": 0, "q4": 0},
            "total_time_s": {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0},
            "total_relations": {"q1": 0, "q2": 0, "q3": 0, "q4": 0},
        }

    def get_params(self) -> dict:
        return {
            "ke": self.ke,
            "te": self.te,
            "limit": self.limit,
            "timeout": self.timeout,
            "language": self.language,
            "model_name": self.model_name,
        }

    # ------------------------------------------------------------------
    # Local predicate filtering

    @staticmethod
    def _valid_relation(uri: str) -> bool:
        """
        Wikidata analogue of the Freebase string filters. Freebase's
        domain.type.property relations are inherently "clean" once you
        exclude a handful of noisy namespaces (wikipedia, _id, kg.,
        dataworld.); the equivalent clean, single-hop, scoreable relation
        on Wikidata is any direct-claim predicate (wdt:Pxxx). wdt:P31
        ("instance of") is additionally dropped, mirroring how the
        Freebase filter excludes type.object.type/instance -- both are
        structural typing edges rather than content relations worth
        scoring against a question.
        """
        if not uri.startswith(_WDT_NS):
            return False
        if uri == f"{_WDT_NS}P31":
            return False
        return True

    @classmethod
    def _valid_relation_pair(cls, r0_uri: str, r1_uri: str) -> bool:
        """
        pair is valid iff both predicates pass the local filter
        """
        return (
            cls._valid_relation(r0_uri)
            and cls._valid_relation(r1_uri)
        )

    # ------------------------------------------------------------------
    # Query builders
    #
    # Same four 2-hop shapes as the Freebase version, unchanged structurally
    # -- only the prefix/entity-var namespace differs. wdt:P31 is excluded
    # at the SPARQL level too (in addition to the Python-side
    # _valid_relation_pair filter) purely to keep result sets smaller,
    # matching how the Freebase version excludes ns:type.object.type in
    # the query itself.

    def _q1(self, ent: str) -> str:
        """
        Entity is object; intermediate is also object elsewhere:

            ?x1 ?r0 entity .
            ?x2 ?r1 ?x1 .
        """
        return PREFIXES + f"""SELECT DISTINCT ?r0 ?r1 WHERE {{
  ?x1 ?r0 {ent} .
  ?x2 ?r1 ?x1 .

  FILTER (?r0 != rdf:type && ?r0 != rdfs:label)
  FILTER (?r1 != rdf:type && ?r1 != rdfs:label)
  FILTER (?r0 != wdt:P31)
  FILTER (?r1 != wdt:P31)
}}
LIMIT {self.limit}"""

    def _q2(self, ent: str) -> str:
        """
        Entity is object; intermediate is subject elsewhere:

            ?x1 ?r0 entity .
            ?x1 ?r1 ?x2 .
        """
        return PREFIXES + f"""SELECT DISTINCT ?r0 ?r1 WHERE {{
  ?x1 ?r0 {ent} .
  ?x1 ?r1 ?x2 .

  FILTER (?x2 != {ent})

  FILTER (?r0 != rdf:type && ?r0 != rdfs:label)
  FILTER (?r1 != rdf:type && ?r1 != rdfs:label)
  FILTER (?r0 != wdt:P31)
  FILTER (?r1 != wdt:P31)
}}
LIMIT {self.limit}"""

    def _q3(self, ent: str) -> str:
        """
        Entity is subject; intermediate is object elsewhere:

            entity ?r0 ?x1 .
            ?x2 ?r1 ?x1 .
        """
        return PREFIXES + f"""SELECT DISTINCT ?r0 ?r1 WHERE {{
  {ent} ?r0 ?x1 .
  ?x2 ?r1 ?x1 .

  FILTER (?r0 != rdf:type && ?r0 != rdfs:label)
  FILTER (?r1 != rdf:type && ?r1 != rdfs:label)
  FILTER (?r0 != wdt:P31)
  FILTER (?r1 != wdt:P31)
}}
LIMIT {self.limit}"""

    def _q4(self, ent: str) -> str:
        """
        Entity is subject; intermediate is also subject elsewhere:

            entity ?r0 ?x1 .
            ?x1 ?r1 ?x2 .
        """
        return PREFIXES + f"""SELECT DISTINCT ?r0 ?r1 WHERE {{
  {ent} ?r0 ?x1 .
  ?x1 ?r1 ?x2 .

  FILTER (?x2 != {ent})

  FILTER (?r0 != rdf:type && ?r0 != rdfs:label)
  FILTER (?r1 != rdf:type && ?r1 != rdfs:label)
  FILTER (?r0 != wdt:P31)
  FILTER (?r1 != wdt:P31)
}}
LIMIT {self.limit}"""

    # ------------------------------------------------------------------
    # Single query executor

    def _run_query(self, query: str, label: str) -> dict:
        debug: dict = {
            "query_label": label,
            "query": query,
            "success": False,
            "error": None,
            "duration_s": None,
            "n_raw_bindings": 0,
            "n_relations": 0,
            "n_filtered_pairs": 0,
            "hit_limit": False,
            "retries_used": 0,
            "status_codes": [],
        }

        relations: set[str] = set()
        attempt_count: int = 0

        def _do_request():
            nonlocal attempt_count

            attempt_count += 1

            resp = requests.post(
                ENDPOINT,
                data={"query": query},
                headers=HEADERS,
                timeout=self.timeout,
                proxies={"http": None, "https": None},
            )

            debug["status_codes"].append(resp.status_code)

            resp.raise_for_status()
            return resp

        t0 = time.perf_counter()

        try:
            resp = call_with_retry(
                _do_request,
                retries=3,
                base_delay=5.0,
                backoff=2.0,
                exceptions=(requests.RequestException,),
            )

            debug["duration_s"] = round(
                time.perf_counter() - t0,
                3,
            )

            debug["retries_used"] = attempt_count - 1

            if resp is None:
                debug["error"] = "call_with_retry returned None"

            else:
                bindings = resp.json()["results"]["bindings"]

                debug["n_raw_bindings"] = len(bindings)
                debug["hit_limit"] = len(bindings) == self.limit

                filtered_pairs = 0

                for binding in bindings:
                    r0_uri = binding["r0"]["value"]
                    r1_uri = binding["r1"]["value"]

                    if not self._valid_relation_pair(r0_uri, r1_uri):
                        filtered_pairs += 1
                        continue

                    r0 = (
                        r0_uri[len(_WDT_NS):]
                        if r0_uri.startswith(_WDT_NS)
                        else r0_uri
                    )

                    r1 = (
                        r1_uri[len(_WDT_NS):]
                        if r1_uri.startswith(_WDT_NS)
                        else r1_uri
                    )

                    relations.add(r0)
                    relations.add(r1)

                debug["n_filtered_pairs"] = filtered_pairs
                debug["n_relations"] = len(relations)
                debug["success"] = True

        except Exception as exc:
            debug["duration_s"] = round(
                time.perf_counter() - t0,
                3,
            )
            debug["error"] = str(exc)
            debug["retries_used"] = attempt_count - 1

        return {
            "relations": relations,
            "debug": debug,
        }

    # ------------------------------------------------------------------
    # Entity-level 2-hop fetch

    def _fetch_2hop_relations(
        self,
        entity_id: str,
    ) -> tuple[set[str], dict]:
        """
        Union of all relation local names (PIDs) reachable within 2 hops
        from entity_id (a bare QID, e.g. "Q76"), collected via 4 separate
        queries. Returns (relations, fetch_debug). Results are cached by
        entity QID.
        """

        self._stats["total_fetches"] += 1

        if entity_id in self._entity_cache:
            self._stats["entity_cache_hits"] += 1

            cached = self._entity_cache[entity_id]

            fetch_debug = {
                "entity_id": entity_id,
                "cache_hit": True,
                "pool_size": len(cached["relations"]),
                "queries": cached["query_debug"],
            }

            return cached["relations"], fetch_debug

        self._stats["entity_cache_misses"] += 1

        ent = f"<{_WD_NS}{entity_id}>"

        queries = [
            ("q1", self._q1(ent)),
            ("q2", self._q2(ent)),
            ("q3", self._q3(ent)),
            ("q4", self._q4(ent)),
        ]

        cumulative: set[str] = set()
        query_debugs: list = []

        for q_label, query in queries:

            pool_before = len(cumulative)

            self._stats["query_attempts"][q_label] += 1

            result = self._run_query(
                query,
                q_label,
            )

            rels = result["relations"]
            q_debug = result["debug"]

            new_rels = rels - cumulative
            cumulative |= rels

            q_debug["n_new_contributed"] = len(new_rels)
            q_debug["pool_size_before"] = pool_before
            q_debug["pool_size_after"] = len(cumulative)

            if q_debug["success"]:

                self._stats["query_successes"][q_label] += 1

                self._stats["total_relations"][q_label] += len(rels)

                self._stats["total_time_s"][q_label] = round(
                    self._stats["total_time_s"][q_label]
                    + q_debug["duration_s"],
                    3,
                )

            else:
                self._stats["query_failures"][q_label] += 1

            query_debugs.append(q_debug)

            time.sleep(1)

        self._entity_cache[entity_id] = {
            "relations": cumulative,
            "query_debug": query_debugs,
        }

        fetch_debug = {
            "entity_id": entity_id,
            "cache_hit": False,
            "pool_size": len(cumulative),
            "queries": query_debugs,
        }

        return cumulative, fetch_debug

    # ------------------------------------------------------------------
    # PID -> label lookup (the step the Freebase version doesn't need)

    def _fetch_relation_labels(self, pids: set[str]) -> dict[str, str]:
        """
        rdfs:label for a property lives on its wd:Pxxx entity URI, not on
        the wdt:Pxxx direct-claim URI used to traverse the graph -- see
        Wikidata.normalize() in src/kb/wikidata.py for the same mapping.
        Falls back to the bare PID string (so scoring still runs, just
        with no semantic signal) if no English label is found or the
        request fails.
        """
        pids = set(pids)
        uncached = [p for p in pids if p not in self._label_cache]

        self._stats["label_cache_hits"] += len(pids) - len(uncached)
        self._stats["label_cache_misses"] += len(uncached)

        if uncached:
            values = " ".join(f"<{_WD_NS}{p}>" for p in uncached)
            query = PREFIXES + f"""SELECT ?p ?label WHERE {{
  VALUES ?p {{ {values} }}
  ?p rdfs:label ?label .
  FILTER(LANG(?label) = "{self.language}")
}}"""

            def _do_request():
                resp = requests.post(
                    ENDPOINT,
                    data={"query": query},
                    headers=HEADERS,
                    timeout=self.timeout,
                    proxies={"http": None, "https": None},
                )
                resp.raise_for_status()
                return resp

            try:
                resp = call_with_retry(
                    _do_request,
                    retries=3,
                    base_delay=5.0,
                    backoff=2.0,
                    exceptions=(requests.RequestException,),
                )
                if resp is not None:
                    bindings = resp.json()["results"]["bindings"]
                    for row in bindings:
                        pid = row["p"]["value"].rsplit("/", 1)[-1]
                        label = row["label"]["value"]
                        self._label_cache[pid] = label
            except Exception:
                pass  # uncached PIDs below fall back to their bare id

        return {p: self._label_cache.get(p, p) for p in pids}

    # ------------------------------------------------------------------
    # Scoring

    def _score(
        self,
        labels: list[str],
        cand_ids: list[str],
        cand_texts: list[str],
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Same cosine-similarity top-k/threshold logic as the Freebase
        version, except candidates are embedded on their fetched label
        text (cand_texts) while the returned tuples reference the
        underlying PID (cand_ids) -- substitute() downstream needs the
        PID, not the label text.
        """

        if not labels or not cand_ids:
            return {
                label: []
                for label in labels
            }

        emb_labels = self.model.encode(
            labels,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=64,
        )

        emb_cands = self.model.encode(
            cand_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=64,
        )

        sim_matrix = util.cos_sim(
            emb_labels,
            emb_cands,
        )

        result: dict[str, list[tuple[str, float]]] = {}

        for i, label in enumerate(labels):

            scored = list(
                zip(
                    cand_ids,
                    sim_matrix[i].tolist(),
                )
            )

            filtered = [
                (pid, s)
                for pid, s in scored
                if s > self.te
            ]

            result[label] = sorted(
                filtered,
                key=lambda x: -x[1],
            )[:self.ke]

        return result

    # ------------------------------------------------------------------
    # BasePredicateLinker interface

    def link(
        self,
        inp: LinkingInput,
        entity_map: dict[str, str],
    ) -> LinkingOutput:

        t_link_start = time.perf_counter()

        pool_key = frozenset(
            entity_map.values()
        )

        score_key = (
            pool_key,
            tuple(sorted(inp.labels)),
        )

        # --------------------------------------------------------------
        # Pool cache

        pool_cache_hit = pool_key in self._pool_cache

        if pool_cache_hit:

            self._stats["pool_cache_hits"] += 1

            cand_ids, pool_debug = self._pool_cache[pool_key]

            entity_fetch_debugs = pool_debug["entity_fetches"]

        else:

            self._stats["pool_cache_misses"] += 1

            cand_pids_set: set[str] = set()
            entity_fetch_debugs: dict = {}

            for entity_id in entity_map.values():

                pids, fetch_debug = self._fetch_2hop_relations(
                    entity_id
                )

                cand_pids_set |= pids

                entity_fetch_debugs[entity_id] = fetch_debug

            cand_ids = list(cand_pids_set)

            pool_debug = {
                "entity_fetches": entity_fetch_debugs,
            }

            self._pool_cache[pool_key] = (
                cand_ids,
                pool_debug,
            )

        # --------------------------------------------------------------
        # Label lookup for the candidate pool (Wikidata-only step)

        label_lookup = self._fetch_relation_labels(set(cand_ids))
        cand_texts = [label_lookup[pid] for pid in cand_ids]

        # --------------------------------------------------------------
        # Score cache

        score_cache_hit = score_key in self._score_cache

        if score_cache_hit:

            self._stats["score_cache_hits"] += 1

            scored_map = self._score_cache[score_key]
            score_duration = 0.0

        else:

            self._stats["score_cache_misses"] += 1

            t_score = time.perf_counter()

            scored_map = self._score(
                inp.labels,
                cand_ids,
                cand_texts,
            )

            score_duration = round(
                time.perf_counter() - t_score,
                3,
            )

            self._score_cache[score_key] = scored_map

        # --------------------------------------------------------------
        # Assemble output

        resolved: dict[str, list[tuple[str, float]]] = {}
        label_map: dict[str, str] = {}
        failed: list[str] = []

        cand_ids_set_fast = set(cand_ids)

        debug: dict = {
            "pool_size": len(cand_ids),
            "pool_cache_hit": pool_cache_hit,
            "score_cache_hit": score_cache_hit,
            "entity_map": entity_map,
            "entity_fetches": entity_fetch_debugs,
            "global_stats": self._stats,
            "score_duration_s": score_duration,
            "link_duration_s": None,
            "per_label": {},
        }

        if not cand_ids:

            for label in inp.labels:
                resolved[label] = []
                failed.append(label)

            debug["link_duration_s"] = round(
                time.perf_counter() - t_link_start,
                3,
            )

            return LinkingOutput(
                label_map=label_map,
                candidates=resolved,
                failed=failed,
                debug=debug,
            )

        for label in inp.labels:

            top = scored_map.get(
                label,
                [],
            )

            debug["per_label"][label] = {
                "top_candidates": [
                    {
                        "id": pid,
                        "label": label_lookup.get(pid, pid),
                        "score": round(score, 6),
                    }
                    for pid, score in top
                ],
                "n_candidates": len(top),
                "best_id": top[0][0] if top else None,
                "best_score": (
                    round(top[0][1], 6)
                    if top
                    else None
                ),
                "in_pool": label in cand_ids_set_fast,
            }

            if top:

                resolved[label] = top
                label_map[label] = top[0][0]

            else:

                resolved[label] = []
                failed.append(label)

        debug["link_duration_s"] = round(
            time.perf_counter() - t_link_start,
            3,
        )

        return LinkingOutput(
            label_map=label_map,
            candidates=resolved,
            failed=failed,
            debug=debug,
        )