from src.linkers.base import BasePredicateLinker, LinkingInput, LinkingOutput
from src.utils.retry import call_with_retry
from sentence_transformers import SentenceTransformer, util
from collections import OrderedDict
import requests
import re
import os

ENDPOINT = os.environ.get("ENDPOINT_URL", "http://localhost:7001/sparql")

HEADERS = {
    "User-Agent": "ChatKBQA-research/0.1 (bachelor thesis; contact: luis.drayer@web.de)",
    "Accept": "application/sparql-results+json",
}


_CLASS_PREDICATES = {
    "instance_of", "subclass_of",
    "P31", "P279",
    "wdt:P31", "wdt:P279",
}

PREFIXES = """
PREFIX wd:       <http://www.wikidata.org/entity/>
PREFIX wdt:      <http://www.wikidata.org/prop/direct/>
PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
"""


# --------------------------------------------
# cache

class BoundedCache(OrderedDict):
    def __init__(self, maxsize=2000):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        if key not in self and len(self) >= self.maxsize:
            self.popitem(last=False)  # evict oldest
        super().__setitem__(key, value)


# --------------------------------------------
# Linker

class Linker(BasePredicateLinker):
    def __init__(
        self,
        ke: int = 5,
        te: float = 0.3,
        neighborhood_limit: int = 300,
        instance_sample: int = 500,
        label_search_limit: int = 50,
    ):
        self.ke = ke
        self.te = te
        self.neighborhood_limit = neighborhood_limit
        self.instance_sample = instance_sample
        self.label_search_limit = label_search_limit

        self.model = SentenceTransformer("princeton-nlp/sup-simcse-roberta-base")

        self._neighborhood_cache = BoundedCache(maxsize=100)
        self._sparql_cache = BoundedCache(maxsize=100)

    # --------------------------------------------
    # Class context detection

    def _is_class_context(self, label: str, entity_id: str, prediction: str) -> bool:
        mention = label.replace("_", " ")
        pred_alts = "|".join(re.escape(p) for p in _CLASS_PREDICATES)

        for term in (label, mention):
            pattern = rf"(?:{pred_alts})\b.*\b{re.escape(term)}\b"
            if re.search(pattern, prediction, re.IGNORECASE):
                return True
        return False

    # --------------------------------------------
    # SPARQL

    def _sparql(self, query: str, tag: str):
        cache_key = query
        if cache_key in self._sparql_cache:
            return self._sparql_cache[cache_key]

        full_query = PREFIXES + query.strip()

        def _do_request():
            resp = requests.get(
                ENDPOINT,
                headers=HEADERS,
                params={"query": full_query},
                timeout=180,
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
            print(f"[predicate_linker] SPARQL failed for: {tag}")
            return None

        bindings = resp.json()["results"]["bindings"]

        results = []
        for b in bindings:
            pid   = b["p"]["value"].split("/")[-1]
            label = b.get("pLabel", {}).get("value", pid)
            results.append({"id": pid, "label": label})

        self._sparql_cache[cache_key] = results
        return results

    # --------------------------------------------
    # Neighborhood fetches

    def _fetch_direct(self, entity_id: str) -> list[dict]:
        key = (entity_id, "direct")
        if key in self._neighborhood_cache:
            return self._neighborhood_cache[key]

        query = f"""
        SELECT DISTINCT ?p ?pLabel WHERE {{
          {{
            wd:{entity_id} ?p ?o .
            FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
          }}
          UNION
          {{
            ?s ?p wd:{entity_id} .
            FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
          }}
          ?prop wikibase:directClaim ?p .
          ?prop rdfs:label ?pLabel .
          FILTER(LANG(?pLabel) = "en")
        }}
        LIMIT {self.neighborhood_limit}
        """

        res = self._sparql(query, entity_id) or []
        self._neighborhood_cache[key] = res
        return res

    def _fetch_instance(self, entity_id: str) -> list[dict]:
        key = (entity_id, "instance")
        if key in self._neighborhood_cache:
            return self._neighborhood_cache[key]

        query = f"""
        SELECT ?p ?pLabel WHERE {{
        {{
            SELECT ?p (COUNT(*) AS ?cnt) WHERE {{
            {{
                SELECT DISTINCT ?instance WHERE {{
                ?instance wdt:P31 wd:{entity_id} .
                }} LIMIT {self.instance_sample}
            }}
            ?instance ?p [] .
            FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
            }}
            GROUP BY ?p
            ORDER BY DESC(?cnt)
            LIMIT {self.neighborhood_limit}
        }}
        ?prop wikibase:directClaim ?p ;
                rdfs:label ?pLabel .
        FILTER(LANG(?pLabel) = "en")
        }}
        """

        res = self._sparql(query, entity_id) or []
        self._neighborhood_cache[key] = res
        return res

    def _fetch_by_label(self, label: str) -> list[dict]:
        mention = label.replace("_", " ").lower()
        key = ("__label_search__", mention)
        if key in self._neighborhood_cache:
            return self._neighborhood_cache[key]

        query = f"""
        SELECT DISTINCT ?p ?pLabel WHERE {{
          ?prop wikibase:directClaim ?p ;
                rdfs:label ?pLabel .
          FILTER(LANG(?pLabel) = "en")
          FILTER(CONTAINS(LCASE(?pLabel), "{mention}"))
        }} LIMIT {self.label_search_limit}
        """

        res = self._sparql(query, f"label_search:{mention}") or []
        self._neighborhood_cache[key] = res
        return res

    # ─────────────────────────────────────────────
    # Combined neighborhood builder

    def _fetch_neighborhood(
        self,
        entity_map: dict[str, str],
        prediction: str,
        labels: list[str],
    ) -> tuple[list[dict], dict]:

        pool: dict[str, dict] = {}
        pool_debug: dict = {
            "total": 0,
            "sources": {},
            "label_search": {},
            "fallback": None,
        }

        # entity neighborhood
        for label, entity_id in entity_map.items():
            source_key = f"{label} ({entity_id})"
            is_class   = self._is_class_context(label, entity_id, prediction)

            entry: dict = {
                "class_context_detected": is_class,
                "direct": None,
                "instance": None,
            }

            # Direct fetch
            cache_hit_direct   = (entity_id, "direct") in self._neighborhood_cache
            direct_props       = self._fetch_direct(entity_id)
            contributed_direct = [p["id"] for p in direct_props if p["id"] not in pool]
            for p in direct_props:
                pool.setdefault(p["id"], p)

            entry["direct"] = {
                "called":          True,
                "cache_hit":       cache_hit_direct,
                "returned":        len(direct_props),
                "contributed":     len(contributed_direct),
                "contributed_ids": contributed_direct,
            }

            # Instance fetch (class context only)
            if is_class:
                cache_hit_instance   = (entity_id, "instance") in self._neighborhood_cache
                instance_props       = self._fetch_instance(entity_id)
                contributed_instance = [p["id"] for p in instance_props if p["id"] not in pool]
                for p in instance_props:
                    pool.setdefault(p["id"], p)

                entry["instance"] = {
                    "called":          True,
                    "cache_hit":       cache_hit_instance,
                    "returned":        len(instance_props),
                    "contributed":     len(contributed_instance),
                    "contributed_ids": contributed_instance,
                }

            pool_debug["sources"][source_key] = entry

        # property label search
        for label in labels:
            mention       = label.replace("_", " ").lower()
            cache_hit_lbl = ("__label_search__", mention) in self._neighborhood_cache
            label_props   = self._fetch_by_label(label)
            contributed   = [p["id"] for p in label_props if p["id"] not in pool]
            for p in label_props:
                pool.setdefault(p["id"], p)

            pool_debug["label_search"][label] = {
                "mention":         mention,
                "cache_hit":       cache_hit_lbl,
                "returned":        len(label_props),
                "contributed":     len(contributed),
                "contributed_ids": contributed,
            }

        pool_debug["total"] = len(pool)
        return list(pool.values()), pool_debug

    # --------------------------------------------
    # Fallback for empty neighborhood (might not be needed anymore)

    def _fallback_global(self, labels: list[str]) -> list[dict]:
        seen = set()
        pool = []

        for label in labels:
            mention = label.replace("_", " ").lower()
            res = self._sparql(f"""
            SELECT DISTINCT ?p ?pLabel WHERE {{
              ?prop wikibase:directClaim ?p .
              ?prop rdfs:label ?pLabel .
              FILTER(LANG(?pLabel) = "en")
              FILTER(CONTAINS(LCASE(?pLabel), "{mention}"))
            }} LIMIT 50
            """, f"fallback:{mention}") or []

            for p in res:
                if p["id"] not in seen:
                    seen.add(p["id"])
                    pool.append(p)

        return pool

    # --------------------------------------------
    # Scoring

    def _lexical_boost(self, mention: str, cand: dict) -> float:
        m = mention.lower().replace("_", " ")
        l = cand.get("label", "").lower()

        if m == l:
            return 0.25
        if m in l or l in m:
            return 0.12
        return 0.0

    def _score(self, question: str, mention: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        q     = f"{question} [RELATION] {mention.replace('_', ' ')}"
        texts = [c.get("label", c["id"]) for c in candidates]

        emb = self.model.encode(
            [q] + texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=64,
        )
        sim = util.cos_sim(emb[0], emb[1:])[0].tolist()
        del emb

        results = []
        for c, s in zip(candidates, sim):
            lex = self._lexical_boost(mention, c)
            results.append({
                "id":       c["id"],
                "label":    c.get("label", c["id"]),
                "semantic": s,
                "lexical":  lex,
                "score":    0.65 * s + 0.35 * lex,
            })

        return results

    def _topk(self, scored: list[dict]) -> list[tuple[str, float]]:
        filtered = [(c["id"], c["score"]) for c in scored if c["score"] >= self.te]
        if not filtered:
            return []
        return sorted(filtered, key=lambda x: -x[1])[: self.ke]

    # --------------------------------------------
    # Main

    def link(self, inp: LinkingInput, entity_map: dict[str, str]) -> LinkingOutput:
        pool, pool_debug = self._fetch_neighborhood(
            entity_map,
            inp.prediction,
            inp.labels,
        )

        if not pool:
            print(f"[predicate_linker] empty neighborhood, falling back to global label search")
            fallback_pool = self._fallback_global(inp.labels)
            fallback_ids  = [p["id"] for p in fallback_pool]
            pool_debug["fallback"] = {
                "reason":          "neighborhood empty after direct+instance+label-search fetch",
                "returned":        len(fallback_pool),
                "contributed":     len(fallback_pool),
                "contributed_ids": fallback_ids,
            }
            pool_debug["total"] = len(fallback_pool)
            pool = fallback_pool
        else:
            pool_debug["fallback"] = None

        resolved       = {}
        candidates_map = {}
        debug_map      = {"pool": pool_debug}
        failed         = []

        for label in inp.labels:
            scored_full = self._score(inp.question, label, pool)
            debug_map[label] = sorted(scored_full, key=lambda x: -x["score"])

            best = self._topk(scored_full)
            candidates_map[label] = best

            if best:
                resolved[label] = best[0][0]
            else:
                failed.append(label)

        return LinkingOutput(
            label_map=resolved,
            candidates=candidates_map,
            failed=failed,
            debug=debug_map,
        )