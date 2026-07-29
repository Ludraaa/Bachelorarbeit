import os
import time
import requests
from collections import OrderedDict
from sentence_transformers import SentenceTransformer, util

from src.linkers.base import BasePredicateLinker, LinkingInput, LinkingOutput
from src.utils.retry import call_with_retry

ENDPOINT = os.environ.get("ENDPOINT_URL", "http://localhost:7001/sparql")

HEADERS = {
    "Accept":     "application/sparql-results+json",
    "User-Agent": "ChatKBQA-repro/1.0",
}

PREFIXES = """PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX ns:   <http://rdf.freebase.com/ns/>
"""

_FB_NS = "http://rdf.freebase.com/ns/"

# Full SPARQL filter block mirroring get_2hop_relations_with_odbc_wo_filter.
_SPARQL_FILTERS = """  FILTER (?r0 != rdf:type  && ?r0 != rdfs:label)
  FILTER (?r1 != rdf:type  && ?r1 != rdfs:label)
  FILTER (?r0 != ns:type.object.type && ?r0 != ns:type.object.instance)
  FILTER (?r1 != ns:type.object.type && ?r1 != ns:type.object.instance)
  FILTER (!REGEX(STR(?r0), "wikipedia", "i"))
  FILTER (!REGEX(STR(?r1), "wikipedia", "i"))
  FILTER (!REGEX(STR(?r0), "_id", "i"))
  FILTER (!REGEX(STR(?r1), "_id", "i"))
  FILTER (!REGEX(STR(?r0), "#type", "i"))
  FILTER (!REGEX(STR(?r1), "#type", "i"))
  FILTER (!REGEX(STR(?r0), "#label", "i"))
  FILTER (!REGEX(STR(?r1), "#label", "i"))
  FILTER (!REGEX(STR(?r0), "/ns/freebase", "i"))
  FILTER (!REGEX(STR(?r1), "/ns/freebase", "i"))
  FILTER (!REGEX(STR(?r0), "ns/kg."))
  FILTER (!REGEX(STR(?r1), "ns/kg."))
  FILTER (!REGEX(STR(?r0), "ns/dataworld."))
  FILTER (!REGEX(STR(?r1), "ns/dataworld."))
  FILTER (REGEX(STR(?r0), "http://rdf.freebase.com/ns/"))
  FILTER (REGEX(STR(?r1), "http://rdf.freebase.com/ns/"))"""


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
    Freebase predicate linker replicating the ChatKBQA 2-hop neighbourhood
    expansion fallback (try_relation in the original codebase).
    """

    def __init__(
        self,
        ke: int = 15,
        te: float = 0.01,
        limit: int = 1000,
        timeout: int = 3000,
        mid_cache_size: int = 200,
        pool_cache_size: int = 500,
        score_cache_size: int = 500,
    ):
        self.ke = ke
        self.te = te
        self.limit = limit
        self.timeout = timeout
        self.model_name = "princeton-nlp/unsup-simcse-roberta-large"
        self.model = SentenceTransformer(self.model_name)

        self._mid_cache: BoundedCache = BoundedCache(maxsize=mid_cache_size)
        self._pool_cache: BoundedCache = BoundedCache(maxsize=pool_cache_size)
        self._score_cache: BoundedCache = BoundedCache(maxsize=score_cache_size)

        self._stats = {
            "total_fetches": 0,
            "mid_cache_hits": 0,
            "mid_cache_misses": 0,
            "pool_cache_hits": 0,
            "pool_cache_misses": 0,
            "score_cache_hits": 0,
            "score_cache_misses": 0,
            "query_attempts": {"q1": 0, "q2": 0, "q3": 0, "q4": 0},
            "query_successes": {"q1": 0, "q2": 0, "q3": 0, "q4": 0},
            "query_failures": {"q1": 0, "q2": 0, "q3": 0, "q4": 0},
            "total_time_s": {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0},
            "total_relations": {"q1": 0, "q2": 0, "q3": 0, "q4": 0},
        }
    
    def get_params(self) -> dict:
        return {"ke": self.ke, "te": self.te, "limit": self.limit, "timeout": self.timeout, "model_name": self.model_name}

    # ------------------------------------------------------------------
    # Query builders

    def _q1(self, ent: str) -> str:
        """entity is object; intermediate is also object elsewhere (?x2 ?r1 ?x1)"""
        return PREFIXES + f"""SELECT DISTINCT ?r0 ?r1 WHERE {{
  ?x1 ?r0 {ent} .
  ?x2 ?r1 ?x1 .
{_SPARQL_FILTERS}
}}
LIMIT {self.limit}"""

    def _q2(self, ent: str) -> str:
        """entity is object; intermediate is subject elsewhere (?x1 ?r1 ?x2)"""
        return PREFIXES + f"""SELECT DISTINCT ?r0 ?r1 WHERE {{
  ?x1 ?r0 {ent} .
  ?x1 ?r1 ?x2 .
  FILTER (?x2 != {ent})
{_SPARQL_FILTERS}
}}
LIMIT {self.limit}"""

    def _q3(self, ent: str) -> str:
        """entity is subject; intermediate is object elsewhere (?x2 ?r1 ?x1)"""
        return PREFIXES + f"""SELECT DISTINCT ?r0 ?r1 WHERE {{
  {ent} ?r0 ?x1 .
  ?x2 ?r1 ?x1 .
{_SPARQL_FILTERS}
}}
LIMIT {self.limit}"""

    def _q4(self, ent: str) -> str:
        """entity is subject; intermediate is also subject elsewhere (?x1 ?r1 ?x2)"""
        return PREFIXES + f"""SELECT DISTINCT ?r0 ?r1 WHERE {{
  {ent} ?r0 ?x1 .
  ?x1 ?r1 ?x2 .
  FILTER (?x2 != {ent})
{_SPARQL_FILTERS}
}}
LIMIT {self.limit}"""

    # ------------------------------------------------------------------
    # Single query executor

    def _run_query(self, query: str, label: str) -> dict:
        debug: dict = {
            "query_label":    label,
            "success":        False,
            "error":          None,
            "duration_s":     None,
            "n_raw_bindings": 0,
            "n_relations":    0,
            "hit_limit":      False,
            "retries_used":   0,
            "status_codes":   [],
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
                base_delay=60.0,
                backoff=2.0,
                exceptions=(requests.RequestException,),
            )
            debug["duration_s"]   = round(time.perf_counter() - t0, 3)
            debug["retries_used"] = attempt_count - 1

            if resp is None:
                debug["error"] = "call_with_retry returned None"
            else:
                bindings = resp.json()["results"]["bindings"]
                debug["n_raw_bindings"] = len(bindings)
                debug["hit_limit"]      = len(bindings) == self.limit

                # SPARQL already filtered; just strip the namespace prefix.
                for binding in bindings:
                    for var in ("r0", "r1"):
                        uri = binding[var]["value"]
                        if uri.startswith(_FB_NS):
                            relations.add(uri[len(_FB_NS):])

                debug["n_relations"] = len(relations)
                debug["success"]     = True

        except Exception as exc:
            debug["duration_s"]   = round(time.perf_counter() - t0, 3)
            debug["error"]        = str(exc)
            debug["retries_used"] = attempt_count - 1

        return {"relations": relations, "debug": debug}

    # ------------------------------------------------------------------
    # MID-level 2-hop fetch

    def _fetch_2hop_relations(self, entity_id: str) -> tuple[set[str], dict]:
        """
        Union of all relation local names reachable within 2 hops from
        entity_id, collected via 4 separate queries.

        Returns (relations, fetch_debug).
        Results are cached by entity MID.
        """
        self._stats["total_fetches"] += 1

        if entity_id in self._mid_cache:
            self._stats["mid_cache_hits"] += 1
            cached = self._mid_cache[entity_id]
            fetch_debug = {
                "entity_id": entity_id,
                "cache_hit": True,
                "pool_size": len(cached["relations"]),
                "queries":   cached["query_debug"],
            }
            return cached["relations"], fetch_debug

        self._stats["mid_cache_misses"] += 1
        ent = f"<{_FB_NS}{entity_id}>"

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

            result = self._run_query(query, q_label)
            rels = result["relations"]
            q_debug = result["debug"]
            new_rels = rels - cumulative
            cumulative |= rels

            q_debug["n_new_contributed"] = len(new_rels)
            q_debug["pool_size_before"]  = pool_before
            q_debug["pool_size_after"]   = len(cumulative)

            if q_debug["success"]:
                self._stats["query_successes"][q_label] += 1
                self._stats["total_relations"][q_label] += len(rels)
                self._stats["total_time_s"][q_label]    = round(
                    self._stats["total_time_s"][q_label] + q_debug["duration_s"], 3
                )
            else:
                self._stats["query_failures"][q_label] += 1

            query_debugs.append(q_debug)
            time.sleep(1)

        self._mid_cache[entity_id] = {
            "relations":   cumulative,
            "query_debug": query_debugs,
        }

        fetch_debug = {
            "entity_id": entity_id,
            "cache_hit": False,
            "pool_size": len(cumulative),
            "queries":   query_debugs,
        }
        return cumulative, fetch_debug

    # ------------------------------------------------------------------
    # Scoring

    def _score(self, labels: list[str], cand_rels: list[str]) -> dict[str, list[tuple[str, float]]]:
        if not labels or not cand_rels:
            return {label: [] for label in labels}

        emb_labels = self.model.encode(labels, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
        emb_cands  = self.model.encode(cand_rels, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)

        sim_matrix = util.cos_sim(emb_labels, emb_cands)

        result: dict[str, list[tuple[str, float]]] = {}
        for i, label in enumerate(labels):
            scored   = list(zip(cand_rels, sim_matrix[i].tolist()))
            filtered = [(rel, s) for rel, s in scored if s > self.te]
            result[label] = sorted(filtered, key=lambda x: -x[1])[: self.ke]
        return result

    # ------------------------------------------------------------------
    # BasePredicateLinker interface

    def link(self, inp: LinkingInput, entity_map: dict[str, str]) -> LinkingOutput:
        t_link_start = time.perf_counter()

        pool_key  = frozenset(entity_map.values())
        score_key = (pool_key, tuple(sorted(inp.labels)))

        # ------------------------------------------------------------------
        # Pool cache

        pool_cache_hit = pool_key in self._pool_cache

        if pool_cache_hit:
            self._stats["pool_cache_hits"] += 1
            cand_rels, pool_debug = self._pool_cache[pool_key]
            entity_fetch_debugs  = pool_debug["entity_fetches"]
            seeded_labels        = pool_debug["pool_seeded_labels"]
        else:
            self._stats["pool_cache_misses"] += 1
            cand_rels_set:       set[str] = set()
            entity_fetch_debugs: dict     = {}

            for entity_id in entity_map.values():
                rels, fetch_debug              = self._fetch_2hop_relations(entity_id)
                cand_rels_set                 |= rels
                entity_fetch_debugs[entity_id] = fetch_debug

            seeded_labels  = [l for l in inp.labels if l not in cand_rels_set]
            cand_rels_set |= set(inp.labels)
            cand_rels      = list(cand_rels_set)

            pool_debug = {
                "entity_fetches":     entity_fetch_debugs,
                "pool_seeded_labels": seeded_labels,
            }
            self._pool_cache[pool_key] = (cand_rels, pool_debug)

        # ------------------------------------------------------------------
        # Score cache

        score_cache_hit = score_key in self._score_cache

        if score_cache_hit:
            self._stats["score_cache_hits"] += 1
            scored_map     = self._score_cache[score_key]
            score_duration = 0.0
        else:
            self._stats["score_cache_misses"] += 1
            t_score        = time.perf_counter()
            scored_map     = self._score(inp.labels, cand_rels)
            score_duration = round(time.perf_counter() - t_score, 3)
            self._score_cache[score_key] = scored_map

        # ------------------------------------------------------------------
        # Assemble output

        resolved:  dict[str, list[tuple[str, float]]] = {}
        label_map: dict[str, str]                     = {}
        failed:    list[str]                          = []

        cand_rels_set_fast = set(cand_rels)

        debug: dict = {
            "pool_size":          len(cand_rels),
            "pool_cache_hit":     pool_cache_hit,
            "score_cache_hit":    score_cache_hit,
            "pool_seeded_labels": seeded_labels,
            "entity_map":         entity_map,
            "entity_fetches":     entity_fetch_debugs,
            "global_stats":       self._stats,
            "score_duration_s":   score_duration,
            "link_duration_s":    None,
            "per_label":          {},
        }

        if not cand_rels:
            for label in inp.labels:
                resolved[label] = []
                failed.append(label)
            debug["link_duration_s"] = round(time.perf_counter() - t_link_start, 3)
            return LinkingOutput(
                label_map=label_map,
                candidates=resolved,
                failed=failed,
                debug=debug,
            )

        for label in inp.labels:
            top = scored_map.get(label, [])
            debug["per_label"][label] = {
                "top_candidates": [{"id": r, "score": round(s, 6)} for r, s in top],
                "n_candidates":   len(top),
                "best_id":        top[0][0] if top else None,
                "best_score":     round(top[0][1], 6) if top else None,
                "was_seeded":     label in seeded_labels,
                "in_pool":        label in cand_rels_set_fast,
            }

            if top:
                resolved[label]  = top
                label_map[label] = top[0][0]
            else:
                resolved[label] = []
                failed.append(label)

        debug["link_duration_s"] = round(time.perf_counter() - t_link_start, 3)

        return LinkingOutput(
            label_map=label_map,
            candidates=resolved,
            failed=failed,
            debug=debug,
        )