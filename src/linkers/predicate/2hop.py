import os
import requests
from sentence_transformers import SentenceTransformer, util

from src.linkers.base import BasePredicateLinker, LinkingInput, LinkingOutput
from src.utils.retry import call_with_retry

ENDPOINT = os.environ.get("ENDPOINT_URL", "http://localhost:7001/sparql")

_HEADERS = {
    "User-Agent": "neighborhood-simcse-linker/0.1",
    "Accept":     "application/sparql-results+json",
}

_WDT_PREFIX = "http://www.wikidata.org/prop/direct/"
_WD_PREFIX  = "http://www.wikidata.org/entity/"


_SIM_THRESHOLD   = 0.01
_TOP_K           = 15

_LIMIT_1HOP      = 5_000   # r0 queries
_LIMIT_2HOP      = 10_000  # r1 queries


class Linker(BasePredicateLinker):

    def __init__(self) -> None:
        self._model = SentenceTransformer("princeton-nlp/unsup-simcse-roberta-large")
        # QID  → frozenset of neighbour PIDs  (avoids re-running 4×SPARQL)
        self._neighbor_cache: dict[str, set[str]] = {}
        # PID  → English label string
        self._label_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # SPARQL helper

    def _sparql(self, query: str) -> list[dict]:
        def _do_request():
            resp = requests.post(
                ENDPOINT,
                data={"query": query},
                headers=_HEADERS,
                timeout=60,
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
        try:
            return resp.json()["results"]["bindings"]
        except (ValueError, KeyError):
            return []

    # ------------------------------------------------------------------
    # 2-hop neighbour fetch
    #
    #   r0-in        ?x1 ?r0 entity                               (shared by q1, q2)
    #   r0-out       entity ?r0 ?x1                               (shared by q3, q4)
    #   r1 q1-in-in  ?x1 ?r0 entity  .  ?x2 ?r1 ?x1
    #   r1 q2-in-out ?x1 ?r0 entity  .  ?x1 ?r1 ?x2  x2!=entity
    #   r1 q3-out-in entity ?r0 ?x1  .  ?x2 ?r1 ?x1
    #   r1 q4-out-out entity ?r0 ?x1 .  ?x1 ?r1 ?x2  x2!=entity

    def _fetch_2hop_neighbors(self, qid: str) -> set[str]:
        if qid in self._neighbor_cache:
            return self._neighbor_cache[qid]

        ent  = f"wd:{qid}"
        wdt  = _WDT_PREFIX
        pids: set[str] = set()

        def _collect(sparql: str, var: str) -> None:
            try:
                for binding in self._sparql(sparql):
                    if var in binding:
                        pids.add(binding[var]["value"].replace(wdt, ""))
            except Exception:
                pass


        # r0-in
        _collect(f"""\
PREFIX wd:  <{_WD_PREFIX}>
SELECT DISTINCT ?r0 WHERE {{
  ?x1 ?r0 {ent} .
  FILTER(STRSTARTS(STR(?r0), "{wdt}"))
}}
LIMIT {_LIMIT_1HOP}
""", "r0")

        # r0-out
        _collect(f"""\
PREFIX wd:  <{_WD_PREFIX}>
SELECT DISTINCT ?r0 WHERE {{
  {ent} ?r0 ?x1 .
  FILTER(STRSTARTS(STR(?r0), "{wdt}"))
}}
LIMIT {_LIMIT_1HOP}
""", "r0")


        # q1 in-in: ?x2 -> ?x1 -> entity, collect the ?x2->?x1 property
        _collect(f"""\
PREFIX wd:  <{_WD_PREFIX}>
SELECT DISTINCT ?r1 WHERE {{
  ?x1 ?r0 {ent} .
  ?x2 ?r1 ?x1 .
  FILTER(STRSTARTS(STR(?r0), "{wdt}"))
  FILTER(STRSTARTS(STR(?r1), "{wdt}"))
}}
LIMIT {_LIMIT_2HOP}
""", "r1")

        # q2 in-out: entity <- ?x1 -> ?x2, collect the ?x1->?x2 property
        _collect(f"""\
PREFIX wd:  <{_WD_PREFIX}>
SELECT DISTINCT ?r1 WHERE {{
  ?x1 ?r0 {ent} .
  ?x1 ?r1 ?x2 .
  FILTER(?x2 != {ent})
  FILTER(STRSTARTS(STR(?r0), "{wdt}"))
  FILTER(STRSTARTS(STR(?r1), "{wdt}"))
}}
LIMIT {_LIMIT_2HOP}
""", "r1")

        # q3 out-in: entity -> ?x1 <- ?x2, collect the ?x2->?x1 property
        _collect(f"""\
PREFIX wd:  <{_WD_PREFIX}>
SELECT DISTINCT ?r1 WHERE {{
  {ent} ?r0 ?x1 .
  ?x2 ?r1 ?x1 .
  FILTER(STRSTARTS(STR(?r0), "{wdt}"))
  FILTER(STRSTARTS(STR(?r1), "{wdt}"))
}}
LIMIT {_LIMIT_2HOP}
""", "r1")

        # q4 out-out: entity -> ?x1 -> ?x2, collect the ?x1->?x2 property
        _collect(f"""\
PREFIX wd:  <{_WD_PREFIX}>
SELECT DISTINCT ?r1 WHERE {{
  {ent} ?r0 ?x1 .
  ?x1 ?r1 ?x2 .
  FILTER(?x2 != {ent})
  FILTER(STRSTARTS(STR(?r0), "{wdt}"))
  FILTER(STRSTARTS(STR(?r1), "{wdt}"))
}}
LIMIT {_LIMIT_2HOP}
""", "r1")

        self._neighbor_cache[qid] = pids
        return pids

    # ------------------------------------------------------------------
    # Property label fetch

    def _fetch_labels(self, pids: set[str]) -> dict[str, str]:
        unknown = pids - self._label_cache.keys()

        if unknown:
            values_clause = " ".join(f"wd:{p}" for p in unknown)
            sparql = f"""\
PREFIX wd:   <{_WD_PREFIX}>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?prop ?label WHERE {{
  VALUES ?prop {{ {values_clause} }}
  ?prop rdfs:label ?label .
  FILTER(LANG(?label) = "en")
}}
"""
            try:
                for binding in self._sparql(sparql):
                    pid   = binding["prop"]["value"].split("/")[-1]
                    label = binding["label"]["value"]
                    self._label_cache[pid] = label
            except Exception:
                pass

            # Fall back to raw PID string for any that returned no label
            for p in unknown:
                self._label_cache.setdefault(p, p)

        return {p: self._label_cache[p] for p in pids}

    # ------------------------------------------------------------------
    # link

    def link(self, inp: LinkingInput, entity_map: dict[str, str]) -> LinkingOutput:
        resolved:       dict[str, str]                     = {}
        candidates_map: dict[str, list[tuple[str, float]]] = {}
        failed:         list[str]                          = []
        debug:          dict                               = {}

        # ----------------------------------------------------------------
        # collect candidate PIDs from 2-hop neighbourhood of all

        qids      = list(entity_map.values())
        cand_pids: set[str] = set()
        for qid in qids:
            cand_pids |= self._fetch_2hop_neighbors(qid)

        if not cand_pids or not inp.labels:
            return LinkingOutput(
                label_map={},
                candidates={label: [] for label in inp.labels},
                failed=list(inp.labels),
                debug={},
            )

        # ----------------------------------------------------------------
        # resolve English labels for candidate PIDs

        pid_to_label = self._fetch_labels(cand_pids)
        pid_list     = list(cand_pids)
        label_list   = [pid_to_label[p] for p in pid_list]   # cand_rels equivalent

        # ----------------------------------------------------------------
        # SimCSE similarity

        emb_a = self._model.encode(list(inp.labels), convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
        emb_b = self._model.encode(label_list,        convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
        similarities = util.cos_sim(emb_a, emb_b)

        # ----------------------------------------------------------------
        # filter (score > 0.01) and keep top-15

        for i, pred_label in enumerate(inp.labels):
            sims = similarities[i]

            scored = [
                (pid_list[j], float(sims[j]))
                for j in range(len(pid_list))
                if float(sims[j]) > _SIM_THRESHOLD
            ]
            scored.sort(key=lambda x: -x[1])
            topk = scored[:_TOP_K]

            candidates_map[pred_label] = topk
            debug[pred_label] = {
                "num_cand_pids": len(pid_list),
                "pid_list":      pid_list,
                "label_list":    label_list,
                "scored":        topk,
            }

            if topk:
                resolved[pred_label] = topk[0][0]
            else:
                failed.append(pred_label)

        return LinkingOutput(
            label_map=resolved,
            candidates=candidates_map,
            failed=failed,
            debug=debug,
        )