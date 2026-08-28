import re

import requests
from sentence_transformers import SentenceTransformer, util

from src.linkers.base import BaseEntityLinker, LinkingInput, LinkingOutput
from src.utils.retry import call_with_retry


class Linker(BaseEntityLinker):
    """
    Wikidata entity linker: candidate generation via the public
    wbsearchentities API, reranked with a SimCSE cross-encoder-style
    lexical+semantic hybrid score.

    No Freebase counterpart to diff against -- this is the Wikidata-only
    analogue of the type_map -> gold -> SimCSE -> FACC1 cascade's
    "generic surface-form fallback" step (FACC1's role on the Freebase
    side). FACC1 itself is not portable: it depends on a Freebase-specific
    prebuilt mention index (entity_list_file/surface_map_file), which has
    no Wikidata equivalent, so this hits the live API instead of a local
    index.
    """

    def __init__(
        self,
        ke: int = 50,
        te: float = 0.01,
        search_limit: int = 50,
        timeout: int = 30,
        semantic_weight: float = 0.55,
        lexical_weight: float = 0.30,
        rank_weight: float = 0.15,
    ):
        self.ke = ke
        self.te = te
        self.search_limit = search_limit
        self.timeout = timeout
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.rank_weight = rank_weight
        self.model_name = "princeton-nlp/sup-simcse-roberta-base"
        self.model = SentenceTransformer(self.model_name)

    def get_params(self) -> dict:
        return {
            "ke": self.ke,
            "te": self.te,
            "search_limit": self.search_limit,
            "semantic_weight": self.semantic_weight,
            "lexical_weight": self.lexical_weight,
            "rank_weight": self.rank_weight,
            "model_name": self.model_name,
        }

    # ---------------------------------------------------------------------------
    # Candidate retrieval

    def _fetch_candidates(self, label: str) -> list[dict]:
        """
        NB divergence from the original draft: wrapped in call_with_retry
        (matching the retry pattern used by the label_search and
        neighborhood predicate linkers) since this hits a live external
        API on every uncached label -- a bare requests.get() here means a
        single transient network hiccup fails the whole label instead of
        retrying, unlike every other network-calling linker in the
        pipeline.
        """

        def _do_request():
            resp = requests.get(
                "https://www.wikidata.org/w/api.php",
                headers={
                    "User-Agent": "ChatKBQA-research/0.1 (bachelor thesis; contact: luis.drayer@web.de)"
                },
                params={
                    "action": "wbsearchentities",
                    "search": label.replace("_", " "),
                    "language": "en",
                    "limit": self.search_limit,
                    "format": "json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp

        try:
            resp = call_with_retry(
                _do_request,
                retries=3,
                base_delay=2.0,
                backoff=2.0,
                exceptions=(requests.RequestException,),
            )
        except requests.RequestException as exc:
            print(f"[wikidata_sim] request failed for '{label}': {exc}")
            return []

        if resp is None:
            print(f"[wikidata_sim] giving up on '{label}' after retries")
            return []

        if not resp.text.strip():
            print(f"[wikidata_sim] empty response body for '{label}'")
            return []

        return resp.json().get("search", [])

    # ---------------------------------------------------------------------------
    # Scoring

    def _lexical_boost(self, mention: str, cand: dict, desc: str) -> float:
        m = mention.lower()
        label = cand.get("label", "").lower()
        desc = desc.lower()

        score = 0.0

        # exact match (strong but not deterministic)
        if m == label:
            score += 0.25

        # substring match
        elif m in label or label in m:
            score += 0.12

        # alias-ish match
        elif re.sub(r"\s+", "", m) == re.sub(r"\s+", "", label):
            score += 0.15

        return score

    def _simi_entities(
        self,
        question: str,
        mention: str,
        candidates: list[dict],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        mention_clean = mention.replace("_", " ").strip().lower()
        question_clean = question.strip()

        query = f"{question_clean} [MENTION] {mention_clean}"

        cand_strings = []
        lexical_scores = []
        rank_scores = []

        for idx, c in enumerate(candidates):
            desc = c.get("description", "") or ""

            cand_strings.append(
                f"{c.get('label', '')} | {desc}"
            )

            lexical_scores.append(
                self._lexical_boost(mention_clean, c, desc)
            )

            # wbsearchentities' own result order is a real relevance/
            # popularity signal (roughly: incoming-link / sitelink weight)
            # that the semantic+lexical terms alone can't recover once
            # several candidates share an identical label -- e.g. "Jamaica"
            # the country vs. "Jamaica" the 1957 musical both get the same
            # exact-match lexical boost and near-identical SimCSE scores
            # against a short description. Rather than discard that
            # ordering, fold it in as a decaying prior.
            rank_scores.append(1.0 / (idx + 1))

        embs = self.model.encode(
            [query] + cand_strings,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        semantic_scores = util.cos_sim(embs[0], embs[1:])[0].tolist()

        results = []
        for c, sem_s, lex_s, rank_s in zip(
            candidates, semantic_scores, lexical_scores, rank_scores
        ):

            # hybrid scoring
            score = (
                (self.semantic_weight * sem_s)
                + (self.lexical_weight * lex_s)
                + (self.rank_weight * rank_s)
            )

            results.append((c["id"], score))

        return results

    # ---------------------------------------------------------------------------
    # Selection

    def _top_k_with_threshold(self, scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
        filtered = [(qid, s) for qid, s in scored if s >= self.te]

        if not filtered:
            filtered = scored  # fallback safety

        return sorted(filtered, key=lambda x: -x[1])[: self.ke]

    # ---------------------------------------------------------------------------
    # Main linking loop

    def link(self, inp: LinkingInput) -> LinkingOutput:
        resolved, candidates_map, failed = {}, {}, []

        for label in inp.labels:
            raw = self._fetch_candidates(label)

            scored = self._simi_entities(
                inp.question,
                label,
                raw,
            )

            candidates = self._top_k_with_threshold(scored)

            candidates_map[label] = candidates

            if candidates:
                resolved[label] = candidates[0][0]
            else:
                failed.append(label)

        return LinkingOutput(
            label_map=resolved,
            candidates=candidates_map,
            failed=failed,
        )