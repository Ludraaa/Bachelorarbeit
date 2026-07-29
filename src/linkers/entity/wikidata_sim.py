from src.linkers.base import BaseEntityLinker, LinkingInput, LinkingOutput
from sentence_transformers import SentenceTransformer, util
import requests
import re


class Linker(BaseEntityLinker):
    def __init__(self, ke: int = 5, te: float = 0.5, search_limit: int = 20):
        self.ke = ke
        self.te = te
        self.search_limit = search_limit
        self.model = SentenceTransformer("princeton-nlp/sup-simcse-roberta-base")

    # ---------------------------------------------------------------------------
    # Candidate retrieval

    def _fetch_candidates(self, label: str) -> list[dict]:
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
        )

        if not resp.ok or not resp.text.strip():
            print(f"[wikidata_sim] HTTP {resp.status_code} for '{label}': {resp.text[:200]!r}")
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

        # penalize noisy entity types
        #if any(x in desc for x in ["song", "album", "recording", "band"]):
        #    score -= 0.10

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

        for c in candidates:
            desc = c.get("description", "") or ""

            cand_strings.append(
                f"{c.get('label', '')} | {desc}"
            )

            lexical_scores.append(
                self._lexical_boost(mention_clean, c, desc)
            )

        embs = self.model.encode(
            [query] + cand_strings,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        semantic_scores = util.cos_sim(embs[0], embs[1:])[0].tolist()

        results = []
        for c, sem_s, lex_s in zip(candidates, semantic_scores, lexical_scores):

            # hybrid scoring 
            score = (0.65 * sem_s) + (0.35 * lex_s)

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