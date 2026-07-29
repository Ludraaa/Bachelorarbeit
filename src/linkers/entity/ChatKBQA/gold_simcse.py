from sentence_transformers import SentenceTransformer, util

from src.linkers.base import BaseEntityLinker, LinkingInput, LinkingOutput


class Linker(BaseEntityLinker):
    """
    Third step of the cascade (if using gold entities): 
    SimCSE similarity against the gold entity map's labels, for mentions the exact-match step missed
    """

    def __init__(
        self,
        model_name: str = "princeton-nlp/unsup-simcse-roberta-large",
        gold_threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.gold_threshold = gold_threshold

    def get_params(self) -> dict:
        return {"gold_threshold": self.gold_threshold, "model_name": self.model_name}

    def link(self, inp: LinkingInput) -> LinkingOutput:
        gold = inp.item.get("gold_entity_map", {})
        gold_map = {label.lower(): mid.split("/")[-1] for mid, label in gold.items()}

        if not gold_map:
            return LinkingOutput(
                label_map={},
                candidates={label: [] for label in inp.labels},
                failed=list(inp.labels),
                debug={},
            )

        gold_labels = list(gold_map.keys())
        gold_embeddings = self.model.encode(
            gold_labels, convert_to_tensor=True, normalize_embeddings=True,
        )

        label_map = {}
        candidates = {}
        failed = []

        for label in inp.labels:
            mention = label.replace("_", " ").lower()

            mention_embedding = self.model.encode(
                [mention], convert_to_tensor=True, normalize_embeddings=True,
            )
            scores = util.cos_sim(mention_embedding[0], gold_embeddings)[0]
            best_idx = scores.argmax().item()
            best_score = scores[best_idx].item()

            if best_score > self.gold_threshold:
                mid = gold_map[gold_labels[best_idx]]
                label_map[label] = mid
                candidates[label] = [(mid, best_score)]
            else:
                failed.append(label)
                candidates[label] = []

        return LinkingOutput(
            label_map=label_map,
            candidates=candidates,
            failed=failed,
            debug={},
        )