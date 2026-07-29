from src.linkers.base import BaseEntityLinker, LinkingInput, LinkingOutput


class Linker(BaseEntityLinker):
    """
    Gold entity linker — uses gold_entity_map from the dataset item.
    Provides an upper bound; requires ground truth annotations.

    Resolution order (mirrors ChatKBQA's denormalize_s_expr_new priority):
      1. type_map  — global type label map, if present in LinkingInput
      2. gold_entity_map — per-item gold annotations
    """

    def link(self, inp: LinkingInput) -> LinkingOutput:
        type_map = inp.type_map  # { "uk constituent country": "m.0hzjlmp" }

        gold = inp.item.get("gold_entity_map", {})
        label_to_qid: dict[str, str] = {}
        for uri, label in gold.items():
            qid        = uri.split("/")[-1]
            normalized = label.lower().replace(" ", "_")
            label_to_qid[normalized] = qid

        resolved: dict[str, str] = {}
        failed:   list[str]      = []

        for label in inp.labels:
            mention = label.replace("_", " ").lower()   # "UK_constituent_country" → "uk constituent country"

            # 1. Type map (highest priority)
            if mention in type_map:
                resolved[label] = type_map[mention]
                continue

            # 2. Gold entity map
            if label.lower() in label_to_qid:
                resolved[label] = label_to_qid[label.lower()]
            else:
                failed.append(label)

        return LinkingOutput(label_map=resolved, failed=failed)