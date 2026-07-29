from src.linkers.base import BaseEntityLinker, LinkingInput, LinkingOutput


class Linker(BaseEntityLinker):
    """
    Second step of the cascade (if using gold entities):
    exact (lowercased) match against this item's gold entity map
    """

    def link(self, inp: LinkingInput) -> LinkingOutput:
        gold = inp.item.get("gold_entity_map", {})
        gold_map = {label.lower(): mid.split("/")[-1] for mid, label in gold.items()}

        label_map = {}
        candidates = {}
        failed = []

        for label in inp.labels:
            mention = label.replace("_", " ").lower()

            if mention in gold_map:
                mid = gold_map[mention]
                label_map[label] = mid
                candidates[label] = [(mid, 1.0)]
            else:
                failed.append(label)
                candidates[label] = []

        return LinkingOutput(
            label_map=label_map,
            candidates=candidates,
            failed=failed,
            debug={},
        )