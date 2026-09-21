from src.linkers.base import BaseEntityLinker, LinkingInput, LinkingOutput


class Linker(BaseEntityLinker):
    """
    Gold entity linker.

    Matches each extracted label against the item's gold_entity_map using
    a lowercased label. If a match is found, returns exactly one candidate
    with score 1.0.
    """

    def link(self, inp: LinkingInput) -> LinkingOutput:
        gold = inp.item.get("gold_entity_map", {})

        gold_map = {
            label.lower(): uri.rsplit("/", 1)[-1]
            for uri, label in gold.items()
        }

        label_map = {}
        candidates = {}
        failed = []

        for label in inp.labels:
            mention = label.replace("_", " ").lower()

            if mention in gold_map:
                entity_id = gold_map[mention]
                label_map[label] = entity_id
                candidates[label] = [(entity_id, 1.0)]
            else:
                failed.append(label)
                candidates[label] = []

        return LinkingOutput(
            label_map=label_map,
            candidates=candidates,
            failed=failed,
            debug={},
        )