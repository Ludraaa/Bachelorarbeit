from src.linkers.base import BasePredicateLinker, LinkingInput, LinkingOutput


class Linker(BasePredicateLinker):
    """
    Gold predicate linker.

    First tries a lowercased label with underscores replaced by spaces.
    If no match is found, falls back to the lowercased label unchanged.
    """

    def link(
        self,
        inp: LinkingInput,
        entity_map: dict[str, str],
    ) -> LinkingOutput:
        gold = inp.item.get("gold_relation_map", {})

        gold_map = {
            label.lower(): uri.rsplit("/", 1)[-1]
            for uri, label in gold.items()
        }

        label_map = {}
        candidates = {}
        failed = []

        for label in inp.labels:
            normalized = label.replace("_", " ").lower()
            literal = label.lower()

            predicate_id = gold_map.get(normalized)

            if predicate_id is None:
                predicate_id = gold_map.get(literal)

            if predicate_id is not None:
                label_map[label] = predicate_id
                candidates[label] = [(predicate_id, 1.0)]
            else:
                failed.append(label)
                candidates[label] = []

        return LinkingOutput(
            label_map=label_map,
            candidates=candidates,
            failed=failed,
            debug={},
        )
