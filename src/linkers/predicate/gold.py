from src.linkers.base import BasePredicateLinker, LinkingInput, LinkingOutput


class Linker(BasePredicateLinker):
    """
    Gold predicate linker — uses gold_relation_map from the dataset item.
    Provides an upper bound; requires ground truth annotations.
    """

    def link(self, inp: LinkingInput, entity_map: dict[str, str]) -> LinkingOutput:
        gold = inp.item.get("gold_relation_map", {})

        # gold_relation_map: {uri -> label}, invert to {normalized_label -> PID}
        label_to_pid = {}
        for uri, label in gold.items():
            pid        = uri.split("/")[-1]          # P31
            normalized = label.replace(" ", "_")     # "instance of" -> "instance_of"
            label_to_pid[normalized] = pid

        resolved, failed = {}, []
        for label in inp.labels:
            if label in label_to_pid:
                resolved[label] = label_to_pid[label]
            else:
                failed.append(label)

        return LinkingOutput(label_map=resolved, failed=failed)