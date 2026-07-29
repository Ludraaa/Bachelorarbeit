from src.linkers.base import BaseEntityLinker, LinkingInput, LinkingOutput


class Linker(BaseEntityLinker):
    """
    First step of the ChatKBQA entity cascade: 
    resolve a label directly against the type label map
    """

    def link(self, inp: LinkingInput) -> LinkingOutput:
        type_map = inp.type_map or {}

        label_map = {}
        candidates = {}
        failed = []

        for label in inp.labels:
            mention = label.replace("_", " ").lower()

            if mention in type_map:
                mid = type_map[mention]
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