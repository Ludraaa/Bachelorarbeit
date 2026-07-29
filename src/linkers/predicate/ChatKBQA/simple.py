from src.linkers.base import BasePredicateLinker, LinkingInput, LinkingOutput


class Linker(BasePredicateLinker):
    """
    Freebase predicate linker:

    Predicates extracted from the predictions are already in
    their canonical fbp: form (e.g. fbp:location.country.languages_spoken)

    Each label is returned as its own top-1 candidate with confidence 1.0.
    """

    def __init__(self):
        pass


    def link(self, inp: LinkingInput, entity_map: dict[str, str]) -> LinkingOutput:
        resolved: dict[str, list[tuple[str, float]]] = {}
        label_map: dict[str, str] = {}
        failed: list[str] = []
        debug: dict = {}

        for label in inp.labels:
            candidates = [(label, 1.0)]
            resolved[label] = candidates
            label_map[label] = label
            debug[label] = [{"id": label, "label": label, "score": 1.0}]

        return LinkingOutput(
            label_map=label_map,
            candidates=resolved,
            failed=failed,
            debug=debug,
        )