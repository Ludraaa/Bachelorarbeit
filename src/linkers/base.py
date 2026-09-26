from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExtractionResult:
    """
    Stores a list of entity and predicate mentions extracted from the prediction beam.
    """
    entity_labels:    list[str] = field(default_factory=list)
    predicate_labels: list[str] = field(default_factory=list)


@dataclass
class LinkingInput:
    """
    Base linking input. Linkers get access to a list of mentions to resolve, the english 
    question associated with the dataset item, the prediction beam itself, as well as
    the whole dataset item (for gold linkers). Additionally, a global type map is
    provided if the specific KB makes use of it.
    """
    labels:     list[str]
    question:   str
    prediction: str
    item:       dict
    # Optional inverted type label map: { label.lower() -> mid }
    type_map:   dict[str, str] = field(default_factory=dict)


@dataclass
class LinkingOutput:
    """
    Base linking output. Linkers return a map with the best possible candidate
    for each input label. They also return a full list of scored candidates per
    label, while labels for which no candidate was found are returned in a separate list.
    """
    label_map:   dict[str, str] #top1
    candidates:  dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    failed:      list[str] = field(default_factory=list)
    debug:       dict[str, list[dict]] = field(default_factory=dict)


class BaseEntityLinker(ABC):
    """
    Abstract entity linker. Defines the link function which takes the base linking in- and
    output, along with a function to get a dictionary of the exact hyperparameter setup
    the linker uses.
    """
    @abstractmethod
    def link(self, inp: LinkingInput) -> LinkingOutput:
        pass

    def get_params(self) -> dict:
        return {}


class BasePredicateLinker(ABC):
    """
    Abstract predicate linker. Identical to entity linkers with one small difference.
    The predicate linking step also receives the current entity permutation produced by
    the entity linkers. This is used for linkers that require entity context, like neighborhood.
    """
    @abstractmethod
    def link(self, inp: LinkingInput, entity_map: dict[str, str]) -> LinkingOutput:
        pass

    def get_params(self) -> dict:
        return {}