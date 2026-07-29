from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExtractionResult:
    entity_labels:    list[str] = field(default_factory=list)
    predicate_labels: list[str] = field(default_factory=list)


@dataclass
class LinkingInput:
    labels:     list[str]
    question:   str
    prediction: str
    item:       dict
    # Optional inverted type label map: { label.lower() -> mid }
    # Populated from <dataset>_<split>_type_label_map.json when available.
    type_map:   dict[str, str] = field(default_factory=dict)


@dataclass
class LinkingOutput:
    label_map:   dict[str, str] #top1
    candidates:  dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    failed:      list[str] = field(default_factory=list)
    debug:       dict[str, list[dict]] = field(default_factory=dict)

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, prediction: str) -> ExtractionResult:
        pass


class BaseEntityLinker(ABC):
    @abstractmethod
    def link(self, inp: LinkingInput) -> LinkingOutput:
        pass

    def get_params(self) -> dict:
        return {}


class BasePredicateLinker(ABC):
    @abstractmethod
    def link(self, inp: LinkingInput, entity_map: dict[str, str]) -> LinkingOutput:
        pass

    def get_params(self) -> dict:
        return {}