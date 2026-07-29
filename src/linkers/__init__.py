import importlib
from src.linkers.base import BaseEntityLinker, BasePredicateLinker, BaseExtractor


def load_entity_linker(name: str, **overrides) -> BaseEntityLinker:
    """
    Auto-discover and instantiate entity linker by name.
    Looks for src/linkers/entity/{name}.py
    Expects the file to contain a class named Linker(BaseEntityLinker).

    Any **overrides are forwarded to the Linker's constructor, letting
    callers (e.g. --linker_params on the CLI) override defaults like
    thresholds or top_k without editing the linker file itself.
    """
    try:
        module = importlib.import_module(f"src.linkers.entity.{name}")
    except ModuleNotFoundError:
        raise ValueError(
            f"Entity linker '{name}' not found. "
            f"Expected file: src/linkers/entity/{name}.py"
        )

    if not hasattr(module, "Linker"):
        raise AttributeError(
            f"src/linkers/entity/{name}.py must define a class named 'Linker' "
            f"that extends BaseEntityLinker."
        )

    try:
        linker = module.Linker(**overrides)
    except TypeError as e:
        raise TypeError(
            f"Failed to construct entity linker '{name}' with overrides "
            f"{overrides}: {e}"
        ) from e

    if not isinstance(linker, BaseEntityLinker):
        raise TypeError(
            f"Linker in src/linkers/entity/{name}.py must extend BaseEntityLinker."
        )

    return linker


def load_predicate_linker(name: str, **overrides) -> BasePredicateLinker:
    """
    Auto-discover and instantiate predicate linker by name.
    Looks for src/linkers/predicate/{name}.py
    Expects the file to contain a class named Linker(BasePredicateLinker).

    Any **overrides are forwarded to the Linker's constructor, letting
    callers (e.g. --linker_params on the CLI) override defaults like
    thresholds or top_k without editing the linker file itself.
    """
    try:
        module = importlib.import_module(f"src.linkers.predicate.{name}")
    except ModuleNotFoundError:
        raise ValueError(
            f"Predicate linker '{name}' not found. "
            f"Expected file: src/linkers/predicate/{name}.py"
        )

    if not hasattr(module, "Linker"):
        raise AttributeError(
            f"src/linkers/predicate/{name}.py must define a class named 'Linker' "
            f"that extends BasePredicateLinker."
        )

    try:
        linker = module.Linker(**overrides)
    except TypeError as e:
        raise TypeError(
            f"Failed to construct predicate linker '{name}' with overrides "
            f"{overrides}: {e}"
        ) from e

    if not isinstance(linker, BasePredicateLinker):
        raise TypeError(
            f"Linker in src/linkers/predicate/{name}.py must extend BasePredicateLinker."
        )

    return linker


def load_extractor(kb: str):
    """
    Returns the extract_from_prediction function for the given KB.
    """
    try:
        module = importlib.import_module(f"src.kb.{kb}")
    except ModuleNotFoundError:
        raise ValueError(
            f"KB module '{kb}' not found. "
            f"Expected file: src/kb/{kb}.py"
        )

    if not hasattr(module, "extract_from_prediction"):
        raise AttributeError(
            f"src/kb/{kb}.py must define extract_from_prediction(prediction: str)"
        )

    return module.extract_from_prediction


def load_substitute(kb: str):
    try:
        module = importlib.import_module(f"src.kb.{kb}")
    except ModuleNotFoundError:
        raise ValueError(f"KB module '{kb}' not found. Expected: src/kb/{kb}.py")

    if not hasattr(module, "substitute"):
        raise AttributeError(f"src/kb/{kb}.py must define substitute(prediction, entity_map, predicate_map)")

    return module.substitute