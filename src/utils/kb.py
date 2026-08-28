import sys
import importlib.util
from pathlib import Path


def load_kb_module(kb_name: str):
    path = Path(f"src/kb/{kb_name}.py")
    if not path.is_file():
        print(f"Error: KB module not found: {path}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location(f"src.kb.{kb_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class_name = kb_name.capitalize()
    kb_class = getattr(module, class_name, None)
    if kb_class is None:
        print(f"Error: {path} does not define a {class_name} class", file=sys.stderr)
        sys.exit(1)

    return kb_class()