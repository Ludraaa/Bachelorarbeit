import sys
import importlib.util
from pathlib import Path

def load_kb_module(kb_name: str):
    path = Path(f"src/kb/{kb_name}.py")
    if not path.is_file():
        print(f"Error: KB module not found: {path}", file=sys.stderr)
        sys.exit(1)
    spec   = importlib.util.spec_from_file_location("kb_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module