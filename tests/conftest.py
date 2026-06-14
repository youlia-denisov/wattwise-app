# conftest.py — auto-loaded by pytest before any test file.
# Puts the project root and src/ on sys.path so every test file
# can do `from src.features import ...` and `import config` without
# repeating the path-setup boilerplate.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
