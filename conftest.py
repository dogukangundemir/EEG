"""pytest configuration: make src/ importable regardless of the invocation directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
