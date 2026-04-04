"""Pytest configuration -- ensure ``src/`` is on sys.path for imports."""

import os
import sys

# Add src/ to path so ``crusades`` package is importable without installation.
_src = os.path.join(os.path.dirname(__file__), os.pardir, "src")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))
