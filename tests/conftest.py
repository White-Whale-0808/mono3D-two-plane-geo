"""Shared test setup.

Puts the project root on sys.path so `libs.*` imports work no matter where
pytest is invoked from, and runs setup_env() first — the repo-wide rule is
that it precedes any C-extension import, and a test module may pull one in
transitively.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.env_setup import setup_env  # noqa: E402

setup_env()
