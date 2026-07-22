"""Import smoke tests.

Every module must import cleanly (or fail only with ImportError for optional
dependencies) on every supported narwhals version. Regression guard for the
narwhals 1.x incident where ``IntoSeries | None`` in a runtime-evaluated
signature raised TypeError at import time: narwhals 1.x exposes
``narwhals.typing`` aliases as plain strings at runtime, so any module using
them in annotations must carry ``from __future__ import annotations``.
"""

import importlib
import pkgutil

import centimators


def test_top_level_import():
    assert hasattr(centimators, "RankTransformer")


def test_all_modules_import():
    prefix = centimators.__name__ + "."
    for mod_info in pkgutil.walk_packages(centimators.__path__, prefix):
        try:
            importlib.import_module(mod_info.name)
        except ImportError:
            # Optional dependencies (jax, dspy, keras, umap) may be absent;
            # anything else (e.g. TypeError from annotation evaluation) fails.
            continue
