"""Load model files saved under earlier internal package names, without rewriting them.

The project's Python package was renamed twice (aerostay → quanterra → quanara). Saved pipelines
(artifacts/models/*/model_*.joblib) were pickled under the package name that was current when they
were trained, so unpickling looks for classes such as `aerostay.models.NonNegative` or
`quanterra.models.SeasonalNaive`. This import hook resolves those legacy module names to the current
package and returns the same module objects, so the artifacts stay byte-identical and the classes are
shared. These two names are internal compatibility identifiers only; they never appear in the UI.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

LEGACY = ("aerostay", "quanterra")
NEW = __name__.split(".")[0]


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self) -> None:
        self._specs: dict[str, object] = {}

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in LEGACY:
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        root = spec.name.split(".")[0]
        module = importlib.import_module(NEW + spec.name[len(root):])
        self._specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module) -> None:
        # importlib stamps the alias spec on the shared module; restore the real one
        for alias, real in list(self._specs.items()):
            if real is not None and getattr(real, "name", None) == module.__name__:
                module.__spec__ = real
                del self._specs[alias]


def install() -> None:
    if not any(isinstance(f, _Alias) for f in sys.meta_path):
        sys.meta_path.insert(0, _Alias())
