"""Deprecated source-tree compatibility wrapper for the profiling script.

This file deliberately retains its own module identity because pytest discovers
its ``test_`` filename. Replacing ``sys.modules`` here would make collection
observe the canonical module's different ``__file__`` and fail before tests run.
"""

import warnings
from importlib import import_module

warnings.warn(
    "src.test_dsa_enhancements is deprecated; import vop_poc_nz.test_dsa_enhancements",
    DeprecationWarning,
    stacklevel=2,
)
_canonical = import_module("vop_poc_nz.test_dsa_enhancements")
for _name, _value in vars(_canonical).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

__test__ = False
