"""Deprecated source-tree alias for :mod:`vop_poc_nz.main`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.main is deprecated; import vop_poc_nz.main", DeprecationWarning, stacklevel=2
)
sys.modules[__name__] = import_module("vop_poc_nz.main")
