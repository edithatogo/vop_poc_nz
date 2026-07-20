"""Deprecated source-tree alias for :mod:`vop_poc_nz.main_analysis`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.main_analysis is deprecated; import vop_poc_nz.main_analysis",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.main_analysis")
