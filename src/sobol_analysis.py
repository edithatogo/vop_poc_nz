"""Deprecated source-tree alias for :mod:`vop_poc_nz.sobol_analysis`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.sobol_analysis is deprecated; import vop_poc_nz.sobol_analysis",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.sobol_analysis")
