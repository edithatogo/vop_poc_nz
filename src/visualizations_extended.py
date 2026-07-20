"""Deprecated source-tree alias for :mod:`vop_poc_nz.visualizations_extended`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.visualizations_extended is deprecated; import vop_poc_nz.visualizations_extended",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.visualizations_extended")
