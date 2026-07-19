"""Deprecated source-tree alias for :mod:`vop_poc_nz.visualizations`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.visualizations is deprecated; import vop_poc_nz.visualizations",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.visualizations")
