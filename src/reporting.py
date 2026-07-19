"""Deprecated source-tree alias for :mod:`vop_poc_nz.reporting`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.reporting is deprecated; import vop_poc_nz.reporting",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.reporting")
