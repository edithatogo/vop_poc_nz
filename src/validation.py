"""Deprecated source-tree alias for :mod:`vop_poc_nz.validation`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.validation is deprecated; import vop_poc_nz.validation",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.validation")
