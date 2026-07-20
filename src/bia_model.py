"""Deprecated source-tree alias for :mod:`vop_poc_nz.bia_model`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.bia_model is deprecated; import vop_poc_nz.bia_model",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.bia_model")
