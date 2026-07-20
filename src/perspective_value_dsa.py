"""Deprecated source-tree alias for :mod:`vop_poc_nz.perspective_value_dsa`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.perspective_value_dsa is deprecated; import vop_poc_nz.perspective_value_dsa",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.perspective_value_dsa")
