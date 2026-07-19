"""Deprecated source-tree alias for :mod:`vop_poc_nz.value_of_information`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.value_of_information is deprecated; import vop_poc_nz.value_of_information",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.value_of_information")
