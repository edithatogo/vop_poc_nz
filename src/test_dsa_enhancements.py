"""Deprecated source-tree alias for :mod:`vop_poc_nz.test_dsa_enhancements`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.test_dsa_enhancements is deprecated; import vop_poc_nz.test_dsa_enhancements",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.test_dsa_enhancements")
