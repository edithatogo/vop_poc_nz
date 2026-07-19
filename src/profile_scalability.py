"""Deprecated source-tree alias for :mod:`vop_poc_nz.profile_scalability`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.profile_scalability is deprecated; import vop_poc_nz.profile_scalability",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.profile_scalability")
