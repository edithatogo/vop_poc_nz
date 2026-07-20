"""Deprecated source-tree alias for :mod:`vop_poc_nz.cea_model_core`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.cea_model_core is deprecated; import vop_poc_nz.cea_model_core",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.cea_model_core")
