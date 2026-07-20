"""Deprecated source-tree alias for :mod:`vop_poc_nz.logging_config`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.logging_config is deprecated; import vop_poc_nz.logging_config",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.logging_config")
