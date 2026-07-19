"""Deprecated source-tree alias for :mod:`vop_poc_nz.policy_brief_generator`."""

import sys
import warnings
from importlib import import_module

warnings.warn(
    "src.policy_brief_generator is deprecated; import vop_poc_nz.policy_brief_generator",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = import_module("vop_poc_nz.policy_brief_generator")
