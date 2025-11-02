"""
Project description.
"""

import importlib_metadata

__author__ = "Jérémie Gince"
__email__ = "gincejeremie@gmail.com"
__copyright__ = "Copyright 2025, Jérémie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/MatchCake/MatchCake-Opt"
__package__ = "matchcake_opt"
__version__ = importlib_metadata.version(__package__)

import warnings

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")

from .datamodules import DataModule
from .datasets import get_dataset_cls_by_name
