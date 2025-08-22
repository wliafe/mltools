"""
mltools.py

此模块提供了一系列机器学习相关的工具函数和类，
涵盖数据处理、模型训练辅助、数据保存与加载等功能，
旨在简化机器学习项目的开发流程。
"""

import tomli
from pathlib import Path
from mltools.draw import images
from mltools.tokenize import Tokenizer
from mltools.data import MyDataset, split_data, iter_data, download_file
from mltools.learn import MachineLearning

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 项目根目录

with open(ROOT / "pyproject.toml", "rb") as f:
    config = tomli.load(f)

__version__ = config["project"]["version"]
__all__ = [
    "images",
    "Tokenizer",
    "MyDataset",
    "split_data",
    "iter_data",
    "download_file",
    "MachineLearning",
]
