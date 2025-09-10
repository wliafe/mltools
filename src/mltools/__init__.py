"""
mltools.py

此模块提供了一系列机器学习相关的工具函数和类，
涵盖数据处理、模型训练辅助、数据保存与加载等功能，
旨在简化机器学习项目的开发流程。
"""

from mltools.utils import Accumulator
from mltools.learn import MachineLearning

__all__ = [
    "Accumulator",
    "MachineLearning",
]
