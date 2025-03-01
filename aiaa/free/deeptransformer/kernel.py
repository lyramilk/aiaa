"""
核心计算函数模块

该模块提供了Transformer模型中使用的核心计算函数。
原始版本包含量化和反量化函数，现已移除，仅保留基本功能。
"""

import torch

def get_block_size():
    """
    获取默认块大小。

    Returns:
        int: 默认块大小。
    """
    return 128
