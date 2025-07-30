#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能配置文件
提供不同模式的配置选项
"""

# 性能模式配置
PERFORMANCE_MODES = {
    "fast": {
        "enable_entity_recognition": False,
        "enable_semantic_matching": False,
        "enable_rag_retrieval": True,  # RAG检索相对较快，可以保留
        "max_keywords": 2,
        "semantic_threshold": 0.8
    },
    "balanced": {
        "enable_entity_recognition": True,
        "enable_semantic_matching": False,  # 关闭语义匹配以提升性能
        "enable_rag_retrieval": True,
        "max_keywords": 3,
        "semantic_threshold": 0.75
    },
    "accurate": {
        "enable_entity_recognition": True,
        "enable_semantic_matching": True,
        "enable_rag_retrieval": True,
        "max_keywords": 5,
        "semantic_threshold": 0.7
    }
}

# 默认模式
DEFAULT_MODE = "fast"

def get_performance_config(mode: str = None):
    """获取性能配置"""
    if mode is None:
        mode = DEFAULT_MODE
    
    if mode not in PERFORMANCE_MODES:
        print(f"警告: 未知的性能模式 '{mode}'，使用默认模式 '{DEFAULT_MODE}'")
        mode = DEFAULT_MODE
    
    return PERFORMANCE_MODES[mode]

def print_performance_modes():
    """打印所有可用的性能模式"""
    print("可用的性能模式:")
    for mode, config in PERFORMANCE_MODES.items():
        print(f"  {mode}:")
        for key, value in config.items():
            print(f"    {key}: {value}")
        print() 