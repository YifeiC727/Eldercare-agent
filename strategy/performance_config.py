#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance configuration file
Provides configuration options for different modes
"""

# Performance mode configuration
PERFORMANCE_MODES = {
    "fast": {
        "enable_entity_recognition": False,
        "enable_semantic_matching": False,
        "enable_rag_retrieval": True,  # RAG retrieval is relatively fast, can be kept
        "max_keywords": 2,
        "semantic_threshold": 0.8
    },
    "balanced": {
        "enable_entity_recognition": True,
        "enable_semantic_matching": False,  # Disable semantic matching to improve performance
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

# Default mode
DEFAULT_MODE = "fast"

def get_performance_config(mode: str = None):
    """Get performance configuration"""
    if mode is None:
        mode = DEFAULT_MODE
    
    if mode not in PERFORMANCE_MODES:
        print(f"Warning: Unknown performance mode '{mode}', using default mode '{DEFAULT_MODE}'")
        mode = DEFAULT_MODE
    
    return PERFORMANCE_MODES[mode]

def print_performance_modes():
    """Print all available performance modes"""
    print("Available performance modes:")
    for mode, config in PERFORMANCE_MODES.items():
        print(f"  {mode}:")
        for key, value in config.items():
            print(f"    {key}: {value}")
        print() 