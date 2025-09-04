#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无音频模式
当PyAudio不可用时提供基础功能
"""

import logging
import os
import tempfile
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class NoAudioMode:
    """无音频模式类"""
    
    def __init__(self):
        self.available = False
        self.mode = "no_audio"
        logger.info("🔇 启用无音频模式")
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return False
    
    def get_audio_info(self) -> Dict[str, Any]:
        """获取音频信息"""
        return {
            "available": False,
            "mode": "no_audio",
            "device_count": 0,
            "default_device": None,
            "message": "音频功能不可用，请使用文本输入"
        }
    
    def record_audio(self, duration: int = 5, **kwargs) -> Optional[bytes]:
        """录制音频（无音频模式）"""
        logger.warning("⚠️ 音频录制功能不可用，请使用文本输入")
        return None
    
    def save_audio_to_file(self, audio_data: bytes, filename: str, **kwargs) -> bool:
        """保存音频到文件（无音频模式）"""
        logger.warning("⚠️ 音频保存功能不可用")
        return False
    
    def get_installation_help(self) -> str:
        """获取安装帮助"""
        return """
🔇 无音频模式说明:

当前系统运行在无音频模式下，语音识别功能不可用。

如需启用语音功能，请:
1. 解决PyAudio安装问题
2. 或使用文本输入方式与系统交互

系统其他功能（文本处理、情绪分析等）正常工作。
"""
    
    def close(self):
        """关闭资源"""
        logger.info("🔇 无音频模式已关闭")

# 全局实例
_no_audio_manager = None

def get_no_audio_manager() -> NoAudioMode:
    """获取无音频管理器实例"""
    global _no_audio_manager
    if _no_audio_manager is None:
        _no_audio_manager = NoAudioMode()
    return _no_audio_manager

def is_audio_available() -> bool:
    """检查音频是否可用"""
    return False

def record_audio_safe(duration: int = 5) -> Optional[bytes]:
    """安全的音频录制"""
    return None

def save_audio_safe(audio_data: bytes, filename: str) -> bool:
    """安全的音频保存"""
    return False

if __name__ == "__main__":
    print("🔇 无音频模式测试")
    
    manager = get_no_audio_manager()
    info = manager.get_audio_info()
    
    print(f"模式: {info['mode']}")
    print(f"可用性: {'✅ 可用' if info['available'] else '❌ 不可用'}")
    print(f"消息: {info['message']}")
    
    print(manager.get_installation_help())


