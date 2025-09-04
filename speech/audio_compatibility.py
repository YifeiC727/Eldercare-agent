#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频兼容性模块
处理PyAudio和其他音频库的兼容性问题
"""

import os
import sys
import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

# 尝试导入PyAudio
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    logger.info("✅ PyAudio导入成功")
except ImportError as e:
    PYAUDIO_AVAILABLE = False
    logger.warning(f"⚠️ PyAudio导入失败: {e}")
    pyaudio = None
except Exception as e:
    PYAUDIO_AVAILABLE = False
    logger.error(f"❌ PyAudio导入错误: {e}")
    pyaudio = None

class AudioCompatibilityManager:
    """音频兼容性管理器"""
    
    def __init__(self):
        self.pyaudio_available = PYAUDIO_AVAILABLE
        self.audio_device_count = 0
        self.default_device = None
        
        if self.pyaudio_available:
            self._initialize_pyaudio()
        else:
            self._setup_fallback()
    
    def _initialize_pyaudio(self):
        """初始化PyAudio"""
        try:
            self.audio = pyaudio.PyAudio()
            self.audio_device_count = self.audio.get_device_count()
            self.default_device = self.audio.get_default_input_device_info()
            logger.info(f"✅ PyAudio初始化成功，设备数量: {self.audio_device_count}")
        except Exception as e:
            logger.error(f"❌ PyAudio初始化失败: {e}")
            self.pyaudio_available = False
            self._setup_fallback()
    
    def _setup_fallback(self):
        """设置回退方案"""
        logger.info("🔄 设置音频回退方案")
        self.audio = None
        self.audio_device_count = 0
        self.default_device = None
    
    def is_audio_available(self) -> bool:
        """检查音频是否可用"""
        return self.pyaudio_available and self.audio is not None
    
    def get_audio_info(self) -> dict:
        """获取音频设备信息"""
        if not self.is_audio_available():
            return {
                "available": False,
                "device_count": 0,
                "default_device": None,
                "error": "PyAudio不可用"
            }
        
        try:
            return {
                "available": True,
                "device_count": self.audio_device_count,
                "default_device": {
                    "name": self.default_device.get("name", "Unknown"),
                    "channels": self.default_device.get("maxInputChannels", 0),
                    "sample_rate": self.default_device.get("defaultSampleRate", 44100)
                } if self.default_device else None
            }
        except Exception as e:
            return {
                "available": False,
                "device_count": 0,
                "default_device": None,
                "error": str(e)
            }
    
    def record_audio(self, duration: int = 5, sample_rate: int = 16000, channels: int = 1) -> Optional[bytes]:
        """录制音频"""
        if not self.is_audio_available():
            logger.warning("⚠️ 音频不可用，无法录制")
            return None
        
        try:
            chunk = 1024
            format = pyaudio.paInt16
            
            stream = self.audio.open(
                format=format,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk
            )
            
            logger.info(f"🎤 开始录制音频，时长: {duration}秒")
            frames = []
            
            for i in range(0, int(sample_rate / chunk * duration)):
                data = stream.read(chunk)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            # 合并音频数据
            audio_data = b''.join(frames)
            logger.info(f"✅ 音频录制完成，数据大小: {len(audio_data)} bytes")
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ 音频录制失败: {e}")
            return None
    
    def save_audio_to_file(self, audio_data: bytes, filename: str, sample_rate: int = 16000, channels: int = 1) -> bool:
        """保存音频到文件"""
        if not audio_data:
            logger.warning("⚠️ 没有音频数据可保存")
            return False
        
        try:
            import wave
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            
            logger.info(f"✅ 音频保存成功: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 音频保存失败: {e}")
            return False
    
    def get_installation_help(self) -> str:
        """获取安装帮助信息"""
        system = sys.platform.lower()
        
        if system == "darwin":  # macOS
            return """
🔧 macOS PyAudio安装帮助:

1. 安装PortAudio:
   brew install portaudio

2. 重新安装PyAudio:
   pip uninstall pyaudio
   pip install pyaudio

3. 如果仍有问题，尝试:
   pip install --upgrade pip
   pip install pyaudio --no-cache-dir
"""
        elif system == "linux":
            return """
🔧 Linux PyAudio安装帮助:

1. Ubuntu/Debian:
   sudo apt-get install portaudio19-dev python3-pyaudio
   pip install pyaudio

2. CentOS/RHEL:
   sudo yum install portaudio-devel
   pip install pyaudio

3. 如果仍有问题:
   pip install --upgrade pip
   pip install pyaudio --no-cache-dir
"""
        elif system == "win32":
            return """
🔧 Windows PyAudio安装帮助:

1. 下载预编译版本:
   pip install pipwin
   pipwin install pyaudio

2. 或者从官方源安装:
   pip install pyaudio

3. 如果仍有问题，尝试:
   pip install --upgrade pip
   pip install pyaudio --no-cache-dir
"""
        else:
            return """
🔧 PyAudio安装帮助:

1. 确保安装了音频开发库
2. 重新安装PyAudio:
   pip uninstall pyaudio
   pip install pyaudio

3. 如果仍有问题:
   pip install --upgrade pip
   pip install pyaudio --no-cache-dir
"""
    
    def close(self):
        """关闭音频资源"""
        if self.audio:
            try:
                self.audio.terminate()
                logger.info("🔌 PyAudio资源已释放")
            except Exception as e:
                logger.warning(f"⚠️ 释放PyAudio资源时出错: {e}")

# 全局实例
_audio_manager = None

def get_audio_manager() -> AudioCompatibilityManager:
    """获取全局音频管理器实例"""
    global _audio_manager
    if _audio_manager is None:
        _audio_manager = AudioCompatibilityManager()
    return _audio_manager

def check_audio_system():
    """检查音频系统状态"""
    manager = get_audio_manager()
    info = manager.get_audio_info()
    
    print("🎵 音频系统检查:")
    print(f"  可用性: {'✅ 可用' if info['available'] else '❌ 不可用'}")
    print(f"  设备数量: {info['device_count']}")
    
    if info['default_device']:
        device = info['default_device']
        print(f"  默认设备: {device['name']}")
        print(f"  声道数: {device['channels']}")
        print(f"  采样率: {device['sample_rate']}")
    
    if not info['available']:
        print(f"  错误: {info.get('error', '未知错误')}")
        print(manager.get_installation_help())
    
    return info['available']

if __name__ == "__main__":
    # 测试音频兼容性
    print("🎵 音频兼容性测试")
    print("=" * 50)
    
    available = check_audio_system()
    
    if available:
        print("\n🧪 测试音频录制...")
        manager = get_audio_manager()
        
        # 录制3秒音频
        audio_data = manager.record_audio(duration=3)
        
        if audio_data:
            print(f"✅ 音频录制成功，数据大小: {len(audio_data)} bytes")
            
            # 保存到文件
            if manager.save_audio_to_file(audio_data, "test_audio.wav"):
                print("✅ 音频保存成功")
            else:
                print("❌ 音频保存失败")
        else:
            print("❌ 音频录制失败")
    
    manager.close()


