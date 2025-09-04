#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频功能回退模块
当PyAudio不可用时提供基础功能
"""

import logging
import wave
import tempfile
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class AudioFallback:
    """音频功能回退类"""
    
    def __init__(self):
        self.available = False
        self._check_availability()
    
    def _check_availability(self):
        """检查音频功能可用性"""
        try:
            import pyaudio
            self.available = True
            logger.info("✅ PyAudio可用")
        except ImportError as e:
            logger.warning(f"⚠️ PyAudio不可用: {e}")
            self.available = False
        except Exception as e:
            logger.warning(f"⚠️ PyAudio初始化失败: {e}")
            self.available = False
    
    def record_audio(self, duration: int = 5, sample_rate: int = 16000, channels: int = 1) -> Optional[bytes]:
        """录制音频（回退版本）"""
        if not self.available:
            logger.warning("⚠️ 音频录制功能不可用，请使用文本输入")
            return None
        
        try:
            import pyaudio
            
            chunk = 1024
            format = pyaudio.paInt16
            
            audio = pyaudio.PyAudio()
            stream = audio.open(
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
            audio.terminate()
            
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
    
    def get_audio_info(self) -> dict:
        """获取音频设备信息"""
        if not self.available:
            return {
                "available": False,
                "error": "PyAudio不可用"
            }
        
        try:
            import pyaudio
            audio = pyaudio.PyAudio()
            
            device_count = audio.get_device_count()
            default_device = audio.get_default_input_device_info()
            
            audio.terminate()
            
            return {
                "available": True,
                "device_count": device_count,
                "default_device": {
                    "name": default_device.get("name", "Unknown"),
                    "channels": default_device.get("maxInputChannels", 0),
                    "sample_rate": default_device.get("defaultSampleRate", 44100)
                }
            }
            
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }

# 全局实例
_audio_fallback = None

def get_audio_fallback() -> AudioFallback:
    """获取音频回退实例"""
    global _audio_fallback
    if _audio_fallback is None:
        _audio_fallback = AudioFallback()
    return _audio_fallback

def is_audio_available() -> bool:
    """检查音频功能是否可用"""
    return get_audio_fallback().available

def record_audio_safe(duration: int = 5) -> Optional[bytes]:
    """安全的音频录制"""
    return get_audio_fallback().record_audio(duration)

def save_audio_safe(audio_data: bytes, filename: str) -> bool:
    """安全的音频保存"""
    return get_audio_fallback().save_audio_to_file(audio_data, filename)

if __name__ == "__main__":
    # 测试音频回退功能
    print("🎵 测试音频回退功能")
    
    fallback = get_audio_fallback()
    info = fallback.get_audio_info()
    
    print(f"音频可用性: {'✅ 可用' if info['available'] else '❌ 不可用'}")
    
    if info['available']:
        print(f"设备数量: {info['device_count']}")
        print(f"默认设备: {info['default_device']['name']}")
        
        # 测试录制
        audio_data = fallback.record_audio(duration=2)
        if audio_data:
            print(f"✅ 录制成功，数据大小: {len(audio_data)} bytes")
        else:
            print("❌ 录制失败")
    else:
        print(f"错误: {info.get('error', '未知错误')}")


