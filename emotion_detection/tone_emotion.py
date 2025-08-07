try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("警告: pyaudio 未安装，实时音频功能将不可用")
    print("请运行: pip install pyaudio")

import numpy as np
import librosa
import threading
import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import queue
import logging
from typing import List, Dict, Callable, Any


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceEmotionAnalyzer:
    def __init__(self, emotion_labels: List[str] = None):
        # 情绪标签 - 与emotion_recognizer保持一致
        self.emotion_labels = emotion_labels or ["joy", "sadness", "anger", "intensity"]
        
        # 音频参数
        if PYAUDIO_AVAILABLE:
            self.FORMAT = pyaudio.paInt16
        else:
            self.FORMAT = None
        self.CHANNELS = 1
        self.RATE = 16000  # 采样率
        self.CHUNK = 1024  # 缓冲大小
        self.RECORD_SECONDS = 3  # 每次分析的音频片段长度
        self.SILENCE_THRESHOLD = 0.001  #  silence detection threshold
        
        # 特征提取和模型相关
        self.scaler = StandardScaler()
        self.model = SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
        self._is_recording = False
        self._audio_queue = queue.Queue()
        self._callback = None
        
        # 初始音频输入
        if PYAUDIO_AVAILABLE:
            self.audio = pyaudio.PyAudio()
        else:
            self.audio = None

    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """从音频数据中提取音调和音色特征"""
        # 转换为 librosa 的格式
        y = audio_data.astype(np.float32) / 32768.0  # 转换为 [-1, 1] 范围
        
        # 音调 (pitch)
        f0, _, _ = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=self.RATE
        )
        pitch_mean = np.nanmean(f0) if not np.isnan(f0).all() else 0
        pitch_std = np.nanstd(f0) if not np.isnan(f0).all() else 0
        pitch_max = np.nanmax(f0) if not np.isnan(f0).all() else 0
        pitch_min = np.nanmin(f0) if not np.isnan(f0).all() else 0
        
        # 音色 (timbre)
        # 梅尔频率倒谱系数
        mfccs = librosa.feature.mfcc(y=y, sr=self.RATE, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # 频谱质心
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.RATE).mean()
        
        # 频谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.RATE).mean()
        
        # 频谱滚降点
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.RATE).mean()
        
        # 零交叉率
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        
        # 整合所有特征
        features = np.concatenate([
            [pitch_mean, pitch_std, pitch_max, pitch_min],
            mfccs_mean,
            [spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate]
        ])
        
        return features

    def detect_emotion(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """分析音频数据&返回情绪结果 - 与emotion_recognizer格式完全一致"""
        # 提取特征
        features = self.extract_features(audio_data)
        
        # 特征标准化
        features_scaled = self.scaler.transform([features])
        
        # 预测情绪
        emotion_idx = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # 直接输出与emotion_recognizer完全一致的格式
        result = {
            "joy": float(probabilities[0]) if len(probabilities) > 0 else 0.0,
            "sadness": float(probabilities[1]) if len(probabilities) > 1 else 0.0,
            "anger": float(probabilities[2]) if len(probabilities) > 2 else 0.0,
            "intensity": float(max(probabilities)) if len(probabilities) > 0 else 0.0
        }
        
        # 确保所有值都在0-1范围内
        for key in ["joy", "sadness", "anger", "intensity"]:
            result[key] = max(0.0, min(1.0, result[key]))
        
        return result

    def analyze_audio_file(self, audio_path: str, segment_duration: int = 3) -> Dict[str, float]:
        """
        分析整段音频文件，支持长音频分段分析并取均值
        
        Args:
            audio_path: 音频文件路径
            segment_duration: 分段时长（秒），默认3秒
            
        Returns:
            整段音频的平均情绪分析结果
        """
        try:
            # 加载音频
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            # 如果音频长度合适，直接分析
            if len(audio_data) <= segment_duration * sr:
                return self.detect_emotion(audio_data)
            
            # 长音频分段分析
            segment_samples = int(segment_duration * sr)
            results = []
            
            for i in range(0, len(audio_data), segment_samples):
                segment = audio_data[i:i+segment_samples]
                # 确保分段至少有50%的长度
                if len(segment) >= segment_samples * 0.5:
                    try:
                        result = self.detect_emotion(segment)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"分段 {i//segment_samples} 分析失败: {e}")
                        continue
            
            if not results:
                logger.error("没有成功分析任何音频分段")
                return {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "intensity": 0.0}
            
            # 计算平均结果
            return self._calculate_average_emotion(results)
            
        except Exception as e:
            logger.error(f"音频文件分析失败: {e}")
            return {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "intensity": 0.0}
    
    def _calculate_average_emotion(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        计算多段音频的平均情绪结果
        
        Args:
            results: 多段音频的情绪分析结果列表
            
        Returns:
            平均情绪结果
        """
        if not results:
            return {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "intensity": 0.0}
        
        # 计算各情绪维度的平均值
        avg_joy = np.mean([r.get("joy", 0.0) for r in results])
        avg_sadness = np.mean([r.get("sadness", 0.0) for r in results])
        avg_anger = np.mean([r.get("anger", 0.0) for r in results])
        avg_intensity = np.mean([r.get("intensity", 0.0) for r in results])
        
        # 确保所有值都在0-1范围内
        result = {
            "joy": max(0.0, min(1.0, avg_joy)),
            "sadness": max(0.0, min(1.0, avg_sadness)),
            "anger": max(0.0, min(1.0, avg_anger)),
            "intensity": max(0.0, min(1.0, avg_intensity))
        }
        
        logger.info(f"整段音频分析完成，共分析 {len(results)} 个分段")
        logger.info(f"平均结果: joy={result['joy']:.2f}, sadness={result['sadness']:.2f}, anger={result['anger']:.2f}, intensity={result['intensity']:.2f}")
        
        return result

    def _audio_capture_loop(self):
        """音频捕获循环，持续从麦克风读取数据"""
        if not PYAUDIO_AVAILABLE or self.audio is None:
            logger.error("pyaudio不可用，无法进行音频捕获")
            return
            
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        logger.info("开始音频捕获...")
        
        try:
            while self._is_recording:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                self._audio_queue.put(audio_chunk)
                
        except Exception as e:
            logger.error(f"音频捕获错误: {str(e)}")
        finally:
            stream.stop_stream()
            stream.close()
            logger.info("音频捕获已停止")

    def _processing_loop(self):
        """处理循环，定期从队列中获取音频并分析"""
        logger.info("开始情绪分析处理...")
        
        while self._is_recording:
            # 收集足够时长的音频
            audio_frames = []
            start_time = time.time()
            
            while len(audio_frames) < (self.RATE // self.CHUNK) * self.RECORD_SECONDS:
                if not self._is_recording:
                    break
                try:
                    frame = self._audio_queue.get(timeout=1)
                    audio_frames.append(frame)
                    self._audio_queue.task_done()
                except queue.Empty:
                    continue
            
            if not audio_frames or not self._is_recording:
                continue
            
            # 合并音频帧
            audio_data = np.concatenate(audio_frames)
            
            # 简单的静音检测
            rms = np.sqrt(np.mean(np.square(audio_data)))
            if rms < self.SILENCE_THRESHOLD:
                logger.info("检测到静音，跳过分析")
                continue
            
            # 分析情绪
            try:
                result = self.detect_emotion(audio_data)
                logger.info(f"音调情绪分析结果: joy={result['joy']:.2f}, sadness={result['sadness']:.2f}, anger={result['anger']:.2f}, intensity={result['intensity']:.2f}")
                
                # 调用回调函数
                if self._callback:
                    self._callback(result)
            except Exception as e:
                logger.error(f"情绪分析出错: {str(e)}")
            
            # 控制分析频率
            elapsed = time.time() - start_time
            if elapsed < self.RECORD_SECONDS:
                time.sleep(self.RECORD_SECONDS - elapsed)

    def start_analysis(self, callback: Callable[[Dict[str, Any]], None] = None):
        """开始实时情绪分析"""
        if self._is_recording:
            logger.warning("已经在进行分析")
            return
            
        self._is_recording = True
        self._callback = callback
        
        # 启动音频捕获线程
        self._capture_thread = threading.Thread(target=self._audio_capture_loop)
        self._capture_thread.daemon = True
        self._capture_thread.start()
        
        # 启动处理线程
        self._processing_thread = threading.Thread(target=self._processing_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        logger.info("实时语音情绪分析已启动 (按 Ctrl+C 停止)")

    def stop_analysis(self):
        """停止实时情绪分析"""
        self._is_recording = False
        
        if hasattr(self, '_capture_thread'):
            self._capture_thread.join(timeout=2.0)
        if hasattr(self, '_processing_thread'):
            self._processing_thread.join(timeout=2.0)
            
        if PYAUDIO_AVAILABLE and self.audio is not None:
            self.audio.terminate()
        logger.info("实时语音情绪分析已停止")

    def train_with_demo_data(self):
        """使用演示数据训练模型（实际应用中应使用真实标注数据集）"""
        logger.info("使用演示数据训练模型...")
        
        # 生成演示数据（实际应用中应替换为真实的带标签音频特征）
        # 这里模拟5种情绪，每种情绪10个样本
        np.random.seed(42)
        X = []
        y = []
        
        for emotion_idx in range(len(self.emotion_labels)):
            # 为每种情绪生成特征（实际应用中应从真实音频提取）
            for _ in range(10):
                # 生成类似真实特征分布的随机数据
                pitch_features = np.random.normal(loc=emotion_idx*50, scale=30, size=4)
                mfcc_features = np.random.normal(loc=emotion_idx*0.5, scale=1.0, size=13)
                spectral_features = np.random.normal(loc=emotion_idx*100, scale=50, size=4)
                
                features = np.concatenate([pitch_features, mfcc_features, spectral_features])
                X.append(features)
                y.append(emotion_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        # 标准化特征
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # 训练模型
        self.model.fit(X_scaled, y)
        logger.info("模型训练完成（演示数据）")

if __name__ == "__main__":
    try:
        # 创建分析器实例
        analyzer = VoiceEmotionAnalyzer()
        
        # 使用演示数据训练模型（实际应用中应替换为真实训练数据）
        analyzer.train_with_demo_data()
        
        # 测试整段音频分析功能
        print("=== 测试整段音频分析功能 ===")
        
        # 生成测试音频数据（模拟）
        import numpy as np
        test_audio = np.random.randn(16000 * 5)  # 5秒的测试音频
        
        # 保存测试音频文件
        test_audio_path = None
        try:
            import soundfile as sf
            test_audio_path = "test_audio.wav"
            sf.write(test_audio_path, test_audio, 16000)
        except ImportError:
            print("警告: soundfile 未安装，跳过音频文件测试")
            print("请运行: pip install soundfile")
        else:
            # 分析整段音频
            result = analyzer.analyze_audio_file(test_audio_path)
            print(f"整段音频分析结果:")
            print(f"  喜悦 (joy): {result['joy']:.2f}")
            print(f"  悲伤 (sadness): {result['sadness']:.2f}")
            print(f"  愤怒 (anger): {result['anger']:.2f}")
            print(f"  整体强度 (intensity): {result['intensity']:.2f}")
            
            # 清理测试文件
            import os
            if os.path.exists(test_audio_path):
                os.remove(test_audio_path)
        
        print("\n=== 实时分析测试 ===")
        
        # 定义结果处理回调函数
        def handle_result(result):
            print("\n===== 音调情绪分析结果 =====")
            print(f"喜悦 (joy): {result['joy']:.2f}")
            print(f"悲伤 (sadness): {result['sadness']:.2f}")
            print(f"愤怒 (anger): {result['anger']:.2f}")
            print(f"整体强度 (intensity): {result['intensity']:.2f}")
            print("=======================\n")
        
        # 启动实时分析
        analyzer.start_analysis(handle_result)
        
        # 保持程序运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n用户中断程序")
        analyzer.stop_analysis()
    except Exception as e:
        print(f"程序出错: {str(e)}")
        if 'analyzer' in locals():
            analyzer.stop_analysis()