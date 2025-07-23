from flask import Flask, render_template, request, jsonify, session
from strategy.llm_generator import LLMGenerator
from strategy.strategy_selector import StrategySelector
import secrets
import os
import time
from werkzeug.utils import secure_filename
from speech.baidu_speech_recognizer import BaiduSpeechRecognizer
from emotion_detection.emotion_recognizer import EmotionRecognizer
from pydub import AudioSegment

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # 用于 session 管理

generator = LLMGenerator()
selector = StrategySelector()
speech_recognizer = BaiduSpeechRecognizer()
emotion_recognizer = EmotionRecognizer()

@app.route("/")
def index():
    session["history"] = []
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if not request.json:
            return jsonify({"error": "无效的请求格式"}), 400
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "消息内容不能为空"}), 400

        history = session.get("history", [])
        history.append(user_input)

        try:
            emotion_scores = emotion_recognizer.analyze_emotion_deepseek(user_input)
            emotion_intensity = emotion_scores.get("intensity", 0.5)
            liwc_score = emotion_recognizer.liwc_score(user_input)
            liwc_score = {k: float(v) for k, v in liwc_score.items()}
        except Exception as e:
            print(f"情感分析失败: {e}")
            emotion_scores = {"sadness": 0.2, "joy": 0.6, "anger": 0.1, "intensity": 0.5}
            emotion_intensity = 0.5
            liwc_score = {}

        strategy = selector.select_strategy(emotion_scores, emotion_intensity, history, liwc_score, user_input)
        reply = generator.generate_response(user_input, strategy)

        history.append(reply)
        session["history"] = history

        return jsonify({
            "reply": reply,
            "emotion": emotion_scores,
            "liwc": liwc_score
        })
    except Exception as e:
        return jsonify({"error": f"处理消息失败: {str(e)}"}), 500

@app.route("/chat_audio", methods=["POST"])
def chat_audio():
    temp_path = None  # 先定义 temp_path，避免 finally 里引用未赋值
    try:
        if 'audio' not in request.files:
            return jsonify({'error': '没有上传音频文件'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 检查文件大小（限制为10MB）
        audio_file.seek(0, 2)  # 移动到文件末尾
        file_size = audio_file.tell()
        audio_file.seek(0)  # 重置到文件开头
        print(f"收到音频请求，原始文件大小: {file_size} 字节")
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return jsonify({'error': '音频文件过大，请选择小于10MB的文件'}), 400
        
        # 使用项目下的 tmp 目录保存临时文件
        tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        filename = secure_filename(audio_file.filename or 'audio.wav')
        temp_path = os.path.join(tmp_dir, filename)
        audio_file.save(temp_path)
        print(f"保存到临时文件: {temp_path}")
        print(f"保存后文件大小: {os.path.getsize(temp_path)} 字节")
        
        # 自动转码为16kHz单声道WAV
        try:
            t1 = time.time()
            audio = AudioSegment.from_file(temp_path)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio.export(temp_path, format='wav')
            t2 = time.time()
            print(f"转码后文件大小: {os.path.getsize(temp_path)} 字节，转码耗时: {t2-t1:.2f} 秒")
        except Exception as e:
            print(f'音频转码失败: {e}')
        
        # 语音识别
        try:
            t3 = time.time()
            text = speech_recognizer.recognize_file(temp_path)
            t4 = time.time()
            print(f"识别耗时: {t4-t3:.2f} 秒，识别结果: {text}")
        except Exception as e:
            text = None
            print(f"语音识别失败: {e}")
        
        if not text:
            return jsonify({
                'error': '语音识别失败，请检查音频质量或重新录制',
                'text': '',
                'reply': '抱歉，我没有听清楚您说的话，请您重新说一遍。'
            }), 200
        
        # 情感识别
        try:
            emotion_scores = emotion_recognizer.analyze_emotion_deepseek(text)
            emotion_intensity = emotion_scores.get("intensity", 0.5) if emotion_scores else 0.5
            liwc_score = emotion_recognizer.liwc_score(text)
            liwc_score = {k: float(v) for k, v in liwc_score.items()}
        except Exception as e:
            print(f"情感分析失败: {e}")
            emotion_scores = {"sadness": 0.2, "joy": 0.6, "anger": 0.1, "intensity": 0.5}
            emotion_intensity = 0.5
            liwc_score = {}
        
        # 策略选择
        history = session.get("history", [])
        history.append(text)
        strategy = selector.select_strategy(emotion_scores, emotion_intensity, history, liwc_score, text)
        reply = generator.generate_response(text, strategy)
        history.append(reply)
        session["history"] = history
        
        return jsonify({
            'text': text,
            'reply': reply,
            'emotion': emotion_scores,
            'liwc': liwc_score
        })
        
    except Exception as e:
        return jsonify({'error': f'处理音频失败: {str(e)}'}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True)
