from flask import Flask, render_template, request, jsonify, url_for, session, redirect
from strategy.llm_generator import LLMGenerator
from strategy.strategy_selector import StrategySelector
from user_bio.user_info_manager import UserInfoManager
from bson import ObjectId
import secrets
import os
import time
import csv
import numpy as np
from werkzeug.utils import secure_filename
from speech.baidu_speech_recognizer import BaiduSpeechRecognizer
from emotion_detection.emotion_recognizer import EmotionRecognizer

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available, audio processing will be limited")

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

generator = LLMGenerator()
selector = StrategySelector()
speech_recognizer = BaiduSpeechRecognizer()
emotion_recognizer = EmotionRecognizer()
user_info_manager = UserInfoManager()

# 延迟初始化的tone_emotion分析器
tone_analyzer = None

def get_tone_analyzer():
    """延迟初始化tone_emotion分析器"""
    global tone_analyzer
    if tone_analyzer is None:
        try:
            from emotion_detection.tone_emotion import VoiceEmotionAnalyzer
            tone_analyzer = VoiceEmotionAnalyzer()
            tone_analyzer.train_with_demo_data()
            print("✅ Tone emotion analyzer initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize tone emotion analyzer: {e}")
            tone_analyzer = None
    return tone_analyzer

def get_dynamic_weights(text_length: int, audio_quality: float = 0.5, emotion_type: str = None):
    """
    根据具体情况动态调整权重
    
    Args:
        text_length: 文本长度
        audio_quality: 音频质量评分 (0-1)，默认0.5
        emotion_type: 情绪类型（可选）
    """
    base_text_weight = 0.7
    base_tone_weight = 0.3
    
    # 根据文本长度调整
    if text_length < 10:  # 短文本，增加音调权重
        text_weight = base_text_weight - 0.2
        tone_weight = base_tone_weight + 0.2
    elif text_length > 50:  # 长文本，增加文本权重
        text_weight = base_text_weight + 0.1
        tone_weight = base_tone_weight - 0.1
    else:
        text_weight = base_text_weight
        tone_weight = base_tone_weight
    
    # 根据音频质量调整
    if audio_quality < 0.5:  # 低质量音频，增加文本权重
        text_weight += 0.1
        tone_weight -= 0.1
    elif audio_quality > 0.8:  # 高质量音频，增加音调权重
        text_weight -= 0.1
        tone_weight += 0.1
    
    # 根据情绪类型调整
    if emotion_type in ['anger', 'excitement']:  # 音调明显的情绪
        text_weight -= 0.1
        tone_weight += 0.1
    
    # 确保权重在合理范围内
    text_weight = max(0.1, min(0.9, text_weight))
    tone_weight = max(0.1, min(0.9, tone_weight))
    
    return text_weight, tone_weight

def combine_emotions(text_emotion: dict, tone_emotion: dict, text_weight: float, tone_weight: float) -> dict:
    """
    结合文本情绪和音调情绪
    
    Args:
        text_emotion: 文本情绪分析结果
        tone_emotion: 音调情绪分析结果
        text_weight: 文本情绪权重
        tone_weight: 音调情绪权重
    
    Returns:
        结合后的情绪结果
    """
    if not tone_emotion:
        return text_emotion
    
    combined = {}
    for key in ["joy", "sadness", "anger", "intensity"]:
        text_val = text_emotion.get(key, 0.0)
        tone_val = tone_emotion.get(key, 0.0)
        
        # 加权平均
        combined[key] = (text_val * text_weight + tone_val * tone_weight) / (text_weight + tone_weight)
        combined[key] = max(0.0, min(1.0, combined[key]))
    
    return combined

def estimate_audio_quality(audio_path: str) -> float:
    """
    估算音频质量评分 (0-1)
    
    Args:
        audio_path: 音频文件路径
    
    Returns:
        音频质量评分
    """
    try:
        import librosa
        audio_data, sr = librosa.load(audio_path, sr=16000)
        
        # 计算音频质量指标
        # 1. 信噪比（简化版）
        rms = np.sqrt(np.mean(np.square(audio_data)))
        snr_score = min(1.0, rms * 10)  # 简化的信噪比评分
        
        # 2. 音频长度
        duration = len(audio_data) / sr
        duration_score = min(1.0, duration / 10)  # 10秒为满分
        
        # 3. 频谱能量分布
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr).mean()
        spectral_score = min(1.0, spectral_centroid / 2000)  # 2000Hz为满分
        
        # 综合评分
        quality_score = (snr_score * 0.4 + duration_score * 0.3 + spectral_score * 0.3)
        
        return max(0.1, min(1.0, quality_score))
        
    except Exception as e:
        print(f"音频质量评估失败: {e}")
        return 0.5  # 默认中等质量

# 用户情绪历史数据存储 (内存中，生产环境建议使用数据库)
user_emotion_history = {}

# 用户预警历史数据存储 (内存中，生产环境建议使用数据库)
user_warning_history = {}

@app.route("/")
def index():
    # 检查用户是否已登录
    if "user_id" in session:
        return render_template("index.html")
    else:
        return render_template("login.html")

@app.route("/register")
def register():
    return render_template("minimal_registration.html")

@app.route("/chat")
def chat_page():
    if "user_id" not in session:
        return redirect("/")
    return render_template("index.html")

@app.route("/api/users", methods=["POST"])
def create_user():
    """创建新用户"""
    data = request.json
    user_ip = request.remote_addr
    
    # 验证必填字段
    if not all(key in data for key in ["name", "age", "gender", "password"]):
        return jsonify({"error": "请填写所有必填信息"}), 400
    
    # 创建用户
    user_id = user_info_manager.create_user(data, user_ip)
    session["user_id"] = user_id
    session["history"] = []
    
    # 初始化用户情绪历史
    user_emotion_history[user_id] = []
    user_warning_history[user_id] = []
    
    return jsonify({
        "status": "success",
        "user_id": user_id,
        "message": "用户创建成功"
    }), 201

@app.route("/api/login", methods=["POST"])
def login():
    """用户登录"""
    data = request.json
    
    if not all(key in data for key in ["username", "password"]):
        return jsonify({"error": "请输入用户名和密码"}), 400
    
    # 验证用户
    user = user_info_manager.authenticate_user(data["username"], data["password"])
    
    if user:
        user_id = user["_id"]
        session["user_id"] = user_id
        session["history"] = []
        
        # 初始化用户情绪历史（如果不存在）
        if user_id not in user_emotion_history:
            user_emotion_history[user_id] = []
        if user_id not in user_warning_history:
            user_warning_history[user_id] = []
        
        return jsonify({
            "status": "success",
            "message": "登录成功"
        }), 200
    else:
        return jsonify({"error": "用户名或密码错误"}), 401

@app.route("/api/logout")
def logout():
    """用户登出"""
    session.clear()
    return jsonify({"status": "success", "message": "已登出"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if not request.json:
            return jsonify({"error": "无效的请求格式"}), 400
        
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "消息内容不能为空"}), 400

        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "用户未登录"}), 401

        history = session.get("history", [])
        # 确保history中的所有元素都是字符串
        if not isinstance(history, list):
            history = []
        # 过滤掉非字符串元素
        history = [item for item in history if isinstance(item, str)]
        history.append(user_input)

        # 情绪分析
        try:
            print(f"\n=== 情绪分析开始 ===")
            print(f"用户输入: '{user_input}'")
            
            emotion_scores = emotion_recognizer.analyze_emotion_deepseek(user_input)
            print(f"🧠 情绪检测结果: {emotion_scores}")
            
            emotion_intensity = emotion_scores.get("intensity", 0.5)
            print(f"📊 情绪强度: {emotion_intensity}")
            
            liwc_score = emotion_recognizer.liwc_score(user_input)
            liwc_score = {k: float(v) for k, v in liwc_score.items()}
            print(f"🔍 LIWC分析结果: {liwc_score}")
            print(f"=== 情绪分析结束 ===\n")
            
            # 写入情绪趋势数据
            user_info_manager.save_emotion_data(user_id, emotion_scores)
            
            # 更新用户情绪历史数据
            if user_id not in user_emotion_history:
                user_emotion_history[user_id] = []
            
            # 添加当前情绪数据到历史记录
            current_sadness = emotion_scores.get("sadness", 0.0)
            user_emotion_history[user_id].append(current_sadness)
            
            # 保持最近20次对话的情绪数据
            if len(user_emotion_history[user_id]) > 20:
                user_emotion_history[user_id] = user_emotion_history[user_id][-20:]
            
            # 更新长期悲伤日志
            selector.log_long_term_sadness(current_sadness)
            
        except Exception as e:
            print(f"情感分析失败: {e}")
            emotion_scores = {"sadness": 0.2, "joy": 0.6, "anger": 0.1, "intensity": 0.5}
            emotion_intensity = 0.5
            liwc_score = {}
            current_sadness = 0.2

        # 获取用户信息
        user_info = user_info_manager.get_user(user_id)
        print(f"DEBUG: user_id = {user_id}")
        print(f"DEBUG: user_info = {user_info}")
        
        # 获取用户情绪历史窗口数据
        window_sadness_scores = user_emotion_history.get(user_id, [])
        
        # 策略选择（增加window_sadness_scores参数）
        print(f"=== 策略选择开始 ===")
        strategy = selector.select_strategy(
            emotion_scores, emotion_intensity, history, liwc_score, user_input, window_sadness_scores, user_info
        )
        print(f"🎯 选择的策略: {strategy.get('matched_rule', 'Unknown')}")
        print(f"💬 引导语: {strategy.get('引导语', 'N/A')}")
        
        # 检查关键词预警结果
        keyword_warning_result = selector.check_critical_keywords(user_input)
        print(f"⚠️ 关键词预警: {keyword_warning_result}")
        
        # 检查早期预警结果
        early_warning_result = selector.check_early_warning(
            window_sadness_scores, 
            emotion_scores.get("sadness", 0.0), 
            liwc_score
        )
        print(f"🚨 早期预警: {early_warning_result}")
        print(f"=== 策略选择结束 ===\n")
        
        # 记录预警历史（用户隔离）
        from datetime import datetime
        current_time = datetime.now()
        
        # 如果触发关键词预警，记录到用户预警历史
        if keyword_warning_result["triggered"]:
            warning_record = {
                "timestamp": current_time,
                "type": "keyword_warning",
                "level": keyword_warning_result["level"],
                "reason": keyword_warning_result["reason"],
                "keywords": keyword_warning_result.get("keywords", []),
                "user_input": user_input
            }
            user_warning_history[user_id].append(warning_record)
        
        # 如果触发早期预警，记录到用户预警历史
        if early_warning_result["triggered"]:
            warning_record = {
                "timestamp": current_time,
                "type": "early_warning",
                "level": early_warning_result["level"],
                "reason": early_warning_result["reason"],
                "user_input": user_input
            }
            user_warning_history[user_id].append(warning_record)
        
        # 根据预警级别调整响应策略
        if keyword_warning_result["triggered"]:
            # 关键词预警：最高优先级，直接使用策略选择器返回的结果
            strategy["keyword_warning"] = keyword_warning_result
        elif early_warning_result["triggered"]:
            warning_level = early_warning_result["level"]
            warning_reason = early_warning_result["reason"]
            
            # 根据预警级别选择相应的响应策略
            if warning_level == "severe":
                # 严重预警：紧急关切
                strategy["引导语"] = f"我注意到{warning_reason}，这让我非常担心你的状态。你愿意和我详细聊聊吗？如果需要的话，我建议我们可以联系专业的心理支持资源。"
                strategy["语气"] = "紧急关切"
                strategy["目标"] = "立即情绪干预，建议转介专业支持"
                strategy["early_warning"] = {
                    "level": warning_level,
                    "reason": warning_reason,
                    "action": "立即关注，建议人工介入"
                }
            elif warning_level == "moderate":
                # 中等预警：关切引导
                strategy["引导语"] = f"我注意到{warning_reason}，你最近是不是遇到了一些困难？愿意和我聊聊吗？我会一直陪着你。"
                strategy["语气"] = "关切引导"
                strategy["目标"] = "主动关怀，预防情绪恶化"
                strategy["early_warning"] = {
                    "level": warning_level,
                    "reason": warning_reason,
                    "action": "需要持续关注，建议增加关怀频率"
                }
            elif warning_level == "mild":
                # 轻微预警：温和关怀
                strategy["引导语"] = f"我注意到{warning_reason}，你最近心情怎么样？有什么想和我分享的吗？"
                strategy["语气"] = "温和关怀"
                strategy["目标"] = "增加关怀频率，预防问题发展"
                strategy["early_warning"] = {
                    "level": warning_level,
                    "reason": warning_reason,
                    "action": "长期情绪偏低，建议定期关怀"
                }
        
        # 生成回复
        reply = generator.generate_response(user_input, strategy)
        
        # 检查是否需要询问用户信息
        next_question = strategy.get("next_question")
        if next_question:
            reply = user_info_manager.integrate_question_naturally(reply, next_question, user_input)
        
        # 建议特殊情况下填写问卷
        if strategy.get("recommend_gds", False):
            reply += "\n📝 建议你填写一个简短的自评问卷（GDS），这有助于我们更好地了解你的情绪状态。"

        # 保存对话到数据库
        user_info_manager.save_conversation(user_id, user_input, reply, emotion_scores)
        
        # 确保reply是字符串类型
        if isinstance(reply, str):
            history.append(reply)
        session["history"] = history

        # 返回响应，包含预警信息
        response_data = {
            "reply": reply,
            "emotion": emotion_scores,
            "liwc": liwc_score,
            "next_question": next_question
        }
        
        # 如果有关键词预警，添加到响应中
        if keyword_warning_result["triggered"]:
            response_data["keyword_warning"] = keyword_warning_result
            response_data["show_alert"] = True
            response_data["alert_message"] = f"⚠️ 检测到危险关键词！\n{keyword_warning_result['reason']}\n建议立即人工介入。"
        
        # 如果有早期预警，添加到响应中
        elif early_warning_result["triggered"]:
            response_data["early_warning"] = early_warning_result
            response_data["show_alert"] = True
            response_data["alert_message"] = f"⚠️ 情绪预警！\n{early_warning_result['reason']}\n{early_warning_result['suggested_action']}"

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": f"处理消息失败: {str(e)}"}), 500

@app.route("/chat_audio", methods=["POST"])
def chat_audio():
    temp_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': '没有上传音频文件'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 检查文件大小
        audio_file.seek(0, 2)
        file_size = audio_file.tell()
        audio_file.seek(0)
        
        if file_size > 10 * 1024 * 1024:
            return jsonify({'error': '音频文件过大，请选择小于10MB的文件'}), 400
        
        # 保存临时文件
        tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        filename = secure_filename(audio_file.filename or 'audio.wav')
        temp_path = os.path.join(tmp_dir, filename)
        audio_file.save(temp_path)
        
        # 音频转码
        try:
            audio = AudioSegment.from_file(temp_path)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio.export(temp_path, format='wav')
        except Exception as e:
            print(f'音频转码失败: {e}')
        
        # 语音识别
        try:
            text = speech_recognizer.recognize_file(temp_path)
        except Exception as e:
            text = None
            print(f"语音识别失败: {e}")
        
        if not text:
            return jsonify({
                'error': '语音识别失败，请检查音频质量或重新录制',
                'text': '',
                'reply': '抱歉，我没有听清楚您说的话，请您重新说一遍。'
            }), 200
        
        # ===== 新增：音调情绪分析 =====
        tone_emotion_result = None
        try:
            analyzer = get_tone_analyzer()  # 延迟初始化
            if analyzer:
                tone_emotion_result = analyzer.analyze_audio_file(temp_path)
                print(f"🎵 音调情绪分析结果: {tone_emotion_result}")
            else:
                print("⚠️ Tone emotion analyzer not available")
        except Exception as e:
            print(f"❌ 音调情绪分析失败: {e}")
            tone_emotion_result = None
        
        # ===== 文本情绪分析 =====
        try:
            text_emotion = emotion_recognizer.analyze_emotion_deepseek(text)
            print(f"📝 文本情绪分析结果: {text_emotion}")
            
            # ===== 动态权重计算 =====
            text_length = len(text)
            audio_quality = estimate_audio_quality(temp_path)
            
            # 根据文本情绪判断情绪类型
            emotion_type = None
            if text_emotion.get("anger", 0) > 0.6:
                emotion_type = "anger"
            elif text_emotion.get("joy", 0) > 0.6:
                emotion_type = "excitement"
            
            text_weight, tone_weight = get_dynamic_weights(text_length, audio_quality, emotion_type)
            print(f"⚖️ 动态权重 - 文本: {text_weight:.2f}, 音调: {tone_weight:.2f}")
            
            # ===== 结合文本和音调情绪 =====
            if tone_emotion_result:
                combined_emotion = combine_emotions(text_emotion, tone_emotion_result, text_weight, tone_weight)
                emotion_scores = combined_emotion
                print(f"🎯 结合后情绪结果: {emotion_scores}")
            else:
                emotion_scores = text_emotion
                print(f"📝 仅使用文本情绪: {emotion_scores}")
            
            emotion_intensity = emotion_scores.get("intensity", 0.5)
            liwc_score = emotion_recognizer.liwc_score(text)
            liwc_score = {k: float(v) for k, v in liwc_score.items()}
            
            # 保存情绪数据
            user_id = session.get("user_id")
            if user_id:
                user_info_manager.save_emotion_data(user_id, emotion_scores)
            
        except Exception as e:
            print(f"❌ 情感分析失败: {e}")
            emotion_scores = {"sadness": 0.2, "joy": 0.6, "anger": 0.1, "intensity": 0.5}
            emotion_intensity = 0.5
            liwc_score = {}
        
        # 策略选择 - 使用优化后的对话历史
        user_id = session.get("user_id")
        if user_id:
            # 从数据库获取最近的对话历史，避免session中的历史信息过载
            recent_conversation_text = user_info_manager.get_recent_conversation_text(user_id, limit=5)
            # 将当前用户输入添加到历史中
            history = [recent_conversation_text, text] if recent_conversation_text else [text]
        else:
            # 如果没有用户ID，使用session中的历史（向后兼容）
            history = session.get("history", [])
        if not isinstance(history, list):
            history = []
        history = [item for item in history if isinstance(item, str)]
        history.append(text)
        
        user_info = user_info_manager.get_user(user_id) if user_id else None
        
        strategy = selector.select_strategy(
            emotion_scores, emotion_intensity, history, liwc_score, text, user_info
        )
        reply = generator.generate_response(text, strategy)
        
        # 检查是否需要询问用户信息
        next_question = strategy.get("next_question")
        if next_question:
            reply = user_info_manager.integrate_question_naturally(reply, next_question, text)
        
        # 保存对话记录到数据库
        if user_id:
            user_info_manager.save_conversation(user_id, text, reply, emotion_scores)
        
        # 更新session中的历史（保持向后兼容）
        if isinstance(reply, str):
            session_history = session.get("history", [])
            if not isinstance(session_history, list):
                session_history = []
            session_history.append(text)
            session_history.append(reply)
            # 限制session历史长度，避免内存占用过大
            session_history = session_history[-10:]  # 只保留最近10轮对话
            session["history"] = session_history
        
        return jsonify({
            'text': text,
            'reply': reply,
            'emotion': emotion_scores,
            'liwc': liwc_score,
            'tone_emotion': tone_emotion_result,  # 新增：音调情绪结果
            'text_emotion': text_emotion,  # 新增：原始文本情绪
            'weights': {  # 新增：使用的权重信息
                'text_weight': text_weight,
                'tone_weight': tone_weight,
                'audio_quality': audio_quality,
                'text_length': text_length
            },
            'next_question': next_question
        })
        
    except Exception as e:
        return jsonify({'error': f'处理音频失败: {str(e)}'}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route("/api/users/<user_id>/info", methods=["PUT"])
def update_user_info(user_id):
    """更新用户信息（处理问题回答和手动编辑）"""
    try:
        data = request.json
        
        # 检查是否是手动编辑模式
        if "field" in data and "value" in data:
            # 手动编辑模式
            field = data.get("field")
            value = data.get("value")
            
            # 映射字段到问题键
            field_to_question_map = {
                "basic_info.name": "name",
                "basic_info.age": "age", 
                "basic_info.gender": "gender",
                "family_relation.children_count": "children_count",
                "family_relation.spouse_status": "spouse_status",
                "family_relation.living_alone": "living_alone",
                "living_habits.hobbies": "hobbies",
                "living_habits.health_status": "health_status"
            }
            
            question_key = field_to_question_map.get(field)
            if not question_key:
                return jsonify({"error": "无效的字段"}), 400
            
            # 创建更新数据
            update_data = {
                "question_key": question_key,
                "answer": value
            }
            
            success = user_info_manager.update_user_info(user_id, update_data)
        else:
            # 原有的问题回答模式
            success = user_info_manager.update_user_info(user_id, data)
        
        if success:
            return jsonify({"status": "success", "message": "信息更新成功"}), 200
        else:
            return jsonify({"error": "用户不存在或更新失败"}), 400
            
    except Exception as e:
        return jsonify({"error": f"更新失败: {str(e)}"}), 500

@app.route("/api/users/<user_id>/bio", methods=["GET"])
def get_user_bio(user_id):
    """获取用户详细信息"""
    try:
        user_info = user_info_manager.get_user(user_id)
        if user_info:
            return jsonify({"status": "success", "user": user_info}), 200
        else:
            return jsonify({"error": "用户不存在"}), 404
            
    except Exception as e:
        return jsonify({"error": f"获取失败: {str(e)}"}), 500

@app.route("/trend")
def trend():
    # 检查用户是否已登录
    user_id = session.get("user_id")
    if not user_id:
        return redirect("/")
    return render_template("trend_chart.html")

@app.route("/api/trend_data")
def trend_data():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "用户未登录"}), 401
    
    from visualization.trend_plotter import get_emotion_trend
    trend_data = get_emotion_trend(user_id)
    return jsonify(trend_data)

@app.route("/api/chat_history")
def get_chat_history():
    """获取用户对话历史"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "用户未登录"}), 401
    
    history = session.get("history", [])
    return jsonify({"history": history})

@app.route("/api/warning_data")
def warning_data():
    """获取用户预警历史数据"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "用户未登录"}), 401
    
    # 获取用户预警历史
    warnings = user_warning_history.get(user_id, [])
    
    # 按日期分组统计
    from collections import defaultdict
    from datetime import datetime
    
    daily_warnings = defaultdict(lambda: {"keyword_warnings": 0, "early_warnings": 0, "total": 0})
    
    for warning in warnings:
        if isinstance(warning, dict) and "timestamp" in warning:
            if isinstance(warning["timestamp"], datetime):
                date_str = warning["timestamp"].strftime("%Y-%m-%d")
            else:
                # 如果是字符串，尝试解析
                try:
                    date_obj = datetime.fromisoformat(warning["timestamp"].replace('Z', '+00:00'))
                    date_str = date_obj.strftime("%Y-%m-%d")
                except:
                    continue
        else:
            continue
        
        daily_warnings[date_str]["total"] += 1
        
        if warning.get("type") == "keyword_warning":
            daily_warnings[date_str]["keyword_warnings"] += 1
        elif warning.get("type") == "early_warning":
            daily_warnings[date_str]["early_warnings"] += 1
    
    # 转换为列表格式
    warning_data = []
    for date, counts in sorted(daily_warnings.items()):
        warning_data.append({
            "date": date,
            "keyword_warnings": counts["keyword_warnings"],
            "early_warnings": counts["early_warnings"],
            "total": counts["total"]
        })
    
    return jsonify({
        "warning_data": warning_data,
        "total_warnings": len(warnings)
    })

@app.route("/user_bio")
def user_bio_page():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "用户未登录"}), 401
    
    return render_template("user_bio.html", user_id=user_id)

@app.route("/questionnaire", methods=["GET", "POST"])
def questionnaire():
    if request.method == "POST":
        answers = request.form.to_dict()

        # 如果有人没填完会报错
        if len(answers) < 15 or any(v == '' for v in answers.values()):
            return render_template("questionnaire.html", error_message="⚠️ 请回答完所有题目再提交哦～")
        
        score = sum(int(value) for value in answers.values())

        # 假设你已经登录并有 user_id 存在 session 中
        user_id = session.get('user_id', 'anonymous')

        # 保存到 session，方便跳转后使用
        session['gds_answers'] = answers
        session['gds_score'] = score
        session['gds_user'] = user_id

        return redirect(url_for('gds_result'))
    
    return render_template("questionnaire.html")

@app.route("/gds_result")
def gds_result():
    score = session.get("gds_score", 0)
    answers = session.get("gds_answers", {})
    user_id = session.get("gds_user", "anonymous")

    # 保存结果
    file_path = 'user_bio/data/gds_results.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            header = ['username'] + [f'q{i}' for i in range(1, 16)] + ['score']
            writer.writerow(header)

        row = [user_id] + [answers.get(f'q{i}', '') for i in range(1, 16)] + [score]
        writer.writerow(row)

    # 根据得分生成文字描述
    score = int(score)
    if score <= 4:
        description = "状态良好 😊"
    elif score <= 8:
        description = "有轻度抑郁倾向 😐"
    else:
        description = "可能存在明显抑郁，建议进一步评估 😟"

    return render_template("gds_result.html", score=score, description=description)

if __name__ == "__main__":
    app.run(debug=True, port=5001)