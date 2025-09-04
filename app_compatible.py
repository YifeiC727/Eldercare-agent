#!/usr/bin/env python3
"""
兼容版养老护理系统 - 渐进式启用高级功能
"""

from flask import Flask, render_template, request, jsonify, url_for, session, redirect
import secrets
import os
import time
import csv
import json
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from bson import ObjectId

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ 环境变量已加载")
except ImportError:
    print("⚠️ python-dotenv未安装，跳过环境变量加载")

# 渐进式导入高级功能模块
ADVANCED_FEATURES = {
    'strategy_selector': False,
    'llm_generator': False,
    'emotion_recognizer': False,
    'user_info_manager': False,
    'speech_recognizer': False
}

# 尝试导入策略选择器
try:
    from strategy.llm_generator import LLMGenerator
    from strategy.strategy_selector import StrategySelector
    ADVANCED_FEATURES['strategy_selector'] = True
    ADVANCED_FEATURES['llm_generator'] = True
    print("✅ 策略选择器模块导入成功")
except Exception as e:
    print(f"⚠️ 策略选择器模块导入失败: {e}")

# 尝试导入情感识别器
try:
    from emotion_detection.emotion_recognizer import EmotionRecognizer
    ADVANCED_FEATURES['emotion_recognizer'] = True
    print("✅ 情感识别器模块导入成功")
except Exception as e:
    print(f"⚠️ 情感识别器模块导入失败: {e}")

# 尝试导入用户信息管理器
try:
    from user_bio.improved_user_info_manager import ImprovedUserInfoManager
    ADVANCED_FEATURES['user_info_manager'] = True
    print("✅ 用户信息管理器模块导入成功")
except Exception as e:
    print(f"⚠️ 用户信息管理器模块导入失败: {e}")

# 尝试导入语音识别器
try:
    from speech.baidu_speech_recognizer import BaiduSpeechRecognizer
    ADVANCED_FEATURES['speech_recognizer'] = True
    print("✅ 语音识别器模块导入成功")
except Exception as e:
    print(f"⚠️ 语音识别器模块导入失败: {e}")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available, audio processing will be limited")

# 自定义JSON编码器处理ObjectId
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.json_encoder = CustomJSONEncoder

# 确保JSON响应正确处理中文字符
app.config['JSON_AS_ASCII'] = False

# 初始化核心组件
generator = None
selector = None
emotion_recognizer = None
user_info_manager = None
speech_recognizer = None

if ADVANCED_FEATURES['llm_generator']:
    try:
        generator = LLMGenerator()
        print("✅ LLM生成器初始化成功")
    except Exception as e:
        print(f"⚠️ LLM生成器初始化失败: {e}")

if ADVANCED_FEATURES['strategy_selector']:
    try:
        selector = StrategySelector()
        print("✅ 策略选择器初始化成功")
    except Exception as e:
        print(f"⚠️ 策略选择器初始化失败: {e}")

if ADVANCED_FEATURES['emotion_recognizer']:
    try:
        emotion_recognizer = EmotionRecognizer()
        print("✅ 情感识别器初始化成功")
    except Exception as e:
        print(f"⚠️ 情感识别器初始化失败: {e}")

if ADVANCED_FEATURES['user_info_manager']:
    try:
        user_info_manager = ImprovedUserInfoManager()
        print("✅ 用户信息管理器初始化成功")
    except Exception as e:
        print(f"⚠️ 用户信息管理器初始化失败: {e}")

if ADVANCED_FEATURES['speech_recognizer']:
    try:
        speech_recognizer = BaiduSpeechRecognizer()
        print("✅ 语音识别器初始化成功")
    except Exception as e:
        print(f"⚠️ 语音识别器初始化失败: {e}")

# 简单的用户数据存储（降级方案）
USERS_FILE = 'users.json'

def load_users():
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"加载用户数据失败: {e}")
        return {}

def save_users(users):
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存用户数据失败: {e}")
        return False

def hash_password(password):
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

# User emotion history data storage (in memory, recommend using database in production)
user_emotion_history = {}

# User warning history data storage (in memory, recommend using database in production)
user_warning_history = {}

@app.route("/")
def index():
    # Check if user is logged in
    if "user_id" in session:
        return render_template("index.html")
    else:
        return render_template("login.html")

@app.route('/health')
def health_check():
    """健康检查端点"""
    try:
        # 检查数据库连接
        db_status = "healthy"
        try:
            if user_info_manager and hasattr(user_info_manager, 'data_manager'):
                stats = user_info_manager.data_manager.get_stats()
                db_status = "healthy" if stats else "unhealthy"
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
        
        # 检查核心组件
        components = {
            "database": db_status,
            "emotion_recognizer": "healthy" if emotion_recognizer else "unhealthy",
            "strategy_selector": "healthy" if selector else "unhealthy",
            "llm_generator": "healthy" if generator else "unhealthy",
            "speech_recognizer": "healthy" if speech_recognizer else "unavailable",
            "advanced_features": ADVANCED_FEATURES
        }
        
        # 总体状态
        overall_status = "healthy" if all(
            status in ["healthy", "unavailable"] for status in components.values() if isinstance(status, str)
        ) else "unhealthy"
        
        return jsonify({
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": components,
            "version": "2.0.0-compatible"
        }), 200 if overall_status == "healthy" else 503
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

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
    """Create new user"""
    data = request.json
    user_ip = request.remote_addr
    
    # Validate required fields
    if not all(key in data for key in ["name", "age", "gender", "password"]):
        return jsonify({"error": "Please fill in all required information"}), 400
    
    # Create user
    if user_info_manager:
        try:
            user_id = user_info_manager.create_user(data, user_ip)
        except Exception as e:
            print(f"高级用户创建失败，降级到简单模式: {e}")
            user_id = create_simple_user(data)
    else:
        user_id = create_simple_user(data)
    
    # 确保user_id是字符串类型，避免ObjectId序列化问题
    session["user_id"] = str(user_id)
    session["history"] = []
    
    # Initialize user emotion history
    user_emotion_history[user_id] = []
    user_warning_history[user_id] = []
    
    return jsonify({
        "status": "success",
        "user_id": user_id,
        "message": "User created successfully"
    }), 201

def create_simple_user(data):
    """创建简单用户（降级方案）"""
    users = load_users()
    user_id = str(len(users) + 1)
    
    new_user = {
        'id': user_id,
        'name': data['name'],
        'age': data['age'],
        'gender': data['gender'],
        'password': hash_password(data['password']),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'last_login': None
    }
    
    users[user_id] = new_user
    save_users(users)
    return user_id

@app.route("/api/login", methods=["POST"])
def login():
    """User login"""
    data = request.json
    
    if not all(key in data for key in ["username", "password"]):
        return jsonify({"error": "Please enter username and password"}), 400
    
    # Verify user
    if user_info_manager:
        try:
            user = user_info_manager.authenticate_user(data["username"], data["password"])
        except Exception as e:
            print(f"高级用户验证失败，降级到简单模式: {e}")
            user = authenticate_simple_user(data["username"], data["password"])
    else:
        user = authenticate_simple_user(data["username"], data["password"])
    
    if user:
        user_id = user["_id"] if isinstance(user, dict) and "_id" in user else user.get("id", "unknown")
        # 确保user_id是字符串类型，避免ObjectId序列化问题
        session["user_id"] = str(user_id)
        session["history"] = []
        
        # Initialize user emotion history (if not exists)
        if user_id not in user_emotion_history:
            user_emotion_history[user_id] = []
        if user_id not in user_warning_history:
            user_warning_history[user_id] = []
        
        return jsonify({
            "status": "success",
            "message": "Login successful"
        }), 200
    else:
        return jsonify({"error": "Username or password incorrect"}), 401

def authenticate_simple_user(username, password):
    """简单用户验证（降级方案）"""
    users = load_users()
    password_hash = hash_password(password)
    
    for user_id, user_data in users.items():
        if user_data['name'] == username and user_data['password'] == password_hash:
            return {"_id": user_id, "name": user_data['name']}
    return None

@app.route("/api/logout")
def logout():
    """User logout"""
    session.clear()
    return jsonify({"status": "success", "message": "Logged out"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if not request.json:
            return jsonify({"error": "Invalid request format"}), 400
        
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "Message content cannot be empty"}), 400

        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "User not logged in"}), 401

        history = session.get("history", [])
        # Ensure all elements in history are strings
        if not isinstance(history, list):
            history = []
        # Filter out non-string elements
        history = [item for item in history if isinstance(item, str)]
        history.append(user_input)

        # Emotion analysis
        try:
            print(f"\n=== Emotion Analysis Started ===")
            print(f"User input: '{user_input}'")
            
            if emotion_recognizer:
                emotion_scores = emotion_recognizer.analyze_emotion_deepseek(user_input)
                print(f"🧠 Emotion detection results: {emotion_scores}")
                
                emotion_intensity = emotion_scores.get("intensity", 0.5)
                print(f"📊 Emotion intensity: {emotion_intensity}")
                
                liwc_score = emotion_recognizer.liwc_score(user_input)
                liwc_score = {k: float(v) for k, v in liwc_score.items()}
                print(f"🔍 LIWC analysis results: {liwc_score}")
            else:
                # 降级到简单情感分析
                emotion_scores = analyze_emotion_simple(user_input)
                emotion_intensity = emotion_scores.get("intensity", 0.5)
                liwc_score = {}
                print("⚠️ 使用简单情感分析")
            
            print(f"=== Emotion Analysis Ended ===\n")
            
            # Write emotion trend data
            if user_info_manager:
                try:
                    user_info_manager.save_emotion_data(user_id, emotion_scores)
                except Exception as e:
                    print(f"保存情感数据失败: {e}")
            
            # Update user emotion history data
            if user_id not in user_emotion_history:
                user_emotion_history[user_id] = []
            
            # Add current emotion data to history
            current_sadness = emotion_scores.get("sadness", 0.0)
            user_emotion_history[user_id].append(current_sadness)
            
            # Keep recent 20 conversations' emotion data
            if len(user_emotion_history[user_id]) > 20:
                user_emotion_history[user_id] = user_emotion_history[user_id][-20:]
            
            # Update long-term sadness log
            if selector:
                try:
                    selector.log_long_term_sadness(current_sadness)
                except Exception as e:
                    print(f"更新长期悲伤记录失败: {e}")
            
        except Exception as e:
            print(f"情感分析失败: {e}")
            emotion_scores = {"sadness": 0.2, "joy": 0.6, "anger": 0.1, "intensity": 0.5}
            emotion_intensity = 0.5
            liwc_score = {}
            current_sadness = 0.2

        # Get user information
        user_info = None
        if user_info_manager:
            try:
                user_info = user_info_manager.get_user(user_id)
            except Exception as e:
                print(f"获取用户信息失败: {e}")
        
        print(f"DEBUG: user_id = {user_id}")
        print(f"DEBUG: user_info = {user_info}")
        
        # Get user emotion history window data
        window_sadness_scores = user_emotion_history.get(user_id, [])
        
        # Strategy selection (add window_sadness_scores parameter)
        print(f"=== 策略选择开始 ===")
        if selector:
            try:
                strategy = selector.select_strategy(
                    emotion_scores, emotion_intensity, history, liwc_score, user_input, window_sadness_scores, user_info
                )
                print(f"🎯 选择的策略: {strategy.get('matched_rule', 'Unknown')}")
                print(f"💬 引导语: {strategy.get('引导语', 'N/A')}")
                
                # Check keyword warning results
                keyword_warning_result = selector.check_critical_keywords(user_input)
                print(f"⚠️ 关键词预警: {keyword_warning_result}")
                
                # Check early warning results
                early_warning_result = selector.check_early_warning(
                    window_sadness_scores, 
                    emotion_scores.get("sadness", 0.0), 
                    liwc_score
                )
                print(f"🚨 早期预警: {early_warning_result}")
            except Exception as e:
                print(f"策略选择失败，降级到简单模式: {e}")
                strategy = create_simple_strategy(emotion_scores, user_input)
                keyword_warning_result = {"triggered": False}
                early_warning_result = {"triggered": False}
        else:
            # 降级到简单策略
            strategy = create_simple_strategy(emotion_scores, user_input)
            keyword_warning_result = {"triggered": False}
            early_warning_result = {"triggered": False}
            print("⚠️ 使用简单策略选择")
        
        print(f"=== 策略选择结束 ===\n")
        
        # Record warning history (user isolation)
        from datetime import datetime
        current_time = datetime.now()
        
        # If keyword warning is triggered, record to user warning history
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
        
        # If early warning is triggered, record to user warning history
        if early_warning_result["triggered"]:
            warning_record = {
                "timestamp": current_time,
                "type": "early_warning",
                "level": early_warning_result["level"],
                "reason": early_warning_result["reason"],
                "user_input": user_input
            }
            user_warning_history[user_id].append(warning_record)
        
        # Adjust response strategy based on warning level
        if keyword_warning_result["triggered"]:
            # Keyword warning: highest priority, directly use strategy selector's result
            strategy["keyword_warning"] = keyword_warning_result
        elif early_warning_result["triggered"]:
            warning_level = early_warning_result["level"]
            warning_reason = early_warning_result["reason"]
            
            # Select corresponding response strategy based on warning level
            if warning_level == "severe":
                # Severe warning: urgent concern
                strategy["引导语"] = f"我注意到{warning_reason}，这让我非常担心你的状态。你愿意和我详细聊聊吗？如果需要的话，我建议我们可以联系专业的心理支持资源。"
                strategy["语气"] = "紧急关切"
                strategy["目标"] = "立即情绪干预，建议转介专业支持"
                strategy["early_warning"] = {
                    "level": warning_level,
                    "reason": warning_reason,
                    "action": "立即关注，建议人工介入"
                }
            elif warning_level == "moderate":
                # Moderate warning: caring guidance
                strategy["引导语"] = f"我注意到{warning_reason}，你最近是不是遇到了一些困难？愿意和我聊聊吗？我会一直陪着你。"
                strategy["语气"] = "关切引导"
                strategy["目标"] = "主动关怀，预防情绪恶化"
                strategy["early_warning"] = {
                    "level": warning_level,
                    "reason": warning_reason,
                    "action": "需要持续关注，建议增加关怀频率"
                }
            elif warning_level == "mild":
                # Mild warning: gentle care
                strategy["引导语"] = f"我注意到{warning_reason}，你最近心情怎么样？有什么想和我分享的吗？"
                strategy["语气"] = "温和关怀"
                strategy["目标"] = "增加关怀频率，预防问题发展"
                strategy["early_warning"] = {
                    "level": warning_level,
                    "reason": warning_reason,
                    "action": "长期情绪偏低，建议定期关怀"
                }
        
        # Generate reply
        print(f"🔧 开始生成回复...")
        print(f"🔧 策略信息: {strategy}")
        if generator:
            try:
                reply = generator.generate_response(user_input, strategy)
            except Exception as e:
                print(f"LLM生成失败，降级到简单回复: {e}")
                reply = strategy.get("引导语", "我理解您的感受，请告诉我更多。")
        else:
            # 降级到简单回复
            reply = strategy.get("引导语", "我理解您的感受，请告诉我更多。")
        print(f"🔧 生成的回复: {reply}")
        
        # Check if need to ask user information
        next_question = strategy.get("next_question")
        if next_question and user_info_manager:
            try:
                reply = user_info_manager.integrate_question_naturally(reply, next_question, user_input)
            except Exception as e:
                print(f"整合问题失败: {e}")
        
        # Suggest questionnaire in special cases
        if strategy.get("recommend_gds", False):
            reply += "\n📝 建议你填写一个简短的自评问卷（GDS），这有助于我们更好地了解你的情绪状态。"

        # Save conversation to database
        if user_info_manager:
            try:
                user_info_manager.save_conversation(user_id, user_input, reply, emotion_scores)
            except Exception as e:
                print(f"保存对话失败: {e}")
        
        # Ensure reply is string type
        if isinstance(reply, str):
            history.append(reply)
        session["history"] = history

        # Return response with warning information
        response_data = {
            "reply": reply,
            "emotion": emotion_scores,
            "liwc": liwc_score,
            "next_question": next_question
        }
        
        # If there's keyword warning, add to response
        if keyword_warning_result["triggered"]:
            response_data["keyword_warning"] = keyword_warning_result
            response_data["show_alert"] = True
            response_data["alert_message"] = f"⚠️ 检测到危险关键词！\n{keyword_warning_result['reason']}\n建议立即人工介入。"
        
        # If there's early warning, add to response
        elif early_warning_result["triggered"]:
            response_data["early_warning"] = early_warning_result
            response_data["show_alert"] = True
            response_data["alert_message"] = f"⚠️ 情绪预警！\n{early_warning_result['reason']}\n{early_warning_result['suggested_action']}"

        return jsonify(response_data)
    except Exception as e:
        import traceback
        print(f"❌ 聊天处理异常: {str(e)}")
        print(f"❌ 异常详情: {traceback.format_exc()}")
        return jsonify({"error": f"处理消息失败: {str(e)}"}), 500

def analyze_emotion_simple(user_input):
    """简单情感分析（降级方案）"""
    # 基于关键词的简单情感分析
    sadness_keywords = ['难过', '伤心', '孤独', '寂寞', '痛苦', '绝望', '想死', '不想活']
    joy_keywords = ['开心', '高兴', '快乐', '愉快', '兴奋', '满意']
    anger_keywords = ['生气', '愤怒', '恼火', '烦躁', '讨厌', '恨']
    
    text_lower = user_input.lower()
    
    sadness_score = sum(1 for word in sadness_keywords if word in text_lower) / len(sadness_keywords)
    joy_score = sum(1 for word in joy_keywords if word in text_lower) / len(joy_keywords)
    anger_score = sum(1 for word in anger_keywords if word in text_lower) / len(anger_keywords)
    
    # 归一化
    total = sadness_score + joy_score + anger_score
    if total > 0:
        sadness_score /= total
        joy_score /= total
        anger_score /= total
    else:
        sadness_score = 0.2
        joy_score = 0.6
        anger_score = 0.1
    
    intensity = max(sadness_score, joy_score, anger_score)
    
    return {
        "sadness": min(1.0, sadness_score),
        "joy": min(1.0, joy_score),
        "anger": min(1.0, anger_score),
        "intensity": min(1.0, intensity)
    }

def create_simple_strategy(emotion_scores, user_input):
    """创建简单策略（降级方案）"""
    sadness_score = emotion_scores.get("sadness", 0.2)
    joy_score = emotion_scores.get("joy", 0.6)
    
    if sadness_score > 0.6:
        return {
            "matched_rule": "high_sadness",
            "引导语": "我注意到您可能有些难过，我在这里陪伴您。有什么想和我分享的吗？",
            "语气": "温和关怀",
            "目标": "提供情感支持"
        }
    elif joy_score > 0.7:
        return {
            "matched_rule": "high_joy",
            "引导语": "很高兴看到您心情不错！继续保持这种积极的状态。",
            "语气": "积极鼓励",
            "目标": "维持积极情绪"
        }
    else:
        return {
            "matched_rule": "neutral",
            "引导语": "我理解您的感受，请告诉我更多。",
            "语气": "温和倾听",
            "目标": "深入了解"
        }

# 其他路由保持不变...
@app.route("/api/users/<user_id>/info", methods=["PUT"])
def update_user_info(user_id):
    """更新用户信息（处理问题回答和手动编辑）"""
    try:
        data = request.json
        
        if not user_info_manager:
            return jsonify({"error": "用户信息管理功能不可用"}), 503
        
        # Check if it's manual edit mode
        if "field" in data and "value" in data:
            # Manual edit mode
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
        if not user_info_manager:
            return jsonify({"error": "用户信息管理功能不可用"}), 503
            
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
    
    try:
        from visualization.trend_plotter import get_emotion_trend
        trend_data = get_emotion_trend(user_id)
        return jsonify(trend_data)
    except Exception as e:
        print(f"获取趋势数据失败: {e}")
        return jsonify({
            "dates": [],
            "sadness": [],
            "joy": [],
            "anger": [],
            "intensity": []
        })

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
    # 根据环境变量决定是否启用调试模式
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, port=5001, host='0.0.0.0')
