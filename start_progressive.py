#!/usr/bin/env python3
"""
渐进式启动脚本 - 根据可用依赖逐步启用功能
"""

import os
import sys
import traceback

# 设置环境变量
os.environ.setdefault('FLASK_ENV', 'production')

def check_dependency(module_name):
    """检查单个依赖是否可用"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def get_available_features():
    """获取可用功能列表"""
    features = {
        'basic_web': True,  # Flask基础功能
        'user_management': False,
        'emotion_analysis': False,
        'ai_chat': False,
        'database': False,
        'speech_recognition': False
    }
    
    # 检查用户管理功能
    if check_dependency('bson'):
        features['user_management'] = True
        features['database'] = True
    
    # 检查情感分析功能
    if check_dependency('numpy') and check_dependency('pandas'):
        features['emotion_analysis'] = True
    
    # 检查AI聊天功能
    if check_dependency('openai') and check_dependency('jieba'):
        features['ai_chat'] = True
    
    # 检查语音识别功能
    if check_dependency('pydub'):
        features['speech_recognition'] = True
    
    return features

def create_progressive_app():
    """创建渐进式Flask应用"""
    from flask import Flask, jsonify, request, render_template_string, session, redirect, url_for
    import json
    import hashlib
    from datetime import datetime
    
    app = Flask(__name__)
    app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # 获取可用功能
    features = get_available_features()
    print(f"🔍 可用功能: {[k for k, v in features.items() if v]}")
    
    # 简单的用户数据存储
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
        return hashlib.sha256(password.encode()).hexdigest()
    
    # 主页模板
    INDEX_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>养老护理智能助手</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: 'Microsoft YaHei', Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 900px; 
                margin: 0 auto; 
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .content {
                padding: 30px;
            }
            .status { 
                color: #28a745; 
                font-weight: bold; 
                font-size: 18px;
                margin: 20px 0;
            }
            .feature-list {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .feature-list h3 {
                color: #495057;
                margin-top: 0;
            }
            .feature-list ul {
                list-style: none;
                padding: 0;
            }
            .feature-list li {
                padding: 8px 0;
                border-bottom: 1px solid #dee2e6;
            }
            .feature-list li:before {
                content: "✅ ";
                color: #28a745;
            }
            .feature-list li.unavailable:before {
                content: "⚠️ ";
                color: #ffc107;
            }
            .user-info {
                background: #e3f2fd;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
            }
            .btn {
                display: inline-block;
                padding: 12px 24px;
                background: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 6px;
                margin: 5px;
                transition: background 0.3s;
            }
            .btn:hover {
                background: #0056b3;
            }
            .btn-success {
                background: #28a745;
            }
            .btn-success:hover {
                background: #1e7e34;
            }
            .btn-danger {
                background: #dc3545;
            }
            .btn-danger:hover {
                background: #c82333;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 养老护理智能助手</h1>
                <p>专业的老年人健康管理和情感陪伴服务</p>
            </div>
            <div class="content">
                <div class="status">✅ 系统运行正常</div>
                
                {% if user %}
                <div class="user-info">
                    <h3>👤 欢迎回来，{{ user.name }}！</h3>
                    <p>用户ID: {{ user.id }}</p>
                    <p>注册时间: {{ user.created_at }}</p>
                    <a href="/logout" class="btn btn-danger">退出登录</a>
                </div>
                {% else %}
                <div class="user-info">
                    <h3>🔐 用户登录</h3>
                    <p>请登录以使用完整功能</p>
                    <a href="/login" class="btn">登录</a>
                    <a href="/register" class="btn btn-success">注册新用户</a>
                </div>
                {% endif %}
                
                <div class="feature-list">
                    <h3>🚀 系统功能状态</h3>
                    <ul>
                        <li><a href="/health">系统健康检查</a></li>
                        <li><a href="/status">系统状态信息</a></li>
                        {% if features.user_management %}
                        <li><a href="/profile">个人资料管理</a></li>
                        {% endif %}
                        {% if features.ai_chat %}
                        <li><a href="/chat">智能对话</a></li>
                        {% endif %}
                        {% if features.emotion_analysis %}
                        <li><a href="/emotion">情感分析</a></li>
                        {% endif %}
                        {% if features.speech_recognition %}
                        <li><a href="/speech">语音识别</a></li>
                        {% endif %}
                    </ul>
                </div>
                
                <div class="feature-list">
                    <h3>📋 系统信息</h3>
                    <ul>
                        <li>版本: 1.4.0 (渐进式启动版)</li>
                        <li>状态: 运行中</li>
                        <li>用户数: {{ user_count }}</li>
                        <li>可用功能: {{ available_features_count }}/6</li>
                        <li>最后更新: {{ current_time }}</li>
                    </ul>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 注册页面模板 (使用原有的设计)
    REGISTER_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Welcome - ElderCare Companion</title>
        <style>
            body {
                font-family: 'Microsoft YaHei', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                padding: 40px;
                width: 90%;
                max-width: 500px;
                text-align: center;
            }
            
            h1 {
                color: #333;
                margin-bottom: 30px;
                font-size: 28px;
                font-weight: 300;
            }
            
            .form-group {
                margin-bottom: 25px;
                text-align: left;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                color: #555;
                font-weight: 500;
            }
            
            input[type="text"], select {
                width: 100%;
                padding: 12px 15px;
                border: 2px solid #e1e5e9;
                border-radius: 10px;
                font-size: 16px;
                transition: border-color 0.3s;
                box-sizing: border-box;
            }
            
            input[type="text"]:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .range-container {
                position: relative;
                margin-top: 10px;
            }
            
            input[type="range"] {
                width: 100%;
                height: 6px;
                border-radius: 3px;
                background: #e1e5e9;
                outline: none;
                -webkit-appearance: none;
            }
            
            input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #667eea;
                cursor: pointer;
            }
            
            input[type="range"]::-moz-range-thumb {
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #667eea;
                cursor: pointer;
                border: none;
            }
            
            .age-display {
                text-align: center;
                margin-top: 10px;
                font-size: 18px;
                color: #667eea;
                font-weight: bold;
            }
            
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 25px;
                font-size: 18px;
                cursor: pointer;
                transition: transform 0.2s;
                width: 100%;
                margin-top: 20px;
            }
            
            button:hover {
                transform: translateY(-2px);
            }
            
            .welcome-text {
                color: #666;
                margin-bottom: 30px;
                line-height: 1.6;
            }
            
            .error {
                color: #e74c3c;
                font-size: 14px;
                margin-top: 5px;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ElderCare Companion</h1>
            <p class="welcome-text">
                Welcome! To personalize your experience, please provide a few basic details.
            </p>
            
            <form id="registrationForm">
                <div class="form-group">
                    <label for="name">Name *</label>
                    <input type="text" id="name" name="name" required placeholder="Enter your name">
                    <div class="error" id="nameError">Please enter your name</div>
                </div>
                
                <div class="form-group">
                    <label for="age">Age *</label>
                    <div class="range-container">
                        <input type="range" id="age" name="age" min="50" max="100" value="70">
                        <div class="age-display" id="ageDisplay">70</div>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="gender">Gender *</label>
                    <select id="gender" name="gender" required>
                        <option value="">Select gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                    <div class="error" id="genderError">Please select your gender</div>
                </div>
                
                <div class="form-group">
                    <label for="password">Set Password *</label>
                    <input type="password" id="password" name="password" required placeholder="Set your password">
                    <div class="error" id="passwordError">Please set a password</div>
                </div>
                
                <div class="form-group">
                    <label for="confirm_password">Confirm Password *</label>
                    <input type="password" id="confirm_password" name="confirm_password" required placeholder="Re-enter your password">
                    <div class="error" id="confirmPasswordError">Please confirm your password</div>
                </div>
                
                <button type="submit">Start Chat</button>
            </form>
        </div>

        <script>
            // Age slider display
            const ageSlider = document.getElementById('age');
            const ageDisplay = document.getElementById('ageDisplay');
            
            ageSlider.addEventListener('input', function() {
                ageDisplay.textContent = this.value;
            });
            
            // Form submission
            document.getElementById('registrationForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                // Clear previous errors
                document.querySelectorAll('.error').forEach(el => el.style.display = 'none');
                
                // Get form data
                const formData = {
                    name: document.getElementById('name').value.trim(),
                    age: parseInt(document.getElementById('age').value),
                    gender: document.getElementById('gender').value,
                    password: document.getElementById('password').value,
                    confirm_password: document.getElementById('confirm_password').value
                };
                
                // Validation
                let hasError = false;
                
                if (!formData.name) {
                    document.getElementById('nameError').style.display = 'block';
                    hasError = true;
                }
                
                if (!formData.gender) {
                    document.getElementById('genderError').style.display = 'block';
                    hasError = true;
                }
                
                if (!formData.password) {
                    document.getElementById('passwordError').style.display = 'block';
                    hasError = true;
                }
                
                if (!formData.confirm_password) {
                    document.getElementById('confirmPasswordError').style.display = 'block';
                    hasError = true;
                }
                
                if (formData.password !== formData.confirm_password) {
                    document.getElementById('confirmPasswordError').style.display = 'block';
                    document.getElementById('confirmPasswordError').textContent = 'Passwords do not match';
                    hasError = true;
                }
                
                if (hasError) return;
                
                // Submit data
                try {
                    const response = await fetch('/api/users', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(formData)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        // Redirect to chat page
                        window.location.href = '/chat';
                    } else {
                        alert('Registration failed: ' + result.error);
                    }
                } catch (error) {
                    alert('Network error, please try again');
                    console.error('Error:', error);
                }
            });
        </script>
    </body>
    </html>
    """
    
    @app.route('/')
    def index():
        """主页"""
        users = load_users()
        user = None
        if 'user_id' in session:
            user = users.get(session['user_id'])
        
        available_features_count = sum(1 for v in features.values() if v)
        
        return render_template_string(INDEX_TEMPLATE, 
                                    user=user, 
                                    user_count=len(users),
                                    available_features_count=available_features_count,
                                    features=features,
                                    current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    @app.route('/register')
    def register():
        """用户注册页面"""
        return render_template_string(REGISTER_TEMPLATE)
    
    @app.route('/api/users', methods=['POST'])
    def create_user():
        """创建用户"""
        data = request.json
        user_ip = request.remote_addr
        
        # 验证必需字段
        if not all(key in data for key in ["name", "age", "gender", "password"]):
            return jsonify({"error": "Please fill in all required information"}), 400
        
        if data["password"] != data.get("confirm_password", ""):
            return jsonify({"error": "Passwords do not match"}), 400
        
        # 创建用户
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
        
        if save_users(users):
            session['user_id'] = user_id
            return jsonify({
                "status": "success",
                "user_id": user_id,
                "message": "User created successfully"
            }), 201
        else:
            return jsonify({"error": "Failed to save user"}), 500
    
    @app.route('/api/login', methods=['POST'])
    def login():
        """用户登录"""
        data = request.json
        
        if not all(key in data for key in ["username", "password"]):
            return jsonify({"error": "Please enter username and password"}), 400
        
        users = load_users()
        for user_id, user_data in users.items():
            if user_data['name'] == data['username'] and user_data['password'] == hash_password(data['password']):
                session['user_id'] = user_id
                return jsonify({
                    "status": "success",
                    "user_id": user_id,
                    "message": "Login successful"
                }), 200
        
        return jsonify({"error": "Invalid username or password"}), 401
    
    @app.route('/health')
    def health():
        """健康检查"""
        users = load_users()
        return jsonify({
            "status": "healthy",
            "message": "Progressive system running",
            "version": "1.4.0",
            "features": features,
            "user_count": len(users)
        }), 200
    
    @app.route('/status')
    def status():
        """系统状态"""
        users = load_users()
        return jsonify({
            "service": "Eldercare Agent",
            "version": "1.4.0",
            "status": "running",
            "mode": "progressive",
            "environment": os.environ.get('FLASK_ENV', 'development'),
            "python_version": sys.version,
            "features": features,
            "statistics": {
                "total_users": len(users),
                "active_sessions": len(session) if hasattr(session, 'keys') else 0
            }
        }), 200
    
    return app

if __name__ == '__main__':
    try:
        # 尝试启动完整系统
        print("🚀 尝试启动完整系统...")
        from app import app
        print("✅ 完整系统启动成功")
        
        port = int(os.environ.get('PORT', 5000))
        
        # 使用Gunicorn启动
        try:
            from gunicorn.app.base import BaseApplication
            
            class StandaloneApplication(BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()
            
                def load_config(self):
                    config = {key: value for key, value in self.options.items()
                              if key in self.cfg.settings and value is not None}
                    for key, value in config.items():
                        self.cfg.set(key.lower(), value)
            
                def load(self):
                    return self.application
            
            options = {
                'bind': f'0.0.0.0:{port}',
                'workers': 1,
                'timeout': 120,
                'keepalive': 2,
                'max_requests': 1000,
                'max_requests_jitter': 100
            }
            
            print("✅ 使用Gunicorn启动完整系统")
            StandaloneApplication(app, options).run()
            
        except ImportError:
            print("⚠️ Gunicorn不可用，使用Flask开发服务器")
            app.run(host='0.0.0.0', port=port, debug=False)
            
    except Exception as e:
        print(f"❌ 完整系统启动失败: {e}")
        print("🔄 启动渐进式系统...")
        
        # 启动渐进式系统
        app = create_progressive_app()
        port = int(os.environ.get('PORT', 5000))
        
        try:
            from gunicorn.app.base import BaseApplication
            
            class StandaloneApplication(BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()
            
                def load_config(self):
                    config = {key: value for key, value in self.options.items()
                              if key in self.cfg.settings and value is not None}
                    for key, value in config.items():
                        self.cfg.set(key.lower(), value)
            
                def load(self):
                    return self.application
            
            options = {
                'bind': f'0.0.0.0:{port}',
                'workers': 1,
                'timeout': 120,
                'keepalive': 2,
                'max_requests': 1000,
                'max_requests_jitter': 100
            }
            
            print("✅ 使用Gunicorn启动渐进式系统")
            StandaloneApplication(app, options).run()
            
        except ImportError:
            print("⚠️ Gunicorn不可用，使用Flask开发服务器")
            app.run(host='0.0.0.0', port=port, debug=False)
