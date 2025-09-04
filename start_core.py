#!/usr/bin/env python3
"""
核心功能启动脚本 - 只包含基础依赖的功能
"""

import os
import sys
import json
import hashlib
from datetime import datetime

# 设置环境变量
os.environ.setdefault('FLASK_ENV', 'production')

try:
    from flask import Flask, jsonify, request, render_template_string, session, redirect, url_for
    
    # 创建Flask应用
    app = Flask(__name__)
    app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # 简单的用户数据存储（文件存储）
    USERS_FILE = 'users.json'
    
    def load_users():
        """加载用户数据"""
        try:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"加载用户数据失败: {e}")
            return {}
    
    def save_users(users):
        """保存用户数据"""
        try:
            with open(USERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存用户数据失败: {e}")
            return False
    
    def hash_password(password):
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def simple_emotion_analysis(text):
        """简单的情感分析（基于关键词）"""
        positive_words = ['好', '棒', '开心', '高兴', '满意', '喜欢', '爱', '幸福', '快乐', '不错']
        negative_words = ['坏', '差', '难过', '伤心', '失望', '讨厌', '恨', '痛苦', '糟糕', '不好']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return {"emotion": "positive", "score": 0.7, "confidence": 0.8}
        elif negative_count > positive_count:
            return {"emotion": "negative", "score": -0.7, "confidence": 0.8}
        else:
            return {"emotion": "neutral", "score": 0.0, "confidence": 0.6}
    
    def simple_chat_response(user_input, user_context=None):
        """简单的聊天回复"""
        # 基础回复模板
        responses = {
            "问候": ["您好！很高兴见到您。", "您好！我是您的养老护理助手。", "您好！有什么可以帮助您的吗？"],
            "健康": ["健康很重要，建议您定期体检。", "保持良好的生活习惯对健康很有帮助。", "如果身体不适，请及时就医。"],
            "情感": ["我理解您的感受。", "每个人都会有这样的情绪，这是正常的。", "您可以尝试做一些让自己开心的事情。"],
            "生活": ["生活需要慢慢品味。", "保持积极的心态很重要。", "多和家人朋友交流会让生活更美好。"],
            "默认": ["我明白了。", "这很有趣。", "请告诉我更多。", "我在这里倾听您。"]
        }
        
        # 简单的关键词匹配
        if any(word in user_input for word in ['你好', '您好', 'hi', 'hello']):
            return responses["问候"][0]
        elif any(word in user_input for word in ['健康', '身体', '病', '药']):
            return responses["健康"][0]
        elif any(word in user_input for word in ['心情', '感觉', '情绪', '开心', '难过']):
            return responses["情感"][0]
        elif any(word in user_input for word in ['生活', '日常', '习惯']):
            return responses["生活"][0]
        else:
            return responses["默认"][0]
    
    # HTML模板
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
            .user-info {
                background: #e3f2fd;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
            }
            .chat-container {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .chat-messages {
                height: 300px;
                overflow-y: auto;
                border: 1px solid #dee2e6;
                padding: 15px;
                background: white;
                border-radius: 8px;
                margin-bottom: 15px;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 8px;
            }
            .user-message {
                background: #007bff;
                color: white;
                margin-left: 20%;
            }
            .bot-message {
                background: #e9ecef;
                color: #495057;
                margin-right: 20%;
            }
            .chat-input {
                display: flex;
                gap: 10px;
            }
            .chat-input input {
                flex: 1;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 16px;
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
                border: none;
                cursor: pointer;
                font-size: 16px;
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
            .emotion-display {
                background: #fff3cd;
                padding: 10px;
                border-radius: 6px;
                margin: 10px 0;
                border-left: 4px solid #ffc107;
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
                <div class="status">✅ 核心服务运行正常</div>
                
                {% if user %}
                <div class="user-info">
                    <h3>👤 欢迎回来，{{ user.name }}！</h3>
                    <p>用户ID: {{ user.id }}</p>
                    <p>注册时间: {{ user.created_at }}</p>
                    <a href="/logout" class="btn btn-danger">退出登录</a>
                </div>
                
                <div class="chat-container">
                    <h3>💬 智能对话</h3>
                    <div class="chat-messages" id="chatMessages">
                        <div class="message bot-message">
                            <strong>助手:</strong> 您好！我是您的养老护理助手，有什么可以帮助您的吗？
                        </div>
                    </div>
                    <div class="chat-input">
                        <input type="text" id="chatInput" placeholder="请输入您的问题..." onkeypress="handleKeyPress(event)">
                        <button class="btn" onclick="sendMessage()">发送</button>
                    </div>
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
                    <h3>🚀 可用功能</h3>
                    <ul>
                        <li><a href="/health">系统健康检查</a></li>
                        <li><a href="/status">系统状态信息</a></li>
                        <li><a href="/dependencies">依赖检查</a></li>
                        {% if user %}
                        <li><a href="/profile">个人资料</a></li>
                        <li><a href="/chat">智能对话</a></li>
                        <li><a href="/emotion">情感分析</a></li>
                        {% endif %}
                    </ul>
                </div>
                
                <div class="feature-list">
                    <h3>📋 系统信息</h3>
                    <ul>
                        <li>版本: 1.3.0 (核心功能版)</li>
                        <li>状态: 核心模式运行中</li>
                        <li>用户数: {{ user_count }}</li>
                        <li>最后更新: {{ current_time }}</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <script>
            function sendMessage() {
                const input = document.getElementById('chatInput');
                const message = input.value.trim();
                if (!message) return;
                
                const chatMessages = document.getElementById('chatMessages');
                
                // 添加用户消息
                const userMessage = document.createElement('div');
                userMessage.className = 'message user-message';
                userMessage.innerHTML = '<strong>您:</strong> ' + message;
                chatMessages.appendChild(userMessage);
                
                // 清空输入框
                input.value = '';
                
                // 滚动到底部
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                // 发送到服务器
                fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({message: message})
                })
                .then(response => response.json())
                .then(data => {
                    // 添加机器人回复
                    const botMessage = document.createElement('div');
                    botMessage.className = 'message bot-message';
                    botMessage.innerHTML = '<strong>助手:</strong> ' + data.response;
                    chatMessages.appendChild(botMessage);
                    
                    // 如果有情感分析结果
                    if (data.emotion) {
                        const emotionDiv = document.createElement('div');
                        emotionDiv.className = 'emotion-display';
                        emotionDiv.innerHTML = '<strong>情感分析:</strong> ' + data.emotion.emotion + ' (置信度: ' + (data.emotion.confidence * 100).toFixed(0) + '%)';
                        chatMessages.appendChild(emotionDiv);
                    }
                    
                    // 滚动到底部
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                })
                .catch(error => {
                    console.error('Error:', error);
                    const errorMessage = document.createElement('div');
                    errorMessage.className = 'message bot-message';
                    errorMessage.innerHTML = '<strong>助手:</strong> 抱歉，我遇到了一些问题，请稍后再试。';
                    chatMessages.appendChild(errorMessage);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                });
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
        </script>
    </body>
    </html>
    """
    
    LOGIN_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>用户登录 - 养老护理智能助手</title>
        <meta charset="utf-8">
        <style>
            body { 
                font-family: 'Microsoft YaHei', Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .login-container { 
                max-width: 400px; 
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                padding: 40px;
            }
            .form-group {
                margin: 20px 0;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="text"], input[type="password"] {
                width: 100%;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 16px;
                box-sizing: border-box;
            }
            .btn {
                width: 100%;
                padding: 12px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                cursor: pointer;
                margin: 10px 0;
            }
            .btn:hover {
                background: #0056b3;
            }
            .btn-secondary {
                background: #6c757d;
            }
            .btn-secondary:hover {
                background: #545b62;
            }
            .error {
                color: #dc3545;
                background: #f8d7da;
                padding: 10px;
                border-radius: 6px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="login-container">
            <h2 style="text-align: center; margin-bottom: 30px;">🔐 用户登录</h2>
            
            {% if error %}
            <div class="error">{{ error }}</div>
            {% endif %}
            
            <form method="POST">
                <div class="form-group">
                    <label for="username">用户名:</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">密码:</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <button type="submit" class="btn">登录</button>
            </form>
            
            <a href="/register" class="btn btn-secondary">注册新用户</a>
            <a href="/" class="btn btn-secondary">返回首页</a>
        </div>
    </body>
    </html>
    """
    
    REGISTER_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>用户注册 - 养老护理智能助手</title>
        <meta charset="utf-8">
        <style>
            body { 
                font-family: 'Microsoft YaHei', Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .register-container { 
                max-width: 400px; 
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                padding: 40px;
            }
            .form-group {
                margin: 20px 0;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="text"], input[type="password"], input[type="email"] {
                width: 100%;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 16px;
                box-sizing: border-box;
            }
            .btn {
                width: 100%;
                padding: 12px;
                background: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                cursor: pointer;
                margin: 10px 0;
            }
            .btn:hover {
                background: #1e7e34;
            }
            .btn-secondary {
                background: #6c757d;
            }
            .btn-secondary:hover {
                background: #545b62;
            }
            .error {
                color: #dc3545;
                background: #f8d7da;
                padding: 10px;
                border-radius: 6px;
                margin: 10px 0;
            }
            .success {
                color: #155724;
                background: #d4edda;
                padding: 10px;
                border-radius: 6px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="register-container">
            <h2 style="text-align: center; margin-bottom: 30px;">📝 用户注册</h2>
            
            {% if error %}
            <div class="error">{{ error }}</div>
            {% endif %}
            
            {% if success %}
            <div class="success">{{ success }}</div>
            {% endif %}
            
            <form method="POST">
                <div class="form-group">
                    <label for="username">用户名:</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="email">邮箱:</label>
                    <input type="email" id="email" name="email" required>
                </div>
                <div class="form-group">
                    <label for="name">姓名:</label>
                    <input type="text" id="name" name="name" required>
                </div>
                <div class="form-group">
                    <label for="password">密码:</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <div class="form-group">
                    <label for="confirm_password">确认密码:</label>
                    <input type="password" id="confirm_password" name="confirm_password" required>
                </div>
                <button type="submit" class="btn">注册</button>
            </form>
            
            <a href="/login" class="btn btn-secondary">已有账号？登录</a>
            <a href="/" class="btn btn-secondary">返回首页</a>
        </div>
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
        
        return render_template_string(INDEX_TEMPLATE, 
                                    user=user, 
                                    user_count=len(users),
                                    current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """用户登录"""
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not username or not password:
                return render_template_string(LOGIN_TEMPLATE, error='请填写用户名和密码')
            
            users = load_users()
            for user_id, user_data in users.items():
                if user_data['username'] == username and user_data['password'] == hash_password(password):
                    session['user_id'] = user_id
                    return redirect(url_for('index'))
            
            return render_template_string(LOGIN_TEMPLATE, error='用户名或密码错误')
        
        return render_template_string(LOGIN_TEMPLATE)
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """用户注册"""
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            name = request.form.get('name')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            if not all([username, email, name, password, confirm_password]):
                return render_template_string(REGISTER_TEMPLATE, error='请填写所有字段')
            
            if password != confirm_password:
                return render_template_string(REGISTER_TEMPLATE, error='密码确认不匹配')
            
            if len(password) < 6:
                return render_template_string(REGISTER_TEMPLATE, error='密码长度至少6位')
            
            users = load_users()
            
            # 检查用户名是否已存在
            for user_data in users.values():
                if user_data['username'] == username:
                    return render_template_string(REGISTER_TEMPLATE, error='用户名已存在')
                if user_data['email'] == email:
                    return render_template_string(REGISTER_TEMPLATE, error='邮箱已被注册')
            
            # 创建新用户
            user_id = str(len(users) + 1)
            new_user = {
                'id': user_id,
                'username': username,
                'email': email,
                'name': name,
                'password': hash_password(password),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_login': None
            }
            
            users[user_id] = new_user
            
            if save_users(users):
                return render_template_string(REGISTER_TEMPLATE, success='注册成功！请登录')
            else:
                return render_template_string(REGISTER_TEMPLATE, error='注册失败，请重试')
        
        return render_template_string(REGISTER_TEMPLATE)
    
    @app.route('/logout')
    def logout():
        """用户退出"""
        session.pop('user_id', None)
        return redirect(url_for('index'))
    
    @app.route('/profile')
    def profile():
        """用户资料页面"""
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        users = load_users()
        user = users.get(session['user_id'])
        
        if not user:
            return redirect(url_for('login'))
        
        return jsonify({
            'user_id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'name': user['name'],
            'created_at': user['created_at'],
            'last_login': user.get('last_login', '从未登录')
        })
    
    @app.route('/chat', methods=['POST'])
    def chat():
        """智能对话"""
        if 'user_id' not in session:
            return jsonify({'error': '请先登录'}), 401
        
        data = request.get_json()
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({'error': '请输入消息'}), 400
        
        # 简单的情感分析
        emotion = simple_emotion_analysis(user_input)
        
        # 简单的聊天回复
        response = simple_chat_response(user_input)
        
        return jsonify({
            'response': response,
            'emotion': emotion,
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/emotion', methods=['POST'])
    def emotion_analysis():
        """情感分析"""
        if 'user_id' not in session:
            return jsonify({'error': '请先登录'}), 401
        
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '请输入文本'}), 400
        
        emotion = simple_emotion_analysis(text)
        
        return jsonify({
            'emotion': emotion,
            'text': text,
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/health')
    def health():
        """健康检查"""
        users = load_users()
        return jsonify({
            "status": "healthy",
            "message": "核心功能系统正常运行",
            "version": "1.3.0",
            "features": ["user_management", "simple_chat", "emotion_analysis"],
            "user_count": len(users)
        }), 200
    
    @app.route('/status')
    def status():
        """系统状态"""
        users = load_users()
        return jsonify({
            "service": "Eldercare Agent",
            "version": "1.3.0",
            "status": "running",
            "mode": "core",
            "environment": os.environ.get('FLASK_ENV', 'development'),
            "python_version": sys.version,
            "features": {
                "user_management": True,
                "simple_chat": True,
                "emotion_analysis": True,
                "advanced_ai": False,
                "database": False
            },
            "statistics": {
                "total_users": len(users),
                "active_sessions": len(session) if hasattr(session, 'keys') else 0
            }
        }), 200
    
    @app.route('/dependencies')
    def dependencies():
        """依赖检查"""
        return jsonify({
            "available_dependencies": ["flask", "werkzeug"],
            "missing_dependencies": ["bson", "numpy", "pandas", "sklearn", "jieba", "requests", "openai", "sentence_transformers", "faiss", "pymongo", "matplotlib", "seaborn"],
            "dependency_count": {
                "total": 14,
                "available": 2,
                "missing": 12
            },
            "core_features": {
                "user_management": True,
                "simple_chat": True,
                "emotion_analysis": True
            }
        }), 200
    
    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        print(f"🚀 启动核心功能系统，端口: {port}")
        
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
            
            print("✅ 使用Gunicorn启动核心系统")
            StandaloneApplication(app, options).run()
            
        except ImportError:
            print("⚠️ Gunicorn不可用，使用Flask开发服务器")
            app.run(host='0.0.0.0', port=port, debug=False)
            
except Exception as e:
    print(f"❌ 核心系统启动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
