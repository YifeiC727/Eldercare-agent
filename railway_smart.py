#!/usr/bin/env python3
"""
Railway智能启动脚本
只包含核心功能，使用API而不是本地模型
"""

import os
import json
import requests
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime

# 创建智能Flask应用
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# 简化的情感分析（使用DeepSeek API）
def analyze_emotion_with_api(text):
    """使用DeepSeek API进行情感分析"""
    api_key = os.environ.get('CAMELLIA_KEY')
    if not api_key:
        return {"error": "API key not configured"}
    
    try:
        # 调用DeepSeek API
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个情感分析专家。请分析用户输入的情感，返回JSON格式：{\"emotion\": \"情感类型\", \"intensity\": 0.5, \"confidence\": 0.8}"
                    },
                    {
                        "role": "user", 
                        "content": f"请分析这句话的情感：{text}"
                    }
                ],
                "temperature": 0.3
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content', '{}')
        else:
            return {"error": f"API调用失败: {response.status_code}"}
            
    except Exception as e:
        return {"error": f"情感分析失败: {str(e)}"}

# 简化的对话生成
def generate_response_with_api(user_input, emotion_data):
    """使用DeepSeek API生成回复"""
    api_key = os.environ.get('CAMELLIA_KEY')
    if not api_key:
        return "抱歉，系统配置不完整，无法生成回复。"
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的养老护理助手，具有温暖、耐心、关爱的特质。请根据用户的情感状态，提供合适的回复。回复要简洁、温暖、有针对性。"
                    },
                    {
                        "role": "user",
                        "content": f"用户说：{user_input}\n情感分析结果：{emotion_data}\n请生成合适的回复。"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 150
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content', '抱歉，我暂时无法回复。')
        else:
            return "抱歉，系统暂时无法回复，请稍后再试。"
            
    except Exception as e:
        return f"抱歉，系统出现错误：{str(e)}"

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>养老护理系统 - 智能版</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                font-family: 'Microsoft YaHei', Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .chat-container {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                min-height: 300px;
                background: #f9f9f9;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
            }
            .user-message {
                background: #e3f2fd;
                text-align: right;
            }
            .bot-message {
                background: #f3e5f5;
            }
            .input-group {
                display: flex;
                gap: 10px;
                margin: 20px 0;
            }
            input[type="text"] {
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            button {
                padding: 10px 20px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background: #0056b3;
            }
            .status {
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .status.success { background: #d4edda; color: #155724; }
            .status.error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 养老护理系统 - 智能版</h1>
            <p>基于DeepSeek API的智能对话系统，轻量级部署</p>
            
            <div id="status" class="status success">
                ✅ 系统运行正常，API连接正常
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="message bot-message">
                    <strong>系统：</strong>您好！我是您的养老护理助手。有什么可以帮助您的吗？
                </div>
            </div>
            
            <div class="input-group">
                <input type="text" id="userInput" placeholder="请输入您想说的话..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">发送</button>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <a href="/health" style="color: #007bff;">健康检查</a> | 
                <a href="/api/status" style="color: #007bff;">系统状态</a>
            </div>
        </div>
        
        <script>
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            function sendMessage() {
                const input = document.getElementById('userInput');
                const message = input.value.trim();
                if (!message) return;
                
                // 显示用户消息
                addMessage('user', message);
                input.value = '';
                
                // 发送到后端
                fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({message: message})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        addMessage('bot', '抱歉，系统出现错误：' + data.error);
                    } else {
                        addMessage('bot', data.reply);
                    }
                })
                .catch(error => {
                    addMessage('bot', '抱歉，网络连接出现问题。');
                });
            }
            
            function addMessage(type, content) {
                const container = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + (type === 'user' ? 'user-message' : 'bot-message');
                messageDiv.innerHTML = '<strong>' + (type === 'user' ? '您：' : '助手：') + '</strong>' + content;
                container.appendChild(messageDiv);
                container.scrollTop = container.scrollHeight;
            }
        </script>
    </body>
    </html>
    ''')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({"error": "消息不能为空"})
        
        # 情感分析
        emotion_result = analyze_emotion_with_api(user_input)
        
        # 生成回复
        reply = generate_response_with_api(user_input, emotion_result)
        
        return jsonify({
            "reply": reply,
            "emotion": emotion_result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "message": "养老护理系统运行正常",
        "version": "1.0.0-smart",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "web_interface": True,
            "chat_api": True,
            "emotion_analysis": True,
            "api_based": True
        }
    })

@app.route('/api/status')
def status():
    api_key = os.environ.get('CAMELLIA_KEY')
    return jsonify({
        "service": "养老护理系统",
        "version": "1.0.0-smart",
        "status": "running",
        "api_configured": bool(api_key),
        "features": {
            "web_interface": True,
            "chat_api": True,
            "emotion_analysis": True,
            "api_based": True,
            "lightweight": True
        },
        "dependencies": {
            "flask": "3.1.2",
            "requests": "2.32.5",
            "pymongo": "4.14.1"
        }
    })

if __name__ == "__main__":
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 启动智能养老护理系统...")
    print(f"📡 监听地址: {host}:{port}")
    print(f"🌐 访问地址: http://{host}:{port}")
    print(f"💾 镜像大小: < 500MB (智能轻量级)")
    print(f"🤖 使用DeepSeek API进行智能对话")
    
    app.run(host=host, port=port, debug=False)
