#!/usr/bin/env python3
"""
统一启动脚本
"""

import os
from flask import Flask, jsonify, request

# 创建Flask应用
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>养老护理系统</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
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
            .success { color: #28a745; font-weight: bold; }
            .info { color: #007bff; }
            h1 { color: #333; text-align: center; }
            .btn { 
                display: inline-block; 
                padding: 10px 20px; 
                background: #007bff; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
                margin: 5px; 
            }
            .btn:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎉 养老护理系统</h1>
            <p class="success">✅ 系统运行正常</p>
            <p class="info">📡 访问地址: {}</p>
            <p class="info">🏥 健康检查: {}/health</p>
            
            <h2>功能状态：</h2>
            <ul>
                <li class="success">✅ Web服务正常运行</li>
                <li class="success">✅ 健康检查功能正常</li>
                <li class="info">📊 版本: 1.0.0-basic</li>
            </ul>
            
            <div style="text-align: center; margin-top: 30px;">
                <a href="/health" class="btn">健康检查</a>
                <a href="/api/status" class="btn">系统状态</a>
            </div>
            
            <div style="text-align: center; margin-top: 30px; color: #666;">
                <p>养老护理系统 - 让科技温暖每一个家庭 ❤️</p>
            </div>
        </div>
    </body>
    </html>
    '''.format(request.url_root, request.url_root)

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "message": "养老护理系统运行正常",
        "version": "1.0.0-basic",
        "timestamp": "2024-01-01T12:00:00Z"
    })

@app.route('/api/status')
def status():
    return jsonify({
        "service": "养老护理系统",
        "version": "1.0.0-basic",
        "status": "running",
        "features": {
            "web_interface": True,
            "health_check": True,
            "api_endpoints": True
        }
    })

if __name__ == "__main__":
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 启动养老护理系统...")
    print(f"📡 监听地址: {host}:{port}")
    print(f"🌐 访问地址: http://{host}:{port}")
    
    app.run(host=host, port=port, debug=False)
