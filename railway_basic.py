#!/usr/bin/env python3
"""
Railway最基础启动脚本
只包含最基本的Flask应用，确保能够启动
"""

import os
from flask import Flask, jsonify, request

# 创建最基础的Flask应用
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
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f0f0; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
            .success { color: green; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎉 养老护理系统</h1>
            <p class="success">✅ 系统运行正常</p>
            <p>访问地址: {}</p>
            <p><a href="/health">健康检查</a></p>
        </div>
    </body>
    </html>
    '''.format(request.url_root)

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "message": "养老护理系统运行正常",
        "version": "1.0.0-basic"
    })

if __name__ == "__main__":
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 启动基础版养老护理系统...")
    print(f"📡 监听地址: {host}:{port}")
    
    app.run(host=host, port=port, debug=False)
