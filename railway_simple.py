#!/usr/bin/env python3
"""
Railway简化启动脚本
只包含核心功能，避免复杂依赖
"""

import os
import sys
from flask import Flask, request

# 创建简化的Flask应用
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
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: 0 auto; }
            .success { color: green; }
            .info { color: blue; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎉 养老护理系统部署成功！</h1>
            <p class="success">✅ 系统已成功部署到Railway</p>
            <p class="info">📡 访问地址: {}</p>
            <p class="info">🏥 健康检查: {}/health</p>
            <h2>功能状态：</h2>
            <ul>
                <li>✅ Web服务正常运行</li>
                <li>✅ 基础功能可用</li>
                <li>⚠️ 部分高级功能需要完整部署</li>
            </ul>
            <h2>下一步：</h2>
            <p>如需完整功能，请配置环境变量并重新部署完整版本。</p>
        </div>
    </body>
    </html>
    '''.format(request.url_root, request.url_root)

@app.route('/health')
def health():
    return {
        "status": "healthy",
        "message": "养老护理系统运行正常",
        "version": "1.0.0-simple"
    }

@app.route('/api/test')
def test():
    return {"message": "API测试成功", "status": "ok"}

if __name__ == "__main__":
    # 获取配置
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 启动简化版养老护理系统...")
    print(f"📡 监听地址: {host}:{port}")
    print(f"🌐 访问地址: http://{host}:{port}")
    
    # 启动应用
    app.run(host=host, port=port, debug=False)