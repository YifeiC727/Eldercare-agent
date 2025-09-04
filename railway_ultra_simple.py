#!/usr/bin/env python3
"""
Railway超轻量级启动脚本
只使用Flask核心功能，无任何重型依赖
"""

import os
from flask import Flask, request, jsonify

# 创建超轻量级Flask应用
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>养老护理系统 - 轻量版</title>
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
            .success { color: #28a745; font-weight: bold; }
            .info { color: #007bff; }
            .warning { color: #ffc107; }
            h1 { color: #333; text-align: center; }
            .status-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; 
                margin: 20px 0; 
            }
            .status-card { 
                padding: 15px; 
                border-radius: 8px; 
                text-align: center; 
            }
            .card-success { background: #d4edda; border: 1px solid #c3e6cb; }
            .card-warning { background: #fff3cd; border: 1px solid #ffeaa7; }
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
            <h1>🎉 养老护理系统部署成功！</h1>
            <p class="success">✅ 系统已成功部署到Railway</p>
            <p class="info">📡 访问地址: {}</p>
            
            <div class="status-grid">
                <div class="status-card card-success">
                    <h3>✅ Web服务</h3>
                    <p>正常运行</p>
                </div>
                <div class="status-card card-success">
                    <h3>✅ 健康检查</h3>
                    <p>系统正常</p>
                </div>
                <div class="status-card card-warning">
                    <h3>⚠️ 功能状态</h3>
                    <p>轻量版运行</p>
                </div>
                <div class="status-card card-warning">
                    <h3>📊 版本信息</h3>
                    <p>v1.0.0-ultra-simple</p>
                </div>
            </div>
            
            <h2>🔗 快速链接</h2>
            <a href="/health" class="btn">健康检查</a>
            <a href="/api/test" class="btn">API测试</a>
            <a href="/api/status" class="btn">系统状态</a>
            
            <h2>📋 功能说明</h2>
            <ul>
                <li class="success">✅ Web服务正常运行</li>
                <li class="success">✅ 基础API接口可用</li>
                <li class="success">✅ 健康检查功能正常</li>
                <li class="warning">⚠️ 智能对话功能需要完整版</li>
                <li class="warning">⚠️ 情感分析功能需要完整版</li>
                <li class="warning">⚠️ 数据存储功能需要完整版</li>
            </ul>
            
            <h2>🚀 升级到完整版</h2>
            <p>如需完整功能，请：</p>
            <ol>
                <li>升级Railway计划到Hobby或Pro</li>
                <li>修改Procfile使用完整版启动脚本</li>
                <li>重新部署应用</li>
            </ol>
            
            <div style="text-align: center; margin-top: 30px; color: #666;">
                <p>养老护理系统 - 让科技温暖每一个家庭 ❤️</p>
            </div>
        </div>
    </body>
    </html>
    '''.format(request.url_root)

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "message": "养老护理系统运行正常",
        "version": "1.0.0-ultra-simple",
        "timestamp": "2024-01-01T12:00:00Z",
        "components": {
            "web_server": "healthy",
            "api_endpoints": "healthy"
        }
    })

@app.route('/api/test')
def test():
    return jsonify({
        "message": "API测试成功", 
        "status": "ok",
        "version": "1.0.0-ultra-simple"
    })

@app.route('/api/status')
def status():
    return jsonify({
        "service": "养老护理系统",
        "version": "1.0.0-ultra-simple",
        "status": "running",
        "features": {
            "web_interface": True,
            "health_check": True,
            "api_endpoints": True,
            "smart_chat": False,
            "emotion_analysis": False,
            "data_storage": False
        },
        "upgrade_required": "Hobby plan or higher for full features"
    })

if __name__ == "__main__":
    # 获取配置
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 启动超轻量级养老护理系统...")
    print(f"📡 监听地址: {host}:{port}")
    print(f"🌐 访问地址: http://{host}:{port}")
    print(f"💾 镜像大小: < 1GB (超轻量级)")
    
    # 启动应用
    app.run(host=host, port=port, debug=False)
