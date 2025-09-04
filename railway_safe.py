#!/usr/bin/env python3
"""
Railway安全启动脚本
包含错误处理和降级机制
"""

import os
import sys
import traceback

def start_safe_server():
    """安全启动服务器"""
    
    # 设置环境变量
    os.environ.setdefault('FLASK_ENV', 'production')
    os.environ.setdefault('FLASK_DEBUG', 'False')
    
    # 获取配置
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 启动Railway养老护理系统...")
    print(f"📡 监听地址: {host}:{port}")
    print(f"🌐 环境: {os.environ.get('RAILWAY_ENVIRONMENT', 'production')}")
    
    # 尝试导入完整应用
    try:
        print("📦 尝试导入完整应用...")
        from app import app
        print("✅ 完整应用导入成功")
        
        # 检查环境变量
        api_key = os.environ.get('CAMELLIA_KEY')
        if api_key:
            print("✅ API密钥已配置")
        else:
            print("⚠️ API密钥未配置，部分功能可能受限")
        
        # 启动完整应用
        print("🌐 启动完整版养老护理系统...")
        app.run(host=host, port=port, debug=False)
        
    except ImportError as e:
        print(f"⚠️ 完整应用导入失败: {e}")
        print("🔄 启动简化版应用...")
        start_simple_app(host, port)
        
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        print("📋 错误详情:")
        traceback.print_exc()
        print("🔄 启动简化版应用...")
        start_simple_app(host, port)

def start_simple_app(host, port):
    """启动简化版应用"""
    try:
        from flask import Flask, request, jsonify
        
        # 创建简化应用
        simple_app = Flask(__name__)
        simple_app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
        
        @simple_app.route('/')
        def home():
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>养老护理系统 - 简化版</title>
                <meta charset="UTF-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                    .container { max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
                    .success { color: green; }
                    .warning { color: orange; }
                    .error { color: red; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎉 养老护理系统部署成功！</h1>
                    <p class="success">✅ 系统已成功部署到Railway</p>
                    <p class="warning">⚠️ 当前运行简化版，部分功能受限</p>
                    <h2>功能状态：</h2>
                    <ul>
                        <li class="success">✅ Web服务正常运行</li>
                        <li class="success">✅ 健康检查功能正常</li>
                        <li class="warning">⚠️ 完整功能需要修复依赖</li>
                    </ul>
                    <h2>快速链接：</h2>
                    <p><a href="/health">健康检查</a> | <a href="/api/status">系统状态</a></p>
                </div>
            </body>
            </html>
            '''
        
        @simple_app.route('/health')
        def health():
            return jsonify({
                "status": "healthy",
                "message": "养老护理系统运行正常（简化版）",
                "version": "1.0.0-simple",
                "mode": "fallback"
            })
        
        @simple_app.route('/api/status')
        def status():
            return jsonify({
                "service": "养老护理系统",
                "version": "1.0.0-simple",
                "status": "running",
                "mode": "fallback",
                "message": "完整版功能正在修复中"
            })
        
        print("🌐 启动简化版应用...")
        simple_app.run(host=host, port=port, debug=False)
        
    except Exception as e:
        print(f"❌ 简化版应用也启动失败: {e}")
        print("📋 错误详情:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    start_safe_server()
