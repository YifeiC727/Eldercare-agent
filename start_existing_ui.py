#!/usr/bin/env python3
"""
使用现有UI和用户系统的启动脚本
保持原有的用户注册流程和界面设计
"""

import os
import sys
import traceback

# 设置环境变量
os.environ.setdefault('FLASK_ENV', 'production')

try:
    # 尝试导入现有的完整系统
    print("🚀 尝试启动现有完整系统...")
    from app import app
    print("✅ 成功导入现有完整系统")
    print("✅ 包含功能:")
    print("  - 原有用户注册流程 (姓名、年龄、性别、密码)")
    print("  - 智能对话功能") 
    print("  - 情感分析")
    print("  - 个性化推荐")
    print("  - 健康监测")
    print("  - 数据存储")
    
    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        print(f"🌐 启动完整系统，端口: {port}")
        
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
    print("🔄 回退到基础版本...")
    
    # 如果完整系统启动失败，回退到基础版本
    try:
        from flask import Flask, jsonify, render_template_string
        
        # 创建基础Flask应用作为回退
        fallback_app = Flask(__name__)
        fallback_app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
        
        @fallback_app.route('/')
        def index():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>养老护理系统 - 维护模式</title>
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
                    .container { 
                        max-width: 600px; 
                        margin: 0 auto; 
                        background: white;
                        border-radius: 15px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                        padding: 40px;
                        text-align: center;
                    }
                    .status { 
                        color: #ff6b6b; 
                        font-weight: bold; 
                        font-size: 18px;
                        margin: 20px 0;
                    }
                    .btn {
                        display: inline-block;
                        padding: 12px 24px;
                        background: #007bff;
                        color: white;
                        text-decoration: none;
                        border-radius: 6px;
                        margin: 10px;
                        transition: background 0.3s;
                    }
                    .btn:hover {
                        background: #0056b3;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🏥 养老护理智能助手</h1>
                    <p class="status">⚠️ 系统正在维护中</p>
                    <p>完整功能暂时不可用，请稍后再试。</p>
                    <p>基础服务正常运行。</p>
                    <a href="/health" class="btn">健康检查</a>
                    <a href="/status" class="btn">系统状态</a>
                </div>
            </body>
            </html>
            """
        
        @fallback_app.route('/health')
        def health():
            return jsonify({
                "status": "maintenance",
                "message": "系统维护中，基础服务正常",
                "version": "1.0.0-fallback"
            }), 200
        
        @fallback_app.route('/status')
        def status():
            return jsonify({
                "service": "Eldercare Agent",
                "status": "maintenance",
                "message": "完整系统启动失败，运行在维护模式",
                "error": str(e)
            }), 200
        
        if __name__ == '__main__':
            port = int(os.environ.get('PORT', 5000))
            print(f"🚀 启动回退版本，端口: {port}")
            fallback_app.run(host='0.0.0.0', port=port, debug=False)
            
    except Exception as fallback_error:
        print(f"❌ 回退版本也启动失败: {fallback_error}")
        traceback.print_exc()
        sys.exit(1)
