#!/usr/bin/env python3
"""
最简化的Flask应用启动脚本
确保在Railway上能够正常运行
"""

import os
import sys

# 设置环境变量
os.environ.setdefault('FLASK_ENV', 'production')

try:
    from flask import Flask, jsonify
    
    # 创建Flask应用
    app = Flask(__name__)
    app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    @app.route('/')
    def index():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Eldercare Agent</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .status { color: green; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏥 Eldercare Agent</h1>
                <p class="status">✅ 服务运行正常</p>
                <p>这是一个基础的养老护理智能助手服务。</p>
                <h2>可用功能：</h2>
                <ul>
                    <li><a href="/health">健康检查</a></li>
                    <li><a href="/status">系统状态</a></li>
                </ul>
            </div>
        </body>
        </html>
        """
    
    @app.route('/health')
    def health():
        return jsonify({
            "status": "healthy",
            "message": "Eldercare Agent is running",
            "version": "1.0.0"
        }), 200
    
    @app.route('/status')
    def status():
        return jsonify({
            "service": "Eldercare Agent",
            "status": "running",
            "environment": os.environ.get('FLASK_ENV', 'development'),
            "python_version": sys.version
        }), 200
    
    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        print(f"🚀 Starting Eldercare Agent on port {port}")
        
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
            
            print("✅ Starting with Gunicorn")
            StandaloneApplication(app, options).run()
            
        except ImportError:
            print("⚠️ Gunicorn not available, using Flask dev server")
            app.run(host='0.0.0.0', port=port, debug=False)
            
except Exception as e:
    print(f"❌ Error starting application: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
