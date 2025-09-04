#!/usr/bin/env python3
"""
智能启动脚本 - 逐步尝试启动不同版本的系统
"""

import os
import sys
import traceback

# 设置环境变量
os.environ.setdefault('FLASK_ENV', 'production')

def try_import_module(module_name, description):
    """尝试导入模块"""
    try:
        __import__(module_name)
        print(f"✅ {description} - 可用")
        return True
    except ImportError as e:
        print(f"⚠️ {description} - 不可用: {e}")
        return False
    except Exception as e:
        print(f"❌ {description} - 错误: {e}")
        return False

def check_dependencies():
    """检查依赖"""
    print("🔍 检查系统依赖...")
    
    dependencies = [
        ("flask", "Flask框架"),
        ("werkzeug", "Werkzeug工具"),
        ("bson", "BSON支持"),
        ("numpy", "NumPy数值计算"),
        ("pandas", "Pandas数据处理"),
        ("sklearn", "Scikit-learn机器学习"),
        ("jieba", "中文分词"),
        ("requests", "HTTP请求"),
        ("openai", "OpenAI API"),
        ("sentence_transformers", "句子转换器"),
        ("faiss", "FAISS向量搜索"),
        ("pymongo", "MongoDB连接"),
        ("matplotlib", "Matplotlib绘图"),
        ("seaborn", "Seaborn统计绘图")
    ]
    
    available_deps = []
    missing_deps = []
    
    for module, desc in dependencies:
        if try_import_module(module, desc):
            available_deps.append(module)
        else:
            missing_deps.append(module)
    
    print(f"📊 依赖检查完成: {len(available_deps)}/{len(dependencies)} 可用")
    
    if missing_deps:
        print(f"⚠️ 缺失依赖: {', '.join(missing_deps)}")
    
    return available_deps, missing_deps

def start_full_system():
    """启动完整系统"""
    print("🚀 尝试启动完整系统...")
    
    try:
        # 检查关键依赖
        critical_deps = ["flask", "bson", "numpy", "pandas"]
        available_deps, missing_deps = check_dependencies()
        
        missing_critical = [dep for dep in critical_deps if dep in missing_deps]
        if missing_critical:
            print(f"❌ 缺少关键依赖: {missing_critical}")
            return False
        
        # 尝试导入完整系统
        print("📦 导入完整系统模块...")
        from app import app
        print("✅ 完整系统导入成功")
        
        # 启动应用
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
            return True
            
        except ImportError:
            print("⚠️ Gunicorn不可用，使用Flask开发服务器")
            app.run(host='0.0.0.0', port=port, debug=False)
            return True
            
    except Exception as e:
        print(f"❌ 完整系统启动失败: {e}")
        traceback.print_exc()
        return False

def start_basic_system():
    """启动基础系统"""
    print("🔄 启动基础系统...")
    
    try:
        from flask import Flask, jsonify
        
        # 创建基础Flask应用
        app = Flask(__name__)
        app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
        
        @app.route('/')
        def index():
            return """
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
                        max-width: 800px; 
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
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🏥 养老护理智能助手</h1>
                        <p>专业的老年人健康管理和情感陪伴服务</p>
                    </div>
                    <div class="content">
                        <div class="status">✅ 基础服务运行正常</div>
                        
                        <div class="feature-list">
                            <h3>🚀 可用功能</h3>
                            <ul>
                                <li><a href="/health">系统健康检查</a></li>
                                <li><a href="/status">系统状态信息</a></li>
                                <li><a href="/dependencies">依赖检查</a></li>
                            </ul>
                        </div>
                        
                        <div class="feature-list">
                            <h3>📋 系统信息</h3>
                            <ul>
                                <li>版本: 1.2.0 (智能启动版)</li>
                                <li>状态: 基础模式运行中</li>
                                <li>启动时间: """ + str(os.environ.get('START_TIME', '未知')) + """</li>
                            </ul>
                        </div>
                        
                        <p style="color: #6c757d; font-size: 14px;">
                            完整功能正在加载中，请稍后刷新页面或联系管理员。
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """
        
        @app.route('/health')
        def health():
            return jsonify({
                "status": "healthy",
                "message": "基础服务正常运行",
                "version": "1.2.0-basic",
                "mode": "basic"
            }), 200
        
        @app.route('/status')
        def status():
            available_deps, missing_deps = check_dependencies()
            return jsonify({
                "service": "Eldercare Agent",
                "version": "1.2.0",
                "status": "running",
                "mode": "basic",
                "environment": os.environ.get('FLASK_ENV', 'development'),
                "python_version": sys.version,
                "dependencies": {
                    "available": available_deps,
                    "missing": missing_deps,
                    "total": len(available_deps) + len(missing_deps)
                }
            }), 200
        
        @app.route('/dependencies')
        def dependencies():
            available_deps, missing_deps = check_dependencies()
            return jsonify({
                "available_dependencies": available_deps,
                "missing_dependencies": missing_deps,
                "dependency_count": {
                    "total": len(available_deps) + len(missing_deps),
                    "available": len(available_deps),
                    "missing": len(missing_deps)
                }
            }), 200
        
        if __name__ == '__main__':
            port = int(os.environ.get('PORT', 5000))
            print(f"🚀 启动基础系统，端口: {port}")
            app.run(host='0.0.0.0', port=port, debug=False)
            
    except Exception as e:
        print(f"❌ 基础系统启动失败: {e}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # 设置启动时间
    os.environ['START_TIME'] = str(os.environ.get('START_TIME', '未知'))
    
    print("🚀 智能启动系统...")
    print("=" * 50)
    
    # 首先尝试启动完整系统
    if not start_full_system():
        print("=" * 50)
        print("🔄 完整系统启动失败，回退到基础系统...")
        start_basic_system()
