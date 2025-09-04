#!/usr/bin/env python3
"""
Railway专用启动脚本 - 确保所有功能正常加载
"""

import os
import sys
import traceback

# 设置环境变量
os.environ.setdefault('FLASK_ENV', 'production')

def main():
    """主启动函数"""
    print("🚀 Railway启动脚本开始...")
    
    try:
        # 尝试导入兼容版应用
        print("📦 尝试导入兼容版应用...")
        from app_compatible import app
        print("✅ 兼容版应用导入成功")
        
        # 检查高级功能状态
        from app_compatible import ADVANCED_FEATURES
        print("🔧 高级功能状态:")
        for feature, status in ADVANCED_FEATURES.items():
            print(f"   - {feature}: {'✅' if status else '❌'}")
        
        # 启动应用
        port = int(os.environ.get('PORT', 5000))
        print(f"🌐 启动应用，端口: {port}")
        
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
            
            print("🚀 使用Gunicorn启动...")
            StandaloneApplication(app, options).run()
            
        except ImportError:
            print("⚠️ Gunicorn不可用，使用Flask开发服务器...")
            app.run(host='0.0.0.0', port=port, debug=False)
            
    except Exception as e:
        print(f"❌ 兼容版应用导入失败: {e}")
        print("🔄 回退到简化版...")
        
        # 回退到简化版
        try:
            from start_simple_complete import create_fallback_app
            app = create_fallback_app()
            print("✅ 简化版应用启动成功")
            
            port = int(os.environ.get('PORT', 5000))
            app.run(host='0.0.0.0', port=port, debug=False)
            
        except Exception as e2:
            print(f"❌ 简化版应用也启动失败: {e2}")
            print("💥 系统启动失败")
            sys.exit(1)

if __name__ == '__main__':
    main()