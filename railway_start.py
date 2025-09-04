#!/usr/bin/env python3
"""
Railway部署专用启动脚本
优化了Railway平台的配置
"""

import os
import sys
from app import app

def start_railway_server():
    """启动Railway服务器"""
    
    # 设置Railway环境变量
    os.environ.setdefault('FLASK_ENV', 'production')
    os.environ.setdefault('FLASK_DEBUG', 'False')
    
    # Railway配置
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 启动Railway服务器...")
    print(f"📡 监听地址: {host}:{port}")
    print(f"🌐 环境: {os.environ.get('RAILWAY_ENVIRONMENT', 'production')}")
    
    # 检查关键环境变量
    required_vars = ['CAMELLIA_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️ 缺少环境变量: {', '.join(missing_vars)}")
        print("请在Railway控制台设置这些环境变量")
    
    # 启动应用
    try:
        app.run(host=host, port=port, debug=False)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_railway_server()
