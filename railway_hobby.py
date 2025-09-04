#!/usr/bin/env python3
"""
Railway Hobby计划专用启动脚本
8GB资源，支持完整功能
"""

import os
import sys
from app import app

def start_hobby_server():
    """启动Hobby计划服务器"""
    
    # 设置Hobby环境变量
    os.environ.setdefault('FLASK_ENV', 'production')
    os.environ.setdefault('FLASK_DEBUG', 'False')
    
    # Hobby配置
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 启动Hobby计划养老护理系统...")
    print(f"📡 监听地址: {host}:{port}")
    print(f"🌐 环境: {os.environ.get('RAILWAY_ENVIRONMENT', 'production')}")
    print(f"💾 资源: 8GB RAM (Hobby计划)")
    
    # 检查关键环境变量
    required_vars = ['CAMELLIA_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️ 缺少环境变量: {', '.join(missing_vars)}")
        print("请在Railway控制台设置这些环境变量")
    else:
        print("✅ 环境变量配置完整")
    
    # 检查可选依赖
    optional_deps = {
        'pyaudio': '音频录制功能',
        'librosa': '音频处理功能', 
        'hanlp': '高级NLP功能',
        'faiss': '向量搜索功能'
    }
    
    for dep, desc in optional_deps.items():
        try:
            __import__(dep)
            print(f"✅ {desc}可用")
        except ImportError:
            print(f"⚠️ {desc}不可用，将使用降级模式")
    
    # 启动应用
    try:
        print("🌐 启动完整版养老护理系统...")
        app.run(host=host, port=port, debug=False)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_hobby_server()
