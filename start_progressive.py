#!/usr/bin/env python3
"""
渐进式启动脚本 - 根据可用功能自动配置
"""

import os
import sys
import json

def load_deployment_config():
    """加载部署配置"""
    try:
        with open('deployment_config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'version': '1.0.0-basic',
            'features': {
                'basic_chat': True,
                'emotion_analysis': False,
                'strategy_selection': False,
                'user_management': False,
                'speech_recognition': False,
                'data_visualization': False
            }
        }

def main():
    """主启动函数"""
    config = load_deployment_config()
    
    print("🚀 启动养老护理系统...")
    print(f"📊 版本: {config.get('version', 'unknown')}")
    print("🔧 可用功能:")
    
    features = config.get('features', {})
    for feature, available in features.items():
        status = "✅" if available else "❌"
        print(f"   {status} {feature}")
    
    # 根据配置选择启动方式
    if features.get('emotion_analysis') and features.get('strategy_selection'):
        print("\n🎯 启动增强版系统...")
        os.system("python app_enhanced.py")
    else:
        print("\n🎯 启动基础版系统...")
        os.system("python app.py")

if __name__ == "__main__":
    main()
