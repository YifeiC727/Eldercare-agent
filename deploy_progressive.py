#!/usr/bin/env python3
"""
渐进式部署脚本 - 逐步启用高级功能
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """检查依赖包是否可用"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        'flask', 'requests', 'jieba', 'sentence_transformers', 
        'numpy', 'pandas', 'python-dotenv'
    ]
    
    optional_packages = [
        'baidu-aip', 'pydub', 'pymongo', 'bson'
    ]
    
    available_packages = []
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            available_packages.append(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            available_packages.append(package)
            print(f"✅ {package} (可选)")
        except ImportError:
            print(f"⚠️ {package} (可选，未安装)")
    
    return available_packages, missing_packages

def check_advanced_modules():
    """检查高级功能模块是否可用"""
    print("\n🔍 检查高级功能模块...")
    
    modules = {
        'strategy': ['strategy_selector', 'llm_generator'],
        'emotion_detection': ['emotion_recognizer'],
        'user_bio': ['improved_user_info_manager'],
        'speech': ['baidu_speech_recognizer'],
        'visualization': ['trend_plotter']
    }
    
    available_modules = {}
    
    for module_group, module_list in modules.items():
        available_modules[module_group] = []
        for module in module_list:
            try:
                if module_group == 'strategy':
                    exec(f"from strategy.{module} import *")
                elif module_group == 'emotion_detection':
                    exec(f"from emotion_detection.{module} import *")
                elif module_group == 'user_bio':
                    exec(f"from user_bio.{module} import *")
                elif module_group == 'speech':
                    exec(f"from speech.{module} import *")
                elif module_group == 'visualization':
                    exec(f"from visualization.{module} import *")
                
                available_modules[module_group].append(module)
                print(f"✅ {module_group}.{module}")
            except Exception as e:
                print(f"❌ {module_group}.{module}: {e}")
    
    return available_modules

def create_deployment_config(available_packages, available_modules):
    """创建部署配置"""
    print("\n📝 创建部署配置...")
    
    config = {
        'version': '2.0.0-progressive',
        'available_packages': available_packages,
        'available_modules': available_modules,
        'features': {
            'basic_chat': True,
            'emotion_analysis': 'emotion_detection' in available_modules and available_modules['emotion_detection'],
            'strategy_selection': 'strategy' in available_modules and available_modules['strategy'],
            'user_management': 'user_bio' in available_modules and available_modules['user_bio'],
            'speech_recognition': 'speech' in available_modules and available_modules['speech'],
            'data_visualization': 'visualization' in available_modules and available_modules['visualization']
        }
    }
    
    # 保存配置
    import json
    with open('deployment_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("✅ 部署配置已保存到 deployment_config.json")
    return config

def generate_startup_script(config):
    """生成启动脚本"""
    print("\n📝 生成启动脚本...")
    
    script_content = f'''#!/usr/bin/env python3
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
        return {{
            'version': '1.0.0-basic',
            'features': {{
                'basic_chat': True,
                'emotion_analysis': False,
                'strategy_selection': False,
                'user_management': False,
                'speech_recognition': False,
                'data_visualization': False
            }}
        }}

def main():
    """主启动函数"""
    config = load_deployment_config()
    
    print("🚀 启动养老护理系统...")
    print(f"📊 版本: {{config.get('version', 'unknown')}}")
    print("🔧 可用功能:")
    
    features = config.get('features', {{}})
    for feature, available in features.items():
        status = "✅" if available else "❌"
        print(f"   {{status}} {{feature}}")
    
    # 根据配置选择启动方式
    if features.get('emotion_analysis') and features.get('strategy_selection'):
        print("\\n🎯 启动增强版系统...")
        os.system("python app_enhanced.py")
    else:
        print("\\n🎯 启动基础版系统...")
        os.system("python app.py")

if __name__ == "__main__":
    main()
'''
    
    with open('start_progressive.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ 启动脚本已生成: start_progressive.py")

def main():
    """主函数"""
    print("🚀 渐进式部署检查开始...")
    print("=" * 50)
    
    # 检查依赖包
    available_packages, missing_packages = check_dependencies()
    
    # 检查高级功能模块
    available_modules = check_advanced_modules()
    
    # 创建部署配置
    config = create_deployment_config(available_packages, available_modules)
    
    # 生成启动脚本
    generate_startup_script(config)
    
    # 显示总结
    print("\n" + "=" * 50)
    print("📊 部署检查总结:")
    print("=" * 50)
    
    features = config['features']
    enabled_features = [k for k, v in features.items() if v]
    disabled_features = [k for k, v in features.items() if not v]
    
    print(f"✅ 已启用功能: {{', '.join(enabled_features)}}")
    if disabled_features:
        print(f"❌ 未启用功能: {{', '.join(disabled_features)}}")
    
    print(f"\\n📦 可用依赖包: {{len(available_packages)}} 个")
    if missing_packages:
        print(f"⚠️ 缺失依赖包: {{', '.join(missing_packages)}}")
    
    print(f"\\n🔧 可用模块: {{sum(len(modules) for modules in available_modules.values())}} 个")
    
    # 建议
    print("\\n💡 建议:")
    if len(enabled_features) >= 3:
        print("   🎉 系统功能丰富，建议使用增强版部署")
    elif len(enabled_features) >= 1:
        print("   ✅ 基础功能可用，建议使用渐进式部署")
    else:
        print("   ⚠️ 功能有限，建议检查依赖包安装")
    
    print("\\n🚀 下一步:")
    print("   1. 提交代码到GitHub")
    print("   2. 在Railway中重新部署")
    print("   3. 系统会自动选择最佳启动方式")

if __name__ == "__main__":
    main()
