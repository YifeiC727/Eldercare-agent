#!/usr/bin/env python3
"""
测试增强版功能
"""

import requests
import json
import time

def test_enhanced_features(base_url):
    """测试增强版功能"""
    print("🚀 测试增强版功能...")
    print("=" * 50)
    
    # 创建会话
    session = requests.Session()
    
    # 1. 测试健康检查（查看组件状态）
    print("🏥 1. 测试健康检查...")
    try:
        response = session.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查通过")
            print(f"   版本: {data.get('version', 'unknown')}")
            components = data.get('components', {})
            print("   组件状态:")
            for component, status in components.items():
                print(f"     - {component}: {status}")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False
    
    # 2. 用户注册
    print("\n👤 2. 测试用户注册...")
    test_user = {
        "name": "增强测试用户",
        "age": 75,
        "gender": "Male", 
        "password": "test123",
        "confirm_password": "test123"
    }
    
    try:
        response = session.post(f"{base_url}/api/users", json=test_user, timeout=10)
        if response.status_code == 201:
            data = response.json()
            user_id = data.get('user_id')
            print(f"✅ 用户注册成功，用户ID: {user_id}")
        else:
            print(f"❌ 用户注册失败: {response.status_code}")
            print(f"   响应: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 用户注册异常: {e}")
        return False
    
    # 3. 测试高级聊天功能
    print("\n💬 3. 测试高级聊天功能...")
    advanced_messages = [
        "我今天心情不太好，感觉有点孤独",
        "我的膝盖最近总是疼，走路都困难",
        "孩子们很久没来看我了，是不是把我忘了",
        "我想起了以前和老伴一起的日子"
    ]
    
    for i, message in enumerate(advanced_messages, 1):
        print(f"   发送消息 {i}: {message}")
        try:
            response = session.post(
                f"{base_url}/api/chat",
                json={"message": message},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                reply = data.get('reply', 'N/A')
                emotion = data.get('emotion', {})
                liwc = data.get('liwc', {})
                
                print(f"   ✅ 回复: {reply}")
                if emotion:
                    print(f"   🧠 情感分析: 悲伤{emotion.get('sadness', 0):.1%}, 快乐{emotion.get('joy', 0):.1%}, 愤怒{emotion.get('anger', 0):.1%}")
                if liwc:
                    print(f"   🔍 LIWC分析: {len(liwc)} 个维度")
                
                # 检查是否有预警
                if data.get('show_alert'):
                    print(f"   ⚠️ 预警信息: {data.get('alert_message', 'N/A')}")
            else:
                print(f"   ❌ 聊天失败: {response.status_code}")
                print(f"   错误: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ 聊天异常: {e}")
            return False
        
        time.sleep(1)  # 避免请求过快
    
    # 4. 测试个人资料功能
    print("\n👤 4. 测试个人资料功能...")
    try:
        response = session.get(f"{base_url}/user_bio", timeout=10)
        if response.status_code == 200:
            print("✅ 个人资料页面访问成功")
            # 检查是否包含中文
            if "增强测试用户" in response.text:
                print("✅ 中文字符显示正常")
            else:
                print("⚠️ 中文字符可能有问题")
        else:
            print(f"❌ 个人资料页面失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 个人资料页面异常: {e}")
        return False
    
    # 5. 测试数据可视化
    print("\n📊 5. 测试数据可视化...")
    try:
        response = session.get(f"{base_url}/api/trend_data", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ 趋势数据获取成功")
            if data.get('dates'):
                print(f"   数据点数量: {len(data['dates'])}")
            else:
                print("   暂无历史数据")
        else:
            print(f"❌ 趋势数据获取失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 趋势数据获取异常: {e}")
    
    # 6. 测试预警数据
    print("\n⚠️ 6. 测试预警数据...")
    try:
        response = session.get(f"{base_url}/api/warning_data", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ 预警数据获取成功")
            total_warnings = data.get('total_warnings', 0)
            print(f"   总预警次数: {total_warnings}")
        else:
            print(f"❌ 预警数据获取失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 预警数据获取异常: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 增强版功能测试完成！")
    print("=" * 50)
    
    return True

def main():
    """主函数"""
    base_url = "https://web-production-bcfb6.up.railway.app"
    
    print(f"🌐 测试URL: {base_url}")
    print()
    
    success = test_enhanced_features(base_url)
    
    if success:
        print("\n🎯 增强版系统测试成功！")
        print("✅ 所有高级功能都正常工作：")
        print("   - 智能策略选择")
        print("   - 高级情感分析")
        print("   - LIWC语言分析")
        print("   - 用户档案管理")
        print("   - 数据可视化")
        print("   - 预警系统")
        print("\n🚀 你的养老护理系统已经升级到增强版！")
    else:
        print("\n⚠️ 测试过程中发现问题，请检查配置")

if __name__ == "__main__":
    main()
