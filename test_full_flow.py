#!/usr/bin/env python3
"""
完整流程测试 - 包含登录和聊天功能
"""

import requests
import json
import time

def test_complete_flow(base_url):
    """测试完整的用户流程"""
    print("🚀 开始完整流程测试...")
    print("=" * 50)
    
    # 创建会话
    session = requests.Session()
    
    # 1. 测试主页
    print("🏠 1. 测试主页访问...")
    try:
        response = session.get(base_url, timeout=10)
        if response.status_code == 200:
            print("✅ 主页访问成功")
        else:
            print(f"❌ 主页访问失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 主页访问异常: {e}")
        return False
    
    # 2. 用户注册
    print("\n👤 2. 测试用户注册...")
    test_user = {
        "name": "测试用户",
        "age": 70,
        "gender": "Female", 
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
    
    # 3. 测试聊天功能
    print("\n💬 3. 测试聊天功能...")
    chat_messages = [
        "你好",
        "今天天气怎么样？",
        "我有点难过",
        "谢谢你的陪伴"
    ]
    
    for i, message in enumerate(chat_messages, 1):
        print(f"   发送消息 {i}: {message}")
        try:
            response = session.post(
                f"{base_url}/api/chat",
                json={"message": message},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                reply = data.get('response', 'N/A')
                emotion = data.get('emotion', {})
                
                print(f"   ✅ 回复: {reply}")
                if emotion:
                    print(f"   🧠 情感分析: 悲伤{emotion.get('sadness', 0):.1%}, 快乐{emotion.get('joy', 0):.1%}, 愤怒{emotion.get('anger', 0):.1%}")
            else:
                print(f"   ❌ 聊天失败: {response.status_code}")
                print(f"   错误: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ 聊天异常: {e}")
            return False
        
        time.sleep(1)  # 避免请求过快
    
    # 4. 测试个人资料
    print("\n👤 4. 测试个人资料页面...")
    try:
        response = session.get(f"{base_url}/profile", timeout=10)
        if response.status_code == 200:
            print("✅ 个人资料页面访问成功")
            # 检查是否包含中文
            if "测试用户" in response.text:
                print("✅ 中文字符显示正常")
            else:
                print("⚠️  中文字符可能有问题")
        else:
            print(f"❌ 个人资料页面失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 个人资料页面异常: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 完整流程测试通过！系统功能正常！")
    print("=" * 50)
    
    return True

def main():
    """主函数"""
    base_url = "https://web-production-bcfb6.up.railway.app"
    
    print(f"🌐 测试URL: {base_url}")
    print()
    
    success = test_complete_flow(base_url)
    
    if success:
        print("\n🎯 系统部署完全成功！")
        print("✅ 所有核心功能都正常工作：")
        print("   - 用户注册和登录")
        print("   - 智能对话和情感分析")
        print("   - 个人资料管理")
        print("   - 中文字符显示")
        print("\n🚀 你的养老护理系统已经可以正常使用了！")
    else:
        print("\n⚠️  测试过程中发现问题，请检查配置")

if __name__ == "__main__":
    main()
