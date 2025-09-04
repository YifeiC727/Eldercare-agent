#!/usr/bin/env python3
"""
部署测试脚本 - 验证系统功能
"""

import requests
import json
import time

def test_health_check(base_url):
    """测试健康检查端点"""
    print("🏥 测试健康检查...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查通过: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

def test_homepage(base_url):
    """测试主页"""
    print("🏠 测试主页...")
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("✅ 主页访问正常")
            return True
        else:
            print(f"❌ 主页访问失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 主页访问异常: {e}")
        return False

def test_user_registration(base_url):
    """测试用户注册"""
    print("👤 测试用户注册...")
    try:
        test_user = {
            "name": "测试用户",
            "age": 70,
            "gender": "Female",
            "password": "test123",
            "confirm_password": "test123"
        }
        
        response = requests.post(
            f"{base_url}/api/users",
            json=test_user,
            timeout=10
        )
        
        if response.status_code == 201:
            data = response.json()
            print(f"✅ 用户注册成功: {data.get('user_id')}")
            return data.get('user_id')
        else:
            print(f"❌ 用户注册失败: {response.status_code}")
            if response.text:
                print(f"   错误信息: {response.text}")
            return None
    except Exception as e:
        print(f"❌ 用户注册异常: {e}")
        return None

def test_chat_functionality(base_url, user_id):
    """测试聊天功能"""
    print("💬 测试聊天功能...")
    try:
        # 创建会话
        session = requests.Session()
        
        # 模拟登录（简化版）
        chat_data = {
            "message": "你好，今天天气怎么样？"
        }
        
        response = session.post(
            f"{base_url}/api/chat",
            json=chat_data,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 聊天功能正常")
            print(f"   回复: {data.get('response', 'N/A')}")
            if 'emotion' in data:
                emotion = data['emotion']
                print(f"   情感分析: 悲伤{emotion.get('sadness', 0):.1%}, 快乐{emotion.get('joy', 0):.1%}")
            return True
        else:
            print(f"❌ 聊天功能失败: {response.status_code}")
            if response.text:
                print(f"   错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 聊天功能异常: {e}")
        return False

def test_profile_page(base_url):
    """测试个人资料页面"""
    print("👤 测试个人资料页面...")
    try:
        response = requests.get(f"{base_url}/profile", timeout=10)
        if response.status_code in [200, 401]:  # 401表示需要登录，这是正常的
            print("✅ 个人资料页面正常")
            return True
        else:
            print(f"❌ 个人资料页面失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 个人资料页面异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试部署的系统...")
    print("=" * 50)
    
    # 获取用户输入的URL
    base_url = input("请输入你的Railway应用URL (例如: https://your-app.railway.app): ").strip()
    
    if not base_url:
        print("❌ 未提供URL，退出测试")
        return
    
    if not base_url.startswith('http'):
        base_url = f"https://{base_url}"
    
    print(f"🌐 测试URL: {base_url}")
    print()
    
    # 执行测试
    tests = [
        ("健康检查", lambda: test_health_check(base_url)),
        ("主页访问", lambda: test_homepage(base_url)),
        ("个人资料页面", lambda: test_profile_page(base_url)),
        ("用户注册", lambda: test_user_registration(base_url)),
    ]
    
    results = []
    user_id = None
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if test_name == "用户注册" and result:
                user_id = result
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
        print()
    
    # 测试聊天功能（如果有用户ID）
    if user_id:
        chat_result = test_chat_functionality(base_url, user_id)
        results.append(("聊天功能", chat_result))
    else:
        print("⚠️  跳过聊天功能测试（用户注册失败）")
        results.append(("聊天功能", False))
    
    # 显示结果
    print("=" * 50)
    print("📊 测试结果汇总:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"📈 测试通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！系统部署成功！")
    elif passed >= total * 0.8:
        print("✅ 大部分测试通过，系统基本正常")
    else:
        print("⚠️  部分测试失败，需要检查配置")
    
    print("\n💡 如果测试失败，请检查:")
    print("   1. Railway部署是否完成")
    print("   2. 环境变量是否正确设置")
    print("   3. DeepSeek API密钥是否有效")
    print("   4. 网络连接是否正常")

if __name__ == "__main__":
    main()
