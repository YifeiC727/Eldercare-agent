#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的用户信息管理器
使用增强的数据管理器
"""

import os
import json
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import csv
from bson import ObjectId

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from user_bio.enhanced_data_manager import get_data_manager, EnhancedDataManager

class ImprovedUserInfoManager:
    """改进的用户信息管理器"""
    
    def __init__(self):
        self.data_manager = get_data_manager()
        print(f"✅ 用户信息管理器初始化完成，使用{self.data_manager.get_stats()['storage_type']}存储")
    
    def create_user(self, data: Dict, user_ip: str) -> str:
        """创建新用户"""
        password_hash = hashlib.sha256(data["password"].encode()).hexdigest()
        
        user_data = {
            "basic_info": {
                "name": data["name"],
                "age": data["age"],
                "gender": data["gender"]
            },
            "family_relation": {},
            "living_habits": {},
            "health_status": {},
            "login_info": {
                "username": data["name"],
                "password_hash": password_hash
            },
            "ip_address": user_ip,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "question_status": {},
            "conversation_count": 0,
            "last_login": datetime.now()
        }
        
        user_id = self.data_manager.insert_one("users", user_data)
        print(f"✅ 用户创建成功: {data['name']} (ID: {user_id})")
        return user_id

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """用户认证"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # 先查找用户名
        user = self.data_manager.find_one("users", {
            "login_info.username": username
        })
        
        if user and user.get("login_info", {}).get("password_hash") == password_hash:
            # 更新最后登录时间
            self.data_manager.update_one("users", {"_id": user["_id"]}, {
                "last_login": datetime.now()
            })
            print(f"✅ 用户认证成功: {username}")
            return user
        else:
            print(f"❌ 用户认证失败: {username}")
            if user:
                print(f"🔍 找到用户但密码不匹配")
            else:
                print(f"🔍 未找到用户: {username}")
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """根据ID获取用户信息"""
        return self.data_manager.find_one("users", {"_id": user_id})

    def update_user_info(self, user_id: str, updates: Dict) -> bool:
        """更新用户信息"""
        updates["updated_at"] = datetime.now()
        success = self.data_manager.update_one("users", {"_id": user_id}, updates)
        if success:
            print(f"✅ 用户信息更新成功: {user_id}")
        else:
            print(f"❌ 用户信息更新失败: {user_id}")
        return success

    def save_conversation(self, user_id: str, user_input: str, reply: str, emotion_scores: Dict = None) -> str:
        """保存对话记录"""
        conversation_data = {
            "user_id": user_id,
            "user_input": user_input,
            "ai_response": reply,
            "emotion_scores": emotion_scores or {},
            "created_at": datetime.now()
        }
        
        conversation_id = self.data_manager.insert_one("conversations", conversation_data)
        print(f"✅ 对话记录保存成功: {conversation_id}")
        return conversation_id

    def get_user_conversations(self, user_id: str, limit: int = 50) -> List[Dict]:
        """获取用户对话记录"""
        conversations = self.data_manager.find_many(
            "conversations", 
            {"user_id": user_id}, 
            limit=limit
        )
        
        # 按时间排序
        conversations.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return conversations

    def save_emotion_trend(self, user_id: str, emotion_data: Dict) -> str:
        """保存情感趋势数据"""
        emotion_data.update({
            "user_id": user_id,
            "timestamp": datetime.now()
        })
        
        trend_id = self.data_manager.insert_one("emotion_trends", emotion_data)
        return trend_id

    def get_emotion_trends(self, user_id: str, days: int = 7) -> List[Dict]:
        """获取情感趋势数据"""
        since_date = datetime.now() - timedelta(days=days)
        
        trends = self.data_manager.find_many("emotion_trends", {
            "user_id": user_id,
            "timestamp": {"$gte": since_date}
        })
        
        # 按时间排序
        trends.sort(key=lambda x: x.get("timestamp", ""))
        return trends

    def save_keyword_memory(self, user_id: str, keyword_data: Dict) -> str:
        """保存关键词记忆"""
        keyword_data.update({
            "user_id": user_id,
            "created_at": datetime.now()
        })
        
        memory_id = self.data_manager.insert_one("keyword_memory", keyword_data)
        return memory_id

    def get_keyword_memory(self, user_id: str, limit: int = 100) -> List[Dict]:
        """获取关键词记忆"""
        memories = self.data_manager.find_many(
            "keyword_memory", 
            {"user_id": user_id}, 
            limit=limit
        )
        
        # 按时间排序
        memories.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return memories

    def get_user_stats(self, user_id: str) -> Dict:
        """获取用户统计信息"""
        user = self.get_user_by_id(user_id)
        if not user:
            return {}
        
        # 统计对话数量
        conversations = self.get_user_conversations(user_id)
        conversation_count = len(conversations)
        
        # 统计情感趋势
        emotion_trends = self.get_emotion_trends(user_id, days=30)
        
        # 统计关键词记忆
        keyword_memories = self.get_keyword_memory(user_id)
        
        return {
            "user_id": user_id,
            "name": user.get("basic_info", {}).get("name", ""),
            "conversation_count": conversation_count,
            "emotion_trends_count": len(emotion_trends),
            "keyword_memories_count": len(keyword_memories),
            "last_login": user.get("last_login"),
            "created_at": user.get("created_at")
        }

    def get_all_users(self) -> List[Dict]:
        """获取所有用户"""
        return self.data_manager.find_many("users")

    def delete_user(self, user_id: str) -> bool:
        """删除用户及其相关数据"""
        try:
            # 删除用户
            user_deleted = self.data_manager.delete_one("users", {"_id": user_id})
            
            # 删除相关对话记录
            conversations = self.data_manager.find_many("conversations", {"user_id": user_id})
            for conv in conversations:
                self.data_manager.delete_one("conversations", {"_id": conv["_id"]})
            
            # 删除情感趋势
            trends = self.data_manager.find_many("emotion_trends", {"user_id": user_id})
            for trend in trends:
                self.data_manager.delete_one("emotion_trends", {"_id": trend["_id"]})
            
            # 删除关键词记忆
            memories = self.data_manager.find_many("keyword_memory", {"user_id": user_id})
            for memory in memories:
                self.data_manager.delete_one("keyword_memory", {"_id": memory["_id"]})
            
            print(f"✅ 用户及其相关数据删除成功: {user_id}")
            return user_deleted
            
        except Exception as e:
            print(f"❌ 删除用户失败: {e}")
            return False

    def get_system_stats(self) -> Dict:
        """获取系统统计信息"""
        stats = self.data_manager.get_stats()
        
        # 添加用户统计
        users = self.get_all_users()
        stats["user_stats"] = {
            "total_users": len(users),
            "active_users": len([u for u in users if u.get("last_login")]),
            "new_users_today": len([
                u for u in users 
                if u.get("created_at") and 
                (u["created_at"].date() == datetime.now().date() if isinstance(u["created_at"], datetime) else False)
            ])
        }
        
        return stats

    def backup_data(self, backup_path: str = None) -> str:
        """备份数据"""
        if not backup_path:
            backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        backup_data = {
            "backup_time": datetime.now().isoformat(),
            "users": self.get_all_users(),
            "conversations": self.data_manager.find_many("conversations"),
            "emotion_trends": self.data_manager.find_many("emotion_trends"),
            "keyword_memory": self.data_manager.find_many("keyword_memory")
        }
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 数据备份完成: {backup_path}")
        return backup_path

    def restore_data(self, backup_path: str) -> bool:
        """恢复数据"""
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # 清空现有数据
            for collection in ["users", "conversations", "emotion_trends", "keyword_memory"]:
                # 这里需要实现清空集合的方法
                pass
            
            # 恢复数据
            for user in backup_data.get("users", []):
                self.data_manager.insert_one("users", user)
            
            for conv in backup_data.get("conversations", []):
                self.data_manager.insert_one("conversations", conv)
            
            for trend in backup_data.get("emotion_trends", []):
                self.data_manager.insert_one("emotion_trends", trend)
            
            for memory in backup_data.get("keyword_memory", []):
                self.data_manager.insert_one("keyword_memory", memory)
            
            print(f"✅ 数据恢复完成: {backup_path}")
            return True
            
        except Exception as e:
            print(f"❌ 数据恢复失败: {e}")
            return False
    
    def save_emotion_data(self, user_id: str, emotion_scores: Dict) -> bool:
        """保存用户情感数据"""
        try:
            emotion_data = {
                "user_id": user_id,
                "emotion_scores": emotion_scores,
                "timestamp": datetime.now(),
                "date": datetime.now().date().isoformat()
            }
            
            result = self.data_manager.insert_one("emotion_trends", emotion_data)
            if result:
                print(f"✅ 情感数据保存成功: {user_id}")
                return True
            else:
                print(f"❌ 情感数据保存失败: {user_id}")
                return False
                
        except Exception as e:
            print(f"❌ 保存情感数据时出错: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """获取用户信息"""
        try:
            if isinstance(user_id, str):
                # 尝试将字符串转换为ObjectId
                try:
                    user_id = ObjectId(user_id)
                except:
                    pass
            
            user = self.data_manager.find_one("users", {"_id": user_id})
            if user:
                # 确保created_at字段是datetime对象
                if isinstance(user.get("created_at"), str):
                    try:
                        user["created_at"] = datetime.fromisoformat(user["created_at"])
                    except:
                        pass
                return user
            return None
            
        except Exception as e:
            print(f"❌ 获取用户信息失败: {e}")
            return None

if __name__ == "__main__":
    # 测试改进的用户信息管理器
    manager = ImprovedUserInfoManager()
    
    # 测试创建用户
    test_user_data = {
        "name": "测试用户",
        "age": "65",
        "gender": "男",
        "password": "test123"
    }
    
    user_id = manager.create_user(test_user_data, "127.0.0.1")
    print(f"✅ 用户创建成功，ID: {user_id}")
    
    # 测试用户认证
    user = manager.authenticate_user("测试用户", "test123")
    if user:
        print(f"✅ 用户认证成功: {user['basic_info']['name']}")
    
    # 测试保存对话
    conversation_data = {
        "user_input": "你好",
        "ai_response": "您好！很高兴见到您。",
        "emotion_scores": {"joy": 0.8, "sadness": 0.1},
        "conversation_id": "test_conv_001"
    }
    
    conv_id = manager.save_conversation(user_id, conversation_data)
    print(f"✅ 对话保存成功，ID: {conv_id}")
    
    # 获取系统统计
    stats = manager.get_system_stats()
    print(f"📊 系统统计: {stats}")
    
    print("✅ 所有测试完成")
