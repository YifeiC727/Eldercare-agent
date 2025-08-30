import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class KeywordMemoryManager:
    def __init__(self, file_path: str = "strategy/keyword_memory.json"):
        self.file_path = file_path
        self.memory = self._load_memory()
        self.expiry_days = 7  # Keyword expiration time

    def _load_memory(self) -> Dict:
        """Load keyword memory"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Failed to load keyword memory: {e}")
        return {}  # Changed to empty dict, no longer using _default

    def _save_memory(self):
        """Save keyword memory"""
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save keyword memory: {e}")

    def _get_user_keywords(self, user_id: str) -> Dict:
        """Get user's keyword storage space"""
        if user_id not in self.memory:
            self.memory[user_id] = {}
        return self.memory[user_id]

    def add_keyword(self, keyword: str, user_id: str, conversation_id: str = None):
        """Add keyword with user isolation support"""
        try:
            if not keyword or len(keyword.strip()) == 0 or not user_id:
                return

            keyword = keyword.strip()
            now = datetime.now()
            
            # Get user's keyword storage space
            user_keywords = self._get_user_keywords(user_id)
            
            # Check if the same keyword already exists
            keyword_exists = False
            for key, data in user_keywords.items():
                if data.get("keyword") == keyword:
                    # Update existing keyword timestamp
                    data["timestamp"] = now.isoformat()
                    if conversation_id:
                        data["conversation_id"] = conversation_id
                    self._save_memory()
                    keyword_exists = True
                    break
            
            # If keyword already exists, return directly
            if keyword_exists:
                return
        
            # Add new keyword
            new_id = str(len(user_keywords) + 1)
            keyword_data = {
                "keyword": keyword,
                "timestamp": now.isoformat(),
                "conversation_id": conversation_id
            }
            
            user_keywords[new_id] = keyword_data
            self._save_memory()
            
        except Exception as e:
            print(f"添加关键词失败: {e}")

    def get_recent_keywords(self, user_id: str, limit: int = 10) -> List[str]:
        """获取指定用户最近的关键词"""
        try:
            if not user_id or user_id not in self.memory:
                return []
                
            keywords = []
            user_keywords = self.memory[user_id]
            
            # Sort by timestamp
            sorted_keywords = sorted(
                user_keywords.items(),
                key=lambda x: x[1].get("timestamp", ""),
                reverse=True
            )
            
            for key, data in sorted_keywords[:limit]:
                keyword = data.get("keyword")
                if keyword:
                    keywords.append(keyword)
            
            return keywords
            
        except Exception as e:
            print(f"获取用户关键词失败: {e}")
            return []

    def get_keyword_info(self, keyword: str, user_id: str) -> Optional[Dict]:
        """获取指定用户关键词的详细信息"""
        try:
            if not user_id or user_id not in self.memory:
                return None
                
            user_keywords = self.memory[user_id]
            for key, data in user_keywords.items():
                if data.get("keyword") == keyword:
                    return data
            return None
        except Exception as e:
            print(f"获取关键词信息失败: {e}")
            return None

    def get_conversation_content(self, conversation_id: str) -> Optional[Dict]:
        """从conversations.json获取对话内容"""
        try:
            conv_file = "user_bio/data/conversations.json"
            if not os.path.exists(conv_file):
                return None
            
            with open(conv_file, 'r', encoding='utf-8') as f:
                conv_data = json.load(f)
            
            for conv in conv_data:
                if conv.get("_id") == conversation_id:
                    return conv
            
            return None
        except Exception as e:
            print(f"获取对话内容失败: {e}")
            return None

    def get_keyword_with_context(self, keyword: str, user_id: str) -> Optional[Dict]:
        """获取关键词及其上下文对话内容"""
        try:
            keyword_info = self.get_keyword_info(keyword, user_id)
            if not keyword_info:
                return None
            
            conversation_id = keyword_info.get("conversation_id")
            if conversation_id:
                conversation = self.get_conversation_content(conversation_id)
                if conversation:
                    return {
            "keyword": keyword,
                        "keyword_info": keyword_info,
                        "conversation": conversation
                    }
            
            return keyword_info
        except Exception as e:
            print(f"获取关键词上下文失败: {e}")
            return None

    def clear_expired(self, user_id: str = None):
        """清理过期的关键词，支持指定用户或全部用户"""
        try:
            now = datetime.now()
            users_to_clean = [user_id] if user_id else list(self.memory.keys())
            
            total_cleaned = 0
            for uid in users_to_clean:
                if uid not in self.memory:
                    continue
                    
                user_keywords = self.memory[uid]
                expired_keys = []
                
                for key, data in user_keywords.items():
                    timestamp_str = data.get("timestamp")
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str)
                            if (now - timestamp).days > self.expiry_days:
                                expired_keys.append(key)
                        except ValueError:
                            # If timestamp format is wrong, delete as well
                            expired_keys.append(key)
                
                # Delete expired keywords
                for key in expired_keys:
                    del user_keywords[key]
                    total_cleaned += 1
                
                # If user has no keywords left, delete user entry
                if not user_keywords:
                    del self.memory[uid]
            
            if total_cleaned > 0:
                self._save_memory()
                print(f"已清理 {total_cleaned} 个过期关键词")
                
        except Exception as e:
            print(f"清理过期关键词失败: {e}")

    def get_keywords_by_user(self, user_id: str) -> List[str]:
        """获取特定用户的所有关键词"""
        try:
            if not user_id or user_id not in self.memory:
                return []
                
            keywords = []
            user_keywords = self.memory[user_id]
            
            for key, data in user_keywords.items():
                keyword = data.get("keyword")
                if keyword:
                    keywords.append(keyword)
            
            return keywords
            
        except Exception as e:
            print(f"获取用户关键词失败: {e}")
            return []

    def remove_keyword(self, keyword: str, user_id: str):
        """删除特定用户的关键词"""
        try:
            if not user_id or user_id not in self.memory:
                return
                
            user_keywords = self.memory[user_id]
            keys_to_remove = []
            
            for key, data in user_keywords.items():
                if data.get("keyword") == keyword:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del user_keywords[key]
            
            if keys_to_remove:
                self._save_memory()
                print(f"已删除用户 {user_id} 的关键词: {keyword}")
                
        except Exception as e:
            print(f"删除关键词失败: {e}")

    def get_memory_stats(self, user_id: str = None) -> Dict:
        """获取记忆统计信息，支持指定用户或全局统计"""
        try:
            if user_id:
                # Single user statistics
                if user_id not in self.memory:
                    return {"total_keywords": 0, "user_id": user_id}
                    
                user_keywords = self.memory[user_id]
                total_keywords = len(user_keywords)
                
                return {
                    "total_keywords": total_keywords,
                    "user_id": user_id
                }
            else:
                # Global statistics
                total_users = len(self.memory)
                total_keywords = sum(len(user_keywords) for user_keywords in self.memory.values())
                
                # User distribution statistics
                user_distribution = {}
                for uid, user_keywords in self.memory.items():
                    user_distribution[uid] = len(user_keywords)
                
                return {
                    "total_keywords": total_keywords,
                    "unique_users": total_users,
                    "user_distribution": user_distribution
                }
            
        except Exception as e:
            print(f"获取记忆统计失败: {e}")
            return {"total_keywords": 0, "unique_users": 0, "user_distribution": {}}

    def delete_user_data(self, user_id: str):
        """删除指定用户的所有关键词数据"""
        try:
            if user_id in self.memory:
                deleted_count = len(self.memory[user_id])
                del self.memory[user_id]
                self._save_memory()
                print(f"已删除用户 {user_id} 的 {deleted_count} 个关键词")
            else:
                print(f"用户 {user_id} 不存在关键词数据")
                
        except Exception as e:
            print(f"删除用户数据失败: {e}")

    def get_all_users(self) -> List[str]:
        """获取所有有关键词记录的用户ID"""
        try:
            return list(self.memory.keys())
        except Exception as e:
            print(f"获取用户列表失败: {e}")
            return []

    def migrate_old_data(self):
        """迁移旧格式数据到新格式（用户隔离）"""
        try:
            if "_default" in self.memory:
                print("检测到旧格式数据，开始迁移...")
                
                # Create default user ID
                default_user_id = "default_user"
                old_data = self.memory["_default"]
                
                # Migrate data
                self.memory[default_user_id] = old_data
                del self.memory["_default"]
                
                self._save_memory()
                print(f"已迁移 {len(old_data)} 个关键词到用户 {default_user_id}")
                
        except Exception as e:
            print(f"数据迁移失败: {e}")