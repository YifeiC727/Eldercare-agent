import os
import json
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import csv
from bson import ObjectId

try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

class UserInfoManager:
    def __init__(self):
        self.use_file_storage = False
        
        # 尝试连接MongoDB
        if MONGODB_AVAILABLE:
            try:
                mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
                self.client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=3000)
                # 测试连接
                self.client.admin.command('ping')
                self.db = self.client["eldercare"]
                self.users_collection = self.db["users"]
                self.questions_collection = self.db["questions"]
                self.conversations_collection = self.db["conversations"]
                print("✅ MongoDB连接成功，使用数据库存储")
                self.use_file_storage = False
            except Exception as e:
                print(f"⚠️ MongoDB连接失败: {e}")
                print("🔄 切换到文件存储模式")
                self.use_file_storage = True
        else:
            print("⚠️ pymongo未安装，使用文件存储模式")
            self.use_file_storage = True
        
        # 初始化文件存储
        if self.use_file_storage:
            self.data_dir = "user_bio/data"
            os.makedirs(self.data_dir, exist_ok=True)
            self.users_file = os.path.join(self.data_dir, "users.json")
            self.questions_file = os.path.join(self.data_dir, "questions.json")
            self.conversations_file = os.path.join(self.data_dir, "conversations.json")
            self._init_file_storage()
            print(f"📁 文件存储路径: {self.data_dir}")

    def _init_file_storage(self):
        """初始化文件存储"""
        for file_path in [self.users_file, self.questions_file, self.conversations_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)

    def _load_json_data(self, file_path: str) -> List[Dict]:
        """从JSON文件加载数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换字符串时间戳为datetime对象
            def convert_timestamps(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key in ['created_at', 'updated_at', 'last_login', 'timestamp'] and isinstance(value, str):
                            try:
                                obj[key] = datetime.fromisoformat(value)
                            except ValueError:
                                pass
                        elif isinstance(value, (dict, list)):
                            convert_timestamps(value)
                elif isinstance(obj, list):
                    for item in obj:
                        convert_timestamps(item)
            
            convert_timestamps(data)
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_json_data(self, file_path: str, data: List[Dict]):
        """保存数据到JSON文件"""
        # 转换datetime对象为字符串
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            else:
                return obj
        
        converted_data = convert_datetime(data)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)

    def _generate_id(self) -> str:
        """生成唯一ID"""
        return str(ObjectId())

    # 问题收集策略
    QUESTION_STRATEGY = {
        "children_count": {
            "trigger": "family_related",
            "text": "您有几个子女呢？",
            "type": "number",
            "priority": 1,
            "keywords": ["孩子", "儿子", "女儿", "子女", "家庭", "家人"]
        },
        "hobbies": {
            "trigger": "leisure_related", 
            "text": "您平时喜欢做什么呢？",
            "type": "text",
            "priority": 2,
            "keywords": ["没事", "空闲", "时间", "喜欢", "爱好", "兴趣"]
        },
        "health_status": {
            "trigger": "health_related",
            "text": "您身体怎么样？有什么需要注意的吗？",
            "type": "text", 
            "priority": 3,
            "keywords": ["身体", "健康", "病", "疼", "不舒服", "吃药"]
        },
        "spouse_status": {
            "trigger": "family_related",
            "text": "您的配偶是否健在？",
            "type": "select",
            "priority": 2,
            "keywords": ["老伴", "配偶", "老公", "老婆", "丈夫", "妻子"]
        },
        "living_alone": {
            "trigger": "living_related",
            "text": "您是一个人住吗？",
            "type": "select",
            "priority": 2,
            "keywords": ["住", "房子", "家", "一个人", "独居"]
        }
    }

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
        
        if self.use_file_storage:
            users_data = self._load_json_data(self.users_file)
            user_data["_id"] = self._generate_id()
            users_data.append(user_data)
            self._save_json_data(self.users_file, users_data)
            print(f"✅ 用户创建成功 (文件存储): {data['name']}")
            return user_data["_id"]
        else:
            result = self.users_collection.insert_one(user_data)
            print(f"✅ 用户创建成功 (MongoDB): {data['name']}")
            return str(result.inserted_id)

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """用户登录验证"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if self.use_file_storage:
            users_data = self._load_json_data(self.users_file)
            for user in users_data:
                if user["login_info"]["username"] == username and user["login_info"]["password_hash"] == password_hash:
                    user["_id"] = str(user["_id"])
                    # 更新最后登录时间
                    user["last_login"] = datetime.now()
                    self._save_json_data(self.users_file, users_data)
                    return user
            return None
        else:
            user = self.users_collection.find_one({
                "login_info.username": username,
                "login_info.password_hash": password_hash
            })
            if user:
                # 更新最后登录时间
                self.users_collection.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"last_login": datetime.now()}}
                )
                user["_id"] = str(user["_id"])
                return user
            return None

    def get_user_by_ip(self, user_ip: str) -> Optional[Dict]:
        """根据IP地址获取用户"""
        if self.use_file_storage:
            users_data = self._load_json_data(self.users_file)
            for user in users_data:
                if user["ip_address"] == user_ip:
                    user["_id"] = str(user["_id"])
                    return user
            return None
        else:
            user = self.users_collection.find_one({"ip_address": user_ip})
            if user:
                user["_id"] = str(user["_id"])
                return user
            return None

    def get_user(self, user_id: str) -> Optional[Dict]:
        """根据用户ID获取用户信息"""
        if self.use_file_storage:
            users_data = self._load_json_data(self.users_file)
            print(f"DEBUG: 文件存储模式，查找用户ID: {user_id}")
            print(f"DEBUG: 用户数据文件中的用户数量: {len(users_data)}")
            for user in users_data:
                user_id_in_file = str(user.get("_id", ""))
                print(f"DEBUG: 比较 {user_id_in_file} == {user_id}")
                if user_id_in_file == user_id:
                    user["_id"] = str(user["_id"])
                    print(f"DEBUG: 找到用户: {user.get('basic_info', {}).get('name', 'Unknown')}")
                    return user
            print(f"DEBUG: 未找到用户ID: {user_id}")
            return None
        else:
            user = self.users_collection.find_one({"_id": ObjectId(user_id)})
            if user:
                user["_id"] = str(user["_id"])
                return user
            return None

    def save_conversation(self, user_id: str, user_message: str, ai_reply: str, emotion_data: Dict = None):
        """保存对话记录，同时更新strategy模块的对话历史"""
        try:
            conversation = {
                "_id": self._generate_id(),
                "user_id": user_id,
                "user_message": user_message,
                "ai_reply": ai_reply,
                "emotion_data": emotion_data or {},
                "timestamp": datetime.now()
            }
            
            if self.use_file_storage:
                conversations_data = self._load_json_data(self.conversations_file)
                conversations_data.append(conversation)
                self._save_json_data(self.conversations_file, conversations_data)
            else:
                self.conversations_collection.insert_one(conversation)
            
            # 同时更新用户对话计数
            self.increment_conversation_count(user_id)
            
            print(f"✅ 对话记录已保存: {user_id}")
            
        except Exception as e:
            print(f"保存对话记录失败: {e}")

    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取用户对话历史，限制数量避免信息过载"""
        try:
            if self.use_file_storage:
                conversations_data = self._load_json_data(self.conversations_file)
                user_conversations = [
                    conv for conv in conversations_data 
                    if conv.get("user_id") == user_id
                ]
                # 按时间排序，取最近的记录
                user_conversations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                return user_conversations[:limit]
            else:
                cursor = self.conversations_collection.find(
                    {"user_id": user_id}
                ).sort("timestamp", -1).limit(limit)
                return list(cursor)
        except Exception as e:
            print(f"获取对话历史失败: {e}")
            return []

    def get_recent_conversation_text(self, user_id: str, limit: int = 5) -> str:
        """获取最近的对话文本，用于strategy模块的history参数"""
        """优化版本：只返回最近的对话文本，避免信息过载"""
        try:
            conversations = self.get_conversation_history(user_id, limit)
            
            # 只提取用户消息和AI回复，不包含元数据
            text_parts = []
            for conv in conversations:
                user_msg = conv.get("user_message", "")
                ai_reply = conv.get("ai_reply", "")
                if user_msg:
                    text_parts.append(user_msg)
                if ai_reply:
                    text_parts.append(ai_reply)
            
            return " ".join(text_parts)
            
        except Exception as e:
            print(f"获取对话文本失败: {e}")
            return ""

    def save_question_record(self, user_id: str, question_key: str, question_text: str, user_answer: str = None):
        """保存问题记录，用于追踪问题收集进度"""
        try:
            question_record = {
                "_id": self._generate_id(),
                "user_id": user_id,
                "question_key": question_key,
                "question_text": question_text,
                "user_answer": user_answer,
                "asked_at": datetime.now(),
                "answered_at": datetime.now() if user_answer else None
            }
            
            if self.use_file_storage:
                questions_data = self._load_json_data(self.questions_file)
                questions_data.append(question_record)
                self._save_json_data(self.questions_file, questions_data)
            else:
                self.questions_collection.insert_one(question_record)
                
        except Exception as e:
            print(f"保存问题记录失败: {e}")

    def get_question_history(self, user_id: str) -> List[Dict]:
        """获取用户的问题历史"""
        try:
            if self.use_file_storage:
                questions_data = self._load_json_data(self.questions_file)
                return [
                    q for q in questions_data 
                    if q.get("user_id") == user_id
                ]
            else:
                cursor = self.questions_collection.find({"user_id": user_id})
                return list(cursor)
        except Exception as e:
            print(f"获取问题历史失败: {e}")
            return []

    def update_user_info(self, user_id: str, data: Dict) -> bool:
        """更新用户信息"""
        try:
            # 解析答案
            question_key = data.get("question_key")
            answer = data.get("answer")
            
            if not question_key or not answer:
                return False
            
            parsed_answer = self.parse_user_answer(question_key, answer)
            
            # 更新用户信息
            update_data = {
                "updated_at": datetime.now(),
                f"question_status.{question_key}": "answered"
            }
            
            # 根据问题类型更新不同字段
            if question_key == "name":
                update_data["basic_info.name"] = parsed_answer
            elif question_key == "age":
                update_data["basic_info.age"] = parsed_answer
            elif question_key == "gender":
                update_data["basic_info.gender"] = parsed_answer
            elif question_key == "children_count":
                update_data["family_relation.children_count"] = parsed_answer
            elif question_key == "spouse_status":
                update_data["family_relation.spouse_status"] = parsed_answer
            elif question_key == "living_alone":
                update_data["family_relation.living_alone"] = parsed_answer
            elif question_key == "hobbies":
                update_data["living_habits.hobbies"] = parsed_answer
            elif question_key == "health_status":
                update_data["living_habits.health_status"] = parsed_answer
            
            if self.use_file_storage:
                users_data = self._load_json_data(self.users_file)
                for user in users_data:
                    if user["_id"] == user_id:
                        # 确保嵌套字典存在
                        if "basic_info" not in user:
                            user["basic_info"] = {}
                        if "family_relation" not in user:
                            user["family_relation"] = {}
                        if "living_habits" not in user:
                            user["living_habits"] = {}
                        if "question_status" not in user:
                            user["question_status"] = {}
                        
                        # 更新嵌套字段
                        for key, value in update_data.items():
                            if "." in key:
                                parts = key.split(".")
                                if len(parts) == 2:
                                    section, field = parts
                                    if section in user:
                                        user[section][field] = value
                            else:
                                user[key] = value
                        
                        self._save_json_data(self.users_file, users_data)
                        return True
                return False
            else:
                # MongoDB更新
                mongo_update = {}
                for key, value in update_data.items():
                    if "." in key:
                        mongo_update[key] = value
                    else:
                        mongo_update[key] = value
                
                result = self.users_collection.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$set": mongo_update}
                )
                return result.modified_count > 0
        except Exception as e:
            print(f"更新用户信息失败: {e}")
            return False

    def should_ask_question(self, user_input: str, user_info: Dict, conversation_round: int) -> bool:
        """判断是否应该询问问题"""
        try:
            # 检查对话轮次
            if conversation_round < 3:
                return False
            
            # 检查是否还有未回答的问题
            available_questions = self.get_available_questions(user_info)
            if not available_questions:
                return False
            
            # 检查话题相关性
            relevant_questions = self.get_relevant_questions(user_input, available_questions)
            if not relevant_questions:
                return False
            
            # 检查是否可以自然融入
            if not self.can_naturally_integrate(user_input):
                return False
            
            # 15%的随机概率
            import random
            if random.random() > 0.15:
                return False
            
            return True
        except Exception as e:
            print(f"判断是否询问问题失败: {e}")
            return False

    def get_available_questions(self, user_info: Dict) -> List[Dict]:
        """获取可用的未回答问题"""
        try:
            question_status = user_info.get("question_status", {})
            available = []
            
            for key, config in self.QUESTION_STRATEGY.items():
                if question_status.get(key) != "answered":
                    available.append({"key": key, **config})
            
            return available
        except Exception as e:
            print(f"获取可用问题失败: {e}")
            return []

    def get_relevant_questions(self, user_input: str, available_questions: List[Dict]) -> List[Dict]:
        """根据用户输入获取相关问题"""
        try:
            relevant = []
            for question in available_questions:
                keywords = question.get("keywords", [])
                if any(keyword in user_input for keyword in keywords):
                    relevant.append(question)
            return relevant
        except Exception as e:
            print(f"获取相关问题失败: {e}")
            return []

    def can_naturally_integrate(self, user_input: str) -> bool:
        """检查是否可以自然融入问题"""
        # 如果用户输入太长，不适合打断
        if len(user_input) > 100:
            return False
        
        # 如果用户以问号结尾，不适合打断
        if user_input.strip().endswith("？") or user_input.strip().endswith("?"):
            return False
        
        # 如果包含强烈情感词汇，不适合打断
        strong_emotion_words = ["很", "非常", "特别", "太", "真的", "确实"]
        if any(word in user_input for word in strong_emotion_words):
            return False
        
        return True

    def get_next_question(self, user_input: str, user_info: Dict, conversation_round: int) -> Optional[Dict]:
        """获取下一个要问的问题"""
        try:
            available = self.get_available_questions(user_info)
            relevant = self.get_relevant_questions(user_input, available)
            
            # 优先选择相关的问题
            if relevant:
                return relevant[0]
            
            # 否则选择优先级最高的问题
            if available:
                return sorted(available, key=lambda x: x.get("priority", 999))[0]
            
            return None
        except Exception as e:
            print(f"获取下一个问题失败: {e}")
            return None

    def integrate_question_naturally(self, ai_response: str, next_question: Dict, user_input: str) -> str:
        """自然融入问题到AI回复中"""
        
        question_text = next_question["text"]
        
        # 策略1：话题相关时自然引入
        if self.is_topic_related(user_input, next_question):
            return f"{ai_response} 说到这个，{question_text}"
        
        # 策略2：在回复结尾自然询问
        if ai_response.endswith("。") or ai_response.endswith("！"):
            return f"{ai_response} 对了，{question_text}"
        
        # 策略3：如果回复较短，直接添加
        if len(ai_response) < 50:
            return f"{ai_response} {question_text}"
        
        # 策略4：如果都不合适，暂时不问
        return ai_response

    def is_topic_related(self, user_input: str, question: Dict) -> bool:
        """检查用户输入是否与问题相关"""
        keywords = question.get("keywords", [])
        return any(keyword in user_input for keyword in keywords)

    def parse_user_answer(self, question_key: str, user_response: str) -> Optional:
        """解析用户回答"""
        
        if question_key in ["children_count", "age"]:
            return self.parse_number_answer(user_response)
        elif question_key in ["spouse_status", "living_alone", "gender"]:
            return self.parse_select_answer(user_response, question_key)
        else:
            return self.parse_text_answer(user_response)

    def parse_number_answer(self, response: str) -> Optional[int]:
        """解析数字答案"""
        numbers = re.findall(r'\d+', response)
        if numbers:
            return int(numbers[0])
        return None

    def parse_select_answer(self, response: str, question_key: str) -> Optional[str]:
        """解析选择答案"""
        option_mapping = {
            "gender": {"男": "男", "女": "女", "male": "男", "female": "女"},
            "spouse_status": {"已婚": "已婚", "离异": "离异", "丧偶": "丧偶", "未婚": "未婚"},
            "living_alone": {"独居": "独居", "与配偶同住": "与配偶同住", "与子女同住": "与子女同住", "与家人同住": "与家人同住"}
        }
        
        mapping = option_mapping.get(question_key, {})
        for key, value in mapping.items():
            if key in response:
                return value
        return None

    def parse_text_answer(self, response: str) -> str:
        """解析文本答案"""
        return response.strip()

    def save_emotion_data(self, user_id: str, emotion_scores: Dict):
        """保存情绪数据到CSV，支持用户隔离"""
        try:
            csv_file = "visualization/emotion_trend.csv"
            fieldnames = ["user_id", "timestamp", "anger", "sadness", "joy", "intensity"]
            timestamp_str = datetime.now().isoformat()
            
            file_exists = os.path.exists(csv_file)
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "user_id": user_id,
                    "timestamp": timestamp_str,
                    "anger": emotion_scores.get("anger", 0),
                    "sadness": emotion_scores.get("sadness", 0),
                    "joy": emotion_scores.get("joy", 0),
                    "intensity": emotion_scores.get("intensity", 0)
                })
        except Exception as e:
            print(f"保存情绪数据失败: {e}")

    def increment_conversation_count(self, user_id: str):
        """增加对话计数"""
        try:
            if self.use_file_storage:
                users_data = self._load_json_data(self.users_file)
                for user in users_data:
                    if user["_id"] == user_id:
                        user["conversation_count"] += 1
                        self._save_json_data(self.users_file, users_data)
                        break
            else:
                self.users_collection.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$inc": {"conversation_count": 1}}
                )
        except Exception as e:
            print(f"更新对话计数失败: {e}") 

    def cleanup_old_data(self, days: int = 30):
        """清理旧数据，避免文件过大"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            if self.use_file_storage:
                # 清理对话记录
                conversations_data = self._load_json_data(self.conversations_file)
                conversations_data = [
                    conv for conv in conversations_data
                    if conv.get("timestamp", datetime.min) > cutoff_date
                ]
                self._save_json_data(self.conversations_file, conversations_data)
                
                # 清理问题记录
                questions_data = self._load_json_data(self.questions_file)
                questions_data = [
                    q for q in questions_data
                    if q.get("asked_at", datetime.min) > cutoff_date
                ]
                self._save_json_data(self.questions_file, questions_data)
                
            else:
                # MongoDB清理
                self.conversations_collection.delete_many({
                    "timestamp": {"$lt": cutoff_date}
                })
                self.questions_collection.delete_many({
                    "asked_at": {"$lt": cutoff_date}
                })
                
            print(f"✅ 已清理{days}天前的数据")
            
        except Exception as e:
            print(f"清理旧数据失败: {e}") 