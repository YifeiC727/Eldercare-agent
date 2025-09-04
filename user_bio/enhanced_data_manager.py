#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的数据管理器
支持MongoDB和文件存储的智能切换
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """数据库配置"""
    mongodb_uri: str = "mongodb://localhost:27017/"
    database_name: str = "eldercare"
    username: str = "eldercare_user"
    password: str = "eldercare_pass"
    connection_timeout: int = 5000
    use_file_fallback: bool = True
    file_storage_path: str = "user_bio/data"

class EnhancedDataManager:
    """增强的数据管理器"""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.use_mongodb = False
        self.client = None
        self.db = None
        self.collections = {}
        
        # 初始化存储
        self._initialize_storage()
    
    def _initialize_storage(self):
        """初始化存储系统"""
        logger.info("🔄 初始化数据存储系统...")
        
        # 尝试连接MongoDB
        if MONGODB_AVAILABLE and self._try_mongodb_connection():
            self.use_mongodb = True
            logger.info("✅ 使用MongoDB存储")
        else:
            self.use_mongodb = False
            self._initialize_file_storage()
            logger.info("📁 使用文件存储模式")
            logger.info("💡 提示: 如需使用MongoDB，请启动Docker服务并运行: docker-compose up -d")
    
    def _try_mongodb_connection(self) -> bool:
        """尝试连接MongoDB"""
        if not MONGODB_AVAILABLE:
            logger.warning("⚠️ pymongo未安装")
            return False
        
        # 尝试多种连接方式
        connection_uris = [
            f"mongodb://{self.config.username}:{self.config.password}@localhost:27017/{self.config.database_name}",
            "mongodb://eldercare_user:eldercare_pass@localhost:27017/eldercare",
            "mongodb://admin:eldercare123@localhost:27017/eldercare"
        ]
        
        for uri in connection_uris:
            try:
                logger.info(f"🔄 尝试连接: {uri.split('@')[0]}@...")
                self.client = MongoClient(
                    uri, 
                    serverSelectionTimeoutMS=self.config.connection_timeout
                )
                
                # 测试连接
                self.client.admin.command('ping')
                
                # 设置数据库和集合
                self.db = self.client[self.config.database_name]
                self.collections = {
                    'users': self.db['users'],
                    'conversations': self.db['conversations'],
                    'questions': self.db['questions'],
                    'emotion_trends': self.db['emotion_trends'],
                    'keyword_memory': self.db['keyword_memory']
                }
                
                # 创建索引
                self._create_indexes()
                
                logger.info("✅ MongoDB连接成功")
                return True
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"❌ 连接失败: {e}")
                continue
            except Exception as e:
                logger.error(f"❌ 连接错误: {e}")
                continue
        
        logger.error("❌ 所有MongoDB连接尝试都失败")
        return False
    
    def _create_indexes(self):
        """创建数据库索引"""
        try:
            # 用户集合索引
            self.collections['users'].create_index("login_info.username", unique=True)
            self.collections['users'].create_index("created_at")
            self.collections['users'].create_index("last_login")
            
            # 对话集合索引
            self.collections['conversations'].create_index("user_id")
            self.collections['conversations'].create_index("created_at")
            self.collections['conversations'].create_index("conversation_id")
            
            # 情感趋势索引
            self.collections['emotion_trends'].create_index([("user_id", 1), ("timestamp", 1)])
            self.collections['emotion_trends'].create_index("timestamp")
            
            # 关键词记忆索引
            self.collections['keyword_memory'].create_index([("user_id", 1), ("keyword", 1)])
            self.collections['keyword_memory'].create_index("created_at")
            
            logger.info("✅ 数据库索引创建完成")
            
        except Exception as e:
            logger.warning(f"⚠️ 索引创建失败: {e}")
    
    def _initialize_file_storage(self):
        """初始化文件存储"""
        if not self.config.use_file_fallback:
            raise Exception("MongoDB连接失败且文件存储被禁用")
        
        self.data_dir = Path(self.config.file_storage_path)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径
        self.files = {
            'users': self.data_dir / 'users.json',
            'conversations': self.data_dir / 'conversations.json',
            'questions': self.data_dir / 'questions.json',
            'emotion_trends': self.data_dir / 'emotion_trends.json',
            'keyword_memory': self.data_dir / 'keyword_memory.json'
        }
        
        # 初始化文件
        for file_path in self.files.values():
            if not file_path.exists():
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
        
        logger.info(f"📁 文件存储初始化完成: {self.data_dir}")
    
    def _load_json_data(self, file_path: Path) -> List[Dict]:
        """加载JSON数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_json_data(self, file_path: Path, data: List[Dict]):
        """保存JSON数据"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def insert_one(self, collection_name: str, document: Dict) -> str:
        """插入单个文档"""
        if self.use_mongodb:
            try:
                result = self.collections[collection_name].insert_one(document)
                return str(result.inserted_id)
            except Exception as e:
                logger.error(f"❌ MongoDB插入失败: {e}")
                if self.config.use_file_fallback:
                    logger.info("🔄 切换到文件存储")
                    self.use_mongodb = False
                    self._initialize_file_storage()
                    return self.insert_one(collection_name, document)
                raise
        
        # 文件存储
        file_path = self.files[collection_name]
        data = self._load_json_data(file_path)
        
        # 生成ID
        document['_id'] = self._generate_id()
        data.append(document)
        
        self._save_json_data(file_path, data)
        return document['_id']
    
    def find_one(self, collection_name: str, query: Dict) -> Optional[Dict]:
        """查找单个文档"""
        if self.use_mongodb:
            try:
                return self.collections[collection_name].find_one(query)
            except Exception as e:
                logger.error(f"❌ MongoDB查询失败: {e}")
                if self.config.use_file_fallback:
                    self.use_mongodb = False
                    self._initialize_file_storage()
                    return self.find_one(collection_name, query)
                raise
        
        # 文件存储
        file_path = self.files[collection_name]
        data = self._load_json_data(file_path)
        
        for doc in data:
            if self._match_query(doc, query):
                return doc
        
        return None
    
    def _match_query(self, doc: Dict, query: Dict) -> bool:
        """检查文档是否匹配查询条件"""
        for key, value in query.items():
            if '.' in key:
                # 处理嵌套字段查询，如 "login_info.username"
                keys = key.split('.')
                current = doc
                try:
                    for k in keys:
                        current = current[k]
                    if current != value:
                        return False
                except (KeyError, TypeError):
                    return False
            else:
                # 处理普通字段查询
                if doc.get(key) != value:
                    return False
        return True
    
    def find_many(self, collection_name: str, query: Dict = None, limit: int = None) -> List[Dict]:
        """查找多个文档"""
        if self.use_mongodb:
            try:
                cursor = self.collections[collection_name].find(query or {})
                if limit:
                    cursor = cursor.limit(limit)
                return list(cursor)
            except Exception as e:
                logger.error(f"❌ MongoDB查询失败: {e}")
                if self.config.use_file_fallback:
                    self.use_mongodb = False
                    self._initialize_file_storage()
                    return self.find_many(collection_name, query, limit)
                raise
        
        # 文件存储
        file_path = self.files[collection_name]
        data = self._load_json_data(file_path)
        
        if query:
            filtered_data = []
            for doc in data:
                if self._match_query(doc, query):
                    filtered_data.append(doc)
            data = filtered_data
        
        if limit:
            data = data[:limit]
        
        return data
    
    def update_one(self, collection_name: str, query: Dict, update: Dict) -> bool:
        """更新单个文档"""
        if self.use_mongodb:
            try:
                result = self.collections[collection_name].update_one(query, {"$set": update})
                return result.modified_count > 0
            except Exception as e:
                logger.error(f"❌ MongoDB更新失败: {e}")
                if self.config.use_file_fallback:
                    self.use_mongodb = False
                    self._initialize_file_storage()
                    return self.update_one(collection_name, query, update)
                raise
        
        # 文件存储
        file_path = self.files[collection_name]
        data = self._load_json_data(file_path)
        
        for i, doc in enumerate(data):
            if all(doc.get(k) == v for k, v in query.items()):
                data[i].update(update)
                self._save_json_data(file_path, data)
                return True
        
        return False
    
    def delete_one(self, collection_name: str, query: Dict) -> bool:
        """删除单个文档"""
        if self.use_mongodb:
            try:
                result = self.collections[collection_name].delete_one(query)
                return result.deleted_count > 0
            except Exception as e:
                logger.error(f"❌ MongoDB删除失败: {e}")
                if self.config.use_file_fallback:
                    self.use_mongodb = False
                    self._initialize_file_storage()
                    return self.delete_one(collection_name, query)
                raise
        
        # 文件存储
        file_path = self.files[collection_name]
        data = self._load_json_data(file_path)
        
        for i, doc in enumerate(data):
            if all(doc.get(k) == v for k, v in query.items()):
                data.pop(i)
                self._save_json_data(file_path, data)
                return True
        
        return False
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        import uuid
        return str(uuid.uuid4())
    
    def get_stats(self) -> Dict:
        """获取存储统计信息"""
        stats = {
            "storage_type": "MongoDB" if self.use_mongodb else "File",
            "collections": {}
        }
        
        for collection_name in ['users', 'conversations', 'questions', 'emotion_trends', 'keyword_memory']:
            try:
                if self.use_mongodb:
                    count = self.collections[collection_name].count_documents({})
                else:
                    data = self._load_json_data(self.files[collection_name])
                    count = len(data)
                
                stats["collections"][collection_name] = count
            except Exception as e:
                stats["collections"][collection_name] = f"Error: {e}"
        
        return stats
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB连接已关闭")

# 全局实例
_data_manager = None

def get_data_manager() -> EnhancedDataManager:
    """获取全局数据管理器实例"""
    global _data_manager
    if _data_manager is None:
        _data_manager = EnhancedDataManager()
    return _data_manager

if __name__ == "__main__":
    # 测试数据管理器
    manager = EnhancedDataManager()
    
    # 测试插入
    test_doc = {
        "name": "测试用户",
        "created_at": datetime.now(),
        "test": True
    }
    
    doc_id = manager.insert_one("users", test_doc)
    print(f"✅ 插入成功，ID: {doc_id}")
    
    # 测试查询
    found_doc = manager.find_one("users", {"test": True})
    print(f"✅ 查询成功: {found_doc}")
    
    # 获取统计信息
    stats = manager.get_stats()
    print(f"📊 存储统计: {stats}")
    
    manager.close()
