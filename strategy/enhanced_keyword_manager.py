#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版关键词管理器
直接从conversations.json实时提取关键词，避免重复存储
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import re

class EnhancedKeywordManager:
    def __init__(self, conv_file: str = "user_bio/data/conversations.json"):
        self.conv_file = conv_file
        self.keyword_templates = {
            "膝盖": "你之前提到膝盖不太舒服，现在好些了吗？",
            "腿疼": "你前面说过腿疼，最近有没有缓解一些？",
            "孩子": "你说孩子最近没来看你，有没有让你有些失落？",
            "朋友": "你说的老朋友我记住了，你最近还有和他们联系吗？",
            "花": "你是不是还是喜欢养花呀？最近阳光不错，阳台上的花开得好吗？",
            "孙子": "你提到孙子孙女时笑得很开心，他们最近有没有做什么有趣的事？",
            "老同事": "你曾提起以前的老同事，有没有想起特别难忘的事？",
            "电视剧": "你喜欢的那个老电视剧最近又重播了，看到的时候会不会有些怀旧的感觉？",
            "老伴": "你前面说起老伴，听起来你们有很多共同的回忆。",
            "年轻时候": "你讲到年轻时候的经历，真是特别有意思。",
            "血压": "你最近量血压了吗？有没有注意到什么变化？",
            "糖尿病": "你平时饮食上会特别注意吗？如果有需要可以和我聊聊。",
            "感冒": "最近天气变化大，你有没有注意保暖，身体还好吗？",
            "睡眠": "你最近睡得怎么样？有没有什么困扰你的地方？",
            "孙女": "你和孙女最近有没有什么开心的事情？",
            "家人": "家人最近有没有来看你？和他们在一起的时候开心吗？",
            "儿子": "你的儿子最近有没有来看你？和他们在一起的时候开心吗？",
            "女儿": "你的女儿最近有没有来看你？和他们在一起的时候开心吗？",
            "唱歌": "你还喜欢唱歌吗？有没有最近常唱的歌？"
        }
        
        # 缓存机制
        self._keyword_cache = {}  # {user_id: {keyword: last_seen_timestamp}}
        self._cache_expiry = timedelta(hours=1)  # 缓存1小时
        self._last_cache_update = None

    def _load_conversations(self) -> List[Dict]:
        """加载对话数据"""
        try:
            if os.path.exists(self.conv_file):
                with open(self.conv_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载对话数据失败: {e}")
        return []

    def _extract_keywords_from_text(self, text: str) -> Set[str]:
        """从文本中提取关键词"""
        keywords = set()
        
        # 1. 模板关键词匹配
        for keyword in self.keyword_templates.keys():
            if keyword in text:
                keywords.add(keyword)
        
        # 2. 实体识别（简单版本）
        # 人名、地名、时间等
        entity_patterns = [
            r'[一-龯]{2,4}(?:老师|医生|护士|朋友|同事|邻居)',
            r'[一-龯]{2,4}(?:医院|公园|超市|菜市场)',
            r'昨天|今天|明天|上周|这周|下周|上个月|这个月|下个月',
            r'早上|中午|下午|晚上|凌晨',
            r'春天|夏天|秋天|冬天'
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, text)
            keywords.update(matches)
        
        return keywords

    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效"""
        if not self._last_cache_update:
            return False
        return datetime.now() - self._last_cache_update < self._cache_expiry

    def _update_cache(self):
        """更新关键词缓存"""
        conversations = self._load_conversations()
        new_cache = {}
        
        for conv in conversations:
            user_id = conv.get("user_id")
            if not user_id:
                continue
                
            if user_id not in new_cache:
                new_cache[user_id] = {}
            
            # 提取用户消息和AI回复中的关键词
            user_message = conv.get("user_message", "")
            ai_reply = conv.get("ai_reply", "")
            timestamp = conv.get("timestamp", "")
            
            # 合并文本并提取关键词
            combined_text = f"{user_message} {ai_reply}"
            keywords = self._extract_keywords_from_text(combined_text)
            
            # 更新缓存
            for keyword in keywords:
                if keyword not in new_cache[user_id]:
                    new_cache[user_id][keyword] = timestamp
                else:
                    # 保留最新的时间戳
                    if timestamp > new_cache[user_id][keyword]:
                        new_cache[user_id][keyword] = timestamp
        
        self._keyword_cache = new_cache
        self._last_cache_update = datetime.now()

    def get_recent_keywords(self, user_id: str, limit: int = 10, hours: int = 24) -> List[str]:
        """获取用户最近的关键词"""
        try:
            # 检查缓存
            if not self._is_cache_valid():
                self._update_cache()
            
            if user_id not in self._keyword_cache:
                return []
            
            # 过滤时间范围内的关键词
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_keywords = []
            
            for keyword, timestamp_str in self._keyword_cache[user_id].items():
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp > cutoff_time:
                        recent_keywords.append((keyword, timestamp))
                except:
                    # 如果时间戳解析失败，仍然包含关键词
                    recent_keywords.append((keyword, datetime.now()))
            
            # 按时间排序，返回最新的关键词
            recent_keywords.sort(key=lambda x: x[1], reverse=True)
            return [kw[0] for kw in recent_keywords[:limit]]
            
        except Exception as e:
            print(f"获取用户关键词失败: {e}")
            return []

    def get_keyword_empathy(self, user_id: str, user_input: str = None, limit: int = 3) -> List[str]:
        """获取关键词共情语句"""
        try:
            recent_keywords = self.get_recent_keywords(user_id, limit=limit * 2)  # 获取更多关键词用于筛选
            
            if not recent_keywords:
                return []
            
            # 如果有用户输入，进行语义匹配
            if user_input:
                # 简单的关键词匹配（可以扩展为语义匹配）
                relevant_keywords = []
                for keyword in recent_keywords:
                    if keyword in user_input or any(kw in keyword for kw in user_input.split()):
                        relevant_keywords.append(keyword)
                
                if relevant_keywords:
                    recent_keywords = relevant_keywords[:limit]
            
            # 生成共情语句
            empathy_statements = []
            for keyword in recent_keywords[:limit]:
                if keyword in self.keyword_templates:
                    template = self.keyword_templates[keyword]
                    
                    # 移除可能造成虚假记忆的词汇
                    problematic_phrases = [
                        "你之前提到", "你前面说过", "你曾提起", "你前面说起",
                        "你讲到", "你刚刚提到的", "你前面说"
                    ]
                    
                    safe_template = template
                    for phrase in problematic_phrases:
                        if phrase in safe_template:
                            safe_template = safe_template.replace(phrase, "关于")
                            break
                    
                    empathy_statements.append(safe_template)
            
            return empathy_statements
            
        except Exception as e:
            print(f"获取关键词共情失败: {e}")
            return []

    def get_user_keyword_stats(self, user_id: str) -> Dict:
        """获取用户关键词统计"""
        try:
            if not self._is_cache_valid():
                self._update_cache()
            
            if user_id not in self._keyword_cache:
                return {"total_keywords": 0, "recent_keywords": 0}
            
            user_keywords = self._keyword_cache[user_id]
            total_keywords = len(user_keywords)
            
            # 计算最近24小时的关键词
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_count = 0
            
            for timestamp_str in user_keywords.values():
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp > cutoff_time:
                        recent_count += 1
                except:
                    pass
            
            return {
                "total_keywords": total_keywords,
                "recent_keywords": recent_count,
                "user_id": user_id
            }
            
        except Exception as e:
            print(f"获取用户关键词统计失败: {e}")
            return {"total_keywords": 0, "recent_keywords": 0}

    def get_all_users(self) -> List[str]:
        """获取所有用户ID"""
        try:
            if not self._is_cache_valid():
                self._update_cache()
            
            return list(self._keyword_cache.keys())
            
        except Exception as e:
            print(f"获取用户列表失败: {e}")
            return []

    def clear_cache(self):
        """清除缓存"""
        self._keyword_cache = {}
        self._last_cache_update = None

    def add_keyword_template(self, keyword: str, template: str):
        """添加新的关键词模板"""
        self.keyword_templates[keyword] = template

    def remove_keyword_template(self, keyword: str):
        """移除关键词模板"""
        if keyword in self.keyword_templates:
            del self.keyword_templates[keyword]

    def get_keyword_templates(self) -> Dict[str, str]:
        """获取所有关键词模板"""
        return self.keyword_templates.copy() 