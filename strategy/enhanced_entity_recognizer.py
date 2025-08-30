#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced entity recognition module
Use professional libraries and dictionaries for more comprehensive entity recognition
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import defaultdict

try:
    import jieba
    import jieba.posseg as pseg
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Warning: jieba not installed, will use basic mode")

try:
    import hanlp
    HANLP_AVAILABLE = True
except ImportError:
    HANLP_AVAILABLE = False
    print("Warning: hanlp not installed, will use basic mode")

class EnhancedEntityRecognizer:
    def __init__(self, use_professional_libs=True, enable_entity_recognition=True, enable_semantic_matching=True):
        """Initialize enhanced entity recognizer"""
        self.use_professional_libs = use_professional_libs
        self.enable_entity_recognition = enable_entity_recognition
        self.enable_semantic_matching = enable_semantic_matching
        
        # Load semantic model
        self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        # Load professional libraries
        if use_professional_libs and enable_entity_recognition:
            self._init_professional_libs()
        
        # Load professional dictionaries
        self._load_dictionaries()
        
        # Precompute embedding vectors
        if enable_semantic_matching:
            self._precompute_embeddings()
    
    def _init_professional_libs(self):
        """Initialize professional libraries"""
        if JIEBA_AVAILABLE:
            # Load custom dictionaries
            self._load_jieba_dicts()
        
        if HANLP_AVAILABLE:
            try:
                # Load HanLP NER model
                self.hanlp_ner = hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH')
            except Exception as e:
                print(f"HanLP加载失败: {e}")
                self.hanlp_ner = None
    
    def _load_jieba_dicts(self):
        """Load jieba custom dictionaries"""
        # Medical health dictionary
        medical_words = [
            "高血压", "糖尿病", "心脏病", "关节炎", "骨质疏松", "白内障", "青光眼", 
            "中风", "癌症", "感冒", "肺炎", "哮喘", "胃炎", "肝炎", "肾炎",
            "降压药", "降糖药", "止痛药", "消炎药", "维生素", "钙片", "鱼油",
            "血压计", "血糖仪", "体温计", "听诊器", "心电图", "B超", "CT", "核磁共振"
        ]
        
        # Common elderly vocabulary
        elder_words = [
            "孙子", "孙女", "外孙", "外孙女", "老伴", "老伴儿", "老同事", "老朋友",
            "广场舞", "太极拳", "麻将", "象棋", "围棋", "钓鱼", "养花", "养鸟",
            "社区", "居委会", "老年大学", "养老院", "敬老院", "福利院",
            "退休金", "养老金", "医保", "社保", "低保"
        ]
        
        # Add to jieba dictionary
        for word in medical_words + elder_words:
            jieba.add_word(word)
    
    def _load_dictionaries(self):
        """Load professional dictionaries"""
        # Medical entity dictionary
        self.medical_dict = {
            "diseases": [
                "高血压", "糖尿病", "心脏病", "冠心病", "心肌梗塞", "脑梗塞", "脑出血",
                "关节炎", "类风湿", "骨质疏松", "腰椎间盘突出", "颈椎病", "肩周炎",
                "白内障", "青光眼", "老花眼", "听力下降", "耳鸣", "失眠", "抑郁症",
                "老年痴呆", "帕金森", "中风", "癌症", "肿瘤", "感冒", "肺炎", "哮喘",
                "支气管炎", "胃炎", "胃溃疡", "肝炎", "肾炎", "尿毒症", "痛风"
            ],
            "symptoms": [
                "疼", "痛", "酸", "麻", "痒", "发烧", "咳嗽", "头痛", "头晕", "恶心",
                "呕吐", "腹泻", "便秘", "失眠", "多梦", "心慌", "胸闷", "气短",
                "乏力", "食欲不振", "消化不良", "口干", "口苦", "耳鸣", "眼花"
            ],
            "body_parts": [
                "头", "脖子", "肩膀", "手臂", "手", "手指", "胸", "背", "腰", "肚子",
                "腿", "膝盖", "脚", "脚趾", "眼睛", "耳朵", "鼻子", "嘴巴", "牙齿",
                "心脏", "肺", "肝", "胃", "肾", "膀胱", "血管", "神经", "关节"
            ],
            "medicines": [
                "药", "片", "胶囊", "冲剂", "中药", "西药", "降压药", "降糖药", 
                "止痛药", "消炎药", "维生素", "钙片", "鱼油", "阿司匹林", "布洛芬",
                "感冒药", "退烧药", "止咳药", "胃药", "安眠药", "抗抑郁药"
            ]
        }
        
        # Elderly life dictionary
        self.elder_life_dict = {
            "family": [
                "儿子", "女儿", "孙子", "孙女", "外孙", "外孙女", "老伴", "老伴儿",
                "老公", "老婆", "爸爸", "妈妈", "爷爷", "奶奶", "外公", "外婆",
                "儿媳", "女婿", "亲家", "兄弟姐妹", "侄子", "侄女"
            ],
            "activities": [
                "散步", "跑步", "太极", "广场舞", "健身", "走路", "活动", "唱歌",
                "跳舞", "书法", "绘画", "摄影", "旅游", "读书", "看报", "下棋",
                "打牌", "钓鱼", "园艺", "养花", "养鸟", "收藏", "编织", "手工"
            ],
            "places": [
                "家", "医院", "公园", "广场", "超市", "银行", "邮局", "学校",
                "单位", "公司", "工厂", "社区", "小区", "老年大学", "养老院",
                "敬老院", "福利院", "居委会", "活动中心", "图书馆", "博物馆"
            ],
            "food": [
                "菜", "汤", "粥", "包子", "饺子", "面条", "米饭", "馒头", "饼",
                "糕", "点心", "肉", "鱼", "鸡蛋", "牛奶", "蔬菜", "水果", "豆腐",
                "豆芽", "白菜", "萝卜", "土豆", "西红柿", "黄瓜", "茄子", "辣椒"
            ]
        }
    
    def _precompute_embeddings(self):
        """Precompute embedding vectors for all keywords"""
        all_keywords = []
        self.keyword_to_category = {}
        
        # Merge all dictionaries
        for category, words in self.medical_dict.items():
            for word in words:
                all_keywords.append(word)
                self.keyword_to_category[word] = f"MEDICAL_{category.upper()}"
        
        for category, words in self.elder_life_dict.items():
            for word in words:
                all_keywords.append(word)
                self.keyword_to_category[word] = f"ELDER_{category.upper()}"
        
        if all_keywords:
            self.keyword_embeddings = self.embed_model.encode(all_keywords, normalize_embeddings=True)
            self.keyword_list = all_keywords
        else:
            self.keyword_embeddings = None
            self.keyword_list = []
    
    def extract_entities_jieba(self, text: str) -> List[Dict]:
        """使用jieba进行实体识别"""
        if not JIEBA_AVAILABLE:
            return []
        
        entities = []
        words = pseg.cut(text)
        
        for word, flag in words:
            if len(word) > 1:  # Filter single characters
                entity_type = self._map_jieba_pos_to_entity_type(flag)
                if entity_type:
                    entities.append({
                        "text": word,
                        "type": entity_type,
                        "method": "jieba",
                        "pos": flag
                    })
        
        return entities
    
    def extract_entities_hanlp(self, text: str) -> List[Dict]:
        """使用HanLP进行实体识别"""
        if not HANLP_AVAILABLE or not self.hanlp_ner:
            return []
        
        try:
            entities = []
            result = self.hanlp_ner([text])
            
            for sent_entities in result:
                for entity in sent_entities:
                    entities.append({
                        "text": entity[0],
                        "type": entity[1],
                        "method": "hanlp",
                        "start": entity[2],
                        "end": entity[3]
                    })
            
            return entities
        except Exception as e:
            print(f"HanLP识别失败: {e}")
            return []
    
    def _map_jieba_pos_to_entity_type(self, pos: str) -> Optional[str]:
        """映射jieba词性到实体类型"""
        pos_mapping = {
                    "nr": "PERSON",      # Person name
        "ns": "LOCATION",    # Place name
        "nt": "ORGANIZATION", # Organization name
        "nz": "OTHER",       # Other proper nouns
        "n": "NOUN",         # Noun
        "v": "VERB",         # Verb
        "a": "ADJECTIVE"     # Adjective
        }
        return pos_mapping.get(pos, None)
    
    def extract_entities_semantic(self, text: str, threshold: float = 0.75) -> List[Dict]:
        """使用语义相似度提取实体"""
        if self.keyword_embeddings is None or len(self.keyword_embeddings) == 0:
            return []
        
        entities = []
        text_embedding = self.embed_model.encode(text, normalize_embeddings=True)
        
        # Calculate similarity
        similarities = util.cos_sim(text_embedding, self.keyword_embeddings)[0].tolist()
        
        # Find keywords with similarity above threshold
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                keyword = self.keyword_list[i]
                entity_type = self.keyword_to_category.get(keyword, "UNKNOWN")
                
                entities.append({
                    "text": keyword,
                    "type": entity_type,
                    "similarity": similarity,
                    "method": "semantic"
                })
        
        return entities
    
    def extract_entities(self, text: str, use_semantic: bool = True, semantic_threshold: float = 0.75) -> List[Dict]:
        """综合提取实体"""
        entities = []
        
        # 1. Professional library recognition
        if self.use_professional_libs and self.enable_entity_recognition:
            jieba_entities = self.extract_entities_jieba(text)
            entities.extend(jieba_entities)
            
            hanlp_entities = self.extract_entities_hanlp(text)
            entities.extend(hanlp_entities)
        
        # 2. Semantic similarity matching
        if use_semantic and self.enable_semantic_matching:
            semantic_entities = self.extract_entities_semantic(text, semantic_threshold)
            entities.extend(semantic_entities)
        
        # 3. Deduplication and sorting
        unique_entities = []
        seen_texts = set()
        
        for entity in entities:
            entity_text = entity.get("text")
            # Ensure entity_text is string type
            if isinstance(entity_text, str) and entity_text not in seen_texts:
                unique_entities.append(entity)
                seen_texts.add(entity_text)
        
        # Sort by similarity (if available)
        unique_entities.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        return unique_entities
    
    def get_entity_summary(self, entities: List[Dict]) -> Dict:
        """获取实体摘要统计"""
        summary = defaultdict(list)
        
        for entity in entities:
            entity_type = entity["type"]
            summary[entity_type].append({
                "text": entity["text"],
                "method": entity.get("method", "unknown"),
                "similarity": entity.get("similarity", 1.0)
            })
        
        return dict(summary)

    # Test function
if __name__ == "__main__":
    recognizer = EnhancedEntityRecognizer(use_professional_libs=True)
    
    test_texts = [
        "我孙子今天考试考得不错，膝盖还是有点疼",
        "最近天气不错，我经常去公园散步，和老朋友一起下棋",
        "血压有点高，医生开了降压药，老伴很担心",
        "社区里有个老年大学，我报名了书法班"
    ]
    
    for text in test_texts:
        print(f"\n文本: {text}")
        entities = recognizer.extract_entities(text)
        summary = recognizer.get_entity_summary(entities)
        
        print("提取的实体:")
        for entity_type, entity_list in summary.items():
            print(f"  {entity_type}: {[e['text'] for e in entity_list]}") 