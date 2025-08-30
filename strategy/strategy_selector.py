from typing import Dict, List
import random
from datetime import datetime
from .RAG_keyword_strategy import RAGKeywordStrategy
from .keyword_memory_manager import KeywordMemoryManager
from .enhanced_entity_recognizer import EnhancedEntityRecognizer
from .performance_config import get_performance_config
import json

from sentence_transformers import SentenceTransformer, util


class StrategySelector:
    def __init__(self, performance_mode="fast"):
        # Load performance configuration
        self.performance_config = get_performance_config(performance_mode)
        self.elder_topics = [
            "跳广场舞", "早晨遛弯", "晒太阳养花", "孩子最近有没有来看你",
            "孙子孙女的趣事", "做饭、做包子、熬汤", "以前的老朋友、老同事", "以前你最喜欢的电视剧/演员"
        ]
        
        # Keyword early warning system
        self.critical_keywords = [
            "想死", "不想活了", "死了算了", "活不下去了", "结束生命", "自杀", "自尽",
            "跳楼", "上吊", "割腕", "吃药", "安眠药", "毒药", "结束自己",
            "离开这个世界", "解脱", "一了百了", "永别", "再见", "告别",
            "我走了", "我要走了", "不想再活了", "活着没意思", "没意思", "没意义"
        ]
        self.topic_blacklist_keywords = {
            "跳广场舞": ["腿疼", "膝盖", "摔", "跌倒"],
            "早晨遛弯": ["腰不好", "不出门", "阴天"]
        }
        self.keyword_to_empathy_templates = {
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
            # Added health-related
            "血压": "你最近量血压了吗？有没有注意到什么变化？",
            "糖尿病": "你平时饮食上会特别注意吗？如果有需要可以和我聊聊。",
            "感冒": "最近天气变化大，你有没有注意保暖，身体还好吗？",
            "睡眠": "你最近睡得怎么样？有没有什么困扰你的地方？",
            # Added family-related
            "老伴": "你和老伴最近有没有一起做什么有趣的事情？",
            "孙女": "你和孙女最近有没有什么开心的事情？",
            '孙子': "你和孙子最近有没有什么开心的事情？",
            "家人": "家人最近有没有来看你？和他们在一起的时候开心吗？",
            "儿子": "你的儿子最近有没有来看你？和他们在一起的时候开心吗？",
            "女儿": "你的女儿最近有没有来看你？和他们在一起的时候开心吗？",
            # Interests and life
            "唱歌": "你还喜欢唱歌吗？有没有最近常唱的歌？",
            "跳舞": "你最近还跳广场舞吗？身体感觉怎么样？",
            "书法": "你最近有没有练书法？写字的时候心情会不会更平静？",
            "摄影": "你最近有没有拍照？拍到什么有趣的画面了吗？",
            "旅游": "你最近有没有出去旅游？有没有什么有趣的见闻？",
            # Diet and daily life
            "做饭": "你最近做了什么好吃的？有没有什么拿手菜？",
            "包子": "你做的包子一定很好吃，有没有什么独家秘诀？",
            "汤": "最近天气凉了，喝点热汤会不会舒服一些？",
            "水果": "你最近喜欢吃什么水果？",
            # Weather and travel
            "下雨": "最近下雨了，出门要注意安全哦。",
            "天气": "最近天气变化大，要注意添衣保暖。",
            "出门": "你最近出门散步了吗？有没有遇到什么有趣的事情？",
            # Festivals and memories
            "春节": "春节快到了，你有什么计划吗？家里会不会很热闹？",
            "中秋": "中秋节的时候有没有和家人一起吃月饼？",
            "回忆": "你刚刚提到的回忆真美好，有没有什么特别想分享的？",
            # Other care
            "孤独": "有时候会觉得孤单吗？如果有心事可以和我聊聊。",
            "开心": "听到你开心我也很高兴，有什么开心的事可以多和我分享。",
            "难过": "如果你有点难过，可以和我说说，我会一直陪着你。"
        }
        self.memory = KeywordMemoryManager()
        self.rag_strategy = RAGKeywordStrategy()
        # Entity recognizer - initialize based on performance configuration
        self.entity_recognizer = EnhancedEntityRecognizer(
            use_professional_libs=True,
            enable_entity_recognition=self.performance_config["enable_entity_recognition"],
            enable_semantic_matching=self.performance_config["enable_semantic_matching"]
        )
        self.long_term_sadness_log = []

        # Initialize embed_model, set to None if failed
        try:
            self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            print(f"Warning: Failed to initialize SentenceTransformer: {e}")
            self.embed_model = None

    def enrich_prompt_with_keyword_empathy(self, base_prompt: str, user_input: str = None, user_id: str = None) -> str:
        """Optimized version: Intelligent keyword empathy with user isolation and false memory prevention"""
        try:
            if not user_id:
                return base_prompt
                
            self.memory.clear_expired(user_id)
            recent_keywords = self.memory.get_recent_keywords(user_id)
            
            if not recent_keywords:
                return base_prompt

            # 1. Semantic similarity ranking (if user input exists)
            relevant_keywords = []
            if user_input is not None and self.embed_model is not None:
                try:
                    input_emb = self.embed_model.encode(user_input, normalize_embeddings=True)
                    keyword_embs = self.embed_model.encode(recent_keywords, normalize_embeddings=True)
                    
                    similarities = util.pytorch_cos_sim(input_emb, keyword_embs)[0]
                    
                    # Only select highly relevant keywords (similarity > 0.3)
                    for i, similarity in enumerate(similarities):
                        if similarity > 0.3:
                            relevant_keywords.append((recent_keywords[i], similarity.item()))
                    
                    # Sort by similarity, take top 3
                    relevant_keywords.sort(key=lambda x: x[1], reverse=True)
                    relevant_keywords = relevant_keywords[:3]
                    
                except Exception as e:
                    print(f"Warning: Semantic matching failed: {e}")
                    # If semantic matching fails, use traditional method
                    relevant_keywords = [(kw, 0.5) for kw in recent_keywords[:3]]
            else:
                # When no user input, use recent keywords
                relevant_keywords = [(kw, 0.5) for kw in recent_keywords[:3]]

            # 2. Build empathy statements, avoid false memories
            empathy_statements = []
            for keyword, similarity in relevant_keywords:
                if keyword in self.keyword_to_empathy_templates:
                    template = self.keyword_to_empathy_templates[keyword]
                    
                    # Check if template contains phrases that might cause false memories
                    problematic_phrases = [
                        "你之前提到", "你前面说过", "你曾提起", "你前面说起",
                        "你讲到", "你刚刚提到的", "你前面说"
                    ]
            
                    # If template contains phrases that might cause false memories, modify them
                    safe_template = template
                    for phrase in problematic_phrases:
                        if phrase in safe_template:
                            # Replace with safer expressions
                            safe_template = safe_template.replace(phrase, "关于")
                            break
                    
                    empathy_statements.append(safe_template)
            
            # 3. Limit number of empathy statements to avoid information overload
            if empathy_statements:
                # Add at most 2 empathy statements
                selected_empathy = empathy_statements[:2]
                empathy_text = " ".join(selected_empathy)
                
                # Check if base prompt is already long
                if len(base_prompt) > 100:
                    # If base prompt is long, simplify empathy statements
                    empathy_text = empathy_statements[0] if empathy_statements else ""
                
                return f"{base_prompt} {empathy_text}".strip()
            
            return base_prompt
            
        except Exception as e:
            print(f"Warning: Failed to enrich prompt with keyword empathy: {e}")
            return base_prompt

    def should_use_rag(self, user_input: str, keywords: List[str]) -> bool:
        """Determine whether RAG retrieval should be used"""
        if not user_input or not keywords:
            return False
        
        # Only use RAG when user explicitly asks about specific topics
        rag_trigger_keywords = [
            "怎么", "如何", "为什么", "什么", "哪些", "建议", "推荐", "方法", "技巧",
            "健康", "锻炼", "饮食", "睡眠", "活动", "兴趣", "爱好", "朋友", "家人"
        ]
        
        user_input_lower = user_input.lower()
        return any(trigger in user_input_lower for trigger in rag_trigger_keywords)

    def is_rag_content_appropriate(self, user_input: str, rag_content: str) -> bool:
        """Determine if RAG content is appropriate"""
        if not user_input or not rag_content:
            return False
        
        # Avoid conflicts between RAG content and user input
        inappropriate_patterns = [
            "您上次提到", "我记得您", "您说过", "您之前"
        ]
        
        rag_content_lower = rag_content.lower()
        return not any(pattern in rag_content_lower for pattern in inappropriate_patterns)

    def insert_elder_topic_safely(self, prompt: str, history: List[str]) -> str:
        try:
            # Ensure all elements in history are strings
            text_parts = []
            for item in history:
                if isinstance(item, str):
                    text_parts.append(item)
            all_text = " ".join(text_parts)
            available = [
                topic for topic in self.elder_topics
                if not any(word in all_text for word in self.topic_blacklist_keywords.get(topic, []))
            ]
            if not available:
                return prompt
            topic = random.choice(available)
            return f"{prompt} 对了，你最近有没有去{topic}呀？我很好奇。"
        except Exception as e:
            print(f"Warning: Failed to insert elder topic: {e}")
            return prompt

    def embed_empathy_mid_dialogue(self, base: str, emotion_type: str = "neutral") -> str:
        prompts = {
            "sadness": ["请加入一句安慰或陪伴的话，让用户感受到你理解他们的情绪。", "在回应中体现你对悲伤情绪的接纳和共情。"],
            "joy": ["加入一句与你一起感到高兴的语句，增强对方情绪反馈。", "体现你对用户好消息的真诚回应。"],
            "anger": ["加入一句平静、理解的语句，承认对方的不满或沮丧。", "请在回应中体现非评判态度，并鼓励继续表达。"],
            "mixed": ["你可以加入一句表示'情绪可能交织也没有关系'的温和提醒。", "提醒用户他们的感觉是被接纳的，可以慢慢说。"],
            "neutral": ["请加入一句能让用户觉得你在认真听他们讲故事的话。", "表现出你对对方回忆的兴趣与尊重。"]
        }
        insert = prompts.get(emotion_type, prompts["neutral"])
        return f"{base} ##Empathy Tips: {insert}"

    def update_context_keywords(self, history: List[str], user_id: str = None, conversation_id: str = None):
        """Optimized version: Intelligently update context keywords, avoiding duplication and irrelevant information"""
        try:
            self.memory.clear_expired()
            
            # Only process the last 3-5 rounds of conversation to avoid historical information interference
            recent_history = history[-6:] if len(history) > 6 else history
            
            # Ensure all elements are strings
            text_parts = []
            for item in recent_history:
                if isinstance(item, str):
                    text_parts.append(item)
            
            if not text_parts:
                return
                
            text = " ".join(text_parts)
            
            # 1. Intelligent keyword extraction, avoiding duplication
            extracted_keywords = set()
            
            # Traditional keyword matching, but with quantity limits
            for keyword in self.keyword_to_empathy_templates:
                if keyword in text and keyword not in extracted_keywords:
                    # Check if keyword appears frequently in recent conversations
                    keyword_count = text.count(keyword)
                    if keyword_count <= 3:  # Avoid excessively repeated keywords
                        self.memory.add_keyword(keyword, conversation_id=conversation_id, user_id=user_id)
                        extracted_keywords.add(keyword)
            
            # 2. Enhanced entity recognition (based on performance configuration)
            if self.performance_config["enable_entity_recognition"] or self.performance_config["enable_semantic_matching"]:
                entities = self.entity_recognizer.extract_entities(
                    text, 
                    use_semantic=self.performance_config["enable_semantic_matching"], 
                    semantic_threshold=self.performance_config["semantic_threshold"]
                )
                
                # Limit entity count to avoid information overload
                entity_count = 0
                for entity in entities:
                    if entity_count >= 5:  # Extract at most 5 entities
                        break
                        
                    entity_text = entity.get("text")
                    if isinstance(entity_text, str) and entity_text not in extracted_keywords:
                        # Avoid repeatedly adding existing keywords
                        if entity_text not in [kw for kw in self.keyword_to_empathy_templates.keys()]:
                            self.memory.add_keyword(entity_text, conversation_id=conversation_id, user_id=user_id)
                            extracted_keywords.add(entity_text)
                            entity_count += 1
                            
        except Exception as e:
            print(f"Warning: Failed to update context keywords: {e}")

    def log_long_term_sadness(self, sadness_score: float):
        try:
            now = datetime.now()
            # 确保long_term_sadness_log是列表类型
            if not isinstance(self.long_term_sadness_log, list):
                self.long_term_sadness_log = []
            
            self.long_term_sadness_log.append((now, sadness_score))
            self.long_term_sadness_log = [entry for entry in self.long_term_sadness_log if (now - entry[0]).days <= 7]
        except Exception as e:
            print(f"Warning: Failed to log long term sadness: {e}")
            # 如果出错，重置为空列表
            self.long_term_sadness_log = []

    def check_early_warning(self, window_sadness_scores: list = None, current_sadness: float = 0.0, liwc_scores: dict = None) -> dict:
        """
        多维度早期预警检测
        返回: {"triggered": bool, "level": str, "reason": str, "suggested_action": str}
        """
        warning_result = {
            "triggered": False,
            "level": "normal",  # normal, mild, moderate, severe
            "reason": "",
            "suggested_action": ""
        }
        
        # 1. Acute high-risk detection (extremely high current sadness)
        if current_sadness > 0.9:
            warning_result.update({
                "triggered": True,
                "level": "severe",
                "reason": f"当前悲伤情绪极高({current_sadness:.2f})",
                "suggested_action": "立即关注，建议人工介入",
                "recommend_gds": True
            })
            return warning_result
        
        # 2. Persistent detection (based on window_sadness_scores)
        if window_sadness_scores is not None and isinstance(window_sadness_scores, list) and len(window_sadness_scores) > 0:
            try:
                recent_scores = window_sadness_scores[-3:] if len(window_sadness_scores) >= 3 else window_sadness_scores
                
                # 2.1 Consecutive high sadness detection
                high_sadness_count = sum(1 for score in recent_scores if score > 0.8)
                if high_sadness_count >= 2:
                    warning_result.update({
                        "triggered": True,
                        "level": "moderate" if high_sadness_count == 2 else "severe",
                        "reason": f"连续{high_sadness_count}次悲伤情绪>0.8",
                        "suggested_action": "需要持续关注，建议增加关怀频率",
                        "recommend_gds": True
                    })
                    return warning_result
                
                # 2.2 趋势恶化检测 (如果有足够历史数据)
                if len(window_sadness_scores) >= 6:
                    recent_avg = sum(recent_scores) / len(recent_scores)
                    # 修复：先创建切片，再计算平均值
                    historical_scores = window_sadness_scores[:-3]
                    historical_avg = sum(historical_scores) / len(historical_scores)
                    
                    if recent_avg - historical_avg > 0.3 and recent_avg > 0.6:
                        warning_result.update({
                            "triggered": True,
                            "level": "moderate",
                            "reason": f"悲伤情绪显著恶化(近期平均{recent_avg:.2f} vs 历史平均{historical_avg:.2f})",
                            "suggested_action": "情绪趋势恶化，建议主动关怀",
                            "recommend_gds": True
                        })
                        return warning_result
            except Exception as e:
                print(f"Warning: Failed to process window_sadness_scores: {e}")
        
        # 3. Long-term trend detection (based on long_term_sadness_log)
        if len(self.long_term_sadness_log) >= 10:
            try:
                # 确保long_term_sadness_log是列表类型
                if isinstance(self.long_term_sadness_log, list):
                    recent_entries = self.long_term_sadness_log[-10:]
                else:
                    recent_entries = []
                # 确保entry是元组或列表，并且有第二个元素
                recent_scores = []
                for entry in recent_entries:
                    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                        recent_scores.append(entry[1])
                    elif isinstance(entry, dict) and 'score' in entry:
                        recent_scores.append(entry['score'])
                    else:
                        # 如果entry是单个数值，直接使用
                        recent_scores.append(float(entry))
                
                if recent_scores:
                    recent_avg = sum(recent_scores) / len(recent_scores)
                    
                    if recent_avg > 0.7:
                        warning_result.update({
                            "triggered": True,
                            "level": "mild",
                            "reason": f"长期悲伤情绪偏高(平均{recent_avg:.2f})",
                            "suggested_action": "长期情绪偏低，建议定期关怀",
                            "recommend_gds": True
                        })
                        return warning_result
            except Exception as e:
                print(f"Warning: Failed to process long_term_sadness_log: {e}")
        
        # 4. Multi-dimensional comprehensive detection (combining LIWC features)
        if liwc_scores and current_sadness > 0.6:
            # High self-focus + Low social + High negative emotions
            if (liwc_scores.get("self_focus_freq", 0) > 0.15 and 
                liwc_scores.get("social_freq", 0) < 0.05 and
                liwc_scores.get("sadness_LIWC_freq", 0) > 0.08):
                warning_result.update({
                    "triggered": True,
                    "level": "moderate",
                    "reason": "高自我关注+低社交+负面情绪组合",
                    "suggested_action": "社交退缩倾向，建议鼓励社交活动",
                    "recommend_gds": True
                })
                return warning_result
        
        return warning_result

    def check_critical_keywords(self, user_input: str) -> dict:
        """Check keyword warnings"""
        if not user_input:
            return {"triggered": False, "keywords": [], "level": "normal"}
        
        found_keywords = []
        for keyword in self.critical_keywords:
            if keyword in user_input:
                found_keywords.append(keyword)
        
        if found_keywords:
            return {
                "triggered": True,
                "keywords": found_keywords,
                "level": "critical",
                "reason": f"检测到危险关键词: {', '.join(found_keywords)}",
                "suggested_action": "立即人工介入，建议联系专业心理危机干预",
                "recommend_gds": True
            }
        
        return {"triggered": False, "keywords": [], "level": "normal"}

    def check_liwc_anomaly(self, liwc: Dict[str, float]) -> bool:
        return (
            liwc.get("sadness_LIWC_freq", 0) > 0.10 and
            liwc.get("social_freq", 0) < 0.05 and
            liwc.get("self_focus_freq", 0) > 0.20
        )

    def is_prompt_relevant(self, user_input: str, strategy_prompt: str, threshold=0.45) -> bool:
        if self.embed_model is None:
            # 如果embed_model不可用，返回True（默认认为相关）
            return True
        
        try:
            emb_user = self.embed_model.encode(user_input, normalize_embeddings=True)
            # 截取策略引导语首句做比对
            first_sentence = strategy_prompt.split("。")[0]
            emb_prompt = self.embed_model.encode(first_sentence, normalize_embeddings=True)
            sim_score = util.cos_sim(emb_user, emb_prompt).item()
            print(f"DEBUG 相似度: {sim_score}, user_input='{user_input}', prompt_first_sentence='{first_sentence}'")
            return sim_score >= threshold
        except Exception as e:
            print(f"Warning: Failed to compute prompt relevance: {e}")
            return True

    def select_strategy(self, emotion_scores: Dict[str, float], emotion_intensity: float, history: List[str], liwc: Dict[str, float], user_input: str = "", window_sadness_scores: list = None, user_info: Dict = None) -> Dict:
        """Select the most appropriate strategy"""
        
        # 调试信息
        print(f"DEBUG: user_input = '{user_input}'")
        print(f"DEBUG: user_info exists = {user_info is not None}")
        print(f"DEBUG: user_info type = {type(user_info)}")
        print(f"DEBUG: user_info bool = {bool(user_info)}")
        if user_info:
            print(f"DEBUG: user_info keys = {list(user_info.keys())}")
        
        # 首先检查是否是个人信息查询
        if user_info is not None and self.is_personal_info_query(user_input):
            print(f"DEBUG: 检测到个人信息查询，用户输入: '{user_input}'")
            return self.handle_personal_info_query(user_input, user_info)
        else:
            print(f"DEBUG: 未检测到个人信息查询，user_info存在: {user_info is not None}, is_personal_info_query结果: {self.is_personal_info_query(user_input) if user_info is not None else 'N/A'}")
        
        # 获取用户ID
        user_id = user_info.get("_id") if user_info else None
        
        # 更新上下文关键词
        from datetime import datetime
        conversation_id = f"conv_{int(datetime.now().timestamp())}"
        self.update_context_keywords(history, user_id=user_id, conversation_id=conversation_id)
        
        # 检查关键词预警（最高优先级）
        keyword_warning = self.check_critical_keywords(user_input)
        
        # 检查早期预警
        early_warning = self.check_early_warning(window_sadness_scores, emotion_scores.get("sadness", 0), liwc)
        
        # 检查LIWC异常
        liwc_anomaly = self.check_liwc_anomaly(liwc)
        
        # 如果触发关键词预警，立即返回紧急响应
        if keyword_warning["triggered"]:
            base = f"我注意到{keyword_warning['reason']}，这让我非常担心你的安全。请立即停止任何危险的想法，你的生命非常宝贵。我建议我们立即联系专业的心理危机干预热线，或者联系你的家人朋友。你愿意和我详细聊聊吗？"
            candidate = {
                "语气": "紧急危机干预",
                "目标": "立即危机干预，防止自伤行为",
                "引导语": base,
                "心理学依据": "危机干预理论：立即关注和干预",
                "避免": "避免轻视或拖延处理",
                "matched_rule": "CriticalKeyword_Alert",
                "rules_prompt": f"关键词预警-危机级别: {keyword_warning['reason']}",
                "keyword_warning": keyword_warning
            }
            
            # 对关键词预警响应进行关键词共情和个性化处理
            candidate["引导语"] = self.enrich_prompt_with_keyword_empathy(candidate["引导语"], user_input, user_id)
            candidate["引导语"] = self.insert_elder_topic_safely(candidate["引导语"], history)
            candidate["引导语"] = self.embed_empathy_mid_dialogue(candidate["引导语"], "crisis")
            
            # 个性化处理
            if user_info:
                candidate["引导语"] = self.personalize_prompt(candidate["引导语"], user_info)
            
            return candidate
        
        # 如果触发早期预警，优先处理预警情况
        if early_warning["triggered"]:
            warning_level = early_warning["level"]
            warning_reason = early_warning["reason"]
            
            # 根据预警级别生成相应的基础响应
            if warning_level == "severe":
                base = f"我注意到{warning_reason}，这让我非常担心你的状态。你愿意和我详细聊聊吗？如果需要的话，我建议我们可以联系专业的心理支持资源。"
                candidate = {
                    "语气": "紧急关切",
                    "目标": "立即情绪干预，建议转介专业支持",
                    "引导语": base,
                    "心理学依据": "急性情绪危机需要立即关注和专业干预",
                    "避免": "避免轻视或拖延处理",
                    "matched_rule": "EarlyWarning_Severe",
                    "rules_prompt": f"早期预警-严重级别: {warning_reason}",
                    "early_warning": early_warning
                }
            elif warning_level == "moderate":
                base = f"我注意到{warning_reason}，你最近是不是遇到了一些困难？愿意和我聊聊吗？我会一直陪着你。"
                candidate = {
                    "语气": "关切引导",
                    "目标": "主动关怀，预防情绪恶化",
                    "引导语": base,
                    "心理学依据": "早期干预可以有效预防情绪问题恶化",
                    "避免": "避免忽视或转移话题",
                    "matched_rule": "EarlyWarning_Moderate",
                    "rules_prompt": f"早期预警-中等级别: {warning_reason}",
                    "early_warning": early_warning
                }
            elif warning_level == "mild":
                base = f"我注意到{warning_reason}，你最近心情怎么样？有什么想和我分享的吗？"
                candidate = {
                    "语气": "温和关怀",
                    "目标": "增加关怀频率，预防问题发展",
                    "引导语": base,
                    "心理学依据": "定期关怀有助于维持情绪稳定",
                    "避免": "避免过度反应",
                    "matched_rule": "EarlyWarning_Mild",
                    "rules_prompt": f"早期预警-轻微级别: {warning_reason}",
                    "early_warning": early_warning
                }
            
            # 对预警响应进行关键词共情和个性化处理
            candidate["引导语"] = self.enrich_prompt_with_keyword_empathy(candidate["引导语"], user_input, user_id)
            candidate["引导语"] = self.insert_elder_topic_safely(candidate["引导语"], history)
            candidate["引导语"] = self.embed_empathy_mid_dialogue(candidate["引导语"], "sadness")
            
            # 个性化处理
            if user_info:
                candidate["引导语"] = self.personalize_prompt(candidate["引导语"], user_info)
            
            return candidate
        
        # 情绪强度阈值
        high_intensity = emotion_intensity > 0.7
        moderate_intensity = 0.4 < emotion_intensity <= 0.7
        low_intensity = emotion_intensity <= 0.4
        
        # 主要情绪
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1] if x[0] != "intensity" else 0)
        sadness_score = emotion_scores.get("sadness", 0)
        joy_score = emotion_scores.get("joy", 0)
        anger_score = emotion_scores.get("anger", 0)
        
        # 规则匹配
        rules_prompt = ""
        
        def is_emotion_neutral():
            return all(score < 0.3 for emotion, score in emotion_scores.items() if emotion != "intensity")
        
        # 规则E：高悲伤 + 高强度
        if sadness_score > 0.6 and high_intensity:
            base = "我感受到你现在心情很低落，这种感受我完全理解。"
            base = self.enrich_prompt_with_keyword_empathy(base, user_input, user_id)
            base = self.insert_elder_topic_safely(base, history)
            base = self.embed_empathy_mid_dialogue(base, "sadness")
            candidate = {
                "语气": "温暖、理解",
                "目标": "提供情感支持、缓解悲伤",
                "引导语": base,
                "心理学依据": "Carl Rogers人本主义理论：无条件积极关注",
                "避免": "避免说教或强行安慰",
                "matched_rule": "RuleE_HighSadness",
                "rules_prompt": rules_prompt
            }
        # 规则D：高愤怒 + 高强度
        elif anger_score > 0.6 and high_intensity:
            base = "我理解你现在很生气，这种情绪是正常的。"
            base = self.enrich_prompt_with_keyword_empathy(base, user_input, user_id)
            base = self.insert_elder_topic_safely(base, history)
            base = self.embed_empathy_mid_dialogue(base, "anger")
            candidate = {
                "语气": "理解、支持",
                "目标": "接纳情绪、引导表达",
                "引导语": base,
                "心理学依据": "情绪调节理论：先接纳再调节",
                "避免": "避免压制或否定情绪",
                "matched_rule": "RuleD_HighAnger",
                "rules_prompt": rules_prompt
            }
        # 规则C：高快乐 + 高强度
        elif joy_score > 0.6 and high_intensity:
            base = "看到你这么开心，我也很高兴！"
            base = self.enrich_prompt_with_keyword_empathy(base, user_input, user_id)
            base = self.insert_elder_topic_safely(base, history)
            base = self.embed_empathy_mid_dialogue(base, "joy")
            candidate = {
                "语气": "分享、共鸣",
                "目标": "分享快乐、强化积极情绪",
                "引导语": base,
                "心理学依据": "积极心理学：强化积极体验",
                "避免": "避免过度兴奋或转移话题",
                "matched_rule": "RuleC_HighJoy",
                "rules_prompt": rules_prompt
            }
        # 规则B：中等强度情绪
        elif moderate_intensity:
            base = "我理解你的感受，能多和我说说吗？"
            base = self.enrich_prompt_with_keyword_empathy(base, user_input, user_id)
            base = self.insert_elder_topic_safely(base, history)
            base = self.embed_empathy_mid_dialogue(base, primary_emotion[0])
            candidate = {
                "语气": "理解、引导",
                "目标": "鼓励表达、深入了解",
                "引导语": base,
                "心理学依据": "非指导式倾听，激发表达",
                "避免": "避免打断或强行转移话题",
                "matched_rule": "RuleB_ModerateIntensity",
                "rules_prompt": rules_prompt
            }
        # 规则A：低强度或中性情绪
        elif low_intensity or is_emotion_neutral():
            base = "我在这儿陪着你，有什么都可以跟我聊聊。"
            base = self.enrich_prompt_with_keyword_empathy(base, user_input, user_id)
            base = self.insert_elder_topic_safely(base, history)
            base = self.embed_empathy_mid_dialogue(base, "neutral")
            candidate = {
                "语气": "中性、平静",
                "目标": "继续陪伴，维持沟通",
                "引导语": base,
                "心理学依据": "Carl Rogers人本主义理论：提供无条件积极关注",
                "避免": "避免强迫话题转移",
                "matched_rule": "RuleA_LowIntensity",
                "rules_prompt": rules_prompt
            }
        else:
            base = "我在这儿陪着你，有什么都可以跟我聊聊。"
            base = self.enrich_prompt_with_keyword_empathy(base, user_input, user_id)
            base = self.insert_elder_topic_safely(base, history)
            base = self.embed_empathy_mid_dialogue(base, "neutral")
            candidate = {
                "语气": "中性、平静",
                "目标": "继续陪伴，维持沟通",
                "引导语": base,
                "心理学依据": "Carl Rogers人本主义理论：提供无条件积极关注",
                "避免": "避免强迫话题转移",
                "matched_rule": "Fallback",
                "rules_prompt": rules_prompt
            }

        # 个性化处理
        if user_info:
            candidate["引导语"] = self.personalize_prompt(candidate["引导语"], user_info)
        
        # 语义相关度检测（早期预警响应跳过此检测）
        if user_input.strip() == "":
            return candidate

        # 如果是早期预警响应，跳过语义相关度检测
        if "EarlyWarning" in candidate.get("matched_rule", ""):
            return candidate

        if not self.is_prompt_relevant(user_input, candidate["引导语"]):
            return {
                "语气": "自然、开放",
                "目标": "理解用户表达，鼓励继续沟通",
                "引导语": "谢谢你告诉我这些，能多和我说说吗？",
                "心理学依据": "非指导式倾听，激发表达",
                "避免": "避免打断或强行转移话题",
                "matched_rule": "OpenResponseFallback",
                "rules_prompt": rules_prompt
            }

        return candidate

    def is_personal_info_query(self, user_input: str) -> bool:
        """Check if it's a personal information query"""
        if not user_input:
            return False
        
        # 个人信息查询关键词
        personal_info_keywords = [
            # 名字相关
            "我的名字", "我叫什么", "我姓什么", "我叫", "我姓", "我名字", "我姓名",
            "我叫什么名字", "我姓什么名字", "我叫什么来着", "我姓什么来着",
            "我忘记我的名字", "我忘了我的名字", "我不记得我的名字",
            "我忘记我叫什么", "我忘了我叫什么", "我不记得我叫什么",
            # 年龄相关
            "我多大", "我年龄", "我几岁", "我多少岁", "我多大年纪", "我年纪",
            "我忘记我多大", "我忘了我多大", "我不记得我多大",
            "我忘记我几岁", "我忘了我几岁", "我不记得我几岁",
            # 性别相关
            "我性别", "我是男是女", "我是男", "我是女", "我男的", "我女的",
            "我忘记我性别", "我忘了我性别", "我不记得我性别",
            # 地址相关
            "我住哪", "我住哪里", "我家在哪", "我家在哪里", "我住在哪",
            "我忘记我住哪", "我忘了我住哪", "我不记得我住哪",
            # 家庭相关
            "我有几个孩子", "我有几个子女", "我孩子", "我子女", "我儿子", "我女儿",
            "我忘记我有几个孩子", "我忘了我有几个孩子", "我不记得我有几个孩子",
            # 配偶相关
            "我老伴", "我配偶", "我老公", "我老婆", "我丈夫", "我妻子",
            "我忘记我老伴", "我忘了我老伴", "我不记得我老伴",
            # 兴趣爱好相关
            "我喜欢什么", "我爱好", "我兴趣", "我喜欢", "我爱好什么",
            "我忘记我喜欢什么", "我忘了我喜欢什么", "我不记得我喜欢什么",
            # 健康相关
            "我身体", "我健康", "我身体怎么样", "我健康状况",
            "我忘记我身体", "我忘了我身体", "我不记得我身体",
            # 通用忘记表达
            "我忘了", "我不记得", "我想不起来", "我忘记了", "我记不清",
            "我忘了我的", "我不记得我的", "我想不起来我的",
            # 个人信息页相关
            "我的个人信息", "我的信息", "个人信息页", "看看我的信息",
            "帮我看看", "查看我的", "我的资料", "我的档案"
        ]
        
        user_input_lower = user_input.lower()
        print(f"DEBUG: 检查个人信息查询，输入: '{user_input_lower}'")
        
        for keyword in personal_info_keywords:
            if keyword in user_input_lower:
                print(f"DEBUG: 匹配到关键词: '{keyword}'")
                return True
        
        print(f"DEBUG: 未匹配到任何个人信息关键词")
        return False

    def handle_personal_info_query(self, user_input: str, user_info: Dict) -> Dict:
        """Handle personal information queries"""
        basic_info = user_info.get("basic_info", {})
        family_relation = user_info.get("family_relation", {})
        living_habits = user_info.get("living_habits", {})
        
        name = basic_info.get("name", "")
        age = basic_info.get("age", "")
        gender = basic_info.get("gender", "")
        children_count = family_relation.get("children_count", "")
        hobbies = living_habits.get("hobbies", "")
        spouse_status = family_relation.get("spouse_status", "")
        
        user_input_lower = user_input.lower()
        
        # 根据查询内容返回相应信息
        if any(keyword in user_input_lower for keyword in ["名字", "叫什么", "姓什么", "姓名", "来着"]):
            if name:
                return {
                    "语气": "温和、提醒",
                    "目标": "提供个人信息",
                    "引导语": f"您的名字是{name}。",
                    "心理学依据": "认知支持，减少记忆负担",
                    "避免": "避免责备或催促",
                    "matched_rule": "PersonalInfo_Name",
                    "rules_prompt": ""
                }
            else:
                return {
                    "语气": "理解、支持",
                    "目标": "安慰并提供帮助",
                    "引导语": "名字不着急想，咱们慢慢来。您要是愿意的话，可以跟我说说您小时候的事情？",
                    "心理学依据": "减少焦虑，转移注意力",
                    "避免": "避免强迫回忆",
                    "matched_rule": "PersonalInfo_Name_NotSet",
                    "rules_prompt": ""
                }
        
        elif any(keyword in user_input_lower for keyword in ["年龄", "多大", "几岁"]):
            if age:
                return {
                    "语气": "温和、提醒",
                    "目标": "提供个人信息",
                    "引导语": f"您今年{age}岁。",
                    "心理学依据": "认知支持，减少记忆负担",
                    "避免": "避免责备或催促",
                    "matched_rule": "PersonalInfo_Age",
                    "rules_prompt": ""
                }
            else:
                return {
                    "语气": "理解、支持",
                    "目标": "安慰并提供帮助",
                    "引导语": "年龄不着急想，咱们慢慢来。您要是愿意的话，可以跟我说说您小时候的事情？",
                    "心理学依据": "减少焦虑，转移注意力",
                    "避免": "避免强迫回忆",
                    "matched_rule": "PersonalInfo_Age_NotSet",
                    "rules_prompt": ""
                }
        
        elif any(keyword in user_input_lower for keyword in ["性别", "男", "女"]):
            if gender:
                return {
                    "语气": "温和、提醒",
                    "目标": "提供个人信息",
                    "引导语": f"您是{gender}性。",
                    "心理学依据": "认知支持，减少记忆负担",
                    "避免": "避免责备或催促",
                    "matched_rule": "PersonalInfo_Gender",
                    "rules_prompt": ""
                }
            else:
                return {
                    "语气": "理解、支持",
                    "目标": "安慰并提供帮助",
                    "引导语": "这个不着急想，咱们慢慢来。您要是愿意的话，可以跟我说说您小时候的事情？",
                    "心理学依据": "减少焦虑，转移注意力",
                    "避免": "避免强迫回忆",
                    "matched_rule": "PersonalInfo_Gender_NotSet",
                    "rules_prompt": ""
                }
        
        elif any(keyword in user_input_lower for keyword in ["孩子", "子女"]):
            if children_count:
                return {
                    "语气": "温和、提醒",
                    "目标": "提供个人信息",
                    "引导语": f"您有{children_count}个子女。",
                    "心理学依据": "认知支持，减少记忆负担",
                    "避免": "避免责备或催促",
                    "matched_rule": "PersonalInfo_Children",
                    "rules_prompt": ""
                }
            else:
                return {
                    "语气": "理解、支持",
                    "目标": "安慰并提供帮助",
                    "引导语": "这个不着急想，咱们慢慢来。您要是愿意的话，可以跟我说说您小时候的事情？",
                    "心理学依据": "减少焦虑，转移注意力",
                    "避免": "避免强迫回忆",
                    "matched_rule": "PersonalInfo_Children_NotSet",
                    "rules_prompt": ""
                }
        
        elif any(keyword in user_input_lower for keyword in ["爱好", "兴趣", "喜欢"]):
            if hobbies:
                return {
                    "语气": "温和、提醒",
                    "目标": "提供个人信息",
                    "引导语": f"您喜欢{hobbies}。",
                    "心理学依据": "认知支持，减少记忆负担",
                    "避免": "避免责备或催促",
                    "matched_rule": "PersonalInfo_Hobbies",
                    "rules_prompt": ""
                }
            else:
                return {
                    "语气": "理解、支持",
                    "目标": "安慰并提供帮助",
                    "引导语": "这个不着急想，咱们慢慢来。您要是愿意的话，可以跟我说说您小时候的事情？",
                    "心理学依据": "减少焦虑，转移注意力",
                    "avoid": "避免强迫回忆",
                    "matched_rule": "PersonalInfo_Hobbies_NotSet",
                    "rules_prompt": ""
                }
        
        # 默认情况
        return {
            "语气": "理解、支持",
            "目标": "安慰并提供帮助",
            "引导语": "这个不着急想，咱们慢慢来。您要是愿意的话，可以跟我说说您小时候的事情？",
            "心理学依据": "减少焦虑，转移注意力",
            "避免": "避免强迫回忆",
            "matched_rule": "PersonalInfo_Default",
            "rules_prompt": ""
        }

    def personalize_prompt(self, base_prompt: str, user_info: Dict) -> str:
        """Personalize prompts based on user information"""
        
        basic_info = user_info.get("basic_info", {})
        family_relation = user_info.get("family_relation", {})
        living_habits = user_info.get("living_habits", {})
        
        name = basic_info.get("name", "")
        age = basic_info.get("age", 0)
        children_count = family_relation.get("children_count", 0)
        hobbies = living_habits.get("hobbies", "")
        spouse_status = family_relation.get("spouse_status", "")
        
        # 个性化称呼
        if name:
            base_prompt = base_prompt.replace("你", f"{name}")
        
        # 根据年龄调整语气
        if age > 80:
            base_prompt = base_prompt.replace("你", "您")
        
        # 根据家庭状况调整话题
        if children_count > 0:
            base_prompt += f" 您提到过您有{children_count}个子女，"
        
        # 根据兴趣爱好调整话题
        if hobbies:
            base_prompt += f" 您喜欢{hobbies}，"
        
        # 根据配偶状况调整话题
        if spouse_status == "是":
            base_prompt += " 您有配偶陪伴，"
        elif spouse_status == "否":
            base_prompt += " 您一个人生活，"
        
        return base_prompt



if __name__ == "__main__":
    selector = StrategySelector()
    print("策略模块已加载")

    # 示例输入，注意传入user_input用于相关度判断
    emotion_scores = {"sadness": 0.3, "joy": 0.6, "anger": 0.1}
    emotion_intensity = 0.65
    history = ["孙子这次考试考得不错", "孩子给我买了新花"]
    liwc = {
        "sadness_LIWC_freq": 1.0,
        "social_freq": 3.5,
        "self_focus_freq": 2.2
    }
    user_input = "孙子这次考试考得不错"

    result = selector.select_strategy(emotion_scores, emotion_intensity, history, liwc, user_input)
    print("策略输出：")
    print(json.dumps(result, ensure_ascii=False, indent=2))