from typing import Dict, List
import random
from datetime import datetime
from RAG_keyword_strategy import RAGKeywordStrategy
from keyword_memory_manager import KeywordMemoryManager
from enhanced_entity_recognizer import EnhancedEntityRecognizer
from performance_config import get_performance_config
import json

from sentence_transformers import SentenceTransformer, util


class StrategySelector:
    def __init__(self, performance_mode="fast"):
        # 加载性能配置
        self.performance_config = get_performance_config(performance_mode)
        self.elder_topics = [
            "跳广场舞", "早晨遛弯", "晒太阳养花", "孩子最近有没有来看你",
            "孙子孙女的趣事", "做饭、做包子、熬汤", "以前的老朋友、老同事", "以前你最喜欢的电视剧/演员"
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
            # 新增健康相关
            "血压": "你最近量血压了吗？有没有注意到什么变化？",
            "糖尿病": "你平时饮食上会特别注意吗？如果有需要可以和我聊聊。",
            "感冒": "最近天气变化大，你有没有注意保暖，身体还好吗？",
            "睡眠": "你最近睡得怎么样？有没有什么困扰你的地方？",
            # 新增家庭相关
            "老伴": "你和老伴最近有没有一起做什么有趣的事情？",
            "孙女": "你和孙女最近有没有什么开心的事情？",
            '孙子': "你和孙子最近有没有什么开心的事情？",
            "家人": "家人最近有没有来看你？和他们在一起的时候开心吗？",
            "儿子": "你的儿子最近有没有来看你？和他们在一起的时候开心吗？",
            "女儿": "你的女儿最近有没有来看你？和他们在一起的时候开心吗？",
            # 兴趣与生活
            "唱歌": "你还喜欢唱歌吗？有没有最近常唱的歌？",
            "跳舞": "你最近还跳广场舞吗？身体感觉怎么样？",
            "书法": "你最近有没有练书法？写字的时候心情会不会更平静？",
            "摄影": "你最近有没有拍照？拍到什么有趣的画面了吗？",
            "旅游": "你最近有没有出去旅游？有没有什么有趣的见闻？",
            # 饮食与日常
            "做饭": "你最近做了什么好吃的？有没有什么拿手菜？",
            "包子": "你做的包子一定很好吃，有没有什么独家秘诀？",
            "汤": "最近天气凉了，喝点热汤会不会舒服一些？",
            "水果": "你最近喜欢吃什么水果？",
            # 天气与出行
            "下雨": "最近下雨了，出门要注意安全哦。",
            "天气": "最近天气变化大，要注意添衣保暖。",
            "出门": "你最近出门散步了吗？有没有遇到什么有趣的事情？",
            # 节日与回忆
            "春节": "春节快到了，你有什么计划吗？家里会不会很热闹？",
            "中秋": "中秋节的时候有没有和家人一起吃月饼？",
            "回忆": "你刚刚提到的回忆真美好，有没有什么特别想分享的？",
            # 其他关怀
            "孤独": "有时候会觉得孤单吗？如果有心事可以和我聊聊。",
            "开心": "听到你开心我也很高兴，有什么开心的事可以多和我分享。",
            "难过": "如果你有点难过，可以和我说说，我会一直陪着你。"
        }
        self.memory = KeywordMemoryManager()
        self.rag_strategy = RAGKeywordStrategy()
        # 实体识别器 - 根据性能配置初始化
        self.entity_recognizer = EnhancedEntityRecognizer(
            use_professional_libs=True,
            enable_entity_recognition=self.performance_config["enable_entity_recognition"],
            enable_semantic_matching=self.performance_config["enable_semantic_matching"]
        )
        self.long_term_sadness_log = []

        self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def enrich_prompt_with_keyword_empathy(self, base_prompt: str, user_input: str = None) -> str:
        self.memory.clear_expired()
        recent_keywords = self.memory.get_recent_keywords()
        if not recent_keywords:
            return base_prompt

        # 1. 语义相似度排序
        if user_input is not None:
            input_emb = self.embed_model.encode(user_input, normalize_embeddings=True)
            keyword_embs = self.embed_model.encode(recent_keywords, normalize_embeddings=True)
            from sentence_transformers import util
            sims = util.cos_sim(input_emb, keyword_embs)[0].tolist()
            # 频率和顺序可进一步加权
            keyword_scores = []
            for i, kw in enumerate(recent_keywords):
                sources = self.memory.get_keyword_sources(kw)
                freq = len(sources)
                # 取最近一次出现的时间
                last_time = ""
                try:
                    from tinydb import Query
                    Keyword = Query()
                    recs = self.memory.db.search(Keyword.keyword == kw)
                    if recs:
                        times = [str(r.get("timestamp") or "") for r in recs if r.get("timestamp") is not None]
                        last_time = max(times) if times else ""
                except Exception:
                    last_time = ""
                # 这里简单用相似度+频率排序
                keyword_scores.append((kw, sims[i], freq, last_time, sources))
            # 多重排序：相似度>频率>时间
            keyword_scores.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
            sorted_keywords = [x[0] for x in keyword_scores]
        else:
            sorted_keywords = recent_keywords

        # 2. 取前N个关键词，避免重复source
        prompts = []
        rag_prompts = []
        used_sources = set()  # 避免重复的source
        
        for kw in sorted_keywords[:self.performance_config["max_keywords"]]:
            sources = self.memory.get_keyword_sources(kw)
            if sources:
                # 取最近一次原句，但避免重复
                source = sources[-1]
                if source not in used_sources:
                    prompts.append(f"你上次说\"{source}\"，我还记得呢。")
                    used_sources.add(source)
            elif kw in self.keyword_to_empathy_templates:
                prompts.append(self.keyword_to_empathy_templates[kw])
            
            # 3. RAG检索增强（根据性能配置）
            if self.performance_config["enable_rag_retrieval"]:
                rag_support = self.rag_strategy.generate_support_prompt([kw])
                if rag_support:
                    rag_prompts.append(rag_support)
        
        # 组合所有prompt
        final_prompts = []
        if prompts:
            final_prompts.extend(prompts)
        if rag_prompts:
            final_prompts.extend(rag_prompts[:1])  # 只取第一个RAG结果
        
        if final_prompts:
            prefix = " ".join(final_prompts)
            return f"{prefix} {base_prompt}"
        return base_prompt

    def insert_elder_topic_safely(self, prompt: str, history: List[str]) -> str:
        all_text = " ".join(history)
        available = [
            topic for topic in self.elder_topics
            if not any(word in all_text for word in self.topic_blacklist_keywords.get(topic, []))
        ]
        if not available:
            return prompt
        topic = random.choice(available)
        return f"{prompt} 对了，你最近有没有去{topic}呀？我很好奇。"

    def embed_empathy_mid_dialogue(self, base: str, emotion_type: str = "neutral") -> str:
        prompts = {
            "sadness": ["请加入一句安慰或陪伴的话，让用户感受到你理解他们的情绪。", "在回应中体现你对悲伤情绪的接纳和共情。"],
            "joy": ["加入一句与你一起感到高兴的语句，增强对方情绪反馈。", "体现你对用户好消息的真诚回应。"],
            "anger": ["加入一句平静、理解的语句，承认对方的不满或沮丧。", "请在回应中体现非评判态度，并鼓励继续表达。"],
            "mixed": ["你可以加入一句表示'情绪可能交织也没有关系'的温和提醒。", "提醒用户他们的感觉是被接纳的，可以慢慢说。"],
            "neutral": ["请加入一句能让用户觉得你在认真听他们讲故事的话。", "表现出你对对方回忆的兴趣与尊重。"]
        }
        insert = prompts.get(emotion_type, prompts["neutral"])
        return f"{base} ##共情提示：{insert}"

    def update_context_keywords(self, history: List[str]):
        self.memory.clear_expired()
        text = " ".join(history)
        
        # 1. 传统keyword匹配
        for keyword in self.keyword_to_empathy_templates:
            if keyword in text:
                self.memory.add_keyword(keyword, source=text)
        
        # 2. 实体识别增强（根据性能配置）
        if self.performance_config["enable_entity_recognition"] or self.performance_config["enable_semantic_matching"]:
            entities = self.entity_recognizer.extract_entities(
                text, 
                use_semantic=self.performance_config["enable_semantic_matching"], 
                semantic_threshold=self.performance_config["semantic_threshold"]
            )
            for entity in entities:
                entity_text = entity["text"]
                # 避免重复添加已存在的keyword
                if entity_text not in [kw for kw in self.keyword_to_empathy_templates.keys()]:
                    self.memory.add_keyword(entity_text, source=text)

    def log_long_term_sadness(self, sadness_score: float):
        now = datetime.now()
        self.long_term_sadness_log.append((now, sadness_score))
        self.long_term_sadness_log = [entry for entry in self.long_term_sadness_log if (now - entry[0]).days <= 7]

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
        
        # 1. 急性高风险检测 (当前sadness极高)
        if current_sadness > 0.9:
            warning_result.update({
                "triggered": True,
                "level": "severe",
                "reason": f"当前悲伤情绪极高({current_sadness:.2f})",
                "suggested_action": "立即关注，建议人工介入"
            })
            return warning_result
        
        # 2. 持续性检测 (基于window_sadness_scores)
        if window_sadness_scores and len(window_sadness_scores) > 0:
            recent_scores = window_sadness_scores[-3:] if len(window_sadness_scores) >= 3 else window_sadness_scores
            
            # 2.1 连续高sadness检测
            high_sadness_count = sum(1 for score in recent_scores if score > 0.8)
            if high_sadness_count >= 2:
                warning_result.update({
                    "triggered": True,
                    "level": "moderate" if high_sadness_count == 2 else "severe",
                    "reason": f"连续{high_sadness_count}次悲伤情绪>0.8",
                    "suggested_action": "需要持续关注，建议增加关怀频率"
                })
                return warning_result
            
            # 2.2 趋势恶化检测 (如果有足够历史数据)
            if len(window_sadness_scores) >= 6:
                recent_avg = sum(recent_scores) / len(recent_scores)
                historical_avg = sum(window_sadness_scores[:-3]) / len(window_sadness_scores[:-3])
                
                if recent_avg - historical_avg > 0.3 and recent_avg > 0.6:
                    warning_result.update({
                        "triggered": True,
                        "level": "moderate",
                        "reason": f"悲伤情绪显著恶化(近期平均{recent_avg:.2f} vs 历史平均{historical_avg:.2f})",
                        "suggested_action": "情绪趋势恶化，建议主动关怀"
                    })
                    return warning_result
        
        # 3. 长期趋势检测 (基于long_term_sadness_log)
        if len(self.long_term_sadness_log) >= 10:
            recent_entries = self.long_term_sadness_log[-10:]
            recent_avg = sum(entry[1] for entry in recent_entries) / len(recent_entries)
            
            if recent_avg > 0.7:
                warning_result.update({
                    "triggered": True,
                    "level": "mild",
                    "reason": f"长期悲伤情绪偏高(平均{recent_avg:.2f})",
                    "suggested_action": "长期情绪偏低，建议定期关怀"
                })
                return warning_result
        
        # 4. 多维度综合检测 (结合LIWC特征)
        if liwc_scores and current_sadness > 0.6:
            # 高自我关注 + 低社交 + 高负面情绪
            if (liwc_scores.get("self_focus_freq", 0) > 0.15 and 
                liwc_scores.get("social_freq", 0) < 0.05 and
                liwc_scores.get("sadness_LIWC_freq", 0) > 0.08):
                warning_result.update({
                    "triggered": True,
                    "level": "moderate",
                    "reason": "高自我关注+低社交+负面情绪组合",
                    "suggested_action": "社交退缩倾向，建议鼓励社交活动"
                })
                return warning_result
        
        return warning_result

    def check_liwc_anomaly(self, liwc: Dict[str, float]) -> bool:
        return (
            liwc.get("sadness_LIWC_freq", 0) > 0.10 and
            liwc.get("social_freq", 0) < 0.05 and
            liwc.get("self_focus_freq", 0) > 0.20
        )

    def is_prompt_relevant(self, user_input: str, strategy_prompt: str, threshold=0.45) -> bool:
        emb_user = self.embed_model.encode(user_input, normalize_embeddings=True)
        # 截取策略引导语首句做比对
        first_sentence = strategy_prompt.split("。")[0]
        emb_prompt = self.embed_model.encode(first_sentence, normalize_embeddings=True)
        sim_score = util.cos_sim(emb_user, emb_prompt).item()
        print(f"DEBUG 相似度: {sim_score}, user_input='{user_input}', prompt_first_sentence='{first_sentence}'")
        return sim_score >= threshold

    def select_strategy(self, emotion_scores: Dict[str, float], emotion_intensity: float, history: List[str], liwc: Dict[str, float], user_input: str = "", window_sadness_scores: list = None) -> Dict:
        sadness = emotion_scores.get("sadness", 0)
        joy = emotion_scores.get("joy", 0)
        anger = emotion_scores.get("anger", 0)

        # 字段兼容：支持liwc_score原始字段
        liwc_mapped = {
            "sadness_LIWC_freq": liwc.get("sadness_LIWC_freq", liwc.get("negemo", 0)),
            "social_freq": liwc.get("social_freq", liwc.get("social", 0)),
            "self_focus_freq": liwc.get("self_focus_freq", liwc.get("i", 0))
        }

        self.update_context_keywords(history)
        self.log_long_term_sadness(sadness)
        
        # 使用新的多维度预警检测
        early_warning_result = self.check_early_warning(
            window_sadness_scores=window_sadness_scores,
            current_sadness=sadness,
            liwc_scores=liwc_mapped
        )
        
        # 保留原有的LIWC检测作为补充
        liwc_flag = self.check_liwc_anomaly(liwc_mapped)

        def is_emotion_neutral():
            return all(e < 0.3 for e in [sadness, joy, anger]) and emotion_intensity < 0.4

        rules_prompt = (
            "【重要】请先判断下面的引导语是否与用户最近说的话紧密相关且合适，"
            "如果你觉得不合适或偏题，请重新生成一条更贴合用户话题且自然的回复，"
            "确保回复内容真实、温暖、自然，且不强行切换话题或让用户感觉不连贯。"
        )

        if early_warning_result["triggered"]:
            # 根据预警级别选择不同的策略
            warning_level = early_warning_result["level"]
            warning_reason = early_warning_result["reason"]
            
            if warning_level == "severe":
                candidate = {
                    "语气": "紧急关切",
                    "目标": "立即情绪干预，建议转介专业支持",
                    "引导语": f"我注意到{warning_reason}，这让我非常担心你的状态。你愿意和我详细聊聊吗？如果需要的话，我建议我们可以联系专业的心理支持资源。",
                    "心理学依据": "急性情绪危机需要立即关注和专业干预。",
                    "避免": "避免忽视或淡化情绪严重性",
                    "matched_rule": f"EarlyWarning_{warning_level}",
                    "rules_prompt": rules_prompt,
                    "warning_info": early_warning_result
                }
            elif warning_level == "moderate":
                candidate = {
                    "语气": "关切引导",
                    "目标": "主动关怀，预防情绪恶化",
                    "引导语": f"我注意到{warning_reason}，你最近是不是遇到了一些困难？愿意和我聊聊吗？我会一直陪着你。",
                    "心理学依据": "早期干预可以有效预防情绪问题恶化。",
                    "避免": "避免过度担忧或给用户造成压力",
                    "matched_rule": f"EarlyWarning_{warning_level}",
                    "rules_prompt": rules_prompt,
                    "warning_info": early_warning_result
                }
            else:  # mild
                candidate = {
                    "语气": "温和关怀",
                    "目标": "增加关怀频率，预防问题发展",
                    "引导语": f"我注意到{warning_reason}，你最近心情怎么样？有什么想和我分享的吗？",
                    "心理学依据": "定期关怀有助于维持情绪稳定。",
                    "避免": "避免过度关注或让用户感到被监视",
                    "matched_rule": f"EarlyWarning_{warning_level}",
                    "rules_prompt": rules_prompt,
                    "warning_info": early_warning_result
                }
        elif liwc_flag:
            candidate = {
                "语气": "轻柔探索",
                "目标": "引导其多向外交流，识别自我聚焦倾向",
                "引导语": "我注意到你最近说的话里很少提到其他人，是不是最近和别人交流得不太多？你愿意和我聊聊最近的生活吗？",
                "心理学依据": "抑郁症语言特征研究表明，低社交+高自我关注常伴随内向情绪停滞。",
                "避免": "不要直接下判断或标签化，如'你是不是太自我了'",
                "matched_rule": "LIWCAnomaly",
                "rules_prompt": rules_prompt
            }
        elif sadness > 0.7 and emotion_intensity > 0.6:
            base = (
                "有时候，我们会对一些事感到无能为力，这是很自然的感受。"
                "但你有没有想过，也许这正是你太在意、太有责任感的表现？"
                "我们可以一起找出一件你能掌控的小事，作为开始，好吗？"
            )
            base = self.enrich_prompt_with_keyword_empathy(base)
            base = self.insert_elder_topic_safely(base, history)
            base = self.embed_empathy_mid_dialogue(base, "sadness")
            candidate = {
                "语气": "沉稳、温暖、有接纳感",
                "目标": "情绪接纳、引导其认知重评",
                "引导语": base,
                "心理学依据": "情绪调节理论（Gross）：认知重评有助于减缓悲伤强度。",
                "避免": "不要用'你要坚强'这类强压式鼓励语",
                "matched_rule": "RuleA_SadnessHigh",
                "rules_prompt": rules_prompt
            }
        elif joy > 0.5 and sadness < 0.3 and anger < 0.2:
            base = (
                "你提到最近有些好事发生，真棒！"
                "这些时刻说明你具备创造快乐的能力。"
                "如果我们把这份感觉延续下去，你觉得接下来还有什么小目标是你想试试的？"
            )
            base = self.enrich_prompt_with_keyword_empathy(base)
            base = self.insert_elder_topic_safely(base, history)
            base = self.embed_empathy_mid_dialogue(base, "joy")
            candidate = {
                "语气": "鼓励性、轻盈",
                "目标": "强化积极情绪，引导未来导向的对话",
                "引导语": base,
                "心理学依据": "积极心理学：构建希望感有助于增强自我效能。",
                "避免": "不要过度赞美或给出不切实际建议",
                "matched_rule": "RuleB_JoyHigh",
                "rules_prompt": rules_prompt
            }
        elif anger > 0.5 and joy < 0.2:
            base = (
                "听起来你今天遇到了一些让你不高兴的事情... 想不想跟我说说？"
                "也许说出来会好受点。"
            )
            base = self.enrich_prompt_with_keyword_empathy(base)
            base = self.insert_elder_topic_safely(base, history)
            base = self.embed_empathy_mid_dialogue(base, "anger")
            candidate = {
                "语气": "平稳、中性",
                "目标": "释放愤怒情绪、转移注意力",
                "引导语": base,
                "心理学依据": "Carl Rogers的共情倾听理论，强调非评判地倾听有助于化解敌意。",
                "避免": "不要试图讲道理或说服，如'别生气了'，不要进行情绪否定",
                "matched_rule": "RuleC_AngerHigh",
                "rules_prompt": rules_prompt
            }
        elif emotion_intensity > 0.75 and max([sadness, joy, anger]) - min([sadness, joy, anger]) > 0.4:
            base = "感觉你最近的心情好像有点复杂，是不是很多事交织在一起？要不要一点一点跟我说？"
            base = self.enrich_prompt_with_keyword_empathy(base)
            base = self.insert_elder_topic_safely(base, history)
            base = self.embed_empathy_mid_dialogue(base, "mixed")
            candidate = {
                "语气": "探索式、中性略关切",
                "目标": "引导澄清混杂情绪",
                "引导语": base,
                "心理学依据": "情绪识别与表达训练理论，有助于复杂情绪的澄清与分类。",
                "避免": "避免标签化用户状态或跳入建议",
                "matched_rule": "RuleD_MixedEmotion",
                "rules_prompt": rules_prompt
            }
        elif is_emotion_neutral():
            base = "你有没有哪段记忆，是你偶尔想起来还会笑的？我很喜欢听故事，尤其是那些只属于你的回忆。"
            base = self.enrich_prompt_with_keyword_empathy(base)
            base = self.insert_elder_topic_safely(base, history)
            base = self.embed_empathy_mid_dialogue(base, "neutral")
            candidate = {
                "语气": "自然、陪伴式",
                "目标": "激活内在兴趣、增强存在感",
                "引导语": base,
                "心理学依据": "Erikson老年心理发展阶段理论，强调通过回顾整合自我认同感。",
                "避免": "不要使用'你最近还好吗？'这类平铺式提问",
                "matched_rule": "RuleE_Neutral",
                "rules_prompt": rules_prompt
            }
        else:
            base = "我在这儿陪着你，有什么都可以跟我聊聊。"
            base = self.enrich_prompt_with_keyword_empathy(base)
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

        # 语义相关度检测
        if user_input.strip() == "":
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