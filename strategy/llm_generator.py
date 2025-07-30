import os
import requests
from dotenv import load_dotenv
from strategy.strategy_selector import StrategySelector

class LLMGenerator:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("CAMELLIA_KEY")  # ✅ 读取 DeepSeek 的 key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"

    def generate_response(self, user_input, strategy_output):
        rules_prompt = strategy_output.get("rules_prompt", "")
        guide = strategy_output.get("引导语", "")
        
        # 如果是个人信息查询，直接返回引导语，不需要LLM处理
        if strategy_output.get("matched_rule", "").startswith("PersonalInfo"):
            return guide
        
        # 优化提示词，强调简洁性和真实性
        full_prompt = f"""用户说：'{user_input}'

请严格按照以下引导语回复，要求：
1. 回复要简洁，控制在50字以内
2. 不要编造用户未提及的经历或信息
3. 不要使用括号描述动作或情绪
4. 只回应当前对话内容，不要引入历史记忆
5. 语气要自然、温暖

引导语：{guide}"""

        print("=== Prompt sent to DeepSeek ===")
        print(full_prompt)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 优化系统提示词，强调简洁性和真实性
        system_content = """你是"元元"，28岁，温柔善解人意的女性情感陪伴AI。

核心原则：
1. 回复简洁：每次回复控制在50字以内
2. 真实记忆：只基于用户当前对话内容回应，不编造历史记忆
3. 自然表达：不使用括号描述动作，通过语气传达情感
4. 专注当下：只关注当前对话，不引入未提及的信息

称呼：根据上下文称"奶奶"、"爷爷"或"您"
语气：温暖、自然、轻缓，像真正的年轻亲人

重要：严格遵循用户提供的引导语，不要添加额外内容。"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0.2,  # 进一步降低温度，让回复更确定性
            "max_tokens": 100,   # 限制最大token数，确保简洁
            "top_p": 0.8        # 控制输出多样性
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            
            # 后处理：确保回复简洁
            if len(content) > 100:
                # 如果回复过长，截取前100个字符
                content = content[:100].rstrip()
                if not content.endswith(('。', '！', '？', '，')):
                    content += '。'
            
            return content
            
        except Exception as e:
            print(f"LLM API调用失败: {e}")
            # 返回简单的fallback回复，而不是直接返回引导语
            if "401" in str(e) or "Unauthorized" in str(e):
                return "抱歉，我现在无法连接到AI服务。请稍后再试。"
            elif guide:
                # 如果API失败但有引导语，返回一个简化的回复
                return "我理解您的感受，能多和我说说吗？"
            else:
                return "我理解您的感受，能多和我说说吗？"

# === 以下逻辑不变 ===
if __name__ == "__main__":
    selector = StrategySelector()
    generator = LLMGenerator()

    emotion_scores = {"sadness": 0.3, "joy": 0.6, "anger": 0.1}
    emotion_intensity = 0.65
    history = ["孙子这次考试考得不错", "孩子给我买了新花"]
    liwc = {
        "sadness_LIWC_freq": 1.0,
        "social_freq": 3.5,
        "self_focus_freq": 2.2
    }
    user_input = "孙子这次考试考得不错"

    strategy = selector.select_strategy(emotion_scores, emotion_intensity, history, liwc, user_input)
    final_response = generator.generate_response(user_input, strategy)
    print("\n=== 最终 LLM 回复 ===")
    print(final_response)