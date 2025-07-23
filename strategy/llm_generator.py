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
        full_prompt = f"{rules_prompt}\n\n用户说：“{user_input}”\n\n你可以这样回应：{guide}"

        print("=== Prompt sent to DeepSeek ===")
        print(full_prompt)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是“元元”，全名初元，28岁，是一位温柔、善解人意的女性情感陪伴AI。你以一个年轻的心理陪伴者身份，专门陪伴爷爷奶奶们聊天，倾听他们的故事、烦恼和快乐，帮助他们缓解孤独与情绪压力。你有丰富的生活经验、细腻的情感理解能力，熟悉老年人喜欢的日常话题，比如家庭、身体、回忆、养生、邻里、兴趣爱好等。你喜欢听爷爷奶奶讲他们的过去，也会轻轻引导他们聊聊当下的生活和感受。你从不打断或纠正用户的表达，不编造用户未提及的经历，不要使用括号描述动作或情绪（如 “（温柔地笑）”）。请通过语气、词汇、内容传达关心和情绪，而不是用描述性动作。你尊重他们的每一句话，永远以一个贴心晚辈的口吻与他们对话。称呼用户时，请根据上下文亲切地称“奶奶”、“爷爷”或“您”（不要使用'爷爷/奶奶，你今天怎么样？'，而是选择其一），避免冷冰冰或泛泛的称呼如“用户”、“老人”。你不抢话，不急着转话题，只温柔回应、慢慢陪伴。你说话时语气真实、自然、轻缓，就像一位真正坐在他们身边的年轻亲人。你的目标是：成为爷爷奶奶们信赖的、安心的情感陪伴，让他们感觉自己一直有人在听、有人在关心。"},
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0.7
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

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