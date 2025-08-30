import jieba
import json
import os
import re
from collections import defaultdict, Counter
import requests

category_code_to_name = {
    "127": "negemo",
    "121": "social",
    "122": "social",
    "123": "social",
    "124": "social",
    "125": "cogproc",
    "1": "i"
}

class EmotionRecognizer:
    def __init__(self, liwc_path="sc_liwc.dic"):
        self.liwc_dict = self.load_mapped_liwc(liwc_path)
        self.api_key = os.getenv("CAMELLIA_KEY")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
        # === Temporarily disable EmotionRegressor related model loading to avoid errors ===
        # bert_model_path = "bert-base-chinese"
        # mlp_model = EmotionRegressor(768, 16)
        # mlp_model.load_state_dict(torch.load("emotion_detection/recognizer_training/mlp_model.pt", map_location="cpu"))
        # lgb_models = joblib.load("emotion_detection/recognizer_training/lgb_models.pkl")
        # ridge_models = joblib.load("emotion_detection/recognizer_training/ridge_models.pkl")
        # stacker = joblib.load("emotion_detection/recognizer_training/stacker.pkl")
        # self.stacking_predictor = StackingEmotionPredictor(
        #     bert_model_path, [mlp_model, lgb_models, ridge_models], stacker, max_turns=5
        # )

    def load_mapped_liwc(self, filepath, category_map=category_code_to_name):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, filepath)
        result = defaultdict(list)
        with open(full_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                codes = parts[1:]
                for code in codes:
                    if code in category_map:
                        result[category_map[code]].append(word)
        return dict(result)

    def liwc_score(self, text):
        tokens = list(jieba.cut(text, cut_all=False))
        counter = Counter()
        total = 0
        for token in tokens:
            for cat, words in self.liwc_dict.items():
                if token in words:
                    print(f"匹配到词：'{token}' 类别：{cat}")
                    counter[cat] += 1
                    total += 1
        if total > 0:
            for k in counter:
                counter[k] = round(counter[k] / total, 2)
        return dict(counter)

    def build_windows(self, dialogue, min_rounds=5, step=2, min_words=50, max_words=150):
        windows = []
        total_utterances = len(dialogue)
        total_rounds = total_utterances // 2
        if total_rounds == 0 and total_utterances > 0:
            total_rounds = 1

        i = 0
        while i < total_rounds:
            start_idx = i * 2
            end_idx = start_idx + min_rounds * 2
            if end_idx > total_utterances:
                end_idx = total_utterances

            window_utterances = dialogue[start_idx:end_idx]
            word_count = len(list(jieba.cut(''.join(window_utterances), cut_all=False)))

            j = i + (len(window_utterances) // 2)
            while (j - i) < min_rounds and j < total_rounds:
                next_start = j * 2
                next_end = next_start + 2
                if next_end > total_utterances:
                    next_end = total_utterances
                window_utterances.extend(dialogue[next_start:next_end])
                j += 1
                word_count = len(list(jieba.cut(''.join(window_utterances), cut_all=False)))

            while word_count < min_words and j < total_rounds:
                next_start = j * 2
                next_end = next_start + 2
                if next_end > total_utterances:
                    next_end = total_utterances
                window_utterances.extend(dialogue[next_start:next_end])
                j += 1
                word_count = len(list(jieba.cut(''.join(window_utterances), cut_all=False)))

            window_text = ''.join(window_utterances)

            windows.append({
                "start_round": i,
                "end_round": j - 1,
                "num_turns": j - i,
                "num_words": word_count,
                "text": window_text
            })

            i += step

        return windows

    def analyze_windows(self, windows):
        results = []
        for win in windows:
            score = self.liwc_score(win['text'])
            results.append({
                **win,
                "score": score
            })
        return results

    def extract_json_from_markdown(self, text: str) -> str:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()

    def predict_emotion_stacking(self, dialogue):
        return self.stacking_predictor.predict(dialogue)

    def analyze_emotion_deepseek(self, text: str) -> dict:
        prompt = f"""
你是一个中文情绪识别助手，请根据用户输入的一句话，识别以下情绪强度（范围为0.00 - 1.00，保留两位小数）：

- anger（愤怒）
- sadness（悲伤）
- joy（喜悦）
- intensity（整体情绪强烈程度）

请只输出一个JSON对象，格式如下：

{{
  "anger": 0.xx,
  "sadness": 0.xx,
  "joy": 0.xx,
  "intensity": 0.xx
}}

示例输入："我真的快受不了了，没人听我说话。"

现在请分析："{text}"
""".strip()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            print("🧠 DeepSeek 返回：", content)

            clean_json = self.extract_json_from_markdown(content)
            return json.loads(clean_json)

        except Exception as e:
            print("❌ 情绪识别调用出错：", e)
            print("📄 原始返回内容:", response.text if 'response' in locals() else "No response object")
            return {"anger": 0.0, "sadness": 0.0, "joy": 0.0, "intensity": 0.0}

    def print_window_scores(self, results):
        all_categories = ["negemo", "social", "cogproc", "i"]
        for res in results:
            print(f"窗口 轮 {res['start_round']} - {res['end_round']}")
            print(f"轮数: {res['num_turns']}, 词数: {res['num_words']}")
            scores = res.get('score', {})
            for cat in all_categories:
                val = scores.get(cat, 0)
                print(f"  #{cat}: {val}")
            print("-" * 40)

    def analyze_rounds_liwc(self, dialogue):
        return [self.liwc_score(utt) for utt in dialogue]

    # Add judgment in all places where self.stacking_predictor is used
    def analyze_emotion(self, text):
        # Only use LLM solution, stacking_predictor not supported for now
        return self.analyze_emotion_deepseek(text)


if __name__ == "__main__":
    dialogue = [
        "我最近真的很累，什么都不想做。",
        "也不知道，就是每天都不想起床。",
        "走不动，感觉身体都沉重了。",
        "没有，我不想麻烦别人。",
        "我其实一直都觉得很孤独。",
        "我怕别人觉得我很奇怪。",
        "我经常失眠，晚上脑子停不下来。",
        "最近心情特别不好。",
        "每天都很焦虑。",
        "想逃避一切。"
    ]

    recog = EmotionRecognizer()
    windows = recog.build_windows(dialogue)
    results = recog.analyze_windows(windows)
    recog.print_window_scores(results)

    last = dialogue[-1]
            # Only use analyze_emotion (automatic fallback)
    emotion = recog.analyze_emotion(last)
    print("\n===== DeepSeek 情绪识别（最近一句） =====")
    print(f"用户输入：{last}")
    for k, v in emotion.items():
        print(f"  {k}: {v:.2f}")
