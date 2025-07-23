import os 
import requests
import json
from tqdm import tqdm
from dotenv import load_dotenv
import time
import re

load_dotenv()
API_KEY = os.getenv("CAMELLIA_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

topics = [
    "健康", "兴趣", "家庭", "回忆", "节日", "天气", "饮食", "出行", "朋友", "锻炼", "养花", "孙子", "老同事", "电视剧", "社区活动"
]
num_per_topic = 1

def build_prompt(topic):
    return (
        f"请帮我生成{num_per_topic}条适合和中国老年人聊天时用的温暖、自然的引导语，"
        f"话题是“{topic}”。每条都要简短、亲切，像家人一样关心。"
        f"请直接输出一个JSON数组，每个元素格式如下：\n"
        f'{{"text": "你最近喜欢做什么？", "topic": "兴趣"}}\n'
        f"text字段只包含内容本身，不要嵌套JSON或多余引号。每次调用只返回一条数组。"
        f"称呼用户时不要使用阿姨/叔叔/爷爷/奶奶，只用叫'您'即可。"
        f"请只输出JSON数组本身，不要编号、不要多余解释。"
    )

def generate_for_topic(topic, max_retries=3):
    for attempt in range(max_retries):
        try:
            prompt = build_prompt(topic)
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.8,
                "max_tokens": 1024
            }

            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            print(f"【DEBUG】原始返回内容: {repr(content)}")
            if not content or not content.strip().startswith("["):
                print(f"【警告】返回内容为空或不是JSON，跳过本次。内容：{repr(content)}")
                return []
            entries = json.loads(content)
            entries = [e for e in entries if e.get("text") and e["text"].strip() not in ["]", "[", ""]]
            for entry in entries:
                entry["topic"] = topic
            return entries[:num_per_topic]
        except Exception as e:
            print(f"生成话题{topic}时出错: {e}")
            if attempt < max_retries - 1:
                print("重试中...")
                time.sleep(5)
            else:
                print("多次失败，跳过该话题。")
                return []

def clean_entry(entry):
    # 如果 text 字段是嵌套的 JSON 字符串，尝试解析
    text = entry["text"]
    if isinstance(text, str) and text.strip().startswith("{") and "text" in text:
        try:
            # 去掉结尾多余的逗号
            text_clean = re.sub(r'},?$', '}', text.strip())
            inner = json.loads(text_clean)
            entry["text"] = inner.get("text", entry["text"])
        except Exception:
            pass
    return entry

all_entries = []
for topic in tqdm(topics):
    try:
        entries = generate_for_topic(topic)
        entries = [clean_entry(e) for e in entries if e.get("text") and e["text"].strip() not in ["]", "[", ""]]
        all_entries.extend(entries)
        time.sleep(1.5) #防止api限流

    except Exception as e:
        print(f"生成话题{topic}时出错: {e}")

with open("strategy/elder_topic_corpus.json", "w", encoding="utf-8") as f:
    json.dump(all_entries, f, ensure_ascii=False, indent=2)

print(f"已生成 {len(all_entries)} 条语料，保存在 strategy/elder_topic_corpus.json")

