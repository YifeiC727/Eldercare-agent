import os
import json
import random
import time
import datetime
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()  # 自动递归查找所有父目录的.env文件
api_key = os.getenv("CAMELLIA_KEY")
print("CAMELLIA_KEY:", repr(api_key))  # 调试用，确认key是否加载成功
api_url = "https://api.deepseek.com/v1/chat/completions"
model = "deepseek-chat"

# 新增：分批输出文件夹
output_dir = "output_batches"
os.makedirs(output_dir, exist_ok=True)

THEMES = [
    "健康", "家庭", "子女", "朋友", "孤独", "兴趣爱好", "回忆童年", "饮食", "锻炼", "天气", "宠物", "邻里关系", "节日", "旅行", "学习", "看病", "购物", "理财", "新闻", "社会", "科技", "娱乐"
]

PROMPT_TEMPLATE = '''
你是一个专业的老年人陪伴AI对话数据生成器。请严格按照如下要求生成一条多轮中文对话数据：

- 本条对话主题为“{theme}”，请确保内容紧扣该主题，且与前述示例不同。
- 总轮数为{n}（必须为奇数），每轮user和AI交替发言，最后一轮必须是user发言。
- 每个user发言都要输出如下情感分数（emotion，四个分数均为0.00-1.00，保留两位小数）：
    - anger（愤怒）：愤怒情绪的强度，0为完全无愤怒，0.3为轻微不满，0.6为明显生气，1为极强愤怒。
    - sadness（悲伤）：悲伤情绪的强度，0为完全无悲伤，0.3为轻微失落，0.6为明显难过，1为极强悲伤。
    - joy（喜悦）：高兴/开心情绪的强度，0为完全无喜悦，0.3为轻微愉快，0.6为明显高兴，1为极强喜悦。
    - intensity（整体情绪强烈程度）：本轮情感表达的总体强度，0为平淡无波，0.3为有些情绪波动，0.6为情绪较强烈，1为极强烈。
- 输出严格为如下JSON格式：

{{
    "dialogue": [
        {{"role": "user", "text": "今天心情不错，儿子来看我了。", "emotion": {{"anger": 0.00, "sadness": 0.00, "joy": 0.90, "intensity": 0.80}}}},
        {{"role": "ai", "text": "真好！和家人在一起总是很开心。"}},
        {{"role": "user", "text": "是啊，感觉很幸福。", "emotion": {{"anger": 0.00, "sadness": 0.00, "joy": 0.80, "intensity": 0.70}}}}
    ]
}}

- 另一个示例（七轮完整对话）：
{{
    "dialogue": [
        {{"role": "user", "text": "最近总觉得有点孤独。", "emotion": {{"anger": 0.00, "sadness": 0.60, "joy": 0.00, "intensity": 0.50}}}},
        {{"role": "ai", "text": "怎么了？家里人最近没来看你吗？"}},
        {{"role": "user", "text": "孩子们都很忙，很少来看我。", "emotion": {{"anger": 0.00, "sadness": 0.70, "joy": 0.00, "intensity": 0.60}}}},
        {{"role": "ai", "text": "你可以和朋友多联系联系。"}},
        {{"role": "user", "text": "朋友们也都很少见面了，感觉有点失落。", "emotion": {{"anger": 0.00, "sadness": 0.65, "joy": 0.00, "intensity": 0.55}}}},
        {{"role": "ai", "text": "要不要我陪你聊聊天？"}},
        {{"role": "user", "text": "有你陪我聊天我就开心多了。", "emotion": {{"anger": 0.00, "sadness": 0.20, "joy": 0.60, "intensity": 0.50}}}}
    ]
}}

- 只输出JSON对象本身，不要输出任何代码块标记（如```json ... ```），也不要输出多余的文字或解释。
- 你的回复必须严格以 {{ 开头，以 }} 结尾，且不能有任何代码块标记或多余内容。
- 请确保每个user发言的emotion分数合理，且与发言内容一致。
- 请确保本条对话与前述示例内容和表达方式尽量不同，避免重复。
'''

# 新增：emotion masking专用prompt
MASKING_PROMPT_TEMPLATE = '''
你是一个专业的老年人陪伴AI对话数据生成器。请严格按照如下要求生成一条多轮中文对话数据：

- 本条对话主题为“{theme}”，请确保内容紧扣该主题，且与前述示例不同。
- 总轮数为{n}（必须为奇数），每轮user和AI交替发言，最后一轮必须是user发言。
- 每个user发言都要输出如下情感分数（emotion，四个分数均为0.00-1.00，保留两位小数），并结合发言内容合理推断（避免情绪与语义割裂）。情绪标注应不仅为标签，而是引导模型训练其理解人心的能力。所以在你生成时，一定是你要自己能够从那些看似中性实则悲伤的句子中感受到那么大的悲伤，才去给打那个分数才行：
    - anger（愤怒）：愤怒情绪的强度，0为完全无愤怒，0.3为轻微不满，0.6为明显生气，1为极强愤怒。
    - sadness（悲伤）：悲伤情绪的强度，0为完全无悲伤，0.3为轻微失落，0.6为明显难过，1为极强悲伤。
    - joy（喜悦）：高兴/开心情绪的强度，0为完全无喜悦，0.3为轻微愉快，0.6为明显高兴，1为极强喜悦。
    - intensity（整体情绪强烈程度）：本轮情感表达的总体强度，0为平淡无波，0.3为有些情绪波动，0.6为情绪较强烈，1为极强烈。
- 输出严格为如下JSON格式：

{{
  "dialogue": [
    {{"role": "user", "text": "邻居家的金毛昨天生了一窝小狗，真可爱啊。", "emotion": {{"anger": 0.0, "sadness": 0.1, "joy": 0.9, "intensity": 0.4}}}},
    {{"role": "ai", "text": "您以前也养过狗吗？"}},
    {{"role": "user", "text": "养过一只博美，后来……算了不说这个了，现在看别人养也挺好的。", "emotion": {{"anger": 0.1, "sadness": 0.85, "joy": 0.2, "intensity": 0.75}}}}
  ]
}}

- 请严格区分以下两类情况：
  1. 句子表面和实际都积极/平静（如“今天社区组织了太极拳活动，我看好多老邻居都去了，挺热闹的。”），sadness/anger 必须低于 0.2。
  2. 只有当句子中出现回避、转折、叹息、‘其实’、‘不过’、‘都挺好’等掩饰性表达时，才允许 sadness/anger 高于 0.5。
- 绝大多数日常正向/中性句子（如“社区今天有义诊活动，医生说我血压控制得不错。”），sadness/anger 必须为0~0.1，joy可高，intensity可适中。
- 反例（不要这样标注）：
  {{"role": "user", "text": "社区今天有义诊活动，医生说我血压控制得不错。", "emotion": {{"anger": 0.0, "sadness": 0.3, "joy": 0.7, "intensity": 0.4}}}}
- 正确标注应为：
  {{"role": "user", "text": "社区今天有义诊活动，医生说我血压控制得不错。", "emotion": {{"anger": 0.0, "sadness": 0.0, "joy": 0.7, "intensity": 0.4}}}}
- 只有当你能明确感受到句子中有回避、转折、叹息、‘其实’、‘不过’、‘都挺好’等掩饰性表达时，才允许 sadness/anger 高于 0.2，否则必须为0~0.1。
- 反例（不要这样标注）：
  {{"role": "user", "text": "今天社区组织了太极拳活动，我看好多老邻居都去了，挺热闹的。", "emotion": {{"anger": 0.0, "sadness": 0.4, "joy": 0.5, "intensity": 0.3}}}}
- 正确标注应为：
  {{"role": "user", "text": "今天社区组织了太极拳活动，我看好多老邻居都去了，挺热闹的。", "emotion": {{"anger": 0.0, "sadness": 0.0, "joy": 0.7, "intensity": 0.5}}}}

- 另一些可参考的user发言示例：
  1. {{"role": "user", "text": "天气真不错，阳光很好。就是有点不知道要干什么，感觉整天都没意思。", "emotion": {{"anger": 0.0, "sadness": 0.6, "joy": 0.3, "intensity": 0.5}}}}
  2. {{"role": "user", "text": "我最近都挺好的，就是晚上睡不着觉，有时候会想一些以前的事。", "emotion": {{"anger": 0.1, "sadness": 0.7, "joy": 0.2, "intensity": 0.6}}}}
  3. {{"role": "user", "text": "我女儿最近挺忙的，不过年轻人嘛，忙点也好。", "emotion": {{"anger": 0.0, "sadness": 0.6, "joy": 0.3, "intensity": 0.4}}}}
- 说明：请生成真实、自然、符合老年人口语风格的多轮对话，尤其关注“表面平静/正向但内含负面情绪”的掩饰现象（emotional masking）。
- masking现象要结合当前主题“{theme}”自然出现。
- 只输出JSON对象本身，不要输出任何代码块标记（如```json ... ```），也不要输出多余的文字或解释。
- 你的回复必须严格以 {{ 开头，以 }} 结尾，且不能有任何代码块标记或多余内容。
- 请确保本条对话与前述示例内容和表达方式尽量不同，避免重复。
'''

# 修改build_prompt逻辑，支持两种prompt

def build_prompt(masking=False):
    n = random.choice([3, 5, 7])
    theme = random.choice(THEMES)
    if masking:
        return MASKING_PROMPT_TEMPLATE.format(n=n, theme=theme), n
    else:
        return PROMPT_TEMPLATE.format(n=n, theme=theme), n

def is_valid_dialogue(data, n):
    if not isinstance(data, dict):
        return False, "不是字典类型"
    dialogue = data.get("dialogue")
    if not isinstance(dialogue, list):
        return False, "dialogue字段不是list"
    if len(dialogue) != n:
        return False, f"对话轮数{len(dialogue)}不等于期望{n}"
    for i, turn in enumerate(dialogue):
        if i % 2 == 0:
            if turn.get("role") != "user":
                return False, f"第{i+1}轮role不是user"
            emo = turn.get("emotion")
            if not isinstance(emo, dict):
                return False, f"第{i+1}轮emotion不是dict"
            for k in ["anger", "sadness", "joy", "intensity"]:
                if k not in emo:
                    return False, f"第{i+1}轮emotion缺少{k}"
                try:
                    score = float(emo[k])
                    if not (0 <= score <= 1):
                        return False, f"第{i+1}轮emotion分数{k}超界:{score}"
                except:
                    return False, f"第{i+1}轮emotion分数{k}无法转为float"
        else:
            if turn.get("role") != "ai":
                return False, f"第{i+1}轮role不是ai"
    if dialogue[-1].get("role") != "user":
        return False, "最后一轮不是user"
    return True, ""

output_prefix = os.path.join(output_dir, "eldercare_15000_dialogues_part")
num_samples = 15000
batch_size = 1000
log_file = "generation_errors.log"
max_workers = 5  # 并发数改为5

# 断点续跑：统计已存在的分批文件中的总条数
existing_count = 0
part_idx = 1
while True:
    part_file = f"{output_prefix}{part_idx}.jsonl"
    if os.path.exists(part_file):
        with open(part_file, "r", encoding="utf-8") as pf:
            lines = sum(1 for _ in pf)
            existing_count += lines
        part_idx += 1
    else:
        break

print(f"已存在数据条数: {existing_count}, 从第{existing_count+1}条开始生成...")

start_time = datetime.datetime.now()

# 修改主生成逻辑，10000条后切换为masking prompt
def generate_one(idx):
    for _ in range(3):
        masking = (existing_count + idx) >= 10000
        prompt, n = build_prompt(masking=masking)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.2
        }
        try:
            time.sleep(1)
            response = requests.post(api_url, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }, json=payload, timeout=15)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            try:
                data = json.loads(content)
                valid, reason = is_valid_dialogue(data, n)
                if valid:
                    return json.dumps(data, ensure_ascii=False)
                else:
                    return f"校验不通过，原因: {reason}，内容: {content}"
            except Exception as e:
                return f"JSON解析失败，内容: {content}"
        except Exception as e:
            return f"API调用失败，错误: {e}"
    return "多次重试后仍失败"

i = existing_count
part_idx = (existing_count // batch_size) + 1
f = open(f"{output_prefix}{part_idx}.jsonl", "a", encoding="utf-8")
logf = open(log_file, 'a', encoding="utf-8")

try:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx in range(i, num_samples):
            futures.append(executor.submit(generate_one, idx))
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result and result.startswith("{"):
                f.write(result + "\n")
                f.flush()
                i += 1
                completed += 1
                if i % batch_size == 0:
                    f.close()
                    part_idx += 1
                    f = open(f"{output_prefix}{part_idx}.jsonl", "a", encoding="utf-8")
                if i % 50 == 0:
                    print(f"已生成{i}条，活跃线程数：{threading.active_count()}")
            else:
                logf.write(f"第{i+1}条{result}\n")
                logf.flush()
except KeyboardInterrupt:
    print("检测到Ctrl+C，正在安全退出...")
finally:
    f.close()
    logf.close()
    end_time = datetime.datetime.now()
    print(f"生成完毕！总耗时: {end_time - start_time}")