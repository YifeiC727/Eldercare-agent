
import jieba
import json
from collections import defaultdict, Counter

# Category code → category name mapping
category_code_to_name = {
    "127": "negemo",  # negative emotion
    "130": "sad",     # sadness
    "121": "social",  # social processes
    "125": "cogproc", # cognitive processes
    "1": "i"          # first-person singular pronoun
}

def load_mapped_liwc(filepath, category_map=category_code_to_name):
    result = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
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

def liwc_score(text, liwc_dict):
    tokens = list(jieba.cut(text))
    counter = Counter()
    for token in tokens:
        for cat, words in liwc_dict.items():
            if token in words:
                counter[cat] += 1
    return dict(counter)
