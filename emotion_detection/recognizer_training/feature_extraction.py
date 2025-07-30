import json
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

def extract_features_from_sample(sample, tokenizer, bert_model, max_turns=5):
    dialogue = sample["dialogue"]
    text = ""
    emotion_feats = []
    user_turns = 0
    for turn in dialogue[:-1]:
        if turn["role"] == "user":
            text += "[USER] " + turn["text"] + " "
            emo = turn["emotion"]
            emotion_feats.extend([
                float(emo["anger"]),
                float(emo["sadness"]),
                float(emo["joy"]),
                float(emo["intensity"])
            ])
            user_turns += 1
        else:
            text += "[AI] " + turn["text"] + " "
    while user_turns < max_turns-1:
        emotion_feats.extend([0.0, 0.0, 0.0, 0.0])
        user_turns += 1
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        text_emb = outputs.pooler_output.squeeze(0).numpy()
    return text_emb, np.array(emotion_feats)

def build_dataset(jsonl_path, tokenizer, bert_model, max_turns=5):
    X_text, X_emo, y = [], [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            text_emb, emo_feats = extract_features_from_sample(sample, tokenizer, bert_model, max_turns)
            X_text.append(text_emb)
            X_emo.append(emo_feats)
            y.append([
                float(sample["target_emotion"]["anger"]),
                float(sample["target_emotion"]["sadness"]),
                float(sample["target_emotion"]["joy"]),
                float(sample["target_emotion"]["intensity"])
            ])
    return np.array(X_text), np.array(X_emo), np.array(y)
