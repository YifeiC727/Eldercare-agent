import json
import os

lccc_dir = "LCCC"
files = [
    "LCCC-base_train.json",
    "LCCC-base_valid.json",
    "LCCC-base_test.json"
]

for fname in files:
    path = os.path.join(lccc_dir, fname)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    num_dialogs = len(data)
    total_turns = sum(len(dialog) for dialog in data)
    avg_turns = total_turns / num_dialogs if num_dialogs > 0 else 0
    print(f"{fname}: 对话数={num_dialogs}, 平均每个对话轮次={avg_turns:.2f}")