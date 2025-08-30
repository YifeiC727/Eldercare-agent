# emotion_detection/recognizer_training/train_stacking.py
import glob
import os
from transformers import BertTokenizer, BertModel
from feature_extraction import build_dataset
from emotion_stacking_model import train_mlp, train_lgb, train_ridge, train_stacking, StackingEmotionPredictor
import joblib
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

bert_model_path = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_model_path)
bert_model = BertModel.from_pretrained(bert_model_path)

def load_multiple_files(file_pattern, max_files=10):
    """Load multiple JSONL files and merge data"""
    X_text_all, X_emo_all, y_all = [], [], []
    
    # Get all matching files
    all_files = glob.glob(file_pattern)
    
    # Sort files by part number order
    def extract_part_number(filename):
        import re
        match = re.search(r'part(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    files = sorted(all_files, key=extract_part_number)[:max_files]
    print(f"Loading {len(files)} files...")
    
    for i, file_path in enumerate(files):
        print(f"Loading file {i+1}/{len(files)}: {os.path.basename(file_path)}")
        try:
            X_text, X_emo, y = build_dataset(file_path, tokenizer, bert_model, max_turns=5)
            X_text_all.append(X_text)
            X_emo_all.append(X_emo)
            y_all.append(y)
            print(f"  - Sample count: {len(X_text)}")
        except Exception as e:
            print(f"  - Loading failed: {e}")
            continue
    
    if not X_text_all:
        raise ValueError("No data files loaded successfully")
    
    # Merge all data
    X_text_combined = np.concatenate(X_text_all, axis=0)
    X_emo_combined = np.concatenate(X_emo_all, axis=0)
    y_combined = np.concatenate(y_all, axis=0)
    
    print(f"Total sample count: {len(X_text_combined)}")
    return X_text_combined, X_emo_combined, y_combined

# Load data from first 10 files
data_pattern = "output_batches/eldercare_15000_dialogues_part*.jsonl"
X_text, X_emo, y = load_multiple_files(data_pattern, max_files=10)

# 8:2 train/validation split
print("\n=== Dataset Split ===")
X_text_train, X_text_val, X_emo_train, X_emo_val, y_train, y_val = train_test_split(
    X_text, X_emo, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training set sample count: {len(X_text_train)}")
print(f"Validation set sample count: {len(X_text_val)}")

# Train base models (only train on training set)
print("\n=== Training Base Models ===")
print("Training MLP model...")
mlp_model = train_mlp(X_text_train, X_emo_train, y_train, epochs=10, lr=1e-3)

print("Training LightGBM model...")
lgb_models = train_lgb(X_text_train, X_emo_train, y_train)

print("Training Ridge model...")
ridge_models = train_ridge(X_text_train, X_emo_train, y_train)

# Stacking fusion (using training set)
print("\n=== Training Stacking Model ===")
base_models = [mlp_model, lgb_models, ridge_models]
stacker = train_stacking(base_models, X_text_train, X_emo_train, y_train)

# Save models
print("\n=== Saving Models ===")
torch.save(mlp_model.state_dict(), "mlp_model.pt")
joblib.dump(lgb_models, "lgb_models.pkl")
joblib.dump(ridge_models, "ridge_models.pkl")
joblib.dump(stacker, "stacker.pkl")
print("Models saved")

# 在训练集上评估
print("\n=== 训练集评估 ===")
X_stack_train = np.concatenate([
    mlp_model(torch.tensor(X_text_train, dtype=torch.float32), torch.tensor(X_emo_train, dtype=torch.float32)).detach().cpu().numpy(),
    np.stack([m.predict(np.concatenate([X_text_train, X_emo_train], axis=1)) for m in lgb_models], axis=1),
    np.stack([m.predict(np.concatenate([X_text_train, X_emo_train], axis=1)) for m in ridge_models], axis=1)
], axis=1)
y_pred_train = stacker.predict(X_stack_train)
mse_train = mean_squared_error(y_train, y_pred_train)
print(f"训练集MSE: {mse_train:.4f}")

# 在验证集上评估
print("\n=== 验证集评估 ===")
X_stack_val = np.concatenate([
    mlp_model(torch.tensor(X_text_val, dtype=torch.float32), torch.tensor(X_emo_val, dtype=torch.float32)).detach().cpu().numpy(),
    np.stack([m.predict(np.concatenate([X_text_val, X_emo_val], axis=1)) for m in lgb_models], axis=1),
    np.stack([m.predict(np.concatenate([X_text_val, X_emo_val], axis=1)) for m in ridge_models], axis=1)
], axis=1)
y_pred_val = stacker.predict(X_stack_val)
mse_val = mean_squared_error(y_val, y_pred_val)
print(f"验证集MSE: {mse_val:.4f}")

# 多标签AUC-ROC评估（验证集）
print("\n=== 验证集AUC-ROC评估 ===")
emotion_names = ["anger", "sadness", "joy", "intensity"]
try:
    aucs = []
    for i in range(4):
        auc = roc_auc_score((y_val[:, i] > 0.5).astype(int), y_pred_val[:, i])
        aucs.append(auc)
        print(f"{emotion_names[i]} AUC-ROC: {auc:.4f}")
    print(f"平均AUC-ROC: {np.mean(aucs):.4f}")
except Exception as e:
    print("AUC-ROC计算失败，原因：", e)

# 各情绪维度的MSE（验证集）
print("\n=== 验证集各情绪维度MSE ===")
for i in range(4):
    mse_dim = mean_squared_error(y_val[:, i], y_pred_val[:, i])
    print(f"{emotion_names[i]} MSE: {mse_dim:.4f}")

# 推理接口示例
print("\n=== 创建推理接口 ===")
predictor = StackingEmotionPredictor(bert_model_path, [mlp_model, lgb_models, ridge_models], stacker, max_turns=5)
print("StackingEmotionPredictor已创建，可用于推理")
# predictor.predict(dialogue)  # 传入对话，返回情绪分布