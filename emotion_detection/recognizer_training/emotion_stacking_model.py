# emotion_detection/recognizer_training/emotion_stacking_model.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from transformers import BertTokenizer, BertModel
import joblib
import lightgbm as lgb

class EmotionRegressor(nn.Module):
    def __init__(self, text_emb_dim, emo_feat_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_emb_dim + emo_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    def forward(self, text_emb, emo_feats):
        x = torch.cat([text_emb, emo_feats], dim=1)
        return self.fc(x)

def train_mlp(X_text, X_emo, y, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_text = torch.tensor(X_text, dtype=torch.float32).to(device)
    X_emo = torch.tensor(X_emo, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    model = EmotionRegressor(X_text.shape[1], X_emo.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_text, X_emo)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return model

def train_lgb(X_text, X_emo, y):
    X = np.concatenate([X_text, X_emo], axis=1)
    models = []
    for i in range(4):
        gbm = lgb.LGBMRegressor()
        gbm.fit(X, y[:, i])
        models.append(gbm)
    return models

def train_ridge(X_text, X_emo, y):
    X = np.concatenate([X_text, X_emo], axis=1)
    models = []
    for i in range(4):
        ridge = Ridge()
        ridge.fit(X, y[:, i])
        models.append(ridge)
    return models

def stacking_predict(base_models, X_text, X_emo):
    X = np.concatenate([X_text, X_emo], axis=1)
    preds = []
    for model in base_models:
        if isinstance(model, list):  # LightGBM/Ridge
            pred = np.stack([m.predict(X) for m in model], axis=1)
            preds.append(pred)
        else:  # torch
            with torch.no_grad():
                pred = model(torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_emo, dtype=torch.float32)).cpu().numpy()
                preds.append(pred)
    return np.concatenate(preds, axis=1)

def train_stacking(base_models, X_text, X_emo, y):
    X_stack = stacking_predict(base_models, X_text, X_emo)
    from sklearn.linear_model import LinearRegression
    stacker = LinearRegression()
    stacker.fit(X_stack, y)
    return stacker

class StackingEmotionPredictor:
    def __init__(self, bert_model_path, base_models, stacker, max_turns=7):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.bert_model = BertModel.from_pretrained(bert_model_path)
        self.base_models = base_models
        self.stacker = stacker
        self.max_turns = max_turns

    def extract_features(self, dialogue):
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
        while user_turns < self.max_turns-1:
            emotion_feats.extend([0.0, 0.0, 0.0, 0.0])
            user_turns += 1
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            text_emb = outputs.pooler_output.squeeze(0).numpy()
        return text_emb, np.array(emotion_feats)

    def predict(self, dialogue):
        text_emb, emo_feats = self.extract_features(dialogue)
        X_text = text_emb.reshape(1, -1)
        X_emo = emo_feats.reshape(1, -1)
        base_preds = []
        for model in self.base_models:
            if isinstance(model, list):
                pred = np.stack([m.predict(np.concatenate([X_text, X_emo], axis=1)) for m in model], axis=1)
                base_preds.append(pred)
            else:
                with torch.no_grad():
                    base_preds.append(model(torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_emo, dtype=torch.float32)).cpu().numpy())
        X_stack = np.concatenate(base_preds, axis=1)
        final_pred = self.stacker.predict(X_stack)[0]
        return {
            "anger": float(final_pred[0]),
            "sadness": float(final_pred[1]),
            "joy": float(final_pred[2]),
            "intensity": float(final_pred[3])
        }