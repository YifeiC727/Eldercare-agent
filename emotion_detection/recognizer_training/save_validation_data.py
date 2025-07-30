# emotion_detection/recognizer_training/save_validation_data.py
import numpy as np
import joblib
from train_stacking import load_multiple_files
from sklearn.model_selection import train_test_split

def save_validation_data():
    """保存train_stacking.py中的验证数据"""
    print("正在保存验证数据...")
    
    # 使用与train_stacking.py相同的数据加载方法
    data_pattern = "../../output_batches/eldercare_15000_dialogues_part*.jsonl"
    X_text, X_emo, y = load_multiple_files(data_pattern, max_files=10)
    
    # 使用相同的train_test_split参数
    X_text_train, X_text_val, X_emo_train, X_emo_val, y_train, y_val = train_test_split(
        X_text, X_emo, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 保存验证数据
    validation_data = {
        'X_text_val': X_text_val,
        'X_emo_val': X_emo_val,
        'y_val': y_val
    }
    
    joblib.dump(validation_data, 'validation_data.pkl')
    
    print(f"验证数据已保存:")
    print(f"  - 验证集样本数: {len(X_text_val)}")
    print(f"  - 文本特征维度: {X_text_val.shape}")
    print(f"  - 情绪特征维度: {X_emo_val.shape}")
    print(f"  - 标签维度: {y_val.shape}")

if __name__ == "__main__":
    save_validation_data() 