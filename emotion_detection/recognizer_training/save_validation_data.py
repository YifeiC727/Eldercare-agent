# emotion_detection/recognizer_training/save_validation_data.py
import numpy as np
import joblib
from train_stacking import load_multiple_files
from sklearn.model_selection import train_test_split

def save_validation_data():
    """Save validation data from train_stacking.py"""
    print("Saving validation data...")
    
    # Use the same data loading method as train_stacking.py
    data_pattern = "output_batches/eldercare_15000_dialogues_part*.jsonl"
    X_text, X_emo, y = load_multiple_files(data_pattern, max_files=10)
    
    # Use the same train_test_split parameters
    X_text_train, X_text_val, X_emo_train, X_emo_val, y_train, y_val = train_test_split(
        X_text, X_emo, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Save validation data
    validation_data = {
        'X_text_val': X_text_val,
        'X_emo_val': X_emo_val,
        'y_val': y_val
    }
    
    joblib.dump(validation_data, 'validation_data.pkl')
    
    print(f"Validation data saved:")
    print(f"  - Validation set sample count: {len(X_text_val)}")
    print(f"  - Text feature dimensions: {X_text_val.shape}")
    print(f"  - Emotion feature dimensions: {X_emo_val.shape}")
    print(f"  - Label dimensions: {y_val.shape}")

if __name__ == "__main__":
    save_validation_data() 