# emotion_detection/recognizer_training/compare_models_final.py
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import mean_squared_error, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set font support for English
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# Define MLP model class (same as in train_stacking.py)
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

def load_data_and_models():
    """Load validation data and trained models"""
    print("Loading validation data and models...")
    
    # Load validation data
    validation_data = joblib.load('validation_data.pkl')
    X_text_val = validation_data['X_text_val']
    X_emo_val = validation_data['X_emo_val']
    y_val = validation_data['y_val']
    
    # Load trained models
    # For MLP, need to create model instance first, then load weights
    mlp_model = EmotionRegressor(X_text_val.shape[1], X_emo_val.shape[1])
    mlp_model.load_state_dict(torch.load("mlp_model.pt", map_location='cpu'))
    mlp_model.eval()  # Set to evaluation mode
    
    lgb_models = joblib.load("lgb_models.pkl")
    ridge_models = joblib.load("ridge_models.pkl")
    stacker = joblib.load("stacker.pkl")
    
    print(f"Validation set samples: {len(X_text_val)}")
    print("Models loaded successfully")
    
    return X_text_val, X_emo_val, y_val, mlp_model, lgb_models, ridge_models, stacker

def evaluate_single_model(model, X_text, X_emo, y, model_name):
    """Evaluate single model"""
    print(f"Evaluating {model_name}...")
    
    if model_name == "MLP":
        with torch.no_grad():
            y_pred = model(torch.tensor(X_text, dtype=torch.float32), 
                          torch.tensor(X_emo, dtype=torch.float32)).cpu().numpy()
    else:
        X_combined = np.concatenate([X_text, X_emo], axis=1)
        y_pred = np.stack([m.predict(X_combined) for m in model], axis=1)
    
    # Calculate overall MSE
    mse_overall = mean_squared_error(y, y_pred)
    
    # Calculate MSE and AUC-ROC for each dimension
    emotion_names = ["Anger", "Sadness", "Joy", "Intensity"]
    results = {}
    
    for i, emotion in enumerate(emotion_names):
        mse_dim = mean_squared_error(y[:, i], y_pred[:, i])
        try:
            auc = roc_auc_score((y[:, i] > 0.5).astype(int), y_pred[:, i])
        except:
            auc = 0.0
        results[emotion] = {"mse": mse_dim, "auc": auc}
    
    # Calculate average AUC-ROC
    avg_auc = np.mean([results[emotion]["auc"] for emotion in emotion_names])
    
    return {
        "model_name": model_name,
        "mse_overall": mse_overall,
        "avg_auc": avg_auc,
        "details": results
    }

def evaluate_stacking_model(mlp_model, lgb_models, ridge_models, stacker, X_text, X_emo, y):
    """Evaluate Stacking model"""
    print("Evaluating Stacking Ensemble...")
    
    # Get base model predictions
    with torch.no_grad():
        mlp_pred = mlp_model(torch.tensor(X_text, dtype=torch.float32), 
                            torch.tensor(X_emo, dtype=torch.float32)).cpu().numpy()
    
    X_combined = np.concatenate([X_text, X_emo], axis=1)
    lgb_pred = np.stack([m.predict(X_combined) for m in lgb_models], axis=1)
    ridge_pred = np.stack([m.predict(X_combined) for m in ridge_models], axis=1)
    
    # Stacking prediction
    X_stack = np.concatenate([mlp_pred, lgb_pred, ridge_pred], axis=1)
    y_pred = stacker.predict(X_stack)
    
    # Calculate overall MSE
    mse_overall = mean_squared_error(y, y_pred)
    
    # Calculate MSE and AUC-ROC for each dimension
    emotion_names = ["Anger", "Sadness", "Joy", "Intensity"]
    results = {}
    
    for i, emotion in enumerate(emotion_names):
        mse_dim = mean_squared_error(y[:, i], y_pred[:, i])
        try:
            auc = roc_auc_score((y[:, i] > 0.5).astype(int), y_pred[:, i])
        except:
            auc = 0.0
        results[emotion] = {"mse": mse_dim, "auc": auc}
    
    # Calculate average AUC-ROC
    avg_auc = np.mean([results[emotion]["auc"] for emotion in emotion_names])
    
    return {
        "model_name": "Stacking Ensemble",
        "mse_overall": mse_overall,
        "avg_auc": avg_auc,
        "details": results
    }

def print_comparison_results(all_results):
    """Print comparison results"""
    print("\n" + "="*80)
    print("Model Performance Comparison Results")
    print("="*80)
    
    # Overall performance comparison
    print("\nOverall Performance Comparison:")
    print(f"{'Model Name':<20} {'MSE':<10} {'Average AUC-ROC':<12}")
    print("-" * 50)
    
    for result in all_results:
        print(f"{result['model_name']:<20} {result['mse_overall']:<10.4f} {result['avg_auc']:<12.4f}")
    
    # Detailed comparison by emotion dimension
    emotion_names = ["Anger", "Sadness", "Joy", "Intensity"]
    
    print(f"\nDetailed Comparison by Emotion Dimension:")
    print(f"{'Emotion':<10} {'Model':<20} {'MSE':<10} {'AUC-ROC':<10}")
    print("-" * 60)
    
    for emotion in emotion_names:
        for result in all_results:
            details = result['details'][emotion]
            print(f"{emotion:<10} {result['model_name']:<20} {details['mse']:<10.4f} {details['auc']:<10.4f}")
        print("-" * 60)

def plot_comparison_results(all_results):
    """Plot comparison result charts"""
    print("\nGenerating comparison charts...")
    
    # Prepare data
    model_names = [result['model_name'] for result in all_results]
    mse_values = [result['mse_overall'] for result in all_results]
    auc_values = [result['avg_auc'] for result in all_results]
    
    # Create comparison charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MSE comparison
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    bars1 = ax1.bar(model_names, mse_values, color=colors[:len(model_names)])
    ax1.set_title('Overall MSE Comparison (Lower is Better)', fontsize=14)
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add values on bar charts
    for bar, value in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom')
    
    # AUC-ROC comparison
    bars2 = ax2.bar(model_names, auc_values, color=colors[:len(model_names)])
    ax2.set_title('Average AUC-ROC Comparison (Higher is Better)', fontsize=14)
    ax2.set_ylabel('AUC-ROC')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add values on bar charts
    for bar, value in zip(bars2, auc_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison_final.png', dpi=300, bbox_inches='tight')
    print("Comparison chart saved as 'model_comparison_final.png'")
    
    # Emotion dimension heatmaps
    emotion_names = ["Anger", "Sadness", "Joy", "Intensity"]
    
    # Prepare heatmap data
    mse_matrix = []
    auc_matrix = []
    
    for result in all_results:
        mse_row = [result['details'][emotion]['mse'] for emotion in emotion_names]
        auc_row = [result['details'][emotion]['auc'] for emotion in emotion_names]
        mse_matrix.append(mse_row)
        auc_matrix.append(auc_row)
    
    # Draw heatmaps
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
    
    # MSE heatmap
    sns.heatmap(mse_matrix, annot=True, fmt='.4f', cmap='Reds_r', 
                xticklabels=emotion_names, yticklabels=model_names, ax=ax3)
    ax3.set_title('MSE Heatmap by Emotion Dimension (Darker = Worse)', fontsize=14)
    ax3.set_xlabel('Emotion Dimensions')
    ax3.set_ylabel('Models')
    
    # AUC-ROC heatmap
    sns.heatmap(auc_matrix, annot=True, fmt='.4f', cmap='Greens', 
                xticklabels=emotion_names, yticklabels=model_names, ax=ax4)
    ax4.set_title('AUC-ROC Heatmap by Emotion Dimension (Darker = Better)', fontsize=14)
    ax4.set_xlabel('Emotion Dimensions')
    ax4.set_ylabel('Models')
    
    plt.tight_layout()
    plt.savefig('emotion_dimension_comparison_final.png', dpi=300, bbox_inches='tight')
    print("Emotion dimension comparison chart saved as 'emotion_dimension_comparison_final.png'")

def main():
    """Main function"""
    try:
        # Load validation data and models
        X_text_val, X_emo_val, y_val, mlp_model, lgb_models, ridge_models, stacker = load_data_and_models()
        
        # Evaluate all models
        all_results = []
        
        # Evaluate base models
        all_results.append(evaluate_single_model(mlp_model, X_text_val, X_emo_val, y_val, "MLP"))
        all_results.append(evaluate_single_model(lgb_models, X_text_val, X_emo_val, y_val, "LightGBM"))
        all_results.append(evaluate_single_model(ridge_models, X_text_val, X_emo_val, y_val, "Ridge"))
        
        # Evaluate Stacking model
        all_results.append(evaluate_stacking_model(mlp_model, lgb_models, ridge_models, stacker, 
                                                 X_text_val, X_emo_val, y_val))
        
        # Print comparison results
        print_comparison_results(all_results)
        
        # Plot comparison charts
        plot_comparison_results(all_results)
        
        print("\n=== Model Comparison Completed ===")
        
    except Exception as e:
        print(f"Error occurred during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 