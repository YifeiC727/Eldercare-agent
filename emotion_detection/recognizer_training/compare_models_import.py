# emotion_detection/recognizer_training/compare_models_import.py
import numpy as np
import torch
import joblib
from sklearn.metrics import mean_squared_error, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_trained_models():
    """加载已训练好的模型"""
    print("正在加载已训练好的模型...")
    
    # 加载基础模型
    mlp_model = torch.load("mlp_model.pt", map_location='cpu')
    lgb_models = joblib.load("lgb_models.pkl")
    ridge_models = joblib.load("ridge_models.pkl")
    stacker = joblib.load("stacker.pkl")
    
    print("模型加载完成")
    return mlp_model, lgb_models, ridge_models, stacker

def get_validation_data_from_train_stacking():
    """从train_stacking.py中获取验证数据"""
    print("正在从train_stacking.py中获取验证数据...")
    
    # 导入train_stacking模块
    import train_stacking
    
    # 重新运行train_stacking的数据加载部分
    # 这样我们就能获得相同的验证集
    data_pattern = "../../output_batches/eldercare_15000_dialogues_part*.jsonl"
    X_text, X_emo, y = train_stacking.load_multiple_files(data_pattern, max_files=10)
    
    # 使用相同的train_test_split参数
    from sklearn.model_selection import train_test_split
    X_text_train, X_text_val, X_emo_train, X_emo_val, y_train, y_val = train_test_split(
        X_text, X_emo, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"验证集样本数: {len(X_text_val)}")
    return X_text_val, X_emo_val, y_val

def evaluate_single_model(model, X_text, X_emo, y, model_name):
    """评估单个模型"""
    print(f"评估 {model_name}...")
    
    if model_name == "MLP":
        with torch.no_grad():
            y_pred = model(torch.tensor(X_text, dtype=torch.float32), 
                          torch.tensor(X_emo, dtype=torch.float32)).cpu().numpy()
    else:
        X_combined = np.concatenate([X_text, X_emo], axis=1)
        y_pred = np.stack([m.predict(X_combined) for m in model], axis=1)
    
    # 计算整体MSE
    mse_overall = mean_squared_error(y, y_pred)
    
    # 计算各维度的MSE和AUC-ROC
    emotion_names = ["anger", "sadness", "joy", "intensity"]
    results = {}
    
    for i, emotion in enumerate(emotion_names):
        mse_dim = mean_squared_error(y[:, i], y_pred[:, i])
        try:
            auc = roc_auc_score((y[:, i] > 0.5).astype(int), y_pred[:, i])
        except:
            auc = 0.0
        results[emotion] = {"mse": mse_dim, "auc": auc}
    
    # 计算平均AUC-ROC
    avg_auc = np.mean([results[emotion]["auc"] for emotion in emotion_names])
    
    return {
        "model_name": model_name,
        "mse_overall": mse_overall,
        "avg_auc": avg_auc,
        "details": results
    }

def evaluate_stacking_model(mlp_model, lgb_models, ridge_models, stacker, X_text, X_emo, y):
    """评估Stacking模型"""
    print("评估 Stacking Ensemble...")
    
    # 获取基础模型预测
    with torch.no_grad():
        mlp_pred = mlp_model(torch.tensor(X_text, dtype=torch.float32), 
                            torch.tensor(X_emo, dtype=torch.float32)).cpu().numpy()
    
    X_combined = np.concatenate([X_text, X_emo], axis=1)
    lgb_pred = np.stack([m.predict(X_combined) for m in lgb_models], axis=1)
    ridge_pred = np.stack([m.predict(X_combined) for m in ridge_models], axis=1)
    
    # Stacking预测
    X_stack = np.concatenate([mlp_pred, lgb_pred, ridge_pred], axis=1)
    y_pred = stacker.predict(X_stack)
    
    # 计算整体MSE
    mse_overall = mean_squared_error(y, y_pred)
    
    # 计算各维度的MSE和AUC-ROC
    emotion_names = ["anger", "sadness", "joy", "intensity"]
    results = {}
    
    for i, emotion in enumerate(emotion_names):
        mse_dim = mean_squared_error(y[:, i], y_pred[:, i])
        try:
            auc = roc_auc_score((y[:, i] > 0.5).astype(int), y_pred[:, i])
        except:
            auc = 0.0
        results[emotion] = {"mse": mse_dim, "auc": auc}
    
    # 计算平均AUC-ROC
    avg_auc = np.mean([results[emotion]["auc"] for emotion in emotion_names])
    
    return {
        "model_name": "Stacking Ensemble",
        "mse_overall": mse_overall,
        "avg_auc": avg_auc,
        "details": results
    }

def print_comparison_results(all_results):
    """打印对比结果"""
    print("\n" + "="*80)
    print("模型性能对比结果")
    print("="*80)
    
    # 整体性能对比
    print("\n整体性能对比:")
    print(f"{'模型名称':<20} {'MSE':<10} {'平均AUC-ROC':<12}")
    print("-" * 50)
    
    for result in all_results:
        print(f"{result['model_name']:<20} {result['mse_overall']:<10.4f} {result['avg_auc']:<12.4f}")
    
    # 各情绪维度详细对比
    emotion_names = ["anger", "sadness", "joy", "intensity"]
    
    print(f"\n各情绪维度详细对比:")
    print(f"{'情绪':<10} {'模型':<20} {'MSE':<10} {'AUC-ROC':<10}")
    print("-" * 60)
    
    for emotion in emotion_names:
        for result in all_results:
            details = result['details'][emotion]
            print(f"{emotion:<10} {result['model_name']:<20} {details['mse']:<10.4f} {details['auc']:<10.4f}")
        print("-" * 60)

def plot_comparison_results(all_results):
    """绘制对比结果图表"""
    print("\n正在生成对比图表...")
    
    # 准备数据
    model_names = [result['model_name'] for result in all_results]
    mse_values = [result['mse_overall'] for result in all_results]
    auc_values = [result['avg_auc'] for result in all_results]
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MSE对比
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    bars1 = ax1.bar(model_names, mse_values, color=colors[:len(model_names)])
    ax1.set_title('整体MSE对比 (越低越好)', fontsize=14)
    ax1.set_ylabel('MSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值
    for bar, value in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom')
    
    # AUC-ROC对比
    bars2 = ax2.bar(model_names, auc_values, color=colors[:len(model_names)])
    ax2.set_title('平均AUC-ROC对比 (越高越好)', fontsize=14)
    ax2.set_ylabel('AUC-ROC')
    ax2.tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值
    for bar, value in zip(bars2, auc_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison_import.png', dpi=300, bbox_inches='tight')
    print("对比图表已保存为 'model_comparison_import.png'")
    
    # 各情绪维度热力图
    emotion_names = ["anger", "sadness", "joy", "intensity"]
    
    # 准备热力图数据
    mse_matrix = []
    auc_matrix = []
    
    for result in all_results:
        mse_row = [result['details'][emotion]['mse'] for emotion in emotion_names]
        auc_row = [result['details'][emotion]['auc'] for emotion in emotion_names]
        mse_matrix.append(mse_row)
        auc_matrix.append(auc_row)
    
    # 绘制热力图
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
    
    # MSE热力图
    sns.heatmap(mse_matrix, annot=True, fmt='.4f', cmap='Reds_r', 
                xticklabels=emotion_names, yticklabels=model_names, ax=ax3)
    ax3.set_title('各情绪维度MSE热力图 (越红越差)', fontsize=14)
    
    # AUC-ROC热力图
    sns.heatmap(auc_matrix, annot=True, fmt='.4f', cmap='Greens', 
                xticklabels=emotion_names, yticklabels=model_names, ax=ax4)
    ax4.set_title('各情绪维度AUC-ROC热力图 (越绿越好)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('emotion_dimension_comparison_import.png', dpi=300, bbox_inches='tight')
    print("情绪维度对比图已保存为 'emotion_dimension_comparison_import.png'")

def main():
    """主函数"""
    try:
        # 从train_stacking.py中获取验证数据
        X_text_val, X_emo_val, y_val = get_validation_data_from_train_stacking()
        
        # 加载已训练好的模型
        mlp_model, lgb_models, ridge_models, stacker = load_trained_models()
        
        # 评估各个模型
        all_results = []
        
        # 评估基础模型
        all_results.append(evaluate_single_model(mlp_model, X_text_val, X_emo_val, y_val, "MLP"))
        all_results.append(evaluate_single_model(lgb_models, X_text_val, X_emo_val, y_val, "LightGBM/Ridge"))
        all_results.append(evaluate_single_model(ridge_models, X_text_val, X_emo_val, y_val, "Ridge"))
        
        # 评估Stacking模型
        all_results.append(evaluate_stacking_model(mlp_model, lgb_models, ridge_models, stacker, 
                                                 X_text_val, X_emo_val, y_val))
        
        # 打印对比结果
        print_comparison_results(all_results)
        
        # 绘制对比图表
        plot_comparison_results(all_results)
        
        print("\n=== 模型对比完成 ===")
        
    except Exception as e:
        print(f"对比过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 