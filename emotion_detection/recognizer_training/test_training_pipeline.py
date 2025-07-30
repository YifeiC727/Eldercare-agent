#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练流程的脚本
验证数据加载、特征提取、模型训练是否正常工作
"""

import os
import sys
import json
import glob
from pathlib import Path

def test_data_files():
    """测试数据文件是否存在且格式正确"""
    print("=== 测试数据文件 ===")
    
    # 检查前10个文件是否存在
    data_pattern = "../../output_batches/eldercare_15000_dialogues_part*.jsonl"
    files = sorted(glob.glob(data_pattern))[:10]
    
    # 如果没找到，尝试绝对路径
    if not files:
        import os
        current_dir = os.getcwd()
        print(f"当前目录: {current_dir}")
        abs_pattern = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 
                                  "output_batches", "eldercare_15000_dialogues_part*.jsonl")
        print(f"尝试绝对路径: {abs_pattern}")
        files = sorted(glob.glob(abs_pattern))[:10]
    
    if not files:
        print("❌ 未找到数据文件")
        return False
    
    print(f"✅ 找到 {len(files)} 个数据文件")
    
    # 检查第一个文件的格式
    try:
        with open(files[0], 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            sample = json.loads(first_line)
            
            # 检查必要字段
            required_fields = ['dialogue', 'target_emotion']
            for field in required_fields:
                if field not in sample:
                    print(f"❌ 缺少必要字段: {field}")
                    return False
            
            # 检查target_emotion格式
            target_emotion = sample['target_emotion']
            emotion_fields = ['anger', 'sadness', 'joy', 'intensity']
            for field in emotion_fields:
                if field not in target_emotion:
                    print(f"❌ target_emotion缺少字段: {field}")
                    return False
            
            print(f"✅ 数据格式正确，样本数: {len(sample['dialogue'])}")
            print(f"✅ target_emotion: {target_emotion}")
            
    except Exception as e:
        print(f"❌ 数据文件格式错误: {e}")
        return False
    
    return True

def test_feature_extraction():
    """测试特征提取模块"""
    print("\n=== 测试特征提取 ===")
    
    try:
        from transformers import BertTokenizer, BertModel
        from feature_extraction import build_dataset
        
        # 加载BERT模型
        bert_model_path = "bert-base-chinese"
        tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        bert_model = BertModel.from_pretrained(bert_model_path)
        
        # 测试单个文件
        data_pattern = "../../output_batches/eldercare_15000_dialogues_part*.jsonl"
        files = sorted(glob.glob(data_pattern))[:1]  # 只测试第一个文件
        
        if not files:
            print("❌ 未找到数据文件")
            return False
        
        print(f"测试文件: {os.path.basename(files[0])}")
        X_text, X_emo, y = build_dataset(files[0], tokenizer, bert_model, max_turns=5)
        
        print(f"✅ 特征提取成功")
        print(f"  - 文本特征维度: {X_text.shape}")
        print(f"  - 情感特征维度: {X_emo.shape}")
        print(f"  - 标签维度: {y.shape}")
        print(f"  - 样本数: {len(X_text)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        return False

def test_model_imports():
    """测试模型模块导入"""
    print("\n=== 测试模型导入 ===")
    
    try:
        from emotion_stacking_model import (
            EmotionRegressor, train_mlp, train_lgb, train_ridge, 
            train_stacking, StackingEmotionPredictor
        )
        print("✅ 所有模型模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模型模块导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试训练流程...")
    
    # 切换到正确的目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 运行测试
    tests = [
        test_data_files,
        test_feature_extraction,
        test_model_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✅ 所有测试通过，可以开始训练！")
        print("\n运行训练命令:")
        print("cd emotion_detection/recognizer_training")
        print("python train_stacking.py")
    else:
        print("❌ 部分测试失败，请检查问题后重试")

if __name__ == "__main__":
    main() 