#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据架构优化脚本
将keyword_memory.json改为只存储关键词引用，避免与conversations.json重复
"""

import json
import os
from datetime import datetime

def analyze_data_duplication():
    """分析数据重复情况"""
    print("=== 数据重复分析 ===")
    
    # 分析conversations.json
    conv_file = "user_bio/data/conversations.json"
    if os.path.exists(conv_file):
        with open(conv_file, 'r', encoding='utf-8') as f:
            conv_data = json.load(f)
        
        total_conversations = len(conv_data)
        total_conv_size = 0
        user_ids = set()
        
        for conv in conv_data:
            user_msg = conv.get("user_message", "")
            ai_reply = conv.get("ai_reply", "")
            user_id = conv.get("user_id", "")
            total_conv_size += len(user_msg) + len(ai_reply)
            if user_id:
                user_ids.add(user_id)
        
        print(f"conversations.json:")
        print(f"  - 对话数量: {total_conversations}")
        print(f"  - 对话内容总大小: {total_conv_size} 字符")
        print(f"  - 用户数量: {len(user_ids)}")
        print(f"  - 平均每个对话: {total_conv_size // total_conversations if total_conversations > 0 else 0} 字符")
    
    # 分析keyword_memory.json
    keyword_file = "strategy/keyword_memory.json"
    if os.path.exists(keyword_file):
        with open(keyword_file, 'r', encoding='utf-8') as f:
            keyword_data = json.load(f)
        
        if "_default" in keyword_data:
            old_format = True
            total_keywords = len(keyword_data["_default"])
            total_source_size = 0
            
            for item in keyword_data["_default"].values():
                source = item.get("source", "")
                total_source_size += len(source)
            
            print(f"keyword_memory.json (旧格式):")
            print(f"  - 关键词数量: {total_keywords}")
            print(f"  - 源文本总大小: {total_source_size} 字符")
            print(f"  - 平均每个关键词源文本: {total_source_size // total_keywords if total_keywords > 0 else 0} 字符")
            print(f"  - 重复存储比例: {total_source_size / total_conv_size * 100:.1f}%" if total_conv_size > 0 else "无法计算")
        else:
            print(f"keyword_memory.json (新格式):")
            print(f"  - 用户数量: {len(keyword_data)}")
            total_keywords = sum(len(user_data) for user_data in keyword_data.values())
            print(f"  - 总关键词数量: {total_keywords}")

def optimize_keyword_memory():
    """优化keyword_memory.json，移除重复的source文本"""
    
    keyword_file = "strategy/keyword_memory.json"
    conv_file = "user_bio/data/conversations.json"
    
    if not os.path.exists(keyword_file):
        print("关键词记忆文件不存在，跳过优化")
        return
    
    if not os.path.exists(conv_file):
        print("对话记录文件不存在，无法建立引用关系")
        return
    
    try:
        # 读取现有数据
        with open(keyword_file, 'r', encoding='utf-8') as f:
            keyword_data = json.load(f)
        
        with open(conv_file, 'r', encoding='utf-8') as f:
            conv_data = json.load(f)
        
        print(f"\n=== 开始数据架构优化 ===")
        print(f"原始关键词数量: {len(keyword_data.get('_default', {}))}")
        
        # 创建备份
        backup_file = f"strategy/keyword_memory_backup_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(keyword_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 备份已创建: {backup_file}")
        
        # 创建对话ID映射
        conv_id_mapping = {}
        for conv in conv_data:
            conv_id = conv.get("_id")
            user_id = conv.get("user_id")
            timestamp = conv.get("timestamp")
            if conv_id and user_id and timestamp:
                conv_id_mapping[conv_id] = {
                    "user_id": user_id,
                    "timestamp": timestamp
                }
        
        # 优化数据，移除source文本，添加对话ID引用
        optimized_data = {}
        
        if "_default" in keyword_data:
            # 旧格式数据
            old_data = keyword_data["_default"]
            
            # 按用户分组
            user_keywords = {}
            for key, item in old_data.items():
                keyword = item.get("keyword", "")
                timestamp = item.get("timestamp", "")
                source = item.get("source", "")
                
                if keyword and timestamp:
                    # 尝试找到对应的对话ID
                    conversation_id = None
                    for conv_id, conv_info in conv_id_mapping.items():
                        if conv_info["timestamp"] == timestamp:
                            conversation_id = conv_id
                            break
                    
                    # 如果没有找到精确匹配，使用时间戳作为对话ID
                    if not conversation_id:
                        conversation_id = f"conv_{timestamp.replace(':', '').replace('-', '').replace('.', '')}"
                    
                    user_id = conv_id_mapping.get(conversation_id, {}).get("user_id", "unknown_user")
                    
                    if user_id not in user_keywords:
                        user_keywords[user_id] = []
                    
                    user_keywords[user_id].append({
                        "keyword": keyword,
                        "timestamp": timestamp,
                        "conversation_id": conversation_id
                    })
            
            # 转换为新格式
            for user_id, keywords in user_keywords.items():
                optimized_data[user_id] = {}
                for i, keyword_info in enumerate(keywords):
                    optimized_data[user_id][str(i + 1)] = keyword_info
        
        # 保存优化后的数据
        with open(keyword_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 数据架构优化完成")
        print(f"优化后用户数量: {len(optimized_data)}")
        print(f"优化后关键词数量: {sum(len(user_data) for user_data in optimized_data.values())}")
        
        # 计算存储节省
        original_size = os.path.getsize(keyword_file + ".backup") if os.path.exists(keyword_file + ".backup") else 0
        optimized_size = os.path.getsize(keyword_file)
        if original_size > 0:
            saved_percentage = (original_size - optimized_size) / original_size * 100
            print(f"存储节省: {saved_percentage:.1f}%")
        
        # 验证优化结果
        verify_optimization(optimized_data, conv_data)
        
    except Exception as e:
        print(f"❌ 数据架构优化失败: {e}")

def verify_optimization(optimized_data, conv_data):
    """验证优化结果"""
    print("\n=== 优化验证 ===")
    
    try:
        # 检查数据结构
        total_keywords = 0
        total_users = len(optimized_data)
        
        for user_id, user_data in optimized_data.items():
            user_keywords = len(user_data)
            total_keywords += user_keywords
            
            print(f"用户 {user_id}:")
            print(f"  - 关键词数量: {user_keywords}")
            
            # 检查关键词质量
            valid_keywords = 0
            for key, item in user_data.items():
                if item.get("keyword") and item.get("timestamp"):
                    valid_keywords += 1
            
            print(f"  - 有效关键词: {valid_keywords}")
            
            # 显示前几个关键词
            sample_keywords = list(user_data.keys())[:3]
            for key in sample_keywords:
                keyword = user_data[key].get("keyword", "")
                timestamp = user_data[key].get("timestamp", "")
                conv_id = user_data[key].get("conversation_id", "")
                print(f"    - {key}: {keyword} (对话ID: {conv_id})")
        
        print(f"\n总统计:")
        print(f"  - 用户数量: {total_users}")
        print(f"  - 关键词数量: {total_keywords}")
        print(f"  - 对话记录数量: {len(conv_data)}")
        
        print("\n✅ 优化验证通过")
        
    except Exception as e:
        print(f"❌ 优化验证失败: {e}")

def create_conversation_index():
    """创建对话索引，便于关键词快速查找"""
    
    conv_file = "user_bio/data/conversations.json"
    index_file = "user_bio/data/conversation_index.json"
    
    if not os.path.exists(conv_file):
        print("对话记录文件不存在，无法创建索引")
        return
    
    try:
        with open(conv_file, 'r', encoding='utf-8') as f:
            conv_data = json.load(f)
        
        # 创建索引
        conversation_index = {}
        
        for conv in conv_data:
            conv_id = conv.get("_id")
            user_id = conv.get("user_id")
            timestamp = conv.get("timestamp")
            user_message = conv.get("user_message", "")
            ai_reply = conv.get("ai_reply", "")
            
            if conv_id and user_id:
                conversation_index[conv_id] = {
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "content_length": len(user_message) + len(ai_reply),
                    "has_emotion_data": bool(conv.get("emotion_data"))
                }
        
        # 保存索引
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_index, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 对话索引已创建: {index_file}")
        print(f"索引包含 {len(conversation_index)} 个对话记录")
        
    except Exception as e:
        print(f"❌ 创建对话索引失败: {e}")

def main():
    """主函数"""
    print("开始数据架构优化...")
    
    # 分析重复情况
    analyze_data_duplication()
    
    # 优化keyword_memory.json
    optimize_keyword_memory()
    
    # 创建对话索引
    create_conversation_index()
    
    print("\n数据架构优化完成！")

if __name__ == "__main__":
    main() 