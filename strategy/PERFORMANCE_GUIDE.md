# 性能优化指南

## 🚀 性能模式说明

### **fast模式** (推荐用于实时对话)
- ✅ **最快响应**: 关闭实体识别和语义匹配
- ✅ **保留RAG**: 仍能提供知识库支持
- ✅ **基础keyword**: 使用预定义的keyword模板
- ⏱️ **预期响应时间**: < 1秒

### **balanced模式** (推荐用于一般使用)
- ✅ **实体识别**: 启用jieba分词和HanLP NER
- ❌ **语义匹配**: 关闭以提升性能
- ✅ **RAG检索**: 完整的知识库支持
- ⏱️ **预期响应时间**: 2-5秒

### **accurate模式** (推荐用于深度分析)
- ✅ **完整功能**: 所有功能都启用
- ✅ **最高精度**: 实体识别 + 语义匹配 + RAG
- ⏱️ **预期响应时间**: 5-10秒

## 📝 使用方法

### **在app.py中使用**
```python
# 快速模式 (推荐)
selector = StrategySelector(performance_mode="fast")

# 平衡模式
selector = StrategySelector(performance_mode="balanced")

# 精确模式
selector = StrategySelector(performance_mode="accurate")
```

### **动态切换模式**
```python
# 根据用户需求动态切换
if user_needs_quick_response:
    selector = StrategySelector(performance_mode="fast")
elif user_needs_detailed_analysis:
    selector = StrategySelector(performance_mode="accurate")
else:
    selector = StrategySelector(performance_mode="balanced")
```

## 🎯 推荐配置

### **日常对话**: `fast` 模式
- 实时聊天
- 快速回应
- 基础关怀

### **深度交流**: `balanced` 模式  
- 情感分析
- 话题引导
- 记忆增强

### **专业咨询**: `accurate` 模式
- 健康咨询
- 详细分析
- 精准推荐

## ⚡ 性能优化技巧

1. **模型预热**: 首次使用会较慢，后续调用会更快
2. **缓存利用**: 实体识别结果会被缓存
3. **批量处理**: 可以批量处理多个输入
4. **异步处理**: 考虑使用异步处理非关键功能

## 🔧 自定义配置

可以修改 `performance_config.py` 中的配置：

```python
PERFORMANCE_MODES["custom"] = {
    "enable_entity_recognition": True,
    "enable_semantic_matching": False,
    "enable_rag_retrieval": True,
    "max_keywords": 4,
    "semantic_threshold": 0.75
}
``` 