# Multi-dimensional Early Warning System

## 🚨 System Overview

The new early warning system adopts a multi-dimensional detection mechanism that can more accurately identify emotional risks in elderly people and provide graded response strategies.

## 📊 Detection Dimensions

### **1. Acute High-Risk Detection**
- **Trigger Condition**: `current_sadness > 0.9`
- **Response Level**: `severe`
- **Recommended Action**: Immediate attention, recommend human intervention
- **Application Scenario**: User expresses extreme sadness or self-harm tendencies

### **2. Persistent Detection**
- **Continuous High Sadness**: 2-3 consecutive times `sadness > 0.8`
- **Response Level**: `moderate` (2 times) / `severe` (3 times)
- **Recommended Action**: Requires sustained attention, recommend increasing care frequency

### **3. Trend Deterioration Detection**
- **Trigger Condition**: Recent average - Historical average > 0.3 and Recent average > 0.6
- **Response Level**: `moderate`
- **Recommended Action**: Emotional trend deterioration, recommend proactive care
- **Data Requirement**: At least 6 historical data points

### **4. Long-term Trend Detection**
- **Trigger Condition**: Average `sadness > 0.7` for last 10 conversations
- **Response Level**: `mild`
- **Recommended Action**: Long-term low emotions, recommend regular care

### **5. Multi-dimensional Comprehensive Detection**
- **Trigger Conditions**: 
  - `current_sadness > 0.6`
  - `self_focus_freq > 0.15` (High self-focus, above 15%)
  - `social_freq < 0.05` (Low social, below 5%)
  - `sadness_LIWC_freq > 0.08` (Negative emotions, above 8%)
- **Response Level**: `moderate`
- **Recommended Action**: Social withdrawal tendency, recommend encouraging social activities

## 🎯 Response Strategies

### **Severe Level** (Critical)
```python
{
    "语气": "紧急关切",
    "目标": "立即情绪干预，建议转介专业支持",
    "引导语": "我注意到{reason}，这让我非常担心你的状态。你愿意和我详细聊聊吗？如果需要的话，我建议我们可以联系专业的心理支持资源。",
    "心理学依据": "急性情绪危机需要立即关注和专业干预。"
}
```

### **Moderate Level** (Medium)
```python
{
    "Tone": "Concerned guidance",
    "Objective": "Proactive care, prevent emotional deterioration",
    "Guidance": "I noticed {reason}, have you encountered some difficulties recently? Would you like to talk to me? I'll always be here with you.",
    "Psychological Basis": "Early intervention can effectively prevent emotional problems from worsening."
}
```

### **Mild Level** (Light)
```python
{
    "Tone": "Gentle care",
    "Objective": "Increase care frequency, prevent problem development",
    "Guidance": "I noticed {reason}, how have you been feeling lately? Is there anything you'd like to share with me?",
    "Psychological Basis": "Regular care helps maintain emotional stability."
}
```

## 🔧 Technical Implementation

### **Data Structure**
```python
warning_result = {
    "triggered": bool,           # Whether warning is triggered
    "level": str,               # Warning level: normal/mild/moderate/severe
    "reason": str,              # Trigger reason
    "suggested_action": str     # Suggested action
}
```

### **Usage**
```python
# Call in select_strategy
early_warning_result = selector.check_early_warning(
    window_sadness_scores=window_sadness_scores,
    current_sadness=sadness,
    liwc_scores=liwc_mapped
)

# Select strategy based on warning results
if early_warning_result["triggered"]:
    warning_level = early_warning_result["level"]
    # Select response strategy for corresponding level
```

## 📈 Advantages

### **1. 多维度检测**
- 不仅看单一指标，而是综合考虑多个维度
- 减少误报和漏报

### **2. 分级响应**
- 根据风险级别提供不同的响应策略
- 避免过度反应或反应不足

### **3. 趋势分析**
- 不仅看当前状态，还分析历史趋势
- 能够提前发现潜在问题

### **4. 个性化建议**
- 根据具体触发原因提供针对性建议
- 提高干预效果

### **5. 可扩展性**
- 易于添加新的检测维度
- 可以根据实际使用情况调整阈值

## 🎛️ 参数调优

### **可调整的阈值**
```python
# 急性风险阈值
ACUTE_RISK_THRESHOLD = 0.9

# 连续高sadness阈值
HIGH_SADNESS_THRESHOLD = 0.8
HIGH_SADNESS_COUNT = 2

# 趋势恶化阈值
TREND_DETERIORATION_THRESHOLD = 0.3
RECENT_AVG_THRESHOLD = 0.6

# 长期趋势阈值
LONG_TERM_THRESHOLD = 0.7
LONG_TERM_COUNT = 10

# LIWC特征阈值 (百分比形式，0-1)
SELF_FOCUS_THRESHOLD = 0.15  # 15%以上自我关注
SOCIAL_THRESHOLD = 0.05      # 5%以下社交词汇
SADNESS_LIWC_THRESHOLD = 0.08  # 8%以上负面情绪词汇
```

## 🔄 使用流程

1. **数据收集**: 收集情绪分数、LIWC特征、对话历史
2. **多维度检测**: 并行执行各种检测算法
3. **优先级排序**: 按严重程度选择最高级别的预警
4. **策略选择**: 根据预警级别选择相应的响应策略
5. **执行干预**: 生成相应的引导语和建议

## 📝 注意事项

1. **隐私保护**: 预警信息应妥善处理，避免泄露
2. **人工介入**: 严重预警应及时转介人工服务
3. **持续监控**: 预警后应持续关注用户状态
4. **阈值调整**: 根据实际使用效果调整检测阈值
5. **用户反馈**: 收集用户对预警准确性的反馈 