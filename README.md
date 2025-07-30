# AI 陪伴助手 2.0 - 智能情感分析系统

## 项目简介

AI陪伴助手是一个集成了语音识别、情感分析和智能对话的综合性AI系统，专门为老年人提供情感陪伴和智能交互服务。

## 核心功能

### 三种语音输入方式
1. **音频文件上传** - 支持WAV、PCM、AMR、M4A格式
2. **浏览器录音** - 实时录音并自动识别
3. **实时语音识别** - 连续对话模式

### 智能情感分析
- **LIWC分析** - 基于语言学词典的情感词汇分析
- **DeepSeek LLM分析** - 基于大语言模型的深度情感识别
- **多维度情感评估** - 愤怒、悲伤、喜悦、整体强度

### 智能策略选择
- 根据情感状态自动选择对话策略
- 支持多种陪伴模式
- 个性化回复生成

## 系统架构

```
Eldercare-agent 2.0/
├── app.py                    # Flask主应用
├── main.py                   # 主程序入口
├── templates/
│   └── index.html           # 前端界面
├── speech/                   # 语音识别模块
│   └── baidu_speech_recognizer.py
├── emotion_detection/        # 情感分析模块
│   ├── emotion_recognizer.py
│   └── sc_liwc.dic
├── strategy/                 # 策略选择模块
│   ├── strategy_selector.py
│   ├── llm_generator.py
│   └── keyword_memory_manager.py
└── static/
    └── style.css            # 样式文件
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv310

# 激活虚拟环境
# Windows:
venv310\Scripts\activate
# Linux/Mac:
source venv310/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

在环境变量中设置以下API密钥：

```bash
# 百度语音识别API
export BAIDU_APP_ID="your_app_id"
export BAIDU_API_KEY="your_api_key"
export BAIDU_SECRET_KEY="your_secret_key"

# DeepSeek LLM API
export CAMELLIA_KEY="your_deepseek_api_key"
```

### 3. 启动应用

```bash
python app.py
```

### 4. 访问应用

打开浏览器访问：http://localhost:5000

## 使用指南

### 文字对话
1. 在文本框中输入您想说的话
2. 点击"发送消息"或按回车键
3. 系统会分析您的情感状态并给出相应回复

### 语音输入

#### 方式一：上传音频文件
1. 点击"选择文件"按钮
2. 选择音频文件（支持WAV、PCM、AMR、M4A格式）
3. 点击"上传并识别"
4. 系统自动识别语音内容并回复

#### 方式二：浏览器录音
1. 点击"开始录音"按钮
2. 允许浏览器访问麦克风
3. 说出您想说的话
4. 点击"停止录音"按钮
5. 系统自动识别并回复

#### 方式三：实时语音识别
1. 点击"开始实时识别"按钮
2. 系统会持续录音并识别
3. 自动进行对话交互
4. 点击"停止实时识别"结束

## 🔧 技术特性

### 前端特性
- 响应式设计，支持移动端
- 实时状态反馈
- 优雅的错误处理
- 情感分析结果可视化

### 后端特性
- RESTful API设计
- 会话状态管理
- 文件上传安全处理
- 异常处理机制

### AI特性
- 多模态输入处理
- 实时情感分析
- 智能策略选择
- 个性化回复生成

## 测试

运行测试脚本：

```bash
python test_app.py
```

## 情感分析说明

系统使用两种方法进行情感分析：

### LIWC分析
- 基于语言学词典
- 分析情感词汇密度
- 提供客观的语言学指标

### DeepSeek LLM分析
- 基于大语言模型
- 理解上下文语义
- 提供更准确的情感判断

### 情感维度
- **愤怒 (Anger)** - 0.00-1.00
- **悲伤 (Sadness)** - 0.00-1.00  
- **喜悦 (Joy)** - 0.00-1.00
- **强度 (Intensity)** - 整体情感强烈程度

## 安全说明

- API密钥通过环境变量管理
- 文件上传大小限制（10MB）
- 临时文件自动清理
- 输入验证和过滤

## 开发说明

### 添加新的语音识别器
1. 在`speech/`目录下创建新的识别器类
2. 实现`recognize_file()`方法
3. 在`app.py`中更新引用

### 添加新的情感分析方法
1. 在`emotion_detection/`目录下扩展
2. 实现相应的分析方法
3. 在`emotion_recognizer.py`中集成

### 添加新的对话策略
1. 在`strategy/`目录下创建策略类
2. 在`strategy_selector.py`中注册
3. 更新策略选择逻辑

## 更新日志

### v2.0
- 新增实时语音识别功能
- 优化前端界面设计
- 增强情感分析准确性
- 改进错误处理机制

### v1.0
- 基础语音识别功能
- 简单情感分析
- 基础对话系统

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 使用前请确保已正确配置所有必要的API密钥，并遵守相关服务的使用条款。 