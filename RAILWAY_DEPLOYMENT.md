# Railway部署指南

## 🚀 快速部署到Railway

### 步骤1: 准备代码

```bash
# 1. 添加所有文件到Git
git add .

# 2. 提交更改
git commit -m "准备Railway部署 - 优化配置和添加部署文件"

# 3. 推送到GitHub
git push origin main
```

### 步骤2: 部署到Railway

1. **访问Railway**
   - 打开 [Railway.app](https://railway.app)
   - 点击 "Start a New Project"

2. **连接GitHub**
   - 选择 "Deploy from GitHub repo"
   - 授权Railway访问您的GitHub账号
   - 选择 `Eldercare-agent-tone` 仓库

3. **自动部署**
   - Railway会自动检测到Procfile
   - 开始构建和部署过程
   - 等待部署完成（通常需要2-5分钟）

### 步骤3: 配置环境变量

在Railway控制台中设置以下环境变量：

#### 必需的环境变量：
```
CAMELLIA_KEY=your-deepseek-api-key
FLASK_ENV=production
FLASK_DEBUG=False
```

#### 可选的环境变量：
```
BAIDU_APP_ID=your-baidu-app-id
BAIDU_API_KEY=your-baidu-api-key
BAIDU_SECRET_KEY=your-baidu-secret-key
```

### 步骤4: 获取访问URL

部署完成后，Railway会提供一个URL：
- 格式：`https://your-app-name.railway.app`
- 例如：`https://eldercare-agent-tone-production.railway.app`

### 步骤5: 测试部署

```bash
# 测试健康检查
curl https://your-app-name.railway.app/health

# 测试主页
curl https://your-app-name.railway.app/
```

## 🔧 配置说明

### 文件说明：
- `Procfile`: 告诉Railway如何启动应用
- `railway_start.py`: Railway专用启动脚本
- `railway.json`: Railway配置文件
- `requirements.txt`: Python依赖

### 端口配置：
- Railway自动分配端口，通过`PORT`环境变量传递
- 应用监听`0.0.0.0:PORT`

### 数据库：
- Railway提供PostgreSQL数据库
- 当前使用文件存储，后续可迁移到PostgreSQL

## 📊 监控和管理

### Railway控制台功能：
- 查看部署日志
- 监控资源使用
- 管理环境变量
- 查看访问统计

### 日志查看：
```bash
# 在Railway控制台查看实时日志
# 或使用Railway CLI
railway logs
```

## 🚨 故障排除

### 常见问题：

1. **部署失败**
   - 检查requirements.txt中的依赖
   - 查看Railway构建日志
   - 确认Python版本兼容性

2. **应用无法启动**
   - 检查环境变量设置
   - 查看启动日志
   - 确认端口配置

3. **数据库连接失败**
   - 当前使用文件存储
   - 后续可配置Railway PostgreSQL

### 调试命令：
```bash
# 本地测试Railway配置
python railway_start.py

# 检查环境变量
echo $CAMELLIA_KEY
```

## 🎯 部署后优化

### 1. 自定义域名
- 在Railway控制台添加自定义域名
- 配置DNS解析
- 自动获得SSL证书

### 2. 性能优化
- 启用Railway的自动扩缩容
- 配置缓存策略
- 优化数据库查询

### 3. 监控设置
- 设置健康检查
- 配置告警通知
- 监控响应时间

## 💰 费用说明

### Railway免费额度：
- $5/月免费额度
- 足够小型应用使用
- 超出后按使用量计费

### 升级选项：
- Pro计划：$20/月
- 更多资源和服务
- 优先支持

## 🎉 部署成功！

部署完成后，您的养老护理系统将：
- ✅ 拥有公网URL
- ✅ 自动SSL证书
- ✅ 全球可访问
- ✅ 支持自定义域名
- ✅ 自动扩缩容

**现在就开始部署吧！** 🚀
