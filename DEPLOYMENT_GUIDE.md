# 养老护理系统部署指南

## 🚀 Railway 部署步骤

### 1. 准备工作

#### 1.1 获取DeepSeek API密钥
1. 访问 [DeepSeek官网](https://platform.deepseek.com/)
2. 注册账号并获取API密钥
3. 记录你的API密钥（格式类似：`sk-xxxxxxxxxxxxxxxx`）

#### 1.2 准备Railway账号
1. 访问 [Railway官网](https://railway.app/)
2. 使用GitHub账号登录
3. 连接你的GitHub仓库

### 2. 部署步骤

#### 2.1 创建新项目
1. 在Railway控制台点击 "New Project"
2. 选择 "Deploy from GitHub repo"
3. 选择你的 `Eldercare-agent-tone` 仓库

#### 2.2 配置环境变量
在Railway项目设置中添加以下环境变量：

```
CAMELLIA_KEY=你的DeepSeek_API密钥
SECRET_KEY=一个随机字符串（用于Flask会话）
FLASK_ENV=production
```

#### 2.3 部署配置
- **启动命令**: `web: python app.py`
- **端口**: Railway会自动设置PORT环境变量
- **构建命令**: 自动检测requirements.txt

### 3. 验证部署

#### 3.1 检查部署状态
1. 等待构建完成（通常2-5分钟）
2. 查看部署日志确认无错误
3. 访问生成的域名

#### 3.2 功能测试
1. **主页测试**: 访问根路径 `/`
2. **健康检查**: 访问 `/health`
3. **用户注册**: 测试用户注册功能
4. **智能对话**: 测试聊天功能
5. **个人资料**: 测试个人资料显示

### 4. 常见问题解决

#### 4.1 构建失败
- 检查requirements.txt中的依赖版本
- 确保所有必需的依赖都已包含
- 查看构建日志中的具体错误信息

#### 4.2 API调用失败
- 确认CAMELLIA_KEY环境变量已正确设置
- 检查DeepSeek API密钥是否有效
- 查看应用日志中的API调用错误

#### 4.3 中文字符显示问题
- 确认Flask配置中已设置`JSON_AS_ASCII = False`
- 检查响应头是否正确设置UTF-8编码

### 5. 性能优化

#### 5.1 数据库配置（可选）
如果需要持久化数据，可以添加MongoDB：
```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE=eldercare
```

#### 5.2 缓存配置（可选）
添加Redis缓存：
```
REDIS_URL=redis://username:password@host:port/db
```

### 6. 监控和维护

#### 6.1 日志监控
- 在Railway控制台查看实时日志
- 设置日志级别为INFO或DEBUG进行调试

#### 6.2 性能监控
- 监控API响应时间
- 关注内存和CPU使用情况
- 设置自动重启策略

### 7. 安全建议

1. **API密钥安全**
   - 不要在代码中硬编码API密钥
   - 定期轮换API密钥
   - 使用环境变量存储敏感信息

2. **用户数据保护**
   - 实施适当的访问控制
   - 定期备份用户数据
   - 遵循数据保护法规

## 🎉 部署完成

部署成功后，你将获得：
- ✅ 完整的养老护理智能助手系统
- ✅ 用户注册和登录功能
- ✅ 智能对话和情感分析
- ✅ 个人资料管理
- ✅ 响应式Web界面

## 📞 技术支持

如果遇到问题，请检查：
1. Railway部署日志
2. 应用运行日志
3. 环境变量配置
4. API密钥有效性

---

**祝你部署成功！** 🚀