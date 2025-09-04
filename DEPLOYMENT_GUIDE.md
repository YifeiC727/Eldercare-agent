# 养老护理系统部署指南

## 🚀 基础架构优化完成

### ✅ 已解决的问题

1. **MongoDB连接问题** - 实现了智能回退机制
2. **数据存储统一** - 创建了增强的数据管理器
3. **错误处理优化** - 完善的异常处理和日志记录
4. **部署简化** - 提供了多种启动方式

## 📦 数据存储架构

### 智能存储系统
- **主存储**: MongoDB (推荐用于生产环境)
- **备用存储**: JSON文件 (开发环境或MongoDB不可用时)
- **自动切换**: 系统会自动检测并选择可用的存储方式

### 存储结构
```
eldercare/
├── users/           # 用户信息
├── conversations/   # 对话记录
├── questions/       # 问卷数据
├── emotion_trends/  # 情感趋势
└── keyword_memory/  # 关键词记忆
```

## 🛠️ 部署方式

### 方式1: 使用Docker (推荐)

1. **启动MongoDB服务**
```bash
# 启动所有服务
docker-compose up -d

# 或者只启动MongoDB
docker-compose up -d mongodb redis
```

2. **访问管理界面**
- MongoDB管理界面: http://localhost:8081
- 用户名: admin
- 密码: admin123

### 方式2: 使用启动脚本

```bash
# 自动检测和启动
python start_app.py
```

### 方式3: 手动启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
python app.py
```

## 🔧 配置说明

### 环境变量配置
复制 `config.env.example` 为 `.env` 并修改配置：

```bash
cp config.env.example .env
```

主要配置项：
- `MONGODB_URI`: MongoDB连接字符串
- `PERFORMANCE_MODE`: 性能模式 (fast/balanced/accurate)
- `FLASK_ENV`: 运行环境 (development/production)

### 性能模式配置

| 模式 | 响应时间 | 功能 | 适用场景 |
|------|----------|------|----------|
| fast | < 1秒 | 基础功能 | 实时对话 |
| balanced | 2-5秒 | 完整功能 | 一般使用 |
| accurate | 5-10秒 | 最高精度 | 深度分析 |

## 📊 数据管理

### 数据备份
```python
from user_bio.improved_user_info_manager import ImprovedUserInfoManager

manager = ImprovedUserInfoManager()
backup_path = manager.backup_data()
print(f"备份完成: {backup_path}")
```

### 数据恢复
```python
manager.restore_data("backup_20241203_143022.json")
```

### 系统监控
```python
stats = manager.get_system_stats()
print(f"存储类型: {stats['storage_type']}")
print(f"用户数量: {stats['user_stats']['total_users']}")
```

## 🔍 故障排除

### MongoDB连接问题

1. **检查MongoDB状态**
```bash
# 检查Docker容器
docker ps | grep mongodb

# 检查MongoDB进程
ps aux | grep mongod
```

2. **重启MongoDB**
```bash
# Docker方式
docker-compose restart mongodb

# 系统服务方式
sudo brew services restart mongodb-community
```

3. **查看日志**
```bash
# Docker日志
docker logs eldercare-mongodb

# 系统日志
tail -f /usr/local/var/log/mongodb/mongo.log
```

### 文件存储问题

如果MongoDB不可用，系统会自动切换到文件存储：
- 数据文件位置: `user_bio/data/`
- 自动创建目录和文件
- 支持完整的CRUD操作

## 🚀 生产环境部署

### 1. 服务器要求
- **CPU**: 4核心以上
- **内存**: 8GB以上
- **存储**: 100GB以上
- **网络**: 稳定的网络连接

### 2. 推荐配置
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  mongodb:
    image: mongo:7.0
    restart: always
    environment:
      MONGO_INITDB_ROOT_USERNAME: ${MONGO_USERNAME}
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD}
    volumes:
      - mongodb_data:/data/db
    networks:
      - eldercare-network

  app:
    build: .
    restart: always
    ports:
      - "80:5000"
    environment:
      - FLASK_ENV=production
      - MONGODB_URI=mongodb://mongodb:27017/
    depends_on:
      - mongodb
    networks:
      - eldercare-network
```

### 3. 安全配置
- 使用HTTPS
- 配置防火墙
- 定期备份数据
- 监控系统状态

## 📈 性能优化

### 1. 数据库优化
- 创建适当的索引
- 定期清理过期数据
- 使用连接池

### 2. 应用优化
- 启用缓存
- 使用CDN
- 负载均衡

### 3. 监控指标
- 响应时间
- 并发用户数
- 错误率
- 资源使用率

## 🆘 技术支持

如果遇到问题，请检查：
1. 系统日志
2. 数据库连接状态
3. 依赖包版本
4. 配置文件设置

## 📝 更新日志

### v1.1.0 (当前版本)
- ✅ 实现智能数据存储系统
- ✅ 支持MongoDB和文件存储自动切换
- ✅ 优化错误处理和日志记录
- ✅ 提供多种部署方式
- ✅ 完善的数据管理功能

### 下一步计划
- 🔄 实现模型服务化
- 🔄 添加负载均衡
- 🔄 优化性能监控
- 🔄 增强安全防护


