#!/bin/bash

# 养老护理系统服务启动脚本

echo "🚀 启动养老护理系统服务..."

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行，请先启动Docker Desktop"
    exit 1
fi

# 启动MongoDB和Redis服务
echo "📦 启动MongoDB和Redis服务..."
docker-compose up -d mongodb redis

# 等待MongoDB启动
echo "⏳ 等待MongoDB启动..."
sleep 10

# 检查MongoDB是否启动成功
echo "🔍 检查MongoDB连接..."
python -c "
from pymongo import MongoClient
import time

max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        print('✅ MongoDB连接成功!')
        break
    except Exception as e:
        retry_count += 1
        print(f'⏳ 等待MongoDB启动... ({retry_count}/{max_retries})')
        time.sleep(2)

if retry_count >= max_retries:
    print('❌ MongoDB启动失败')
    exit(1)
"

# 启动MongoDB管理界面（可选）
echo "🌐 启动MongoDB管理界面..."
docker-compose up -d mongo-express

echo ""
echo "🎉 服务启动完成!"
echo ""
echo "📊 服务信息:"
echo "  - MongoDB: mongodb://localhost:27017"
echo "  - Redis: redis://localhost:6379"
echo "  - MongoDB管理界面: http://localhost:8081"
echo "    - 用户名: admin"
echo "    - 密码: admin123"
echo ""
echo "🚀 现在可以启动应用:"
echo "  python app.py"
echo ""


