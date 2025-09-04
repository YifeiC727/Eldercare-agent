#!/bin/bash

# 快速部署脚本 - 适合测试环境

echo "🚀 快速部署养老护理系统..."

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 请先安装Docker"
    exit 1
fi

# 创建必要的目录
mkdir -p ssl logs user_bio/data

# 生成自签名SSL证书
if [ ! -f ssl/cert.pem ]; then
    echo "🔐 生成SSL证书..."
    openssl req -x509 -newkey rsa:2048 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=CN/ST=State/L=City/O=Organization/CN=localhost"
fi

# 创建环境变量文件
if [ ! -f .env ]; then
    echo "📝 创建环境配置..."
    cat > .env << EOF
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=eldercare123
MONGO_EXPRESS_PASSWORD=admin123
FLASK_ENV=production
SECRET_KEY=$(openssl rand -hex 32)
EOF
fi

# 启动服务
echo "🚀 启动服务..."
docker-compose -f docker-compose.prod.yml up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 20

# 检查服务状态
echo "🔍 服务状态:"
docker-compose -f docker-compose.prod.yml ps

echo ""
echo "🎉 部署完成！"
echo "🌐 访问地址: https://localhost"
echo "📊 管理界面: http://localhost:8081"
echo "🏥 健康检查: https://localhost/health"
echo ""
echo "📝 管理命令:"
echo "  查看日志: docker-compose -f docker-compose.prod.yml logs -f"
echo "  停止服务: docker-compose -f docker-compose.prod.yml down"


