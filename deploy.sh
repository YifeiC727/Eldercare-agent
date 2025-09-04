#!/bin/bash

# 养老护理系统部署脚本

set -e

echo "🚀 开始部署养老护理系统..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker未安装，请先安装Docker${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose未安装，请先安装Docker Compose${NC}"
    exit 1
fi

# 检查环境变量文件
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️ 未找到.env文件，创建默认配置...${NC}"
    cat > .env << EOF
# 数据库配置
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=eldercare123
MONGO_EXPRESS_PASSWORD=admin123

# 应用配置
FLASK_ENV=production
SECRET_KEY=$(openssl rand -hex 32)

# 第三方服务配置 (可选)
BAIDU_APP_ID=your-baidu-app-id
BAIDU_API_KEY=your-baidu-api-key
BAIDU_SECRET_KEY=your-baidu-secret-key
OPENAI_API_KEY=your-openai-api-key
EOF
    echo -e "${GREEN}✅ 已创建.env配置文件${NC}"
fi

# 创建SSL证书目录
mkdir -p ssl

# 检查SSL证书
if [ ! -f ssl/cert.pem ] || [ ! -f ssl/key.pem ]; then
    echo -e "${YELLOW}⚠️ 未找到SSL证书，生成自签名证书...${NC}"
    openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=CN/ST=State/L=City/O=Organization/CN=localhost"
    echo -e "${GREEN}✅ SSL证书生成完成${NC}"
fi

# 创建日志目录
mkdir -p logs

# 停止现有服务
echo -e "${BLUE}🔄 停止现有服务...${NC}"
docker-compose -f docker-compose.prod.yml down || true

# 构建镜像
echo -e "${BLUE}🔨 构建Docker镜像...${NC}"
docker-compose -f docker-compose.prod.yml build --no-cache

# 启动服务
echo -e "${BLUE}🚀 启动服务...${NC}"
docker-compose -f docker-compose.prod.yml up -d

# 等待服务启动
echo -e "${BLUE}⏳ 等待服务启动...${NC}"
sleep 30

# 检查服务状态
echo -e "${BLUE}🔍 检查服务状态...${NC}"
docker-compose -f docker-compose.prod.yml ps

# 健康检查
echo -e "${BLUE}🏥 执行健康检查...${NC}"
for i in {1..10}; do
    if curl -f http://localhost/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ 服务健康检查通过${NC}"
        break
    else
        echo -e "${YELLOW}⏳ 等待服务启动... ($i/10)${NC}"
        sleep 10
    fi
done

# 显示访问信息
echo -e "${GREEN}🎉 部署完成！${NC}"
echo ""
echo -e "${BLUE}📊 服务访问信息:${NC}"
echo -e "  🌐 主应用: https://localhost"
echo -e "  🏥 健康检查: https://localhost/health"
echo -e "  📊 MongoDB管理: http://localhost:8081"
echo -e "    - 用户名: admin"
echo -e "    - 密码: admin123"
echo ""
echo -e "${BLUE}📝 管理命令:${NC}"
echo -e "  查看日志: docker-compose -f docker-compose.prod.yml logs -f"
echo -e "  停止服务: docker-compose -f docker-compose.prod.yml down"
echo -e "  重启服务: docker-compose -f docker-compose.prod.yml restart"
echo ""
echo -e "${YELLOW}⚠️ 注意事项:${NC}"
echo -e "  1. 请将localhost替换为你的实际域名"
echo -e "  2. 建议使用Let's Encrypt等免费SSL证书"
echo -e "  3. 定期备份MongoDB数据"
echo -e "  4. 监控系统资源使用情况"


