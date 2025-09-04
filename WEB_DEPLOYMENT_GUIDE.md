# 养老护理系统Web部署指南

## 🌐 部署到云服务器

### 推荐的云服务提供商

#### 1. **阿里云** (推荐国内用户)
- **ECS实例**: 2核4GB内存，40GB存储
- **域名**: 可购买.cn域名
- **SSL证书**: 免费SSL证书
- **价格**: 约100-200元/月

#### 2. **腾讯云** (推荐国内用户)
- **CVM实例**: 2核4GB内存，50GB存储
- **域名**: 可购买.cn域名
- **SSL证书**: 免费SSL证书
- **价格**: 约100-200元/月

#### 3. **AWS** (推荐海外用户)
- **EC2实例**: t3.medium (2核4GB)
- **域名**: Route 53
- **SSL证书**: AWS Certificate Manager
- **价格**: 约$50-100/月

#### 4. **DigitalOcean** (推荐海外用户)
- **Droplet**: 2GB RAM, 1 CPU, 50GB SSD
- **域名**: 可连接外部域名
- **SSL证书**: Let's Encrypt
- **价格**: 约$12/月

## 🚀 部署步骤

### 步骤1: 准备云服务器

1. **购买云服务器**
   ```bash
   # 推荐配置
   - CPU: 2核心
   - 内存: 4GB
   - 存储: 40GB SSD
   - 带宽: 5Mbps
   - 操作系统: Ubuntu 20.04 LTS
   ```

2. **配置安全组**
   ```bash
   # 开放端口
   - 22 (SSH)
   - 80 (HTTP)
   - 443 (HTTPS)
   - 8081 (MongoDB管理界面)
   ```

3. **连接服务器**
   ```bash
   ssh root@your-server-ip
   ```

### 步骤2: 安装Docker

```bash
# 更新系统
apt update && apt upgrade -y

# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 安装Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# 启动Docker服务
systemctl start docker
systemctl enable docker
```

### 步骤3: 上传代码

```bash
# 方法1: 使用Git
git clone https://github.com/your-username/eldercare-agent-tone.git
cd eldercare-agent-tone

# 方法2: 使用SCP上传
scp -r /path/to/eldercare-agent-tone root@your-server-ip:/root/
```

### 步骤4: 配置域名和SSL

#### 使用Let's Encrypt (推荐)

```bash
# 安装Certbot
apt install certbot python3-certbot-nginx -y

# 获取SSL证书
certbot --nginx -d your-domain.com

# 自动续期
crontab -e
# 添加以下行
0 12 * * * /usr/bin/certbot renew --quiet
```

#### 使用自签名证书 (测试用)

```bash
# 生成自签名证书
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=CN/ST=State/L=City/O=Organization/CN=your-domain.com"
```

### 步骤5: 部署应用

```bash
# 进入项目目录
cd eldercare-agent-tone

# 运行部署脚本
chmod +x deploy.sh
./deploy.sh
```

### 步骤6: 配置反向代理 (可选)

如果需要使用自定义域名，修改nginx配置：

```bash
# 编辑nginx配置
nano nginx.conf

# 修改server_name
server_name your-domain.com www.your-domain.com;
```

## 🔧 生产环境优化

### 1. 性能优化

```bash
# 修改docker-compose.prod.yml
services:
  web:
    environment:
      - PERFORMANCE_MODE=balanced  # 或 accurate
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### 2. 安全配置

```bash
# 修改.env文件
MONGO_ROOT_PASSWORD=your-strong-password
MONGO_EXPRESS_PASSWORD=your-strong-password
SECRET_KEY=your-very-long-secret-key
```

### 3. 监控配置

```bash
# 安装监控工具
docker run -d --name=prometheus -p 9090:9090 prom/prometheus
docker run -d --name=grafana -p 3000:3000 grafana/grafana
```

## 📊 监控和维护

### 1. 日志查看

```bash
# 查看应用日志
docker-compose -f docker-compose.prod.yml logs -f web

# 查看所有服务日志
docker-compose -f docker-compose.prod.yml logs -f
```

### 2. 数据备份

```bash
# 备份MongoDB数据
docker exec eldercare-mongodb mongodump --out /backup
docker cp eldercare-mongodb:/backup ./mongodb-backup

# 自动备份脚本
cat > backup.sh << EOF
#!/bin/bash
DATE=\$(date +%Y%m%d_%H%M%S)
docker exec eldercare-mongodb mongodump --out /backup/\$DATE
docker cp eldercare-mongodb:/backup/\$DATE ./mongodb-backup/\$DATE
EOF
chmod +x backup.sh
```

### 3. 性能监控

```bash
# 查看资源使用情况
docker stats

# 查看服务状态
docker-compose -f docker-compose.prod.yml ps
```

## 🌍 域名配置

### 1. 购买域名

推荐域名注册商：
- **国内**: 阿里云、腾讯云、万网
- **海外**: GoDaddy、Namecheap、Cloudflare

### 2. DNS配置

```bash
# A记录配置
your-domain.com -> your-server-ip
www.your-domain.com -> your-server-ip

# CNAME记录 (可选)
api.your-domain.com -> your-domain.com
```

### 3. 搜索引擎优化

```html
<!-- 添加到templates/index.html -->
<meta name="description" content="智能养老护理对话系统，为老年人提供情感陪伴和健康关怀">
<meta name="keywords" content="养老护理,智能对话,情感陪伴,老年人,健康关怀">
<meta name="author" content="养老护理系统">
```

## 🔍 SEO优化

### 1. 网站地图

```bash
# 创建sitemap.xml
cat > static/sitemap.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://your-domain.com/</loc>
    <lastmod>$(date -I)</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>
EOF
```

### 2. 搜索引擎提交

- **百度**: https://ziyuan.baidu.com/
- **Google**: https://search.google.com/search-console/
- **必应**: https://www.bing.com/webmasters/

### 3. 社交媒体分享

```html
<!-- 添加到templates/index.html -->
<meta property="og:title" content="智能养老护理对话系统">
<meta property="og:description" content="为老年人提供情感陪伴和健康关怀的智能对话系统">
<meta property="og:image" content="https://your-domain.com/static/images/og-image.jpg">
<meta property="og:url" content="https://your-domain.com/">
```

## 📱 移动端优化

### 1. 响应式设计

确保网站支持移动设备访问，已包含在现有模板中。

### 2. PWA支持

```bash
# 创建manifest.json
cat > static/manifest.json << EOF
{
  "name": "养老护理系统",
  "short_name": "养老护理",
  "description": "智能养老护理对话系统",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#007bff",
  "icons": [
    {
      "src": "/static/images/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    }
  ]
}
EOF
```

## 🚨 故障排除

### 常见问题

1. **服务无法启动**
   ```bash
   # 检查日志
   docker-compose -f docker-compose.prod.yml logs
   
   # 检查端口占用
   netstat -tlnp | grep :80
   ```

2. **SSL证书问题**
   ```bash
   # 重新生成证书
   certbot renew --force-renewal
   ```

3. **数据库连接失败**
   ```bash
   # 检查MongoDB状态
   docker-compose -f docker-compose.prod.yml logs mongodb
   ```

## 📞 技术支持

如果遇到问题，请检查：
1. 服务器资源使用情况
2. 网络连接状态
3. 服务日志信息
4. 配置文件设置

## 🎯 部署检查清单

- [ ] 云服务器配置完成
- [ ] Docker环境安装
- [ ] 代码上传完成
- [ ] 域名解析配置
- [ ] SSL证书安装
- [ ] 应用部署成功
- [ ] 健康检查通过
- [ ] 数据备份配置
- [ ] 监控系统设置
- [ ] SEO优化完成

完成以上步骤后，你的养老护理系统就可以通过互联网访问了！


