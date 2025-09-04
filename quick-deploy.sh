#!/bin/bash

# 养老护理系统快速部署脚本

echo "🚀 开始部署养老护理系统..."

# 检查是否在正确的目录
if [ ! -f "app.py" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    exit 1
fi

# 检查必要的文件
echo "📋 检查部署文件..."
required_files=("app.py" "requirements.txt" "Procfile")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 缺少必要文件: $file"
        exit 1
    fi
done

echo "✅ 所有必要文件检查完成"

# 检查环境变量
echo "🔧 检查环境变量配置..."
if [ -z "$CAMELLIA_KEY" ]; then
    echo "⚠️  警告：未设置CAMELLIA_KEY环境变量"
    echo "   请在Railway控制台中设置你的DeepSeek API密钥"
fi

# 显示部署信息
echo ""
echo "📊 部署配置信息："
echo "   启动文件: app.py"
echo "   依赖文件: requirements.txt"
echo "   启动命令: web: python app.py"
echo ""

# 检查Git状态
if [ -d ".git" ]; then
    echo "📝 Git状态检查..."
    git status --porcelain
    if [ $? -eq 0 ]; then
        echo "✅ Git仓库状态正常"
    else
        echo "⚠️  Git仓库状态异常"
    fi
else
    echo "⚠️  未检测到Git仓库，请确保已初始化Git"
fi

echo ""
echo "🎯 部署准备完成！"
echo ""
echo "📋 下一步操作："
echo "1. 将代码推送到GitHub仓库"
echo "2. 在Railway中创建新项目"
echo "3. 连接GitHub仓库"
echo "4. 设置环境变量 CAMELLIA_KEY"
echo "5. 等待自动部署完成"
echo ""
echo "📖 详细部署指南请查看: DEPLOYMENT_GUIDE.md"
echo ""
echo "🎉 祝你部署成功！"