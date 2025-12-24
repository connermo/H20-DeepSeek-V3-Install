#!/bin/bash
# Deepseek V3模型下载脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Deepseek V3 模型下载脚本"
echo "=========================================="

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}警告: 未检测到虚拟环境，正在激活...${NC}"
    if [ -f "vllm-env/bin/activate" ]; then
        source vllm-env/bin/activate
    else
        echo -e "${RED}错误: 虚拟环境不存在，请先运行 setup_environment.sh${NC}"
        exit 1
    fi
fi

# 设置模型存储路径
DEFAULT_MODEL_DIR="/data/models/deepseek-v3"
read -p "请输入模型存储路径 (默认: ${DEFAULT_MODEL_DIR}): " MODEL_DIR
MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL_DIR}

# 创建目录
echo -e "\n${GREEN}创建模型目录: ${MODEL_DIR}${NC}"
mkdir -p "$MODEL_DIR"

# 检查huggingface-cli
echo -e "\n${GREEN}检查下载工具...${NC}"
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}安装 huggingface-hub...${NC}"
    pip install huggingface-hub
fi

# 询问是否使用镜像
echo -e "\n${YELLOW}是否使用HuggingFace镜像站点？${NC}"
echo "1) 是 (适合国内网络，推荐)"
echo "2) 否 (使用官方站点)"
read -p "请选择 [1/2]: " USE_MIRROR

if [ "$USE_MIRROR" = "1" ]; then
    export HF_ENDPOINT=https://hf-mirror.com
    echo -e "${GREEN}使用镜像站点: ${HF_ENDPOINT}${NC}"
fi

# 开始下载
echo -e "\n${GREEN}开始下载 Deepseek V3 模型...${NC}"
echo -e "${YELLOW}注意: 模型约300GB+，下载时间较长，请保持网络稳定${NC}"

huggingface-cli download \
    deepseek-ai/DeepSeek-V3 \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False \
    --resume-download

# 验证下载
echo -e "\n${GREEN}验证模型文件...${NC}"
if [ -f "$MODEL_DIR/config.json" ]; then
    echo -e "${GREEN}✓ config.json${NC}"
else
    echo -e "${RED}✗ config.json 缺失${NC}"
fi

if [ -f "$MODEL_DIR/tokenizer.json" ]; then
    echo -e "${GREEN}✓ tokenizer.json${NC}"
else
    echo -e "${RED}✗ tokenizer.json 缺失${NC}"
fi

MODEL_FILES=$(find "$MODEL_DIR" -name "*.safetensors" -o -name "*.bin" | wc -l)
echo -e "${GREEN}✓ 找到 ${MODEL_FILES} 个模型权重文件${NC}"

# 显示磁盘使用
DISK_USAGE=$(du -sh "$MODEL_DIR" | cut -f1)
echo -e "\n${GREEN}模型目录大小: ${DISK_USAGE}${NC}"

echo -e "\n${GREEN}=========================================="
echo "模型下载完成！"
echo "==========================================${NC}"
echo ""
echo "模型路径: ${MODEL_DIR}"
echo ""
echo "后续步骤："
echo "1. 运行 start_inference.sh 启动推理服务"
echo "2. 或使用 test_inference.py 测试推理"
