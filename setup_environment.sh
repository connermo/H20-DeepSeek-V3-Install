#!/bin/bash
# H20机器vLLM环境一键配置脚本

set -e

echo "=========================================="
echo "H20 vLLM环境配置脚本"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否以root运行
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}警告: 某些操作可能需要sudo权限${NC}"
fi

# 1. 检查GPU
echo -e "\n${GREEN}[1/7] 检查GPU状态...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}检测到 ${GPU_COUNT} 张GPU${NC}"
else
    echo -e "${RED}错误: 未检测到nvidia-smi，请先安装NVIDIA驱动${NC}"
    exit 1
fi

# 2. 检查CUDA
echo -e "\n${GREEN}[2/7] 检查CUDA版本...${NC}"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo -e "${YELLOW}警告: 未检测到nvcc，将尝试继续...${NC}"
fi

# 3. 检查Python
echo -e "\n${GREEN}[3/7] 检查Python版本...${NC}"
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo -e "${RED}错误: 未找到Python 3，请先安装${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version)
echo -e "${GREEN}使用 ${PYTHON_VERSION}${NC}"

# 4. 创建虚拟环境
echo -e "\n${GREEN}[4/7] 创建Python虚拟环境...${NC}"
if [ -d "vllm-env" ]; then
    echo -e "${YELLOW}虚拟环境已存在，跳过创建${NC}"
else
    $PYTHON_CMD -m venv vllm-env
    echo -e "${GREEN}虚拟环境创建成功${NC}"
fi

# 激活虚拟环境
source vllm-env/bin/activate

# 5. 升级pip
echo -e "\n${GREEN}[5/7] 升级pip...${NC}"
pip install --upgrade pip

# 6. 安装PyTorch
echo -e "\n${GREEN}[6/7] 安装最新版PyTorch (兼容CUDA 13.0/12.x)...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证PyTorch
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 7. 安装vLLM
echo -e "\n${GREEN}[7/7] 安装最新版vLLM (0.13.0+)...${NC}"
pip install --upgrade vllm

# 验证vLLM
python -c "import vllm; print(f'vLLM版本: {vllm.__version__}')"

echo -e "${YELLOW}注意: vLLM 0.13.0包含重要更新:${NC}"
echo -e "  - 1.7x性能提升"
echo -e "  - 优化的执行循环和零开销prefix caching"
echo -e "  - 增强的多模态支持"

# 创建环境变量文件
echo -e "\n${GREEN}创建环境变量配置文件...${NC}"
cat > vllm_env.sh << 'EOF'
# CUDA配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=0

# vLLM配置
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASHINFER

# PyTorch配置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# NCCL配置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
EOF

echo -e "\n${GREEN}=========================================="
echo "环境配置完成！"
echo "==========================================${NC}"
echo ""
echo "后续步骤："
echo "1. 激活虚拟环境: source vllm-env/bin/activate"
echo "2. 加载环境变量: source vllm_env.sh"
echo "3. 下载Deepseek V3模型（参考文档）"
echo "4. 启动推理服务"
echo ""
echo "详细文档: H20_VLLM_DeepSeek_V3_安装指南.md"
