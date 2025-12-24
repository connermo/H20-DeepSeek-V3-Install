#!/bin/bash
# vLLM推理服务启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "vLLM Deepseek V3 推理服务启动脚本"
echo "=========================================="

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}激活虚拟环境...${NC}"
    if [ -f "vllm-env/bin/activate" ]; then
        source vllm-env/bin/activate
    else
        echo -e "${RED}错误: 虚拟环境不存在${NC}"
        exit 1
    fi
fi

# 加载环境变量
if [ -f "vllm_env.sh" ]; then
    echo -e "${GREEN}加载环境变量...${NC}"
    source vllm_env.sh
else
    echo -e "${YELLOW}警告: vllm_env.sh不存在，使用默认配置${NC}"
fi

# 配置参数
DEFAULT_MODEL_PATH="/data/models/deepseek-v3"
read -p "模型路径 (默认: ${DEFAULT_MODEL_PATH}): " MODEL_PATH
MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}

# 验证模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型路径不存在: ${MODEL_PATH}${NC}"
    exit 1
fi

# 其他参数
read -p "端口号 (默认: 8000): " PORT
PORT=${PORT:-8000}

read -p "Tensor并行大小 (默认: 8): " TP_SIZE
TP_SIZE=${TP_SIZE:-8}

read -p "最大序列长度 (默认: 8192): " MAX_LEN
MAX_LEN=${MAX_LEN:-8192}

read -p "GPU显存利用率 (默认: 0.95): " GPU_MEM
GPU_MEM=${GPU_MEM:-0.95}

read -p "KV Cache数据类型 (fp8/bfloat16, 默认: fp8): " KV_DTYPE
KV_DTYPE=${KV_DTYPE:-fp8}

# 显示配置
echo -e "\n${GREEN}配置信息:${NC}"
echo "  模型路径: ${MODEL_PATH}"
echo "  端口: ${PORT}"
echo "  Tensor并行: ${TP_SIZE}"
echo "  最大序列长度: ${MAX_LEN}"
echo "  GPU显存利用率: ${GPU_MEM}"
echo "  KV Cache类型: ${KV_DTYPE} (Deepseek V3原生FP8)"
echo ""

# 确认启动
read -p "确认启动服务? [y/N]: " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "取消启动"
    exit 0
fi

# 检查端口占用
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}错误: 端口 ${PORT} 已被占用${NC}"
    echo "当前占用进程:"
    lsof -i :${PORT}
    exit 1
fi

# 启动服务
echo -e "\n${GREEN}启动vLLM推理服务...${NC}"
echo -e "${YELLOW}按 Ctrl+C 停止服务${NC}\n"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype auto \
    --kv-cache-dtype "$KV_DTYPE" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$PORT" \
    2>&1 | tee vllm_server.log
