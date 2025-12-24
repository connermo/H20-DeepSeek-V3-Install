# H20 8卡机器上安装vLLM并推理Deepseek V3-0324详细指南

## 目录
1. [系统环境要求](#系统环境要求)
2. [环境准备](#环境准备)
3. [安装vLLM](#安装vllm)
4. [下载Deepseek V3模型](#下载deepseek-v3模型)
5. [配置推理环境](#配置推理环境)
6. [启动推理服务](#启动推理服务)
7. [使用示例](#使用示例)
8. [性能优化建议](#性能优化建议)
9. [常见问题排查](#常见问题排查)

---

## 系统环境要求

### 硬件配置
- **GPU**: NVIDIA H20 x 8张
- **显存**: 每卡141GB，总计约1128GB
- **内存**: 建议512GB以上
- **存储**: 建议SSD，至少500GB可用空间（用于模型存储）

### 软件要求
- **操作系统**: Ubuntu 22.04 LTS
- **CUDA**: 13.0 (推荐稳定版)
- **Python**: 3.10-3.13 (vLLM 0.13.0要求)
- **NVIDIA驱动**: 580.65.06+
- **vLLM**: 0.13.0+

---

## 环境准备

### 1. 检查GPU状态

```bash
# 检查GPU是否被正确识别
nvidia-smi

# 查看CUDA版本
nvcc --version

# 检查驱动版本
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

预期输出应显示8张H20 GPU，每张显存约141GB。

### 2. 创建Python虚拟环境

```bash
# 确认Python版本 (Ubuntu 22.04默认是3.10)
python3 --version  # 应该显示3.10.x

# 如需安装其他版本 (可选: 3.10, 3.11, 3.12, 3.13)
# sudo apt update
# sudo apt install python3.11 python3.11-venv python3.11-dev -y

# 创建虚拟环境 (使用系统默认的Python 3.10)
python3 -m venv vllm-env

# 激活虚拟环境
source vllm-env/bin/activate

# 升级pip
pip install --upgrade pip
```

### 3. 安装基础依赖

```bash
# 安装必要的系统包
sudo apt install -y gcc g++ make cmake git

# 安装CUDA相关库（如果系统未预装）
sudo apt install -y cuda-toolkit-12-1
```

---

## 安装vLLM

### 方法一：使用pip安装（推荐）

```bash
# 激活虚拟环境
source vllm-env/bin/activate

# 安装最新版PyTorch（兼容CUDA 13.0/12.x）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装最新版vLLM (0.13.0+)
pip install --upgrade vllm

# 验证安装
python -c "import vllm; print(f'vLLM版本: {vllm.__version__}')"
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
```

**注意**: vLLM 0.13.0 (2025年12月发布) 包含重要更新:
- 1.7x性能提升
- 优化的执行循环
- 零开销prefix caching
- 增强的多模态支持
- 支持DeepSeek-V3等最新模型

### 方法二：从源码安装（适用于需要定制化的场景）

```bash
# 克隆vLLM仓库
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 安装依赖
pip install -e .

# 编译CUDA kernels
python setup.py build_ext --inplace
```

### 验证安装

```bash
# 测试vLLM导入
python -c "import vllm; print('vLLM安装成功！')"

# 检查可用GPU
python -c "import torch; print(f'可用GPU数量: {torch.cuda.device_count()}')"
```

---

## 下载Deepseek V3模型

### 1. 安装模型下载工具

```bash
pip install huggingface-hub
```

### 2. 下载Deepseek V3-0324模型

```bash
# 创建模型存储目录
mkdir -p /data/models/deepseek-v3

# 使用huggingface-cli下载
huggingface-cli download \
    deepseek-ai/DeepSeek-V3 \
    --local-dir /data/models/deepseek-v3 \
    --local-dir-use-symlinks False

# 如果网络受限，可以使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download \
    deepseek-ai/DeepSeek-V3 \
    --local-dir /data/models/deepseek-v3 \
    --local-dir-use-symlinks False
```

### 3. 验证模型下载

```bash
# 检查模型文件
ls -lh /data/models/deepseek-v3/

# 预期看到以下文件：
# - config.json
# - tokenizer.json
# - tokenizer_config.json
# - pytorch_model-*.bin 或 model-*.safetensors
# - special_tokens_map.json
```

---

## 配置推理环境

### 1. 创建配置文件

创建 `vllm_config.json`:

```json
{
    "model": "/data/models/deepseek-v3",
    "tensor_parallel_size": 8,
    "dtype": "auto",
    "kv_cache_dtype": "fp8",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.95,
    "trust_remote_code": true,
    "enforce_eager": false,
    "disable_custom_all_reduce": false
}
```

**重要说明**：
- **Deepseek V3原生使用FP8训练**，模型权重为FP8格式
- `dtype`: "auto" 让vLLM自动检测使用FP8
- `kv_cache_dtype`: "fp8" 使用FP8 KV缓存（推荐，节省显存）
- 如需使用BF16 KV缓存：`kv_cache_dtype: "bfloat16"`（占用更多显存）

### 2. 环境变量配置

```bash
# 创建环境变量配置文件
cat > vllm_env.sh << 'EOF'
# CUDA配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=0

# vLLM配置
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASHINFER  # FlashInfer推荐用于FP8 KV cache

# PyTorch配置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# NCCL配置（用于多GPU通信）
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
EOF

# 加载环境变量
source vllm_env.sh
```

---

## 启动推理服务

### 方法一：OpenAI兼容API服务

```bash
# 激活环境
source vllm-env/bin/activate
source vllm_env.sh

# 启动vLLM服务器（使用FP8，Deepseek V3原生格式）
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/deepseek-v3 \
    --tensor-parallel-size 8 \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000

# 如需使用BF16 KV缓存（更高精度，更多显存）:
# python -m vllm.entrypoints.openai.api_server \
#     --model /data/models/deepseek-v3 \
#     --tensor-parallel-size 8 \
#     --dtype auto \
#     --kv-cache-dtype bfloat16 \
#     ...其他参数相同
```

**FP8 vs BF16 对比**：
- **FP8**（推荐）：
  - 原生格式，性能最优
  - 节省显存（约50%）
  - 3x吞吐量提升
  - 10x显存容量改进
- **BF16**：
  - 稍高精度
  - 占用更多显存
  - 需要更多GPU资源

### 方法二：使用Python脚本直接推理

创建 `inference.py`:

```python
from vllm import LLM, SamplingParams

# 初始化模型（使用FP8，Deepseek V3原生格式）
llm = LLM(
    model="/data/models/deepseek-v3",
    tensor_parallel_size=8,
    dtype="auto",  # 自动检测FP8
    kv_cache_dtype="fp8",  # FP8 KV缓存
    max_model_len=8192,
    gpu_memory_utilization=0.95,
    trust_remote_code=True
)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048
)

# 推理示例
prompts = [
    "请解释什么是大语言模型？",
    "如何优化深度学习模型的推理性能？"
]

outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 80)
```

运行脚本：

```bash
python inference.py
```

### 方法三：使用systemd服务管理（生产环境推荐）

创建服务文件 `/etc/systemd/system/vllm-deepseek.service`:

```ini
[Unit]
Description=vLLM Deepseek V3 Inference Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/work/H20_install
Environment="PATH=/home/your_username/work/H20_install/vllm-env/bin:/usr/local/cuda/bin:$PATH"
Environment="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
ExecStart=/home/your_username/work/H20_install/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/models/deepseek-v3 \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm-deepseek
sudo systemctl start vllm-deepseek
sudo systemctl status vllm-deepseek
```

---

## 使用示例

### 1. 使用curl测试API

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/data/models/deepseek-v3",
        "prompt": "请介绍一下人工智能的发展历史",
        "max_tokens": 500,
        "temperature": 0.7
    }'
```

### 2. 使用OpenAI Python客户端

```python
from openai import OpenAI

# 配置客户端
client = OpenAI(
    api_key="EMPTY",  # vLLM不需要API key
    base_url="http://localhost:8000/v1"
)

# 对话补全
response = client.chat.completions.create(
    model="/data/models/deepseek-v3",
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "什么是张量并行？"}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

### 3. 批量推理示例

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/data/models/deepseek-v3",
    tensor_parallel_size=8,
    dtype="bfloat16"
)

# 批量prompts
prompts = [f"问题{i}: 请给出一个创意故事开头。" for i in range(100)]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512
)

# 批量推理
outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"结果 {i}: {output.outputs[0].text[:100]}...")
```

---

## 性能优化建议

### 1. 张量并行配置

对于8卡H20配置：
- **tensor_parallel_size=8**: 将模型切分到8张卡上
- 适合Deepseek V3这样的超大模型（约685B参数）

### 2. 显存优化

```bash
# 调整GPU显存利用率
--gpu-memory-utilization 0.95  # 使用95%的显存

# 启用KV cache优化
--max-model-len 8192  # 根据实际需求调整上下文长度

# 使用PagedAttention（vLLM默认启用）
# 无需额外配置，自动优化显存使用
```

### 3. 推理性能优化

```python
# 使用continuous batching
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048,
    # 启用speculative decoding（如果支持）
    use_beam_search=False
)

# 调整worker数量
llm = LLM(
    model="/data/models/deepseek-v3",
    tensor_parallel_size=8,
    max_num_batched_tokens=8192,  # 调整批处理token数量
    max_num_seqs=256  # 调整并发序列数
)
```

### 4. 网络通信优化（多GPU）

```bash
# 使用NVLink/NVSwitch加速GPU间通信
export NCCL_P2P_LEVEL=NVL

# 启用GPUDirect RDMA（如果硬件支持）
export NCCL_NET_GDR_LEVEL=5

# 优化NCCL拓扑
export NCCL_TOPO_FILE=/path/to/topology.xml
```

---

## 常见问题排查

### 1. CUDA Out of Memory (OOM)

**问题**: GPU显存不足

**解决方案**:
```bash
# 降低显存使用率
--gpu-memory-utilization 0.85

# 减少最大序列长度
--max-model-len 4096

# 减少并发请求数
--max-num-seqs 128

# 使用量化
--quantization awq  # 或 gptq
```

### 2. 模型加载缓慢

**问题**: 模型加载时间过长

**解决方案**:
```bash
# 使用safetensors格式（更快）
# 在下载模型时确保下载safetensors版本

# 启用模型预加载
--preload-model

# 使用更快的存储（NVMe SSD）
```

### 3. 多GPU通信错误

**问题**: NCCL错误或GPU间通信失败

**解决方案**:
```bash
# 检查GPU拓扑
nvidia-smi topo -m

# 设置NCCL调试模式
export NCCL_DEBUG=INFO

# 禁用IB（如果不使用InfiniBand）
export NCCL_IB_DISABLE=1

# 使用socket通信作为备选
export NCCL_SOCKET_IFNAME=eth0
```

### 4. 推理速度慢

**问题**: 推理吞吐量低于预期

**检查清单**:
```bash
# 1. 验证tensor parallel是否正常工作
# 查看GPU利用率
nvidia-smi dmon -i 0,1,2,3,4,5,6,7

# 2. 检查是否使用了FlashAttention
export VLLM_ATTENTION_BACKEND=FLASHINFER

# 3. 启用CUDA graphs（适用于固定batch size）
--enforce-eager False

# 4. 调整batch size
--max-num-batched-tokens 16384
```

### 5. 模型输出质量问题

**问题**: 生成结果不符合预期

**调整参数**:
```python
sampling_params = SamplingParams(
    temperature=0.7,      # 降低temperature使输出更确定
    top_p=0.9,            # 调整top_p控制多样性
    top_k=50,             # 限制候选token数量
    repetition_penalty=1.1,  # 避免重复
    max_tokens=2048
)
```

### 6. 端口占用问题

**问题**: 端口8000已被占用

**解决方案**:
```bash
# 查看端口占用
lsof -i :8000

# 使用其他端口
--port 8001

# 或杀死占用进程
kill -9 <PID>
```

---

## 监控和日志

### 1. 实时监控GPU

```bash
# 持续监控GPU使用情况
watch -n 1 nvidia-smi

# 或使用更详细的监控
nvidia-smi dmon -s pucvmet -i 0,1,2,3,4,5,6,7
```

### 2. vLLM日志

```bash
# 查看服务日志
journalctl -u vllm-deepseek -f

# 或者直接运行时查看
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/deepseek-v3 \
    --tensor-parallel-size 8 \
    2>&1 | tee vllm.log
```

### 3. 性能指标

```python
# 在Python中获取推理指标
import time
from vllm import LLM, SamplingParams

llm = LLM(model="/data/models/deepseek-v3", tensor_parallel_size=8)

start = time.time()
outputs = llm.generate(["测试prompt"], SamplingParams(max_tokens=100))
duration = time.time() - start

print(f"推理耗时: {duration:.2f}秒")
print(f"吞吐量: {100/duration:.2f} tokens/秒")
```

---

## 参考资源

- [vLLM官方文档](https://docs.vllm.ai/)
- [Deepseek V3模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [NVIDIA H20 GPU规格](https://www.nvidia.com/en-us/data-center/h20/)
- [vLLM GitHub仓库](https://github.com/vllm-project/vllm)

---

## 总结

本文档详细介绍了在H20 8卡机器上部署vLLM和Deepseek V3模型的完整流程。关键要点：

1. **硬件配置**: 8张H20 GPU提供约1128GB显存，足以支持Deepseek V3的推理
2. **张量并行**: 使用tensor_parallel_size=8充分利用多GPU资源
3. **显存管理**: 合理配置gpu_memory_utilization和max_model_len
4. **性能优化**: 启用FlashAttention、continuous batching等特性
5. **生产部署**: 使用systemd服务管理，确保稳定性

如遇到问题，请参考"常见问题排查"章节或查阅官方文档。
