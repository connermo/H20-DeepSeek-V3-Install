# vLLM Deepseek V3 快速参考

## 一分钟快速部署

```bash
# 1. 环境配置
./scripts/setup_environment.sh

# 2. 下载模型
./scripts/download_model.sh

# 3. 启动服务
./scripts/start_inference.sh

# 4. 测试（新终端）
python scripts/test_inference.py --mode api
```

## 常用命令速查

### 服务管理

```bash
# 启动服务
./scripts/start_inference.sh

# 查看日志
tail -f vllm_server.log

# 监控GPU
./scripts/monitor_gpu.sh

# 停止服务
# Ctrl+C 或 kill $(lsof -t -i:8000)
```

### 环境激活

```bash
# 每次使用前需激活
source vllm-env/bin/activate
source vllm_env.sh
```

### API测试

```bash
# Completions API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/data/models/deepseek-v3", "prompt": "你好", "max_tokens": 100}'

# Chat API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/data/models/deepseek-v3", "messages": [{"role": "user", "content": "你好"}]}'
```

### Python客户端

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="/data/models/deepseek-v3",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## 版本信息 (2025年12月)

- **CUDA**: 13.0 (推荐稳定版)
- **NVIDIA驱动**: 580.65.06 
- **vLLM**: 0.13.0
- **PyTorch**: 2.x (latest)

## 参数速查表

### 启动参数（Deepseek V3专用）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--tensor-parallel-size` | 8 | GPU并行数 |
| `--dtype` | auto | 自动检测（FP8） |
| `--kv-cache-dtype` | fp8 | KV缓存类型（FP8原生格式） |
| `--max-model-len` | 8192 | 最大序列长度 |
| `--gpu-memory-utilization` | 0.95 | GPU显存使用率 |
| `--port` | 8000 | 服务端口 |

**Deepseek V3特别说明**：
- 原生FP8训练，使用FP8权重
- FP8比BF16节省约50%显存
- FP8提供3x吞吐量和10x显存容量改进

### 采样参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `temperature` | 0.7 | 控制随机性 (0-2) |
| `top_p` | 0.9 | 核采样阈值 (0-1) |
| `top_k` | 50 | 候选token数量 |
| `max_tokens` | 2048 | 最大生成长度 |
| `repetition_penalty` | 1.0 | 重复惩罚 (1.0-2.0) |

## 故障速查

### OOM (显存不足)

```bash
# 降低显存使用
--gpu-memory-utilization 0.85
--max-model-len 4096
```

### 端口占用

```bash
# 查看占用
lsof -i :8000

# 更换端口
--port 8001
```

### GPU未识别

```bash
# 检查驱动
nvidia-smi

# 检查CUDA
nvcc --version

# 设置可见GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 推理速度慢

```bash
# 检查GPU利用率
nvidia-smi dmon

# 启用优化
export VLLM_ATTENTION_BACKEND=FLASHINFER

# 调整并发
--max-num-seqs 256
```

## 性能基准

### 预期性能指标 (H20 8卡)

- **吞吐量**: 500-1000 tokens/秒 (取决于batch size)
- **首token延迟**: 0.5-2秒
- **最大并发**: 100+ 请求

### 运行基准测试

```bash
python scripts/benchmark.py --model-path /data/models/deepseek-v3
```

## 环境变量速查

```bash
# CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# vLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASHINFER

# NCCL (多GPU)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
```

## 文件位置

```
工作目录/
├── README.md                          # 完整文档
├── H20_VLLM_DeepSeek_V3_安装指南.md  # 详细指南
├── QUICKREF.md                        # 本文件
├── scripts/                          # 脚本目录
│   ├── setup_environment.sh          # 环境配置
│   ├── download_model.sh             # 模型下载
│   ├── start_inference.sh            # 启动服务
│   ├── monitor_gpu.sh                # GPU监控
│   ├── test_inference.py             # 推理测试
│   └── benchmark.py                  # 性能测试
├── vllm-env/                         # 虚拟环境
├── vllm_env.sh                       # 环境变量
└── vllm_server.log                   # 服务日志

模型目录/
└── /data/models/deepseek-v3/         # 模型文件
```

## 监控命令

```bash
# GPU状态
nvidia-smi

# 持续监控
watch -n 1 nvidia-smi

# 详细监控
./scripts/monitor_gpu.sh

# 进程监控
htop

# 网络监控
netstat -tulpn | grep 8000
```

## Python脚本模板

### 直接推理

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/data/models/deepseek-v3", tensor_parallel_size=8)
outputs = llm.generate(["你好"], SamplingParams(max_tokens=100))
print(outputs[0].outputs[0].text)
```

### API调用

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="/data/models/deepseek-v3",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)
```

## 常见问题解答

**Q: 需要多少显存？**
A: Deepseek V3约需要700-800GB，8张H20 (141GB/卡) 足够。

**Q: 支持量化吗？**
A: 支持，使用 `--quantization awq` 或 `--quantization gptq`。

**Q: 如何提高吞吐量？**
A: 增加batch size，调整 `--max-num-seqs` 和 `--max-num-batched-tokens`。

**Q: 支持流式输出吗？**
A: 支持，使用 `stream=True` 参数。

**Q: 如何设置系统提示词？**
A: 在messages中添加 `{"role": "system", "content": "系统提示"}`。

## 链接资源

- [vLLM文档](https://docs.vllm.ai/)
- [Deepseek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [OpenAI API兼容性](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

---

**提示**: 详细信息请参考 `README.md` 和 `H20_VLLM_DeepSeek_V3_安装指南.md`
