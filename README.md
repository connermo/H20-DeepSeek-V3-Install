# H20 8卡机器 vLLM + Deepseek V3 部署工具包

本工具包提供在H20 8卡机器上快速部署vLLM推理引擎和Deepseek V3模型的完整解决方案。

## 快速开始

### 1. 环境配置（一键完成）

```bash
./scripts/setup_environment.sh
```

该脚本会自动完成：
- 检查GPU和CUDA环境
- 创建Python虚拟环境
- 安装PyTorch和vLLM
- 配置环境变量

### 2. 下载模型

```bash
./scripts/download_model.sh
```

该脚本会：
- 交互式选择模型存储路径
- 可选择使用国内镜像站点
- 自动下载Deepseek V3模型（~300GB）
- 验证模型完整性

### 3. 启动推理服务

```bash
./scripts/start_inference.sh
```

该脚本会：
- 交互式配置服务参数（端口、并行度等）
- 启动OpenAI兼容API服务器
- 自动记录日志到 `vllm_server.log`

### 4. 测试推理

```bash
# 方式1: 直接推理模式
python scripts/test_inference.py --mode direct

# 方式2: API模式（需先启动服务）
python scripts/test_inference.py --mode api

# 自定义prompt
python scripts/test_inference.py --mode direct --custom-prompt "你的问题"
```

## 文件说明

### 文档
- **H20_VLLM_DeepSeek_V3_安装指南.md** - 详细的安装和使用文档

### 脚本工具 (scripts/ 目录)
- **setup_environment.sh** - 环境一键配置脚本
- **download_model.sh** - 模型下载脚本
- **start_inference.sh** - 推理服务启动脚本
- **monitor_gpu.sh** - GPU实时监控工具
- **test_inference.py** - 推理功能测试
- **benchmark.py** - 性能基准测试

## 常用命令

### 监控GPU

```bash
# 实时监控
./scripts/monitor_gpu.sh

# 或使用nvidia-smi
watch -n 1 nvidia-smi
```

### 性能测试

```bash
# 运行基准测试
python scripts/benchmark.py --model-path /data/models/deepseek-v3

# 自定义测试参数
python scripts/benchmark.py \
    --num-prompts 100 \
    --prompt-length 128 \
    --output-length 256 \
    --batch-sizes "1,4,8,16,32"
```

### API调用示例

```bash
# 使用curl测试
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/data/models/deepseek-v3",
        "prompt": "介绍一下人工智能",
        "max_tokens": 500
    }'
```

## 配置说明

### 关键参数

- **tensor-parallel-size**: 8 （使用8张GPU）
- **dtype**: bfloat16 （平衡性能和精度）
- **max-model-len**: 8192 （最大序列长度）
- **gpu-memory-utilization**: 0.95 （使用95%显存）

### 环境变量

所有环境变量配置在 `vllm_env.sh` 中：

```bash
source vllm_env.sh  # 加载环境变量
```

## 目录结构建议

```
/data/models/deepseek-v3/     # 模型存储
/home/user/work/H20_install/  # 工作目录
├── vllm-env/                 # Python虚拟环境
├── vllm_env.sh              # 环境变量配置
├── vllm_server.log          # 服务日志
└── benchmark_results_*.json # 性能测试结果
```

## 故障排查

### 问题1: GPU显存不足

```bash
# 降低显存使用率
--gpu-memory-utilization 0.85

# 减少序列长度
--max-model-len 4096
```

### 问题2: 端口被占用

```bash
# 查看占用进程
lsof -i :8000

# 使用其他端口
./start_inference.sh  # 然后输入新端口号
```

### 问题3: 模型下载失败

```bash
# 使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com
./download_model.sh
```

### 问题4: 多GPU通信错误

```bash
# 检查GPU拓扑
nvidia-smi topo -m

# 查看NCCL日志
export NCCL_DEBUG=INFO
```

## 性能优化建议

1. **启用FlashAttention**: 已在 `vllm_env.sh` 中配置
2. **调整batch size**: 使用 `benchmark.py` 找到最佳批次大小
3. **优化NCCL**: 配置 `NCCL_*` 环境变量
4. **使用SSD存储**: 加快模型加载速度

## 生产环境部署

对于生产环境，建议使用systemd服务管理：

```bash
# 参考文档中的systemd配置
sudo nano /etc/systemd/system/vllm-deepseek.service

# 启动服务
sudo systemctl start vllm-deepseek
sudo systemctl enable vllm-deepseek

# 查看状态
sudo systemctl status vllm-deepseek

# 查看日志
journalctl -u vllm-deepseek -f
```

## 技术支持

遇到问题请参考：
1. `H20_VLLM_DeepSeek_V3_安装指南.md` 中的详细文档
2. [vLLM官方文档](https://docs.vllm.ai/)
3. [Deepseek V3模型页面](https://huggingface.co/deepseek-ai/DeepSeek-V3)

## 系统要求

- **GPU**: NVIDIA H20 x 8 (每卡141GB显存)
- **内存**: 512GB+
- **存储**: 500GB+ SSD
- **操作系统**: Ubuntu 22.04 LTS
- **Python**: 3.10-3.13 (vLLM 0.13.0要求)
- **CUDA**: 13.0
- **NVIDIA驱动**: 580.65.06+
- **PyTorch**: 2.9.0+cu130 (CUDA 13.0版本)
- **vLLM**: 0.13.0+

## 许可证

本工具包遵循MIT许可证。使用的第三方软件请遵循其各自的许可证：
- vLLM: Apache 2.0
- Deepseek V3: 请查看模型许可证

---

祝部署顺利！🚀
