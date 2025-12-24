# 版本信息汇总

> 最后更新: 2025年12月24日

## 📦 官方推荐版本

| 组件 | 版本 | 说明 |
|------|------|------|
| **CUDA** | 13.0 | 推荐稳定版 |
| **NVIDIA驱动** | 580.65.06+ <br>580.88+ (Windows) | R580系列，CUDA 13.0必需 |
| **vLLM** | 0.13.0+ | 包含FP8优化 |
| **PyTorch** | 2.x | 最新稳定版 |
| **Python** | 3.9-3.12 | 推荐3.10+ |
| **Ubuntu** | 22.04 LTS | 推荐LTS版本 |
| **NCCL** | 2.x | 最新版 |

## ⚡ Deepseek V3专用配置

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| **模型格式** | FP8 | 原生训练格式 |
| **dtype** | auto | 自动检测FP8 |
| **kv_cache_dtype** | fp8 | 节省50%显存 |
| **attention_backend** | FLASHINFER | FP8优化 |
| **tensor_parallel_size** | 8 | 8卡并行 |
| **gpu_memory_utilization** | 0.95 | 显存使用率 |
| **max_model_len** | 8192 | 上下文长度 |

## 🔄 版本兼容性矩阵

### CUDA与驱动兼容性

| CUDA版本 | 最低驱动版本  | 最低驱动版本 (Windows) | 驱动系列 |
|----------|---------------------|----------------------|----------|
| 13.0 | 580.65.06 | 580.88 | R580 |
| 12.9 | 550.xx | 553.xx | R550 |
| 12.8 | 545.xx | 546.xx | R545 |

### vLLM与PyTorch兼容性

| vLLM版本 | PyTorch版本 | CUDA版本 | 特性 |
|----------|-------------|----------|------|
| 0.13.0 | 2.x | 13.0/12.x | FP8优化，Deepseek V3支持 |
| 0.12.x | 2.x | 12.x | 基础支持 |

## 📊 性能指标

### Deepseek V3 (FP8 vs BF16)

| 指标 | FP8 | BF16 | 改进 |
|------|-----|------|------|
| **显存占用** | ~700GB | ~1400GB | 50%↓ |
| **吞吐量** | 基准×3 | 基准×1 | 300%↑ |
| **延迟** | 低 | 中 | 30%↓ |
| **精度** | 高 | 更高 | 略低 |

### H20 8卡预期性能

- **吞吐量**: 500-1000 tokens/秒 (FP8)
- **首token延迟**: 0.5-2秒
- **最大并发**: 100+ 请求
- **显存使用**: 700-900GB (FP8)
- **NVLink带宽**: 400+ GB/s (8卡总计)

## 🔗 官方文档链接

- [CUDA 13.0下载](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA驱动下载](https://www.nvidia.com/Download/index.aspx)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Deepseek V3 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [vLLM Deepseek V3指南](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3.html)

## ✅ 快速检查命令

```bash
# 检查CUDA版本
nvcc --version | grep release

# 检查NVIDIA驱动
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# 检查vLLM版本
python -c "import vllm; print(vllm.__version__)"

# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"

# 检查Python版本
python --version

# 检查GPU数量
nvidia-smi --query-gpu=name --format=csv,noheader | wc -l

# 检查NVLink状态
nvidia-smi nvlink --status
```

## 📝 版本更新记录

- **2025-12-24**: 
  - 更新CUDA至13.0
  - 更新驱动至580.65.06/580.88
  - 更新vLLM至0.13.0
  - 添加Deepseek V3 FP8原生支持
  - 所有文档版本号对齐

---

**注意**: 此文档随软件更新持续维护，请定期查看最新版本。
