# H20 8卡服务器 vLLM + Deepseek V3 部署文档索引

> 从操作系统安装到推理服务部署的完整解决方案

---

## 📚 文档导航

### 🚀 快速入门

**如果你的服务器已经安装好Ubuntu和NVIDIA驱动，从这里开始：**

1. **[README.md](README.md)** - 快速入门指南
   - 适合: 系统已配置，只需部署vLLM
   - 时间: 30分钟 - 2小时 (取决于模型下载速度)

2. **[QUICKREF.md](QUICKREF.md)** - 命令速查表
   - 适合: 快速查找常用命令和参数
   - 时间: 即时参考

### 📖 完整文档

**如果你需要从零开始配置服务器，按此顺序阅读：**

3. **[完整系统配置指南.md](完整系统配置指南.md)** ⭐ **推荐从这里开始**
   - 适合: 新服务器或重新配置
   - 内容:
     * Ubuntu 22.04 LTS安装
     * NVIDIA驱动安装
     * CUDA Toolkit配置
     * 多核CPU优化
     * NVLink配置与检测
     * 系统性能优化
   - 时间: 2-4小时

4. **[H20_VLLM_DeepSeek_V3_安装指南.md](H20_VLLM_DeepSeek_V3_安装指南.md)**
   - 适合: 详细了解vLLM部署的每个细节
   - 内容:
     * vLLM安装详解
     * Deepseek V3模型部署
     * 推理服务配置
     * 性能优化技巧
     * 常见问题排查
   - 时间: 1-2小时阅读，2-4小时实践

---

## 🛠️ 自动化脚本

### 环境配置

| 脚本 | 功能 | 使用场景 |
|------|------|----------|
| **setup_environment.sh** | 一键配置vLLM Python环境 | 首次安装或重建环境 |
| **download_model.sh** | 下载Deepseek V3模型 | 首次部署或更新模型 |
| **start_inference.sh** | 启动vLLM推理服务 | 每次启动服务 |

### 测试和监控

| 脚本 | 功能 | 使用场景 |
|------|------|----------|
| **test_inference.py** | 测试推理功能 | 验证部署是否成功 |
| **benchmark.py** | 性能基准测试 | 评估系统性能，找最佳配置 |
| **monitor_gpu.sh** | 实时GPU监控 | 运行时监控GPU状态 |

### 系统验证

| 脚本 | 功能 | 创建方式 |
|------|------|----------|
| **system_validation.sh** | 完整系统验证 | 见《完整系统配置指南.md》 |
| **numa_config.sh** | NUMA优化配置 | 见《完整系统配置指南.md》 |
| **nvlink_config.sh** | NVLink优化配置 | 见《完整系统配置指南.md》 |
| **monitor_nvlink.sh** | NVLink实时监控 | 见《完整系统配置指南.md》 |

---

## 📋 部署流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     全新服务器部署流程                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   1. 硬件检查和BIOS配置                 │
         │   参考: 完整系统配置指南 - 硬件准备       │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   2. 安装Ubuntu 22.04 LTS              │
         │   参考: 完整系统配置指南 - Ubuntu安装     │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   3. 系统基础配置                       │
         │   - 网络配置                           │
         │   - SSH配置                            │
         │   - 更新系统                           │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   4. CPU优化配置                        │
         │   - 性能模式                           │
         │   - NUMA优化                           │
         │   - CPU亲和性                          │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   5. 安装NVIDIA驱动                     │
         │   - 禁用nouveau                        │
         │   - 安装驱动(535+)                     │
         │   - 启用持久化模式                      │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   6. 安装CUDA Toolkit 12.1+            │
         │   - 配置环境变量                        │
         │   - 验证安装                           │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   7. NVLink配置和测试                   │
         │   - 检查拓扑                           │
         │   - 带宽测试                           │
         │   - 优化配置                           │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   8. 系统性能优化                       │
         │   - 内存优化                           │
         │   - 磁盘I/O优化                        │
         │   - 网络优化                           │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   9. 运行系统验证                       │
         │   ./system_validation.sh               │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   10. 部署vLLM环境                      │
         │   ./setup_environment.sh               │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   11. 下载Deepseek V3模型               │
         │   ./download_model.sh                  │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   12. 启动推理服务                      │
         │   ./start_inference.sh                 │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   13. 测试和验证                        │
         │   python test_inference.py             │
         │   python benchmark.py                  │
         └────────────────────────────────────────┘
                              │
                              ▼
                        ✅ 部署完成！
```

---

## 🎯 使用场景快速索引

### 场景1: 我有全新的服务器，什么都没有

**阅读顺序:**
1. 📖 [完整系统配置指南.md](完整系统配置指南.md) - 从头到尾完整执行
2. 🚀 [README.md](README.md) - vLLM快速部署

**预计时间:** 4-8小时 (包括模型下载)

---

### 场景2: 服务器已装好Ubuntu和驱动，需要部署vLLM

**阅读顺序:**
1. 🚀 [README.md](README.md) - 快速入门
2. 📖 [H20_VLLM_DeepSeek_V3_安装指南.md](H20_VLLM_DeepSeek_V3_安装指南.md) - 详细参考

**快速命令:**
```bash
./setup_environment.sh
./download_model.sh
./start_inference.sh
```

**预计时间:** 1-3小时 (取决于模型下载速度)

---

### 场景3: vLLM已部署，需要查命令和参数

**参考:**
- 📋 [QUICKREF.md](QUICKREF.md) - 所有常用命令和参数

**预计时间:** 即时查询

---

### 场景4: 遇到问题需要排查

**参考文档的故障排查章节:**
- [完整系统配置指南.md](完整系统配置指南.md#故障排查清单) - 系统级问题
- [H20_VLLM_DeepSeek_V3_安装指南.md](H20_VLLM_DeepSeek_V3_安装指南.md#常见问题排查) - vLLM相关问题

**常见问题快速链接:**
- GPU未检测到 → 完整系统配置指南 - GPU未检测到
- NVLink不工作 → 完整系统配置指南 - NVLink不工作
- 显存不足 → vLLM安装指南 - CUDA Out of Memory
- 推理速度慢 → vLLM安装指南 - 推理速度慢

---

## 📊 系统要求检查表

在开始之前，请确认你的系统满足以下要求：

### 硬件要求

- [ ] **GPU**: NVIDIA H20 x 8张 (每卡141GB显存)
- [ ] **CPU**: 32核心以上 (64核心推荐)
- [ ] **内存**: 512GB+ DDR4/DDR5 ECC
- [ ] **存储**:
  - [ ] 系统盘: 500GB+ NVMe SSD
  - [ ] 数据盘: 2TB+ NVMe SSD (存放模型)
- [ ] **网络**: 10Gbps+ 以太网
- [ ] **NVLink**: 桥接卡已正确安装

### 软件要求

安装后应满足:

- [ ] **操作系统**: Ubuntu 22.04 LTS
- [ ] **内核**: 5.15+
- [ ] **NVIDIA驱动**: 580.65.06+ 
- [ ] **CUDA**: 13.0
- [ ] **Python**: 3.9-3.12
- [ ] **vLLM**: 0.13.0+
- [ ] **PyTorch**: 2.x (最新稳定版)

### Deepseek V3特别要求

- [ ] **模型格式**: FP8 (原生格式)
- [ ] **dtype配置**: auto (自动检测FP8)
- [ ] **kv_cache_dtype**: fp8 (推荐)
- [ ] **attention_backend**: FLASHINFER

---

## 🔧 维护和更新

### 日常维护任务

**每日:**
```bash
# 检查GPU状态
nvidia-smi

# 检查服务状态
systemctl status vllm-deepseek  # 如果使用systemd

# 检查日志
tail -f vllm_server.log
```

**每周:**
```bash
# 运行系统验证
~/system_validation.sh

# 检查磁盘空间
df -h

# 检查系统日志
dmesg | grep -i error
journalctl -p err -b
```

**每月:**
```bash
# 更新系统
sudo apt update && sudo apt upgrade

# 检查NVIDIA驱动更新
ubuntu-drivers devices

# 运行性能基准测试
python benchmark.py
```

### 更新vLLM

```bash
# 激活环境
source vllm-env/bin/activate

# 更新vLLM
pip install --upgrade vllm

# 验证版本
python -c "import vllm; print(vllm.__version__)"

# 重启服务
./start_inference.sh
```

### 更新NVIDIA驱动

```bash
# 停止vLLM服务
# Ctrl+C 或 systemctl stop vllm-deepseek

# 更新驱动
sudo ubuntu-drivers autoinstall

# 重启
sudo reboot

# 验证
nvidia-smi
```

---

## 📞 获取帮助

### 文档内资源

1. **详细文档**: 所有.md文件都有详细的步骤说明
2. **故障排查**: 每个文档都有专门的故障排查章节
3. **脚本注释**: 所有.sh和.py脚本都有详细注释

### 外部资源

- **vLLM官方文档**: https://docs.vllm.ai/
- **Deepseek V3**: https://huggingface.co/deepseek-ai/DeepSeek-V3
- **NVIDIA文档**: https://docs.nvidia.com/
- **Ubuntu文档**: https://help.ubuntu.com/

### 社区支持

- **vLLM GitHub**: https://github.com/vllm-project/vllm/issues
- **NVIDIA Forums**: https://forums.developer.nvidia.com/
- **Stack Overflow**: 标签 `vllm`, `cuda`, `nvidia`

---

## 📈 性能基准参考

### 预期性能指标 (H20 8卡 + Deepseek V3)

| 指标 | 预期值 | 说明 |
|------|--------|------|
| **吞吐量** | 500-1000 tokens/秒 | 取决于batch size |
| **首token延迟** | 0.5-2秒 | 取决于输入长度 |
| **最大并发** | 100+ 请求 | 取决于序列长度 |
| **GPU利用率** | 85-95% | 推理时 |
| **显存使用** | 700-900GB | Deepseek V3约685B参数 |
| **NVLink带宽** | 400+ GB/s | 8卡总带宽 |

### 运行基准测试

```bash
# 完整性能测试
python benchmark.py --model-path /data/models/deepseek-v3

# 快速测试
python benchmark.py --num-prompts 10 --batch-sizes "1,8,16"
```

---

## 📝 版本历史

- **v1.0** (2025-12-24)
  - 初始版本
  - 完整的从OS到vLLM的部署文档
  - 包含Ubuntu 22.04, NVIDIA驱动, CUDA, NVLink配置
  - vLLM和Deepseek V3部署指南

---

## 📄 文件清单

### 文档文件 (Markdown)

```
INDEX.md                               # 本文件 - 总索引
README.md                              # 快速入门指南
QUICKREF.md                            # 命令速查表
完整系统配置指南.md                     # 从OS安装到系统优化
H20_VLLM_DeepSeek_V3_安装指南.md       # vLLM详细部署指南
```

### 脚本文件 (Bash)

```
setup_environment.sh                   # vLLM环境一键配置
download_model.sh                      # 模型下载脚本
start_inference.sh                     # 推理服务启动脚本
monitor_gpu.sh                         # GPU实时监控
```

### Python文件

```
test_inference.py                      # 推理功能测试
benchmark.py                           # 性能基准测试
```

### 配置文件

```
vllm_env.sh                           # 环境变量配置 (运行setup_environment.sh后生成)
vllm_config.json                      # vLLM配置 (可选，根据需要创建)
```

### 日志文件

```
vllm_server.log                       # 服务日志 (运行start_inference.sh后生成)
benchmark_results_*.json              # 性能测试结果 (运行benchmark.py后生成)
```

---

## ✅ 下一步行动

### 如果你是第一次部署:

1. **阅读**: [完整系统配置指南.md](完整系统配置指南.md)
2. **执行**: 按照指南从Ubuntu安装开始
3. **验证**: 运行 `system_validation.sh`
4. **部署vLLM**: 运行 `setup_environment.sh`
5. **测试**: 运行 `test_inference.py`

### 如果系统已配置好:

1. **阅读**: [README.md](README.md)
2. **快速部署**:
   ```bash
   ./setup_environment.sh
   ./download_model.sh
   ./start_inference.sh
   ```
3. **验证**: `python test_inference.py --mode api`

### 如果只需要查询:

1. **参考**: [QUICKREF.md](QUICKREF.md)

---

**祝部署顺利！如有问题，请参考对应文档的故障排查章节。🚀**
