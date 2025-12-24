#!/bin/bash
# GPU监控脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=========================================="
echo "GPU实时监控"
echo "==========================================${NC}"
echo -e "${YELLOW}按 Ctrl+C 退出${NC}\n"

# 选择监控模式
echo "选择监控模式:"
echo "1) 简单模式 (nvidia-smi)"
echo "2) 详细模式 (持续监控)"
echo "3) 性能分析模式 (dmon)"
read -p "请选择 [1-3]: " MODE

case $MODE in
    1)
        watch -n 1 nvidia-smi
        ;;
    2)
        while true; do
            clear
            echo -e "${GREEN}========== GPU状态 ($(date '+%Y-%m-%d %H:%M:%S')) ==========${NC}"
            nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits | \
            awk -F', ' '{printf "GPU %s: %s\n  温度: %s°C | GPU利用率: %s%% | 显存利用率: %s%%\n  显存使用: %s MB / %s MB | 功耗: %s W\n\n", $1, $2, $3, $4, $5, $6, $7, $8}'
            sleep 2
        done
        ;;
    3)
        nvidia-smi dmon -s pucvmet -i 0,1,2,3,4,5,6,7
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
