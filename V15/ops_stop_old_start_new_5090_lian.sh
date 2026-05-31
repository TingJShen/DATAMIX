#!/bin/bash
# ops_stop_old_start_new_5090_lian.sh
# 停止 5090_Lian 上的旧 v13 实验，部署 v13_grad 代码并启动新实验。
# 使用方式：在能 SSH 到 5090_Lian 的机器上执行本脚本。

set -euo pipefail

REMOTE=5090_Lian
BASE_DIR=/zhdd/home/tjshen/260415_ArcherA100

echo "=== Step 1: 停止 5090_Lian 上的旧 v13 训练进程 ==="
ssh "${REMOTE}" bash -c "'
echo \"[INFO] 查找旧 v13 训练进程...\"
OLD_PIDS=\$(ps aux | grep \"dapo.main_dapo\" | grep -v grep | awk \"{print \\\$2}\" || true)
if [ -n \"\${OLD_PIDS}\" ]; then
    echo \"[INFO] 发现旧进程: \${OLD_PIDS}\"
    echo \"\${OLD_PIDS}\" | xargs kill -TERM 2>/dev/null || true
    sleep 5
    echo \"\${OLD_PIDS}\" | xargs kill -9 2>/dev/null || true
    echo \"[INFO] 旧进程已终止\"
else
    echo \"[INFO] 未发现旧 v13 训练进程\"
fi
'"

echo ""
echo "=== Step 2: 停止旧 Ray 集群 ==="
ssh "${REMOTE}" bash -c "'
source /home/tjshen/miniconda3/bin/activate llama2_vllm_copy
timeout 60s ray stop --force 2>/dev/null || true
echo \"[INFO] Ray 已停止\"
'"

echo ""
echo "=== Step 3: 同步 v13_grad 代码到远端 ==="
echo "[INFO] 请确保 ${BASE_DIR}/v13_grad 已在远端就绪。"
echo "[INFO] 如果需要从本地同步，请手动执行："
echo "  rsync -avz --delete code/v13_grad/ ${REMOTE}:${BASE_DIR}/v13_grad/"
echo ""

echo "=== Step 4: 启动 v13_grad 训练 ==="
ssh "${REMOTE}" bash -c "'
cd ${BASE_DIR}/v13_grad
chmod +x launch_v13_grad_5090_lian_qwen3_2b_4gpu.sh
bash launch_v13_grad_5090_lian_qwen3_2b_4gpu.sh
'"

echo ""
echo "=== 完成 ==="
echo "监控命令："
echo "  ssh ${REMOTE} 'tail -f ${BASE_DIR}/diag_5090_Lian_v13_grad_*/launch.log'"
echo "  ssh ${REMOTE} 'nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader -i 0,2,3,5'"
