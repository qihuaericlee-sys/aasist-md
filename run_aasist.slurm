#!/bin/bash
# ===========================================================================
#  AASIST 训练任务 — Slurm 提交脚本  (HKBU DAAI GPU Cluster)
# ===========================================================================
#  用法:
#    sbatch run_aasist.slurm                              # 默认: V100, unseen
#    sbatch --export=GPU_TYPE=A100 run_aasist.slurm        # A100, unseen
#    sbatch --export=GPU_TYPE=A100,EVAL_SPLIT=seen run_aasist.slurm
#    sbatch --export=RESUME=1 run_aasist.slurm              # 从 checkpoint 续训
#
#    srun --gres=gpu:1 --cpus-per-task=4 --time=00:30:00 --pty bash run_aasist.slurm  # 交互式
# ===========================================================================

# ── Slurm 参数 ───────────────────────────────────────────────────────────
#SBATCH --job-name=AASIST_train            # 作业名
#SBATCH --output=/home/comp/25450212/logs/aasist_%j.out  # 标准输出日志 (%j = JobID)
#SBATCH --error=/home/comp/25450212/logs/aasist_%j.err   # 错误日志
#SBATCH --partition=long                   # 分区: long=V100(14天) / short=A100/H20(5天)
#SBATCH --time=7-00:00:00                  # 最大运行时间 (long分区最长14天)
#SBATCH --nodes=1                          # 节点数
#SBATCH --ntasks=1                         # 任务数
#SBATCH --gres=gpu:1                       # GPU 数量 (调试用1; 多卡改 2-4 for V100, 2-8 for A100)
#SBATCH --cpus-per-task=8                  # DataLoader 用 CPU 核心数
#SBATCH --mem=64G                          # 内存
#SBATCH --mail-type=END,FAIL               # 邮件通知
##SBATCH --mail-user=your.email@hkbu.edu.hk  # 取消注释并填写你的邮箱

# ── 环境初始化 ───────────────────────────────────────────────────────────
set -e  # 出错即停

echo "============================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  Partition:   ${SLURM_JOB_PARTITION}"
echo "  GPUs:        ${SLURM_GPUS_ON_NODE:-1}"
echo "  Date:        $(date)"
echo "============================================"

# 创建日志目录
mkdir -p /home/comp/25450212/logs

# ── 加载软件模块 (HKBU DAAI 集群标准环境) ──────────────────────────────
module load gcc/11.2.0
module load cuda/11.4
module load miniconda/py38_4.10.3

echo ""
echo "=== 环境信息 ==="
echo "GCC:       $(gcc --version 2>&1 | head -1)"
echo "CUDA:      $(nvcc --version 2>&1 | tail -1)"
echo "Python:    $(python --version 2>&1)"

# ── 安装 PyTorch 1.12.1 + CUDA 11.3 (兼容集群 CUDA 11.4 驱动) ─────────
# 首次运行或更新依赖时取消注释下面几行:
# pip install --user torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
#     --extra-index-url https://download.pytorch.org/whl/cu113
# pip install --user soundfile numpy

# 验证 PyTorch
echo "PyTorch:   $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "CUDA ok:   $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())' 2>&1)"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "GPU name:  $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>&1)"
fi

# ── 工作目录 (绝对路径, Slurm PWD 不固定) ────────────────────────────────
# 代码在仓库的 aasist/ 子目录中 (main.py, data_utils.py 在此)
WORKDIR=/home/comp/25450212/aasist/aasist
cd "$WORKDIR" || { echo "ERROR: 无法进入 $WORKDIR!"; exit 1; }
echo ""
echo "Working dir: $(pwd)"

# ── 参数 (通过 sbatch --export= 环境变量控制) ────────────────────────────
# GPU_TYPE:  V100 | A100   → 自动选择分区和 batch_size
# EVAL_SPLIT: seen | unseen
# RESUME:     1 = 从 checkpoint 续训
# SEED:       随机种子
GPU_TYPE="${GPU_TYPE:-V100}"                 # V100 (32GB) 或 A100 (80GB)
EVAL_SPLIT="${EVAL_SPLIT:-unseen}"           # seen 或 unseen
RESUME="${RESUME:-0}"                        # 1 = 续训
COMMENT="${COMMENT:-cluster_${GPU_TYPE}}"    # 实验备注
SEED="${SEED:-1234}"

# ── 根据 GPU 类型选择配置 ────────────────────────────────────────────────
if [ "${GPU_TYPE}" = "V100" ]; then
    CONFIG="./config/AASIST_v100.conf"
    BATCH_SIZE=16
    PARTITION="long"
    MAX_TIME="7-00:00:00"
elif [ "${GPU_TYPE}" = "A100" ]; then
    CONFIG="./config/AASIST_a100.conf"
    BATCH_SIZE=64
    PARTITION="short"
    MAX_TIME="5-00:00:00"
else
    echo "ERROR: 未知 GPU_TYPE=${GPU_TYPE}, 请设为 V100 或 A100"
    exit 1
fi

OUTPUT_DIR="./exp_result"

echo ""
echo "=== 运行参数 ==="
echo "GPU type:   ${GPU_TYPE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Config:     ${CONFIG}"
echo "Partition:  ${PARTITION} (max time: ${MAX_TIME})"
echo "Eval split: ${EVAL_SPLIT}"
echo "Resume:     ${RESUME}"
echo "Seed:       ${SEED}"
echo "Comment:    ${COMMENT}"
echo "============================================"
echo ""

# ── 运行训练 ────────────────────────────────────────────────────────────
CMD="python main.py \
  --config ${CONFIG} \
  --output_dir ${OUTPUT_DIR} \
  --seed ${SEED} \
  --eval_split ${EVAL_SPLIT} \
  --comment ${COMMENT}"

if [ "${RESUME}" = "1" ]; then
    CMD="${CMD} --resume"
fi

echo "Running: ${CMD}"
echo ""

${CMD}

EXIT_CODE=$?
echo ""
echo "============================================"
echo "  Finished at: $(date)"
echo "  Exit code:   ${EXIT_CODE}"
echo "============================================"

exit ${EXIT_CODE}
