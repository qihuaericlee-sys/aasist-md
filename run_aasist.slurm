#!/bin/bash
# ===========================================================================
#  AASIST 训练任务 — Slurm 提交脚本
#  HKBU DAAI GPU Cluster
# ===========================================================================
#  用法:
#    sbatch run_aasist.slurm                      # 提交训练 (默认 unseen)
#    sbatch --export=EVAL_SPLIT=seen run_aasist.slurm   # 提交 seen 评估
#    sbatch --export=RESUME=1 run_aasist.slurm     # 从 checkpoint 续训
#    srun --pty bash run_aasist.slurm              # 交互式测试 (见文末说明)
# ===========================================================================

# ── Slurm 参数 ───────────────────────────────────────────────────────────
#SBATCH --job-name=AASIST_train            # 作业名
#SBATCH --output=/home/comp/25450212/logs/aasist_%j.out  # 标准输出日志 (%j = JobID)
#SBATCH --error=/home/comp/25450212/logs/aasist_%j.err   # 错误日志
#SBATCH --partition=short                  # 分区 (short / gpu / long)
#SBATCH --time=02:00:00                    # 最大运行时间 (调试用; 正式训练改 24:00:00)
#SBATCH --nodes=1                          # 节点数
#SBATCH --ntasks=1                         # 任务数
#SBATCH --gres=gpu:1                       # GPU 数量 (调试用1; 多卡改 2-8)
#SBATCH --cpus-per-task=8                  # 每任务 CPU 核心数 (给 DataLoader 用)
#SBATCH --mem=64G                          # 内存 (根据 GPU 显存调整)
#SBATCH --mail-type=END,FAIL               # 邮件通知 (结束后/失败时)
##SBATCH --mail-user=your.email@hkbu.edu.hk  # 你的邮箱 (取消注释后填写)

# ── 环境初始化 ───────────────────────────────────────────────────────────
set -e  # 出错即停

echo "============================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       $(hostname)"
echo "  Partition:  ${SLURM_JOB_PARTITION}"
echo "  GPUs:       ${SLURM_GPUS_ON_NODE:-1}"
echo "  Date:       $(date)"
echo "============================================"

# 创建日志目录
mkdir -p /home/comp/25450212/logs

# ── 加载软件模块 (根据 module avail 结果调整) ──────────────────────────
# 先用 module avail 2>&1 | grep -iE "python|cuda|pytorch|torch" 确认可用版本
module load gcc/11.2.0
module load cuda/11.4
module load miniconda/py38_4.10.3

# 备选方案 (如果上面模块不可用):
#   module load anaconda3
#   conda activate aasist_env   # 需要提前 conda create 好环境
# 或者:
#   pip install --user torch==2.2.0 numpy soundfile

# ── 安装依赖 (首次运行需要; 之后可注释掉) ───────────────────────────────
# 注意: torch 如果已经模块加载，不需要再 pip install
# pip install --user soundfile numpy
# 如果模块没有 torch，取消下面这行注释:
# pip install --user torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"

# ── 工作目录 ────────────────────────────────────────────────────────────
WORKDIR=/home/comp/25450212/aasist
cd "$WORKDIR" || { echo "ERROR: $WORKDIR not found!"; exit 1; }
echo "Working directory: $(pwd)"

# ── 参数 (通过环境变量控制) ─────────────────────────────────────────────
EVAL_SPLIT="${EVAL_SPLIT:-unseen}"          # seen 或 unseen
RESUME="${RESUME:-0}"                       # 1 = 从 checkpoint 续训
COMMENT="${COMMENT:-cluster_run}"           # 实验备注
CONFIG="${CONFIG:-./config/AASIST_cluster.conf}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp_result}"
SEED="${SEED:-1234}"

echo ""
echo "Config:      ${CONFIG}"
echo "Eval split:  ${EVAL_SPLIT}"
echo "Resume:      ${RESUME}"
echo "Seed:        ${SEED}"
echo "============================================"
echo ""

# ── 运行 ────────────────────────────────────────────────────────────────
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

# ── 训练完自动评估 (可选) ───────────────────────────────────────────────
# 如果想训练完立刻用最佳模型评估，取消下面三行注释:
# BEST_MODEL=$(find ${OUTPUT_DIR}/LA_AASIST_cluster_* -name "best.pth" 2>/dev/null | head -1)
# echo "Best model: ${BEST_MODEL}"
# # python main.py --config ${CONFIG} --eval --eval_model_weights ${BEST_MODEL}

exit ${EXIT_CODE}
