#!/bin/bash
#SBATCH --job-name=speechcue_llm
#SBATCH --output=../logs/speechcue_%j.out
#SBATCH --error=../logs/speechcue_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi
cd /local/scratch/yhu383/SpeechCueLLM/vad_evaluation
source ../venv/bin/activate
bash run_vad_evaluation.sh
echo "Job finished at $(date)"
