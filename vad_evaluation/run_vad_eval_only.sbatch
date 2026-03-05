#!/bin/bash
#SBATCH --job-name=speechcue_eval
#SBATCH --output=../logs/speechcue_eval_%j.out
#SBATCH --error=../logs/speechcue_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi
cd /local/scratch/yhu383/SpeechCueLLM/vad_evaluation
source ../venv/bin/activate

CHECKPOINT_DIR="./experiments/with_past_vad/llama3.3-70b/lora/vad/window_12/per_1.0_seed_42"

echo "=========================================="
echo "VAD Eval-Only Configuration"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "=========================================="

deepspeed --num_gpus=1 main_vad.py \
    --model_name_or_path /local/scratch/yhu383/models/llama3.3-70b \
    --data_file ./data/vad_full_dataset.csv \
    --output_dir "$CHECKPOINT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --window_size 12 \
    --data_percent 1.0 \
    --seed 42 \
    --test_session 5 \
    --include_history_vad True \
    --use_filtered_data False \
    --do_train False \
    --do_eval True \
    --zero_shot False \
    --few_shot False \
    --lora True \
    --lora_dim 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_length 1500 \
    --short_prompt True \
    --eval_batch_size 4

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $CHECKPOINT_DIR"
echo "=========================================="
echo "Job finished at $(date)"
