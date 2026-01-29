#!/bin/bash

# =============================================================================
# VAD Evaluation Script
# Supports: Zero-shot, Few-shot, and LoRA fine-tuning for LLaMA models
#          Zero-shot and Few-shot for OpenAI models
# =============================================================================

# Default settings
MODEL_TYPE="openai"  # openai or llama
EXPERIMENT="zero_shot"  # zero_shot, few_shot, lora
MODEL_NAME="gpt-4o-mini"  # OpenAI model name or path to LLaMA model
DATA_FILE="./data/vad_full_dataset.csv"
OUTPUT_BASE="./experiments"
WINDOW_SIZE=12
DATA_PERCENT=1.0
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type)0
            MODEL_TYPE="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --data_file)
            DATA_FILE="$2"
            shift 2
            ;;
        --output_base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --window_size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --data_percent)
            DATA_PERCENT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
MODEL_SHORT_NAME=$(basename "$MODEL_NAME")
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_SHORT_NAME}/${EXPERIMENT}/vad/window_${WINDOW_SIZE}/per_${DATA_PERCENT}_seed_${SEED}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "VAD Evaluation Configuration"
echo "=========================================="
echo "Model Type: $MODEL_TYPE"
echo "Experiment: $EXPERIMENT"
echo "Model: $MODEL_NAME"
echo "Data: $DATA_FILE"
echo "Output: $OUTPUT_DIR"
echo "Window Size: $WINDOW_SIZE"
echo "Data Percent: $DATA_PERCENT"
echo "Seed: $SEED"
echo "=========================================="

# Run based on model type
if [ "$MODEL_TYPE" == "openai" ]; then
    echo "Running OpenAI VAD evaluation..."

    python main_vad_openai.py \
        --data_file "$DATA_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --experiments_setting "$EXPERIMENT" \
        --window_size "$WINDOW_SIZE" \
        --data_percent "$DATA_PERCENT" \
        --seed "$SEED" \
        --batch_delay 0.1

elif [ "$MODEL_TYPE" == "llama" ]; then
    echo "Running LLaMA VAD evaluation..."

    if [ "$EXPERIMENT" == "zero_shot" ]; then
        deepspeed --num_gpus=1 main_vad.py \
            --model_name_or_path "$MODEL_NAME" \
            --data_file "$DATA_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --window_size "$WINDOW_SIZE" \
            --data_percent "$DATA_PERCENT" \
            --seed "$SEED" \
            --do_train False \
            --do_eval True \
            --zero_shot True \
            --few_shot False \
            --lora False \
            --max_length 2048 \
            --eval_batch_size 4

    elif [ "$EXPERIMENT" == "few_shot" ]; then
        deepspeed --num_gpus=1 main_vad.py \
            --model_name_or_path "$MODEL_NAME" \
            --data_file "$DATA_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --window_size "$WINDOW_SIZE" \
            --data_percent "$DATA_PERCENT" \
            --seed "$SEED" \
            --do_train False \
            --do_eval True \
            --zero_shot False \
            --few_shot True \
            --lora False \
            --max_length 2500 \
            --eval_batch_size 4

    elif [ "$EXPERIMENT" == "lora" ]; then
        # For LoRA, use shorter prompts to fit in context window
        deepspeed --num_gpus=1 main_vad.py \
            --model_name_or_path "$MODEL_NAME" \
            --data_file "$DATA_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --window_size "$WINDOW_SIZE" \
            --data_percent "$DATA_PERCENT" \
            --seed "$SEED" \
            --do_train True \
            --do_eval True \
            --zero_shot False \
            --few_shot False \
            --lora True \
            --lora_dim 16 \
            --lora_alpha 16 \
            --lora_dropout 0.05 \
            --learning_rate 3e-4 \
            --num_train_epochs 15 \
            --batch_size 8 \
            --gradient_accumulation_steps 8 \
            --max_length 1500 \
            --short_prompt True \
            --gradient_checkpointing \
            --eval_batch_size 4
    else
        echo "Unknown experiment type: $EXPERIMENT"
        exit 1
    fi
else
    echo "Unknown model type: $MODEL_TYPE"
    exit 1
fi

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
