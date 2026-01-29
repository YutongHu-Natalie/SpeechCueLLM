"""
LLaMA model inference script for VAD (Valence-Arousal-Dominance) prediction.
Supports zero-shot, few-shot, and LoRA fine-tuning modes.
Uses DeepSpeed for efficient inference and training.
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizerFast,
    get_linear_schedule_with_warmup,
)
import torch
import torch.nn as nn
import deepspeed
from dataclasses import dataclass, asdict
import pandas as pd
import json
import logging
import math
import os
import random
import re
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from peft import LoraConfig, get_peft_model
import argparse
from tqdm import tqdm

# Add parent directory for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LLM_code.data_utils.data_utils import (
    get_deepspeed_config_path,
    KeywordsStoppingCriteria,
    get_parameter_number
)


def concordance_correlation_coefficient(y_true, y_pred):
    """Compute Concordance Correlation Coefficient (CCC)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)

    return ccc


def compute_vad_metrics(golds_v, preds_v, golds_a, preds_a, golds_d, preds_d):
    """Compute comprehensive metrics for VAD prediction."""
    metrics = {}

    dimensions = [
        ('valence', golds_v, preds_v),
        ('arousal', golds_a, preds_a),
        ('dominance', golds_d, preds_d)
    ]

    for dim_name, golds, preds in dimensions:
        golds = np.array(golds)
        preds = np.array(preds)

        mse = mean_squared_error(golds, preds)
        metrics[f'{dim_name}_mse'] = round(mse, 4)
        metrics[f'{dim_name}_rmse'] = round(np.sqrt(mse), 4)
        metrics[f'{dim_name}_mae'] = round(mean_absolute_error(golds, preds), 4)

        if len(set(golds)) > 1 and len(set(preds)) > 1:
            pearson_r, pearson_p = pearsonr(golds, preds)
            metrics[f'{dim_name}_pearson_r'] = round(pearson_r, 4)
        else:
            metrics[f'{dim_name}_pearson_r'] = 0.0

        metrics[f'{dim_name}_ccc'] = round(concordance_correlation_coefficient(golds, preds), 4)

    # Average CCC
    metrics['avg_ccc'] = round(
        (metrics['valence_ccc'] + metrics['arousal_ccc'] + metrics['dominance_ccc']) / 3, 4
    )

    return metrics


def extract_vad_from_output(output):
    """Extract VAD values from model output."""
    if not output or output.strip() == '':
        return None, None, None

    # Strategy 1: Extract from <answer> tags
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        try:
            parsed = json.loads(answer_content)
            v = int(parsed.get('v_value', 3))
            a = int(parsed.get('a_value', 3))
            d = int(parsed.get('d_value', 3))
            return float(v), float(a), float(d)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 2: Find JSON object
    json_match = re.search(r'\{[^{}]*"v_value"[^{}]*\}', output)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            v = int(parsed.get('v_value', 3))
            a = int(parsed.get('a_value', 3))
            d = int(parsed.get('d_value', 3))
            return float(v), float(a), float(d)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 3: Regex patterns
    v_match = re.search(r'"?v_value"?\s*:\s*"?(\d)"?', output, re.IGNORECASE)
    a_match = re.search(r'"?a_value"?\s*:\s*"?(\d)"?', output, re.IGNORECASE)
    d_match = re.search(r'"?d_value"?\s*:\s*"?(\d)"?', output, re.IGNORECASE)

    if v_match and a_match and d_match:
        try:
            return float(int(v_match.group(1))), float(int(a_match.group(1))), float(int(d_match.group(1)))
        except ValueError:
            pass

    # Strategy 4: Natural language patterns
    val_match = re.search(r'valence[:\s]+(\d)', output, re.IGNORECASE)
    aro_match = re.search(r'arousal[:\s]+(\d)', output, re.IGNORECASE)
    dom_match = re.search(r'dominance[:\s]+(\d)', output, re.IGNORECASE)

    if val_match and aro_match and dom_match:
        try:
            return float(int(val_match.group(1))), float(int(aro_match.group(1))), float(int(dom_match.group(1)))
        except ValueError:
            pass

    return None, None, None


def get_few_shot_examples():
    """Returns few-shot examples for VAD prediction."""
    examples = [
        {
            "conversation_history": 'Speaker_0: "I\'ve been waiting in this line for two hours."\nSpeaker_1: "Sir, please be patient."',
            "target_utterance": 'Speaker_0: "This is ridiculous! I have places to be!"',
            "audio_features": "high volume, high pitch variation, very high speaking rate",
            "answer": '{"v_value": "2", "a_value": "4", "d_value": "3"}'
        },
        {
            "conversation_history": 'Speaker_0: "What time is the meeting tomorrow?"',
            "target_utterance": 'Speaker_1: "It\'s scheduled for 3pm in conference room B."',
            "audio_features": "moderate volume, low variation, moderate speaking rate",
            "answer": '{"v_value": "3", "a_value": "2", "d_value": "3"}'
        },
        {
            "conversation_history": 'Speaker_0: "I have some news about the job."\nSpeaker_1: "What is it?"',
            "target_utterance": 'Speaker_0: "I got it! They offered me the position!"',
            "audio_features": "high volume, high pitch variation, high speaking rate",
            "answer": '{"v_value": "5", "a_value": "4", "d_value": "4"}'
        },
    ]
    return examples


def create_vad_prompt_zero_shot(conversation_history, target_utterance, audio_features, short_version=False):
    """
    Create zero-shot prompt for VAD prediction.

    Args:
        short_version: If True, use a shorter prompt suitable for LoRA with limited context
    """
    if short_version:
        # Shorter prompt for LoRA to fit in context window
        prompt = f"""Rate the emotion in the target utterance using VAD (1-5 scale):
* Valence: How positive or negative (1 = very negative, 5 = very positive)
* Arousal: The intensity or energy level (1 = very calm, 5 = very excited)
* Dominance: The degree of control or power (1 = very submissive, 5 = very dominant)

Conversation:
{conversation_history}

Target: {target_utterance}
Audio: {audio_features}

Output JSON only: {{"v_value": "X", "a_value": "X", "d_value": "X"}}
<answer>"""
    else:
        prompt = f"""Now you are expert of evaluating emotions for dialogues based on Valence-Arousal-Dominance (VAD) Model. Please rate the emotion on a scale of 1-5 (only give integer) for each dimension.

You will be analyzing a dialogue to evaluate the emotion expressed in a specific target utterance using the VAD Model:
* Valence: How positive or negative (1 = very negative, 5 = very positive)
* Arousal: The intensity or energy level (1 = very calm, 5 = very excited)
* Dominance: The degree of control or power (1 = very submissive, 5 = very dominant)

Conversation history:
{conversation_history}

Target utterance: {target_utterance}

Audio features: {audio_features}

Rate the emotion in the target utterance. Consider both content and audio features.
Provide your ratings in JSON format inside <answer> tags:
<answer>
{{"v_value": "your integer 1-5", "a_value": "your integer 1-5", "d_value": "your integer 1-5"}}
</answer>"""

    return prompt


def create_vad_prompt_few_shot(conversation_history, target_utterance, audio_features):
    """Create few-shot prompt with examples."""
    examples = get_few_shot_examples()
    examples_text = ""

    for i, ex in enumerate(examples, 1):
        examples_text += f"\nExample {i}:\nConversation: {ex['conversation_history']}\nTarget: {ex['target_utterance']}\nAudio: {ex['audio_features']}\n<answer>\n{ex['answer']}\n</answer>\n"

    prompt = f"""Now you are expert of evaluating emotions for dialogues based on VAD Model. Rate emotion on scale 1-5 for each dimension.

VAD dimensions:
* Valence: How positive or negative (1 = very negative, 5 = very positive)
* Arousal: The intensity or energy level (1 = very calm, 5 = very excited)
* Dominance: The degree of control or power (1 = very submissive, 5 = very dominant)
{examples_text}

Now analyze this dialogue:

Conversation history:
{conversation_history}

Target utterance: {target_utterance}

Audio features: {audio_features}

Provide your ratings in JSON format inside <answer> tags:
<answer>"""

    return prompt


def read_vad_data(file_path, percent=1.0, random_seed=42):
    """Read VAD data from file."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.pkl'):
        df = pd.read_pickle(file_path)
    elif file_path.endswith('.jsonl'):
        records = []
        with open(file_path, 'r') as f:
            for line in f:
                records.append(json.loads(line.strip()))
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    if percent < 1.0:
        random.seed(random_seed)
        n_samples = int(len(df) * percent)
        df = df.sample(n=n_samples, random_state=random_seed)

    return df


def build_conversation_history(df, current_idx, window_size=12):
    """Build conversation history for a given utterance."""
    current_row = df.iloc[current_idx]
    dialogue_id = current_row['dialogue_id']
    turn_index = current_row['turn_index']

    dialogue_df = df[df['dialogue_id'] == dialogue_id].sort_values('turn_index')
    start_idx = max(0, turn_index - window_size)
    history_df = dialogue_df[(dialogue_df['turn_index'] >= start_idx) & (dialogue_df['turn_index'] < turn_index)]

    history_lines = []
    for _, row in history_df.iterrows():
        speaker = f"Speaker_{row['speaker']}"
        content = row['content']
        history_lines.append(f'{speaker}: "{content}"')

    conversation_history = '\n'.join(history_lines)
    target_speaker = f"Speaker_{current_row['speaker']}"
    target_content = current_row['content']
    target_utterance = f'{target_speaker}: "{target_content}"'

    return conversation_history, target_utterance


class VADDataset(Dataset):
    """Dataset for VAD prediction."""

    def __init__(self, df, prompt_fn, window_size=12, short_prompt=False):
        self.df = df.reset_index(drop=True)
        self.prompt_fn = prompt_fn
        self.window_size = window_size
        self.short_prompt = short_prompt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        conversation_history, target_utterance = build_conversation_history(
            self.df, idx, self.window_size
        )
        audio_features = "Audio features not available."

        if self.short_prompt:
            prompt = create_vad_prompt_zero_shot(
                conversation_history, target_utterance, audio_features, short_version=True
            )
        else:
            prompt = self.prompt_fn(conversation_history, target_utterance, audio_features)

        # Target for training: the expected JSON output
        target = json.dumps({
            "v_value": str(int(round(row['valence']))),
            "a_value": str(int(round(row['arousal']))),
            "d_value": str(int(round(row['dominance'])))
        })

        return {
            'input': prompt,
            'target': target,
            'gold_v': float(row['valence']),
            'gold_a': float(row['arousal']),
            'gold_d': float(row['dominance']),
            'dialogue_id': row['dialogue_id'],
            'turn_id': row['turn_id'],
            'content': row['content'],
        }


class VADCollator:
    """Collator for VAD dataset."""

    def __init__(self, tokenizer, max_length=2048, mode='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __call__(self, batch):
        inputs = [item['input'] for item in batch]

        if self.mode == 'train':
            targets = [item['target'] for item in batch]

            # Tokenize inputs
            input_encodings = self.tokenizer(
                inputs,
                max_length=self.max_length - 50,
                truncation=True
            )

            # Tokenize targets
            target_encodings = self.tokenizer(
                targets,
                add_special_tokens=False
            )

            # Concatenate input and target
            input_ids_list = []
            attention_mask_list = []
            labels_list = []

            for i in range(len(inputs)):
                input_ids = input_encodings['input_ids'][i]
                target_ids = target_encodings['input_ids'][i]

                concat_ids = input_ids + target_ids + [self.tokenizer.eos_token_id]
                concat_ids = concat_ids[:self.max_length]

                attention_mask = [1] * len(concat_ids)

                # Labels: -100 for input, actual ids for target
                labels = [-100] * len(input_ids) + target_ids + [self.tokenizer.eos_token_id]
                labels = labels[:self.max_length]

                input_ids_list.append(concat_ids)
                attention_mask_list.append(attention_mask)
                labels_list.append(labels)

            # Pad to same length
            max_len = max(len(ids) for ids in input_ids_list)
            for i in range(len(input_ids_list)):
                pad_len = max_len - len(input_ids_list[i])
                input_ids_list[i] = [self.tokenizer.pad_token_id] * pad_len + input_ids_list[i]
                attention_mask_list[i] = [0] * pad_len + attention_mask_list[i]
                labels_list[i] = [-100] * pad_len + labels_list[i]

            return {
                'input_ids': torch.tensor(input_ids_list),
                'attention_mask': torch.tensor(attention_mask_list),
                'labels': torch.tensor(labels_list),
            }

        else:  # eval mode
            encodings = self.tokenizer(
                inputs,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )

            # Include metadata for evaluation
            metadata = {
                'gold_v': [item['gold_v'] for item in batch],
                'gold_a': [item['gold_a'] for item in batch],
                'gold_d': [item['gold_d'] for item in batch],
                'dialogue_id': [item['dialogue_id'] for item in batch],
                'turn_id': [item['turn_id'] for item in batch],
                'content': [item['content'] for item in batch],
            }

            return encodings, metadata


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
world_size = int(os.getenv("WORLD_SIZE", '1'))


@dataclass
class VADModelArgs:
    """Arguments for VAD model."""
    model_name_or_path: str = ""
    data_file: str = ""
    output_dir: str = ""
    checkpoint_dir: str = None
    max_length: int = 2048
    batch_size: int = 8
    eval_batch_size: int = 4
    learning_rate: float = 3e-4
    num_train_epochs: int = 10
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    lora: bool = False
    lora_dim: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_module_name: str = 'q_proj,k_proj,v_proj'
    seed: int = 42
    do_train: bool = False
    do_eval: bool = True
    zero_shot: bool = True
    few_shot: bool = False
    window_size: int = 12
    data_percent: float = 1.0
    short_prompt: bool = False  # Use shorter prompt for LoRA
    gradient_checkpointing: bool = False
    deepspeed_config: str = "auto"
    test_session: int = 5  # Session to use as test set (1-5)
    use_filtered_data: bool = False  # Use filtered dataset (for LoRA only)

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            f.write(json.dumps(asdict(self), indent=2))

    def update(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)


def main():
    parser = argparse.ArgumentParser(description='LLaMA VAD Prediction')
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lora', type=str, default='False')
    parser.add_argument('--lora_dim', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--do_train', type=str, default='False')
    parser.add_argument('--do_eval', type=str, default='True')
    parser.add_argument('--zero_shot', type=str, default='True')
    parser.add_argument('--few_shot', type=str, default='False')
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--data_percent', type=float, default=1.0)
    parser.add_argument('--short_prompt', type=str, default='False')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--deepspeed_config', type=str, default='auto')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--test_session', type=int, default=5,
                        help='Session to use as test set (1-5). For zero-shot/few-shot, only this session is evaluated.')
    parser.add_argument('--use_filtered_data', type=str, default='False',
                        help='For LoRA only: use filtered dataset instead of full dataset')

    cmd_args = parser.parse_args()

    # Convert string bools
    cmd_args.lora = cmd_args.lora == 'True'
    cmd_args.do_train = cmd_args.do_train == 'True'
    cmd_args.do_eval = cmd_args.do_eval == 'True'
    cmd_args.zero_shot = cmd_args.zero_shot == 'True'
    cmd_args.few_shot = cmd_args.few_shot == 'True'
    cmd_args.short_prompt = cmd_args.short_prompt == 'True'
    cmd_args.use_filtered_data = cmd_args.use_filtered_data == 'True'

    args = VADModelArgs()
    args.update(vars(cmd_args))

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-select DeepSpeed config
    if args.deepspeed_config == "auto":
        args.deepspeed_config = get_deepspeed_config_path(
            args.model_name_or_path,
            base_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LLM_code/data_utils")
        )
        print(f"Auto-selected DeepSpeed config: {args.deepspeed_config}")

    with open(args.deepspeed_config, 'r') as f:
        deepspeed_config = json.load(f)
    deepspeed_config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * world_size
    deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Load model and tokenizer
    print(f"Loading model: {args.model_name_or_path}")
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    try:
        tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name_or_path, from_slow=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "left"

    # LoRA setup
    if args.lora:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_dim,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_module_name.split(","),
            bias='none',
        )
        model = get_peft_model(model, lora_config)
        print("LoRA enabled")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Save model params
    num_parameters = get_parameter_number(model)
    with open(os.path.join(args.output_dir, "model_params.json"), 'w') as f:
        json.dump(num_parameters, f, indent=2)

    # Load data
    print(f"Loading data: {args.data_file}")
    df_full = read_vad_data(args.data_file, percent=args.data_percent, random_seed=args.seed)
    print(f"Loaded {len(df_full)} total samples from full dataset")

    # Test data is ALWAYS from the full dataset (Session 5) for consistency across all experiments
    test_df = df_full[df_full['session'] == args.test_session].reset_index(drop=True)
    print(f"Test set: Session {args.test_session} from full dataset = {len(test_df)} samples")

    # Split data based on mode
    if args.lora:
        # For LoRA: determine training data source
        if args.use_filtered_data:
            # Use filtered dataset for training only
            filtered_data_path = args.data_file.replace('full_dataset', 'filtered_dataset')
            if os.path.exists(filtered_data_path):
                df_filtered = read_vad_data(filtered_data_path, percent=args.data_percent, random_seed=args.seed)
                train_df = df_filtered[df_filtered['session'] != args.test_session].reset_index(drop=True)
                print(f"LoRA mode (filtered training): Train={len(train_df)} samples from filtered dataset")
            else:
                print(f"WARNING: Filtered dataset not found at {filtered_data_path}, using full dataset for training")
                train_df = df_full[df_full['session'] != args.test_session].reset_index(drop=True)
                print(f"LoRA mode: Train={len(train_df)} samples from full dataset")
        else:
            # Use full dataset for training
            train_df = df_full[df_full['session'] != args.test_session].reset_index(drop=True)
            print(f"LoRA mode: Train={len(train_df)} samples from full dataset (sessions != {args.test_session})")
    else:
        # For zero-shot and few-shot: no training data needed
        train_df = None
        print(f"Zero-shot/Few-shot mode: No training data (evaluation only)")

    # Select prompt function
    if args.few_shot:
        prompt_fn = create_vad_prompt_few_shot
        print("Using few-shot prompting")
    else:
        prompt_fn = lambda ch, tu, af: create_vad_prompt_zero_shot(ch, tu, af, short_version=args.short_prompt)
        print(f"Using zero-shot prompting (short_prompt={args.short_prompt})")

    # Create datasets
    if args.lora and train_df is not None:
        train_dataset = VADDataset(train_df, prompt_fn, args.window_size, args.short_prompt)
        print(f"Created training dataset with {len(train_dataset)} samples")
    else:
        train_dataset = None

    eval_dataset = VADDataset(test_df, prompt_fn, args.window_size, args.short_prompt)
    print(f"Created evaluation dataset with {len(eval_dataset)} samples")

    if args.do_train:
        print("\n***** Training *****")
        if train_dataset is None:
            raise ValueError("Training requires train_dataset, but running in zero-shot/few-shot mode. Set --lora True for training.")
        train_collator = VADCollator(tokenizer, args.max_length, mode='train')

        optimizer_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(optimizer_params, lr=args.learning_rate)

        t_total = math.ceil(len(train_dataset) / args.batch_size) * args.num_train_epochs
        warmup_steps = int(t_total * args.warmup_ratio)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

        model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
            model=model,
            training_data=train_dataset,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=deepspeed_config,
            collate_fn=train_collator
        )

        best_ccc = -1
        for epoch in range(args.num_train_epochs):
            model_engine.train()
            epoch_loss = 0

            batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
            for step, batch in enumerate(batch_iterator):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model_engine(**batch)
                loss = outputs.loss

                model_engine.backward(loss)
                model_engine.step()

                epoch_loss += loss.item()
                batch_iterator.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

            # Save checkpoint
            model_engine.save_checkpoint(args.output_dir)

        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        args.save(args.output_dir)

    if args.do_eval:
        print("\n***** Evaluating *****")

        if args.checkpoint_dir and not args.zero_shot:
            from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
            model = load_state_dict_from_zero_checkpoint(model, args.checkpoint_dir)

        # Initialize for inference
        dtype = torch.half
        model_engine = deepspeed.init_inference(
            model,
            mp_size=world_size,
            replace_with_kernel_inject=True,
            dtype=dtype,
        )
        model = model_engine.module

        eval_collator = VADCollator(tokenizer, args.max_length, mode='eval')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            sampler=eval_sampler,
            collate_fn=eval_collator,
            num_workers=4
        )

        all_outputs = []
        preds_v, preds_a, preds_d = [], [], []
        golds_v, golds_a, golds_d = [], [], []
        failed_cases = []

        model.eval()
        for batch_idx, (batch, metadata) in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = batch.to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **batch,
                    max_new_tokens=100,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode outputs
            for i in range(len(outputs)):
                output_text = tokenizer.decode(outputs[i], skip_special_tokens=True)

                # Extract predicted VAD
                v, a, d = extract_vad_from_output(output_text)

                # Get gold values
                gold_v = metadata['gold_v'][i]
                gold_a = metadata['gold_a'][i]
                gold_d = metadata['gold_d'][i]

                # Handle failures
                if v is None or a is None or d is None:
                    v = v or 3.0
                    a = a or 3.0
                    d = d or 3.0
                    failed_cases.append(batch_idx * args.eval_batch_size + i)

                # Clamp to valid range
                v = max(1.0, min(5.0, v))
                a = max(1.0, min(5.0, a))
                d = max(1.0, min(5.0, d))

                preds_v.append(v)
                preds_a.append(a)
                preds_d.append(d)
                golds_v.append(gold_v)
                golds_a.append(gold_a)
                golds_d.append(gold_d)

                all_outputs.append({
                    "index": batch_idx * args.eval_batch_size + i,
                    "dialogue_id": metadata['dialogue_id'][i],
                    "turn_id": metadata['turn_id'][i],
                    "content": metadata['content'][i],
                    "output": output_text[-500:],  # Truncate for storage
                    "pred_v": v,
                    "pred_a": a,
                    "pred_d": d,
                    "gold_v": gold_v,
                    "gold_a": gold_a,
                    "gold_d": gold_d,
                })

        # Compute metrics
        metrics = compute_vad_metrics(golds_v, preds_v, golds_a, preds_a, golds_d, preds_d)

        # Save results
        preds_path = os.path.join(args.output_dir, "preds_for_eval.text")
        with open(preds_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VAD PREDICTION RESULTS\n")
            f.write("=" * 80 + "\n\n")

            for dim in ['valence', 'arousal', 'dominance']:
                f.write(f"\n{dim.upper()}:\n")
                f.write(f"  MSE:  {metrics[f'{dim}_mse']:.4f}\n")
                f.write(f"  RMSE: {metrics[f'{dim}_rmse']:.4f}\n")
                f.write(f"  MAE:  {metrics[f'{dim}_mae']:.4f}\n")
                f.write(f"  CCC:  {metrics[f'{dim}_ccc']:.4f}\n")
                f.write(f"  Pearson r: {metrics[f'{dim}_pearson_r']:.4f}\n")

            f.write(f"\nAverage CCC: {metrics['avg_ccc']:.4f}\n")
            f.write(f"Failed cases: {len(failed_cases)}\n")
            f.write("\n" + json.dumps(all_outputs, indent=2, ensure_ascii=False))

        # Save metrics
        with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)

        args.save(args.output_dir)

        # Print results
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Valence  - CCC: {metrics['valence_ccc']:.4f}, RMSE: {metrics['valence_rmse']:.4f}")
        print(f"Arousal  - CCC: {metrics['arousal_ccc']:.4f}, RMSE: {metrics['arousal_rmse']:.4f}")
        print(f"Dominance- CCC: {metrics['dominance_ccc']:.4f}, RMSE: {metrics['dominance_rmse']:.4f}")
        print(f"\nAverage CCC: {metrics['avg_ccc']:.4f}")
        print(f"Failed extractions: {len(failed_cases)}")


if __name__ == "__main__":
    main()
