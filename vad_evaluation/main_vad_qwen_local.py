"""
Local Qwen inference + LoRA fine-tuning script for VAD (Valence-Arousal-Dominance) prediction.
Loads Qwen3.5-35B-A3B directly from a local path using HuggingFace transformers.
No server required.
Supports zero-shot, few-shot, and lora modes.
Text-only input.

Dependencies:
    pip install transformers peft accelerate python-dotenv scikit-learn scipy pandas tqdm
"""

import os
import json
import argparse
import random
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
from dotenv import load_dotenv

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def concordance_correlation_coefficient(y_true, y_pred):
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
    metrics = {}
    dimensions = [
        ('valence', golds_v, preds_v),
        ('arousal', golds_a, preds_a),
        ('dominance', golds_d, preds_d),
    ]
    all_golds = []
    all_preds = []
    for dim_name, golds, preds in dimensions:
        golds = np.array(golds)
        preds = np.array(preds)
        all_golds.extend(golds)
        all_preds.extend(preds)
        mse = mean_squared_error(golds, preds)
        metrics[f'{dim_name}_mse'] = round(mse, 4)
        metrics[f'{dim_name}_rmse'] = round(np.sqrt(mse), 4)
        metrics[f'{dim_name}_mae'] = round(mean_absolute_error(golds, preds), 4)
        if len(set(golds)) > 1 and len(set(preds)) > 1:
            pearson_r, pearson_p = pearsonr(golds, preds)
            metrics[f'{dim_name}_pearson_r'] = round(pearson_r, 4)
            metrics[f'{dim_name}_pearson_p'] = round(pearson_p, 6)
        else:
            metrics[f'{dim_name}_pearson_r'] = 0.0
            metrics[f'{dim_name}_pearson_p'] = 1.0
        metrics[f'{dim_name}_ccc'] = round(concordance_correlation_coefficient(golds, preds), 4)

    all_golds = np.array(all_golds)
    all_preds = np.array(all_preds)
    metrics['overall_mse'] = round(mean_squared_error(all_golds, all_preds), 4)
    metrics['overall_rmse'] = round(np.sqrt(metrics['overall_mse']), 4)
    metrics['overall_mae'] = round(mean_absolute_error(all_golds, all_preds), 4)
    metrics['avg_ccc'] = round(
        (metrics['valence_ccc'] + metrics['arousal_ccc'] + metrics['dominance_ccc']) / 3, 4
    )
    return metrics


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

def get_few_shot_examples():
    return [
        {
            "conversation_history": 'Speaker_0: "I\'ve been waiting in this line for two hours."\nSpeaker_1: "Sir, please be patient. We\'re doing our best."\nSpeaker_0: "This is ridiculous! I have places to be!"',
            "target_utterance": 'Speaker_0: "This is ridiculous! I have places to be!"',
            "audio_features": "high volume with high variation, high pitch with high variation, very high speaking rate",
            "answer": {"v_value": "2", "a_value": "4", "d_value": "3"},
        },
        {
            "conversation_history": 'Speaker_0: "What time is the meeting tomorrow?"\nSpeaker_1: "It\'s scheduled for 3pm in conference room B."',
            "target_utterance": 'Speaker_1: "It\'s scheduled for 3pm in conference room B."',
            "audio_features": "moderate volume with low variation, moderate pitch with low variation, moderate speaking rate",
            "answer": {"v_value": "3", "a_value": "2", "d_value": "3"},
        },
        {
            "conversation_history": 'Speaker_0: "I have some news about the job."\nSpeaker_1: "What is it? Did you hear back?"\nSpeaker_0: "I got it! They offered me the position!"',
            "target_utterance": 'Speaker_0: "I got it! They offered me the position!"',
            "audio_features": "high volume with moderate variation, high pitch with high variation, high speaking rate",
            "answer": {"v_value": "5", "a_value": "4", "d_value": "4"},
        },
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_vad_data(file_path, percent=1.0, random_seed=42):
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
        df = df.sample(n=int(len(df) * percent), random_state=random_seed)

    return df


# ---------------------------------------------------------------------------
# Conversation builder
# ---------------------------------------------------------------------------

def build_conversation_history(df, current_idx, window_size=12,
                                include_history_vad=False,
                                predicted_vad_cache=None,
                                vad_context_size=None):
    current_row = df.iloc[current_idx]
    dialogue_id = current_row['dialogue_id']
    turn_index = current_row['turn_index']

    dialogue_df = df[df['dialogue_id'] == dialogue_id].sort_values('turn_index')
    start_idx = max(0, turn_index - window_size)
    history_df = dialogue_df[
        (dialogue_df['turn_index'] >= start_idx) & (dialogue_df['turn_index'] < turn_index)
    ]

    history_rows = list(history_df.iterrows())
    if include_history_vad and vad_context_size is not None:
        vad_annotated_indices = {row['turn_index'] for _, row in history_rows[-vad_context_size:]}
    else:
        vad_annotated_indices = None

    history_lines = []
    for _, row in history_rows:
        speaker = f"Speaker_{row['speaker']}"
        content = row['content']
        if include_history_vad:
            should_annotate = (vad_annotated_indices is None or row['turn_index'] in vad_annotated_indices)
            cache_key = (row['dialogue_id'], row['turn_index'])
            if should_annotate and predicted_vad_cache is not None and cache_key in predicted_vad_cache:
                pred_v, pred_a, pred_d = predicted_vad_cache[cache_key]
                vad_str = f" [VAD: V={int(round(pred_v))}, A={int(round(pred_a))}, D={int(round(pred_d))}]"
            else:
                vad_str = ""
            history_lines.append(f'{speaker}: "{content}"{vad_str}')
        else:
            history_lines.append(f'{speaker}: "{content}"')

    conversation_history = '\n'.join(history_lines)
    target_utterance = f'Speaker_{current_row["speaker"]}: "{current_row["content"]}"'
    return conversation_history, target_utterance


# ---------------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------------

def extract_vad_from_output(output):
    if not output or output.strip() == '':
        return None, None, None

    # Strip Qwen thinking block
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()

    # Strategy 1: <answer> tags (use last match to skip few-shot examples)
    answer_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', output, re.DOTALL)
    if answer_matches:
        try:
            parsed = json.loads(answer_matches[-1].strip())
            v = int(parsed.get('v_value', 3))
            a = int(parsed.get('a_value', 3))
            d = int(parsed.get('d_value', 3))
            return float(v), float(a), float(d)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 2: JSON object with v_value key
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

    # Strategy 3: Individual key-value regex
    v_match = re.search(r'"?v_value"?\s*:\s*"?(\d)"?', output, re.IGNORECASE)
    a_match = re.search(r'"?a_value"?\s*:\s*"?(\d)"?', output, re.IGNORECASE)
    d_match = re.search(r'"?d_value"?\s*:\s*"?(\d)"?', output, re.IGNORECASE)
    if v_match and a_match and d_match:
        try:
            return float(v_match.group(1)), float(a_match.group(1)), float(d_match.group(1))
        except ValueError:
            pass

    # Strategy 4: Natural language ("Valence: 3")
    val_match = re.search(r'valence[:\s]+(\d)', output, re.IGNORECASE)
    aro_match = re.search(r'arousal[:\s]+(\d)', output, re.IGNORECASE)
    dom_match = re.search(r'dominance[:\s]+(\d)', output, re.IGNORECASE)
    if val_match and aro_match and dom_match:
        try:
            return float(val_match.group(1)), float(aro_match.group(1)), float(dom_match.group(1))
        except ValueError:
            pass

    return None, None, None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _system_message():
    return (
        "Now you are expert of evaluating emotions for dialogues based on Valence-Arousal-Dominance (VAD) Model. "
        "Please rate the emotion on a scale of 1-5 (only give integer) for each dimension."
    )


def _user_message_body(conversation_history, target_utterance, audio_features,
                        include_history_vad=False, examples_text=""):
    history_vad_note = (
        "\nNote: VAD ratings [V=Valence, A=Arousal, D=Dominance] are provided for context utterances."
        if include_history_vad else ""
    )
    return f"""You will be analyzing a dialogue to evaluate the emotion expressed in a specific target utterance using the Valence-Arousal-Dominance (VAD) Model. The VAD model measures emotion across three dimensions:
* Valence: How positive or negative the emotion is (1 = very negative, 5 = very positive)
* Arousal: The intensity or energy level of the emotion (1 = very calm/passive, 5 = very excited/active)
* Dominance: The degree of control or power expressed (1 = very submissive/controlled, 5 = very dominant/in-control)
{examples_text}
Input:
The conversation history leading up to the target utterance:{history_vad_note}
{conversation_history}

Here is the target utterance you need to analyze:
{target_utterance}

Here are the audio features (if available) for the last three utterances in the conversation:
{audio_features}

Your task is to rate the emotion expressed in the target utterance on a scale of 1-5 for each of the three VAD dimensions. You must provide integer values only (no decimals).

Provide your final ratings in valid JSON format inside <answer> tags. The JSON must have exactly this structure:
{{ "v_value": "your integer rating for valence", "a_value": "your integer rating for arousal", "d_value": "your integer rating for dominance" }}

Remember:
* All ratings must be integers from 1 to 5
* Consider both conversational context and audio features (when available)
* The ratings should reflect the emotion in the target utterance specifically, not the overall conversation"""


def build_messages_zero_shot(conversation_history, target_utterance, audio_features,
                              include_history_vad=False):
    return [
        {"role": "system", "content": _system_message()},
        {"role": "user", "content": _user_message_body(
            conversation_history, target_utterance, audio_features, include_history_vad
        )},
    ]


def build_messages_few_shot(conversation_history, target_utterance, audio_features,
                             include_history_vad=False):
    examples = get_few_shot_examples()
    examples_text = "\n\nHere are some examples to guide your analysis:\n"
    for i, ex in enumerate(examples, 1):
        examples_text += f"\n--- Example {i} ---\n"
        examples_text += f"Conversation history:\n{ex['conversation_history']}\n"
        examples_text += f"Target utterance: {ex['target_utterance']}\n"
        examples_text += f"Audio features: {ex['audio_features']}\n"
        examples_text += f"<answer>\n{json.dumps(ex['answer'])}\n</answer>\n"
    examples_text += "\n--- Now analyze this dialogue ---\n"
    return [
        {"role": "system", "content": _system_message()},
        {"role": "user", "content": _user_message_body(
            conversation_history, target_utterance, audio_features,
            include_history_vad, examples_text
        )},
    ]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def apply_chat_template(tokenizer, messages, enable_thinking=False):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def run_local_inference(model, tokenizer, messages, enable_thinking=False,
                        max_new_tokens=512, device="cuda"):
    text = apply_chat_template(tokenizer, messages, enable_thinking)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.5,
        )

    new_tokens = generated_ids[0][model_inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# LoRA dataset
# ---------------------------------------------------------------------------

class VADSFTDataset(Dataset):
    """
    Supervised fine-tuning dataset for VAD regression.
    Each item is a full tokenized sequence: [prompt + target JSON inside <answer> tags].
    Loss is computed only on the completion tokens.
    """

    def __init__(self, df, tokenizer, max_length=2048, enable_thinking=False,
                 window_size=12, include_history_vad=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.items = []

        for idx in range(len(df)):
            row = df.iloc[idx]
            # Build input
            conversation_history, target_utterance = build_conversation_history(
                df, idx, window_size, include_history_vad
            )
            audio_features = "Audio features not available for this sample."
            messages = build_messages_zero_shot(
                conversation_history, target_utterance, audio_features, include_history_vad
            )
            prompt_text = apply_chat_template(tokenizer, messages, enable_thinking)

            # Build expected completion
            v = int(round(float(row['valence'])))
            a = int(round(float(row['arousal'])))
            d = int(round(float(row['dominance'])))
            completion = f'<answer>\n{json.dumps({"v_value": str(v), "a_value": str(a), "d_value": str(d)})}\n</answer>'

            self.items.append((prompt_text, prompt_text + completion))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        prompt_text, full_text = self.items[idx]

        full_enc = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = full_enc["input_ids"][0]
        attention_mask = full_enc["attention_mask"][0]

        prompt_enc = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn(batch, pad_token_id):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# LoRA training
# ---------------------------------------------------------------------------

def train_lora(model, tokenizer, train_df, dev_df, args, device):
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = VADSFTDataset(
        train_df, tokenizer,
        max_length=args.max_train_length,
        enable_thinking=args.enable_thinking,
        window_size=args.window_size,
        include_history_vad=args.include_history_vad,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lora_lr,
        weight_decay=0.01,
    )
    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    best_ccc = -float('inf')
    adapter_save_path = os.path.join(args.output_dir, "lora_adapter")

    print(f"\n***** LoRA Training *****")
    print(f"Train samples: {len(train_dataset)} | Epochs: {args.num_train_epochs} | "
          f"Batch size: {args.train_batch_size} | Grad accum: {args.gradient_accumulation_steps} | "
          f"LR: {args.lora_lr}")

    optimizer.zero_grad()

    for epoch in range(args.num_train_epochs):
        epoch_loss = 0.0
        model.train()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} | Avg loss: {avg_loss:.4f}")

        if dev_df is not None and len(dev_df) > 0:
            print("Evaluating on dev set...")
            metrics = evaluate(model, tokenizer, dev_df, args, device, split="dev")
            avg_ccc = metrics['avg_ccc']
            if avg_ccc > best_ccc:
                best_ccc = avg_ccc
                model.save_pretrained(adapter_save_path)
                tokenizer.save_pretrained(adapter_save_path)
                print(f"  -> New best avg CCC: {best_ccc:.4f} | Adapter saved to {adapter_save_path}")

    if dev_df is None or len(dev_df) == 0:
        model.save_pretrained(adapter_save_path)
        tokenizer.save_pretrained(adapter_save_path)
        print(f"Adapter saved to {adapter_save_path}")

    print(f"\nLoRA training complete. Best dev avg CCC: {best_ccc:.4f}")
    return model, adapter_save_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, df, args, device, split="test"):
    model.eval()

    build_messages = (
        build_messages_few_shot if args.experiments_setting == 'few_shot'
        else build_messages_zero_shot
    )

    preds_v, preds_a, preds_d = [], [], []
    golds_v, golds_a, golds_d = [], [], []
    all_outputs = []
    failed_cases = []
    predicted_vad_cache = {}

    for idx in tqdm(range(len(df)), desc=f"Evaluating [{split}]"):
        row = df.iloc[idx]

        conversation_history, target_utterance = build_conversation_history(
            df, idx, args.window_size, args.include_history_vad,
            predicted_vad_cache=predicted_vad_cache if args.include_history_vad else None,
            vad_context_size=args.vad_context_size,
        )
        audio_features = "Audio features not available for this sample."

        messages = build_messages(
            conversation_history, target_utterance, audio_features, args.include_history_vad
        )
        output = run_local_inference(
            model, tokenizer, messages,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        if output is None:
            output = ""

        v, a, d = extract_vad_from_output(output)
        gold_v = float(row['valence'])
        gold_a = float(row['arousal'])
        gold_d = float(row['dominance'])

        if v is None or a is None or d is None:
            v = v or 3.0
            a = a or 3.0
            d = d or 3.0
            failed_cases.append(idx)

        v = max(1.0, min(5.0, v))
        a = max(1.0, min(5.0, a))
        d = max(1.0, min(5.0, d))

        predicted_vad_cache[(row['dialogue_id'], row['turn_index'])] = (v, a, d)

        preds_v.append(v)
        preds_a.append(a)
        preds_d.append(d)
        golds_v.append(gold_v)
        golds_a.append(gold_a)
        golds_d.append(gold_d)

        all_outputs.append({
            "index": idx,
            "dialogue_id": row['dialogue_id'],
            "turn_id": row.get('turn_id', row['turn_index']),
            "content": row['content'],
            "output": output,
            "pred_v": v, "pred_a": a, "pred_d": d,
            "gold_v": gold_v, "gold_a": gold_a, "gold_d": gold_d,
        })

        if (idx + 1) % 50 == 0 or idx == 0:
            running_metrics = compute_vad_metrics(
                golds_v, preds_v, golds_a, preds_a, golds_d, preds_d
            )
            print(f"\n[{split}] Sample {idx+1}/{len(df)} | "
                  f"Avg CCC: {running_metrics['avg_ccc']:.4f} | "
                  f"V_CCC: {running_metrics['valence_ccc']:.4f}, "
                  f"A_CCC: {running_metrics['arousal_ccc']:.4f}, "
                  f"D_CCC: {running_metrics['dominance_ccc']:.4f}")

    metrics = compute_vad_metrics(golds_v, preds_v, golds_a, preds_a, golds_d, preds_d)

    # Save results
    out_path = os.path.join(args.output_dir, f"preds_for_eval_{split}.text")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"VAD PREDICTION RESULTS [{split}]\n")
        f.write("=" * 80 + "\n\n")
        f.write("METRICS BY DIMENSION:\n")
        f.write("-" * 40 + "\n")
        for dim in ['valence', 'arousal', 'dominance']:
            f.write(f"\n{dim.upper()}:\n")
            f.write(f"  MSE:  {metrics[f'{dim}_mse']:.4f}\n")
            f.write(f"  RMSE: {metrics[f'{dim}_rmse']:.4f}\n")
            f.write(f"  MAE:  {metrics[f'{dim}_mae']:.4f}\n")
            f.write(f"  CCC:  {metrics[f'{dim}_ccc']:.4f}\n")
            f.write(f"  Pearson r: {metrics[f'{dim}_pearson_r']:.4f} (p={metrics[f'{dim}_pearson_p']:.6f})\n")
        f.write("\n" + "-" * 40 + "\n")
        f.write("OVERALL METRICS:\n")
        f.write(f"  Overall MSE:  {metrics['overall_mse']:.4f}\n")
        f.write(f"  Overall RMSE: {metrics['overall_rmse']:.4f}\n")
        f.write(f"  Overall MAE:  {metrics['overall_mae']:.4f}\n")
        f.write(f"  Average CCC:  {metrics['avg_ccc']:.4f}\n")
        f.write("\n" + "-" * 40 + "\n")
        f.write(f"Failed extraction cases: {len(failed_cases)}\n")
        f.write(f"Failed indices: {failed_cases[:50]}{'...' if len(failed_cases) > 50 else ''}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED OUTPUTS:\n")
        f.write("=" * 80 + "\n\n")
        f.write(json.dumps(all_outputs, indent=2, ensure_ascii=False))

    print(f"\n[{split}] Valence  CCC: {metrics['valence_ccc']:.4f}  RMSE: {metrics['valence_rmse']:.4f}  MAE: {metrics['valence_mae']:.4f}")
    print(f"[{split}] Arousal  CCC: {metrics['arousal_ccc']:.4f}  RMSE: {metrics['arousal_rmse']:.4f}  MAE: {metrics['arousal_mae']:.4f}")
    print(f"[{split}] Dominance CCC: {metrics['dominance_ccc']:.4f}  RMSE: {metrics['dominance_rmse']:.4f}  MAE: {metrics['dominance_mae']:.4f}")
    print(f"[{split}] Average CCC: {metrics['avg_ccc']:.4f}")
    print(f"Results saved to: {out_path}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Local Qwen inference + LoRA fine-tuning for VAD prediction'
    )

    # Core
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to VAD data file (CSV, pickle, or JSONL)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save results and adapter')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Local path to the downloaded Qwen model directory')
    parser.add_argument('--experiments_setting', type=str, default='zero_shot',
                        choices=['zero_shot', 'few_shot', 'lora'],
                        help='Experiment setting')

    # Data
    parser.add_argument('--test_session', type=int, default=5,
                        help='Session to use as test set (1-5). Only this session will be evaluated.')
    parser.add_argument('--window_size', type=int, default=12,
                        help='Conversation history window size')
    parser.add_argument('--include_history_vad', action='store_true',
                        help='Include predicted VAD values for utterances in conversation history')
    parser.add_argument('--vad_context_size', type=int, default=None,
                        help='When --include_history_vad is set, only attach predicted VAD to the '
                             'last N utterances in history. Default None attaches to all.')
    parser.add_argument('--data_percent', type=float, default=1.0,
                        help='Fraction of data to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Inference
    parser.add_argument('--enable_thinking', action='store_true',
                        help='Enable Qwen thinking mode (chain-of-thought). Off by default.')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Max new tokens to generate per sample during inference')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['auto', 'float16', 'bfloat16', 'float32'],
                        help='Model dtype for loading')
    parser.add_argument('--device_map', type=str, default='auto',
                        help='Device map for model loading (e.g. "auto", "cuda:0")')

    # LoRA fine-tuning
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str,
                        default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
                        help='Comma-separated list of module names to apply LoRA to')
    parser.add_argument('--lora_lr', type=float, default=3e-4,
                        help='Learning rate for LoRA training')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                        help='Number of training epochs (lora mode)')
    parser.add_argument('--train_batch_size', type=int, default=1,
                        help='Per-device training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Fraction of total steps used for LR warmup')
    parser.add_argument('--max_train_length', type=int, default=2048,
                        help='Max token length for training sequences')
    parser.add_argument('--lora_adapter_path', type=str, default=None,
                        help='Path to a pre-trained LoRA adapter to load for inference '
                             '(skips training, runs evaluation only)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype_map = {
        'auto': 'auto',
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    device = 'cuda' if (args.device_map != 'cpu' and torch.cuda.is_available()) else 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.lora_adapter_path:
        print(f"Loading base model from: {args.model_path}")
        print(f"Loading LoRA adapter from: {args.lora_adapter_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch_dtype, device_map=args.device_map,
        )
        model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    else:
        print(f"Loading model from: {args.model_path}  (dtype={args.dtype}, device_map={args.device_map})")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch_dtype, device_map=args.device_map,
        )

    model.eval()
    print("Model loaded.")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df_all = read_vad_data(args.data_file, percent=args.data_percent, random_seed=args.seed)
    print(f"Loaded {len(df_all)} total samples")

    # Filter test split
    test_df = df_all[df_all['session'] == args.test_session].reset_index(drop=True)
    print(f"Filtered to Session {args.test_session} (test set): {len(test_df)} samples")

    before_nan = len(test_df)
    test_df = test_df.dropna(subset=['valence', 'arousal', 'dominance']).reset_index(drop=True)
    if len(test_df) < before_nan:
        print(f"Removed {before_nan - len(test_df)} NaN VAD samples. Remaining: {len(test_df)}")

    # ------------------------------------------------------------------
    # LoRA: train then evaluate
    # ------------------------------------------------------------------
    if args.experiments_setting == 'lora' and not args.lora_adapter_path:
        # Use all non-test sessions for training, leaving session 4 as dev
        dev_session = args.test_session - 1 if args.test_session > 1 else args.test_session + 1
        train_df = df_all[
            (df_all['session'] != args.test_session) & (df_all['session'] != dev_session)
        ].dropna(subset=['valence', 'arousal', 'dominance']).reset_index(drop=True)
        dev_df = df_all[
            df_all['session'] == dev_session
        ].dropna(subset=['valence', 'arousal', 'dominance']).reset_index(drop=True)

        print(f"Train samples: {len(train_df)} | Dev samples: {len(dev_df)} (Session {dev_session})")

        model, adapter_path = train_lora(model, tokenizer, train_df, dev_df, args, device)

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    setting_label = args.experiments_setting
    if args.lora_adapter_path:
        setting_label = 'lora (pre-trained adapter)'

    print(f"\n***** Running VAD evaluation | setting={setting_label} | "
          f"thinking={'on' if args.enable_thinking else 'off'} *****\n")

    metrics = evaluate(model, tokenizer, test_df, args, device, split="test")

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save experiment config
    config_path = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
