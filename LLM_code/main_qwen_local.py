"""
Local Qwen inference + LoRA fine-tuning script for emotion recognition in conversations.
Loads Qwen3.5-35B-A3B directly from a local path using HuggingFace transformers.
No server required.
Supports zero-shot, few-shot, and lora modes.
Text-only input.

Dependencies:
    pip install transformers peft accelerate python-dotenv scikit-learn tqdm
"""

import os
import json
import argparse
import random
import math
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
import re
from dotenv import load_dotenv

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def get_labels_attr(dataset):
    label_list_set = {
        'iemocap': ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
        'meld': ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'],
        'EmoryNLP': ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared'],
        'dialydailog': ['happy', 'neutral', 'angry', 'sad', 'fear', 'surprise', 'disgust'],
    }
    label_str_set = {
        'iemocap': '"happy", "sad", "neutral", "angry", "excited", "frustrated"',
        'meld': '"neutral", "surprise", "fear", "sad", "joyful", "disgust", "angry"',
        'EmoryNLP': '"Joyful", "Mad", "Peaceful", "Neutral", "Sad", "Powerful", "Scared"',
        'dialydailog': '"happy", "neutral", "angry", "sad", "fear", "surprise", "disgust"',
    }
    emotional_label_dict = {t: i for i, t in enumerate(label_list_set[dataset])}
    return emotional_label_dict, label_str_set[dataset]


def get_few_shot_examples():
    return [
        {
            "input": """Speaker_0: "Guess what? I got the job!"
Speaker_1: "That's amazing! Congratulations!"

Target speech characteristics: high volume with high variation, high pitch with high variation, very high speaking rate.

For Speaker_1: "That's amazing! Congratulations!" """,
            "output": {
                "emotion_label": "excited",
                "reasoning": "The positive congratulations suggest happiness, but the audio features indicate high arousal: very high speaking rate, high volume variation, and high pitch variation all point to energetic enthusiasm rather than calm contentment. This is excited, not happy."
            }
        },
        {
            "input": """Speaker_0: "How's the job search going?"
Speaker_1: "Terrible. I've applied to 50 places and no one even calls back. The whole system is rigged if you don't know someone."

Target speech characteristics: moderate volume with high variation, moderate pitch with moderate variation, moderate speaking rate.

For Speaker_1: "Terrible. I've applied to 50 places and no one even calls back. The whole system is rigged if you don't know someone." """,
            "output": {
                "emotion_label": "frustrated",
                "reasoning": "The content expresses complaint about an unfair situation rather than loss or grief. The moderate-to-high audio variation indicates controlled but emphatic delivery characteristic of exasperation. This is frustrated rather than sad."
            }
        },
        {
            "input": """Speaker_0: "What time is the meeting?"
Speaker_1: "It's at 3pm in conference room B."

Target speech characteristics: moderate volume with low variation, moderate pitch with low variation, moderate speaking rate.

For Speaker_1: "It's at 3pm in conference room B." """,
            "output": {
                "emotion_label": "neutral",
                "reasoning": "The content is purely informational with no emotional valence. The audio features show moderate, stable delivery with low variation. This is truly neutral."
            }
        },
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def report_score(dataset, golds, preds, mode='test'):
    if dataset == 'iemocap':
        target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
        digits = 6
    elif dataset == 'meld':
        target_names = ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry']
        digits = 7
    elif dataset == 'EmoryNLP':
        target_names = ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared']
        digits = 7
    else:
        target_names = ['happy', 'neutral', 'angry', 'sad', 'fear', 'surprise', 'disgust']
        digits = 7

    res = {
        'Acc_SA': round(accuracy_score(golds, preds) * 100, 3),
        'F1_SA': round(f1_score(golds, preds, average='weighted') * 100, 3),
        'mode': mode,
    }
    res_matrix = metrics.classification_report(golds, preds, target_names=target_names, digits=digits)
    return res, res_matrix


# ---------------------------------------------------------------------------
# Label extraction helpers
# ---------------------------------------------------------------------------

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] if s1[i-1] == s2[j-1] else min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    return dp[m][n]


def optimize_output(output, label_set):
    return min(label_set, key=lambda l: edit_distance(output.lower(), l.lower()))


def extract_emotion_from_json(output, label_set):
    if not output or output.strip() == '':
        return None

    # Strip Qwen thinking block
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
    output_lower = output.lower()
    label_set_lower = [l.lower() for l in label_set]

    # Strategy 1: Parse JSON
    try:
        json_match = re.search(r'\{[^{}]*\}', output)
        if json_match:
            parsed = json.loads(json_match.group(0))
            for key in ['emotion_label', 'detected_emotion_label', 'emotion']:
                if key in parsed:
                    detected = parsed[key].strip().lower()
                    if detected in label_set_lower:
                        return detected
                    for label in label_set_lower:
                        if label in detected or detected in label:
                            return label
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass

    # Strategy 2: Regex
    for pattern in [
        r'"emotion_label"\s*:\s*"([^"]+)"',
        r'"detected_emotion_label"\s*:\s*"([^"]+)"',
        r'"emotion"\s*:\s*"([^"]+)"',
        r'emotion_label["\s:]+([a-zA-Z]+)',
    ]:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            detected = match.group(1).strip().lower()
            if detected in label_set_lower:
                return detected
            for label in label_set_lower:
                if label in detected or detected in label:
                    return label

    # Strategy 3: First occurring label
    positions = [(output_lower.find(l), l) for l in label_set_lower if output_lower.find(l) != -1]
    if positions:
        return sorted(positions)[0][1]

    # Strategy 4: Edit distance
    return optimize_output(output_lower, label_set_lower)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_data(file_path, percent=1.0, random_seed=42):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if percent < 1.0:
        random.seed(random_seed)
        data = random.sample(data, int(len(data) * percent))
    return data


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_input(input_text, dataset):
    conv_match = re.search(r'speakers\.\s*###(.+?)###', input_text, re.DOTALL)
    if not conv_match:
        conv_match = re.search(r'###\s*(\t?\s*Speaker_.+?)###', input_text, re.DOTALL)
    conversation_history = conv_match.group(1).strip() if conv_match else ""

    target_match = re.search(r'For\s+(Speaker_\d+:["\'][^"\']+["\'])', input_text)
    if target_match:
        target_utterance = target_match.group(1).replace('<', '').replace('>', '')
    else:
        utterances = re.findall(r'Speaker_\d+:["\'][^"\']+["\']', conversation_history)
        target_utterance = utterances[-1] if utterances else ""

    audio_features = ""
    for pattern in [
        r'Target speech characteristics:\s*([^.]+\.)',
        r'Audio description of target utterance:\s*([^.]+\.)',
    ]:
        m = re.search(pattern, input_text, re.IGNORECASE)
        if m:
            audio_features = m.group(1).strip()
            break

    if not audio_features:
        m = re.search(r'###\s*([^#]+?)(?:\n\n|For\s+Speaker)', input_text, re.DOTALL)
        if m:
            p = m.group(1).strip()
            if any(kw in p.lower() for kw in ['volume', 'pitch', 'speaking rate', 'variation']):
                audio_features = p

    if not audio_features:
        found = re.findall(r'\(([^)]*(?:pitch|volume|speaking rate)[^)]*)\)', input_text, re.IGNORECASE)
        if found:
            audio_features = "; ".join(found)

    return conversation_history, target_utterance, audio_features


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _system_message(label_str):
    return (
        f"Now you are expert of emotional analysis for dialogues. "
        f"Please select one emotion label word from [{label_str}] and always respond in strict json format."
    )


def _user_message_body(conversation_history, target_utterance, audio_features, label_str, examples_text=""):
    return f"""You will be analyzing a dialogue to identify the most dominant emotion expressed in a target utterance. You will need to consider both the conversation context and audio features (if available for the last three utterances) to make your determination.
{examples_text}
input:
Conversation history: {conversation_history}
Target utterance: {target_utterance}
Audio features: {audio_features if audio_features else "Not available"}

Your task is to select the single emotion that best represents the dominant emotion of the target utterance. You must choose from this list of emotions: [{label_str}]

Before providing your answer, use the scratchpad to think through your analysis:

Consider what is being said in the target utterance and how it relates to the conversation context
Analyze the audio features (such as pitch, tone, intensity, speaking rate) and what emotions they might indicate
Consider how the conversation context might influence the emotional state of the speaker
Weigh which single emotion from the list best captures the dominant feeling

Now provide your final answer in the following valid JSON format:
{{
"emotion_label": "your selected one emotion from the provided list here",
"reasoning": "your concise and clear explanation for why this emotion best represents the target utterance that references both the conversation context and audio features"
}}"""


def build_messages_zero_shot(conversation_history, target_utterance, audio_features, label_str):
    return [
        {"role": "system", "content": _system_message(label_str)},
        {"role": "user", "content": _user_message_body(conversation_history, target_utterance, audio_features, label_str)},
    ]


def build_messages_few_shot(conversation_history, target_utterance, audio_features, label_str):
    examples = get_few_shot_examples()
    examples_text = "\n\nHere are some examples to guide your analysis:\n"
    for i, ex in enumerate(examples, 1):
        examples_text += f"\n--- Example {i} ---\nInput:\n{ex['input']}\nOutput:\n{json.dumps(ex['output'], indent=2)}\n"
    examples_text += "\n--- Now analyze this dialogue ---\n"
    return [
        {"role": "system", "content": _system_message(label_str)},
        {"role": "user", "content": _user_message_body(conversation_history, target_utterance, audio_features, label_str, examples_text)},
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

class EmotionSFTDataset(Dataset):
    """
    Supervised fine-tuning dataset.
    Each item is a full tokenized sequence: [prompt + target JSON].
    Loss is computed only on the target (completion) tokens.
    """

    def __init__(self, samples, tokenizer, label_str, dataset_name,
                 max_length=2048, enable_thinking=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.items = []

        for sample in samples:
            input_text = sample['input']
            target_label = sample['target'].strip().lower()
            conv, utt, audio = parse_input(input_text, dataset_name)

            messages = build_messages_zero_shot(conv, utt, audio, label_str)
            prompt_text = apply_chat_template(tokenizer, messages, enable_thinking)

            # Build the expected completion
            completion = json.dumps({"emotion_label": target_label, "reasoning": ""})

            full_text = prompt_text + completion
            self.items.append((prompt_text, full_text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        prompt_text, full_text = self.items[idx]

        # Tokenize full sequence
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = full_enc["input_ids"][0]
        attention_mask = full_enc["attention_mask"][0]

        # Tokenize prompt only to find where completion starts
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        # Labels: -100 for prompt tokens (ignored in loss), real ids for completion
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


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

def train_lora(model, tokenizer, train_data, dev_data, args,
               label_str, emotional_label_dict, device):
    label_set = list(emotional_label_dict.keys())

    # Wrap model with LoRA
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

    # Dataset
    train_dataset = EmotionSFTDataset(
        train_data, tokenizer, label_str, args.dataset,
        max_length=args.max_train_length,
        enable_thinking=args.enable_thinking,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lora_lr,
        weight_decay=0.01,
    )
    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    best_f1 = -1.0
    adapter_save_path = os.path.join(args.output_dir, "lora_adapter")

    print(f"\n***** LoRA Training *****")
    print(f"Train samples: {len(train_dataset)} | Epochs: {args.num_train_epochs} | "
          f"Batch size: {args.train_batch_size} | Grad accum: {args.gradient_accumulation_steps} | "
          f"LR: {args.lora_lr}")

    global_step = 0
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
                global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} | Avg loss: {avg_loss:.4f}")

        # Evaluate on dev set after each epoch
        if dev_data:
            print(f"Evaluating on dev set...")
            f1 = evaluate(model, tokenizer, dev_data, args, label_str,
                          emotional_label_dict, label_set, device, split="dev")
            if f1 > best_f1:
                best_f1 = f1
                model.save_pretrained(adapter_save_path)
                tokenizer.save_pretrained(adapter_save_path)
                print(f"  -> New best F1: {best_f1:.3f} | Adapter saved to {adapter_save_path}")

    # If no dev set, save after last epoch
    if not dev_data:
        model.save_pretrained(adapter_save_path)
        tokenizer.save_pretrained(adapter_save_path)
        print(f"Adapter saved to {adapter_save_path}")

    print(f"\nLoRA training complete. Best dev F1: {best_f1:.3f}")
    return model, adapter_save_path


def evaluate(model, tokenizer, data, args, label_str,
             emotional_label_dict, label_set, device, split="test"):
    """Run inference on a split and return weighted F1."""
    model.eval()
    build_messages = (
        build_messages_few_shot if args.experiments_setting == 'few_shot'
        else build_messages_zero_shot
    )

    preds, golds = [], []
    all_outputs = []
    confuse_case, failed_extraction_case = [], []

    for idx, sample in enumerate(tqdm(data, desc=f"Evaluating [{split}]")):
        input_text = sample['input']
        target = sample['target'].lower()
        conv, utt, audio = parse_input(input_text, args.dataset)
        messages = build_messages(conv, utt, audio, label_str)

        output = run_local_inference(
            model, tokenizer, messages,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        if output is None:
            output = ""

        extracted_label = extract_emotion_from_json(output, label_set)
        if extracted_label is None:
            extracted_label = optimize_output(output, label_set)
            confuse_case.append(idx)
            failed_extraction_case.append(idx)

        golds.append(emotional_label_dict[target])
        preds.append(emotional_label_dict.get(extracted_label.lower(), 0))
        all_outputs.append({
            "index": idx,
            "input": input_text,
            "output": output,
            "target": target,
            "extracted_label": extracted_label,
        })

        if (idx + 1) % 50 == 0 or idx == 0:
            current_acc = sum(1 for g, p in zip(golds, preds) if g == p) / len(golds) * 100
            print(f"\n[{split}] Sample {idx+1}/{len(data)} | Running Acc: {current_acc:.2f}%")

    score, res_matrix = report_score(dataset=args.dataset, golds=golds, preds=preds, mode=split)

    out_path = os.path.join(args.output_dir, f"preds_for_eval_{split}.text")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(score))
        f.write(f'\n{res_matrix}')
        f.write(f'\nconfuse_case: {confuse_case}\n')
        f.write(f'\nThe num of confuse_case is: {len(confuse_case)}\n')
        f.write(f'\nfailed_extraction_case: {failed_extraction_case}\n')
        f.write(f'\nThe num of failed_extraction_case is: {len(failed_extraction_case)}\n')
        f.write(json.dumps(all_outputs, indent=5, ensure_ascii=False))

    print(f"\n[{split}] Accuracy: {score['Acc_SA']}% | Weighted F1: {score['F1_SA']}%")
    print(f"Classification Report:\n{res_matrix}")
    print(f"Results saved to: {out_path}")

    return score['F1_SA']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Local Qwen inference + LoRA fine-tuning for emotion recognition')

    # Core
    parser.add_argument('--dataset', type=str, required=True, choices=['iemocap', 'meld', 'EmoryNLP'],
                        help='Dataset to evaluate')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save results and adapter')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Local path to the downloaded Qwen model directory')
    parser.add_argument('--experiments_setting', type=str, default='zero_shot',
                        choices=['zero_shot', 'few_shot', 'lora'],
                        help='Experiment setting')

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
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha (scaling = lora_alpha / lora_r)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
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
                        help='Gradient accumulation steps (effective batch = batch_size * accum)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Fraction of total steps used for LR warmup')
    parser.add_argument('--max_train_length', type=int, default=2048,
                        help='Max token length for training sequences')
    parser.add_argument('--lora_adapter_path', type=str, default=None,
                        help='Path to a pre-trained LoRA adapter to load for inference '
                             '(skips training, runs evaluation only)')

    # Data
    parser.add_argument('--data_percent', type=float, default=1.0, help='Fraction of data to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

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

    emotional_label_dict, emotional_label_str = get_labels_attr(args.dataset)
    label_set = list(emotional_label_dict.keys())

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.lora_adapter_path:
        # Load base model + pre-trained LoRA adapter (inference only)
        print(f"Loading base model from: {args.model_path}")
        print(f"Loading LoRA adapter from: {args.lora_adapter_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch_dtype, device_map=args.device_map,
        )
        model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    else:
        print(f"Loading model from: {args.model_path}  (dtype={args.dtype}, device_map={args.device_map})")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch_dtype, device_map=args.device_map,
        )

    model.eval()
    print("Model loaded.")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    test_file = os.path.join(args.data_dir, "test.json")
    train_file = os.path.join(args.data_dir, "train.json")
    dev_file = os.path.join(args.data_dir, "dev.json")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    test_data = read_data(test_file, percent=args.data_percent, random_seed=args.seed)
    print(f"Loaded {len(test_data)} test samples")

    # ------------------------------------------------------------------
    # LoRA: train then evaluate
    # ------------------------------------------------------------------
    if args.experiments_setting == 'lora' and not args.lora_adapter_path:
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Train file required for lora mode: {train_file}")

        train_data = read_data(train_file, percent=args.data_percent, random_seed=args.seed)
        dev_data = read_data(dev_file, percent=args.data_percent, random_seed=args.seed) \
            if os.path.exists(dev_file) else []
        print(f"Loaded {len(train_data)} train samples, {len(dev_data)} dev samples")

        model, adapter_path = train_lora(
            model, tokenizer, train_data, dev_data, args,
            emotional_label_str, emotional_label_dict, device,
        )

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    setting_label = args.experiments_setting
    if args.lora_adapter_path:
        setting_label = 'lora (pre-trained adapter)'

    print(f"\n***** Running evaluation | setting={setting_label} | "
          f"thinking={'on' if args.enable_thinking else 'off'} *****\n")

    evaluate(
        model, tokenizer, test_data, args,
        emotional_label_str, emotional_label_dict, label_set,
        device, split="test",
    )

    # Save experiment config
    config_path = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
