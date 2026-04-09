"""
Local Qwen inference script for emotion recognition in conversations.
Loads Qwen3.5-35B-A3B directly from a local path using HuggingFace transformers.
No server required.
Supports zero-shot and few-shot modes.
Text-only input.
"""

import os
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
import re
import time
from dotenv import load_dotenv

load_dotenv(override=True)


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

    emotional_label_dict = {text_label: num_label for num_label, text_label in enumerate(label_list_set[dataset])}
    emotional_label_str = label_str_set[dataset]
    return emotional_label_dict, emotional_label_str


def get_few_shot_examples():
    examples = [
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
    return examples


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

    res = {}
    res['Acc_SA'] = accuracy_score(golds, preds)
    res['F1_SA'] = f1_score(golds, preds, average='weighted')
    res['mode'] = mode
    for k, v in res.items():
        if isinstance(v, float):
            res[k] = round(v * 100, 3)

    res_matrix = metrics.classification_report(golds, preds, target_names=target_names, digits=digits)
    return res, res_matrix


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]


def optimize_output(output, label_set):
    min_distance = float('inf')
    optimized_output = None
    for label in label_set:
        distance = edit_distance(output.lower(), label.lower())
        if distance < min_distance:
            min_distance = distance
            optimized_output = label
    return optimized_output


def extract_emotion_from_json(output, label_set):
    """
    Extract emotion label from model output with multiple fallback strategies.
    Strips Qwen thinking blocks (<think>...</think>) before parsing.
    """
    if not output or output.strip() == '':
        return None

    # Strip thinking block if present
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()

    output_lower = output.lower()
    label_set_lower = [l.lower() for l in label_set]

    # Strategy 1: Parse as valid JSON
    try:
        json_match = re.search(r'\{[^{}]*\}', output)
        if json_match:
            parsed = json.loads(json_match.group(0))
            for key in ['emotion_label', 'detected_emotion_label', 'emotion']:
                if key in parsed:
                    detected_label = parsed[key].strip().lower()
                    if detected_label in label_set_lower:
                        return detected_label
                    for label in label_set_lower:
                        if label in detected_label or detected_label in label:
                            return label
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass

    # Strategy 2: Regex extraction from JSON-like structure
    for pattern in [
        r'"emotion_label"\s*:\s*"([^"]+)"',
        r'"detected_emotion_label"\s*:\s*"([^"]+)"',
        r'"emotion"\s*:\s*"([^"]+)"',
        r'emotion_label["\s:]+([a-zA-Z]+)',
    ]:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            detected_label = match.group(1).strip().lower()
            if detected_label in label_set_lower:
                return detected_label
            for label in label_set_lower:
                if label in detected_label or detected_label in label:
                    return label

    # Strategy 3: First occurring valid label in output
    label_positions = [(output_lower.find(l), l) for l in label_set_lower if output_lower.find(l) != -1]
    if label_positions:
        return sorted(label_positions)[0][1]

    # Strategy 4: Edit distance fallback
    return optimize_output(output_lower, label_set_lower)


def read_data(file_path, percent=1.0, random_seed=42):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    if percent < 1.0:
        random.seed(random_seed)
        data = random.sample(data, int(len(data) * percent))

    return data


def parse_input(input_text, dataset):
    conv_match = re.search(r'speakers\.\s*###(.+?)###', input_text, re.DOTALL)
    if not conv_match:
        conv_match = re.search(r'###\s*(\t?\s*Speaker_.+?)###', input_text, re.DOTALL)
    conversation_history = conv_match.group(1).strip() if conv_match else ""

    target_match = re.search(r'For\s+(Speaker_\d+:["\'][^"\']+["\'])', input_text)
    if target_match:
        target_utterance = target_match.group(1).replace('<', '').replace('>', '')
    else:
        speaker_utterances = re.findall(r'Speaker_\d+:["\'][^"\']+["\']', conversation_history)
        target_utterance = speaker_utterances[-1] if speaker_utterances else ""

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


def build_messages_zero_shot(conversation_history, target_utterance, audio_features, label_str):
    system_message = (
        f"Now you are expert of emotional analysis for dialogues. "
        f"Please select one emotion label word from [{label_str}] and always respond in strict json format."
    )
    user_message = f"""You will be analyzing a dialogue to identify the most dominant emotion expressed in a target utterance. You will need to consider both the conversation context and audio features (if available for the last three utterances) to make your determination.

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

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def build_messages_few_shot(conversation_history, target_utterance, audio_features, label_str):
    system_message = (
        f"Now you are expert of emotional analysis for dialogues. "
        f"Please select one emotion label word from [{label_str}] and always respond in strict json format."
    )

    examples = get_few_shot_examples()
    examples_text = "\n\nHere are some examples to guide your analysis:\n"
    for i, ex in enumerate(examples, 1):
        examples_text += f"\n--- Example {i} ---\nInput:\n{ex['input']}\nOutput:\n{json.dumps(ex['output'], indent=2)}\n"

    user_message = f"""You will be analyzing a dialogue to identify the most dominant emotion expressed in a target utterance. You will need to consider both the conversation context and audio features (if available for the last three utterances) to make your determination.
{examples_text}

--- Now analyze this dialogue ---

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

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def run_local_inference(model, tokenizer, messages, enable_thinking=False,
                        max_new_tokens=4096, device="cuda"):
    """
    Run a single inference pass using the locally loaded model.
    """
    # Apply chat template — enable_thinking controls whether the model
    # produces a <think>...</think> block before the JSON answer.
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
    except TypeError:
        # Older tokenizer versions may not accept chat_template_kwargs
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.5,  # analogous to presence_penalty in API
        )

    # Decode only the newly generated tokens
    new_tokens = generated_ids[0][model_inputs.input_ids.shape[1]:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return output


def main():
    parser = argparse.ArgumentParser(description='Local Qwen inference for emotion recognition')
    parser.add_argument('--dataset', type=str, required=True, choices=['iemocap', 'meld', 'EmoryNLP'],
                        help='Dataset to evaluate')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save results')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Local path to the downloaded Qwen model directory')
    parser.add_argument('--experiments_setting', type=str, default='zero_shot',
                        choices=['zero_shot', 'few_shot'],
                        help='Experiment setting: zero_shot or few_shot')
    parser.add_argument('--enable_thinking', action='store_true',
                        help='Enable Qwen thinking mode (chain-of-thought). '
                             'Disabled by default for faster, direct responses.')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum new tokens to generate per sample')
    parser.add_argument('--dtype', type=str, default='auto',
                        choices=['auto', 'float16', 'bfloat16', 'float32'],
                        help='Model dtype. "auto" lets transformers pick the best option.')
    parser.add_argument('--device_map', type=str, default='auto',
                        help='Device map for model loading (e.g. "auto", "cuda:0", "cpu")')
    parser.add_argument('--data_percent', type=float, default=1.0, help='Percentage of data to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Resolve dtype
    dtype_map = {
        'auto': 'auto',
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Determine primary device for input tensors
    if args.device_map == 'cpu':
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Loading model from: {args.model_path}  (dtype={args.dtype}, device_map={args.device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )
    model.eval()
    print("Model loaded.")

    os.makedirs(args.output_dir, exist_ok=True)

    emotional_label_dict, emotional_label_str = get_labels_attr(args.dataset)
    label_set = list(emotional_label_dict.keys())

    test_file = os.path.join(args.data_dir, "test.json")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    test_data = read_data(test_file, percent=args.data_percent, random_seed=args.seed)
    print(f"Loaded {len(test_data)} test samples")

    build_messages = (
        build_messages_few_shot if args.experiments_setting == 'few_shot'
        else build_messages_zero_shot
    )
    print(f"Setting: {args.experiments_setting} | Thinking: {'on' if args.enable_thinking else 'off'}")

    all_outputs = []
    preds = []
    golds = []
    confuse_case = []
    failed_extraction_case = []

    print(f"\n***** Running local inference with {args.model_path} *****\n")

    for idx, sample in enumerate(tqdm(test_data, desc="Processing")):
        input_text = sample['input']
        target = sample['target'].lower()

        conversation_history, target_utterance, audio_features = parse_input(input_text, args.dataset)
        messages = build_messages(conversation_history, target_utterance, audio_features, emotional_label_str)

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
            print(f"\n{'='*80}")
            print(f"Sample {idx + 1}/{len(test_data)} | Running Accuracy: {current_acc:.2f}%")
            print(f"{'='*80}")
            print(f"Target Sentence: {target_utterance}")
            print(f"Target Emotion: {target}")
            print(f"Predicted Emotion: {extracted_label}")
            print(f"Model Output: {output[:300]}{'...' if len(output) > 300 else ''}")
            print(f"{'='*80}\n")

    score, res_matrix = report_score(dataset=args.dataset, golds=golds, preds=preds)

    preds_for_eval_path = os.path.join(args.output_dir, "preds_for_eval.text")
    with open(preds_for_eval_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(score))
        f.write(f'\n{res_matrix}')
        f.write(f'\nconfuse_case: {confuse_case}\n')
        f.write(f'\nThe num of confuse_case is: {len(confuse_case)}\n')
        f.write(f'\nfailed_extraction_case: {failed_extraction_case}\n')
        f.write(f'\nThe num of failed_extraction_case is: {len(failed_extraction_case)}\n')
        f.write(json.dumps(all_outputs, indent=5, ensure_ascii=False))

    config_path = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_path": args.model_path,
            "dataset": args.dataset,
            "experiments_setting": args.experiments_setting,
            "enable_thinking": args.enable_thinking,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "data_percent": args.data_percent,
            "seed": args.seed,
            "num_samples": len(test_data),
        }, f, indent=2)

    print("\n***** Results *****")
    print(f"Accuracy: {score['Acc_SA']}%")
    print(f"Weighted F1: {score['F1_SA']}%")
    print(f"\nClassification Report:\n{res_matrix}")
    print(f"\nResults saved to: {preds_for_eval_path}")


if __name__ == "__main__":
    main()
