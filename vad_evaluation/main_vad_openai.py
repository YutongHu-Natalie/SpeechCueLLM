"""
OpenAI API inference script for VAD (Valence-Arousal-Dominance) prediction.
Supports GPT-4o-mini, GPT-4o, and other OpenAI models.
Supports both zero-shot and few-shot modes.
"""

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import re
import time
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load environment variables from .env file
load_dotenv(override=True)


def concordance_correlation_coefficient(y_true, y_pred):
    """
    Compute Concordance Correlation Coefficient (CCC).
    CCC measures the agreement between two variables, considering both
    correlation and deviation from the 45-degree line.

    Range: [-1, 1], where 1 indicates perfect agreement.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    # Covariance
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    # CCC formula
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)

    return ccc


def compute_vad_metrics(golds_v, preds_v, golds_a, preds_a, golds_d, preds_d):
    """
    Compute comprehensive metrics for VAD prediction.

    Returns:
        Dictionary with metrics for each dimension and overall
    """
    metrics = {}

    dimensions = [
        ('valence', golds_v, preds_v),
        ('arousal', golds_a, preds_a),
        ('dominance', golds_d, preds_d)
    ]

    all_golds = []
    all_preds = []

    for dim_name, golds, preds in dimensions:
        golds = np.array(golds)
        preds = np.array(preds)

        all_golds.extend(golds)
        all_preds.extend(preds)

        # MSE
        mse = mean_squared_error(golds, preds)
        metrics[f'{dim_name}_mse'] = round(mse, 4)

        # RMSE
        rmse = np.sqrt(mse)
        metrics[f'{dim_name}_rmse'] = round(rmse, 4)

        # MAE
        mae = mean_absolute_error(golds, preds)
        metrics[f'{dim_name}_mae'] = round(mae, 4)

        # Pearson Correlation
        if len(set(golds)) > 1 and len(set(preds)) > 1:
            pearson_r, pearson_p = pearsonr(golds, preds)
            metrics[f'{dim_name}_pearson_r'] = round(pearson_r, 4)
            metrics[f'{dim_name}_pearson_p'] = round(pearson_p, 6)
        else:
            metrics[f'{dim_name}_pearson_r'] = 0.0
            metrics[f'{dim_name}_pearson_p'] = 1.0

        # CCC
        ccc = concordance_correlation_coefficient(golds, preds)
        metrics[f'{dim_name}_ccc'] = round(ccc, 4)

    # Overall metrics (average across dimensions)
    all_golds = np.array(all_golds)
    all_preds = np.array(all_preds)

    metrics['overall_mse'] = round(mean_squared_error(all_golds, all_preds), 4)
    metrics['overall_rmse'] = round(np.sqrt(metrics['overall_mse']), 4)
    metrics['overall_mae'] = round(mean_absolute_error(all_golds, all_preds), 4)

    # Average CCC across dimensions
    metrics['avg_ccc'] = round(
        (metrics['valence_ccc'] + metrics['arousal_ccc'] + metrics['dominance_ccc']) / 3, 4
    )

    return metrics


def get_few_shot_examples():
    """
    Returns few-shot examples for VAD prediction.
    These examples demonstrate the expected reasoning and output format.
    """
    examples = [
        # Example 1: High arousal, low valence (frustrated/angry)
        {
            "conversation_history": """Speaker_0: "I've been waiting in this line for two hours."
Speaker_1: "Sir, please be patient. We're doing our best."
Speaker_0: "This is ridiculous! I have places to be!"
""",
            "target_utterance": 'Speaker_0: "This is ridiculous! I have places to be!"',
            "audio_features": "high volume with high variation, high pitch with high variation, very high speaking rate",
            "answer": {"v_value": "2", "a_value": "4", "d_value": "3"}
        },
        # Example 2: Low arousal, moderate valence (calm/neutral)
        {
            "conversation_history": """Speaker_0: "What time is the meeting tomorrow?"
Speaker_1: "It's scheduled for 3pm in conference room B."
""",
            "target_utterance": 'Speaker_1: "It\'s scheduled for 3pm in conference room B."',
            "audio_features": "moderate volume with low variation, moderate pitch with low variation, moderate speaking rate",
            "answer": {"v_value": "3", "a_value": "2", "d_value": "3"}
        },
        # Example 3: High valence, high arousal (excited/happy)
        {
            "conversation_history": """Speaker_0: "I have some news about the job."
Speaker_1: "What is it? Did you hear back?"
Speaker_0: "I got it! They offered me the position!"
""",
            "target_utterance": 'Speaker_0: "I got it! They offered me the position!"',
            "audio_features": "high volume with moderate variation, high pitch with high variation, high speaking rate",
            "answer": {"v_value": "5", "a_value": "4", "d_value": "4"}
        },
    ]
    return examples


def read_vad_data(file_path, percent=1.0, random_seed=42):
    """
    Read VAD data from CSV or pickle file.

    Args:
        file_path: Path to data file (CSV or pickle)
        percent: Percentage of data to use
        random_seed: Random seed for sampling

    Returns:
        DataFrame with VAD data
    """
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
    """
    Build conversation history for a given utterance.

    Args:
        df: DataFrame with VAD data
        current_idx: Index of the target utterance
        window_size: Number of previous utterances to include

    Returns:
        Tuple of (conversation_history, target_utterance)
    """
    current_row = df.iloc[current_idx]
    dialogue_id = current_row['dialogue_id']
    turn_index = current_row['turn_index']

    # Get all utterances from the same dialogue
    dialogue_df = df[df['dialogue_id'] == dialogue_id].sort_values('turn_index')

    # Get the utterances before the current one (within window)
    start_idx = max(0, turn_index - window_size)
    history_df = dialogue_df[(dialogue_df['turn_index'] >= start_idx) & (dialogue_df['turn_index'] < turn_index)]

    # Build conversation history string
    history_lines = []
    for _, row in history_df.iterrows():
        speaker = f"Speaker_{row['speaker']}"
        content = row['content']
        history_lines.append(f'{speaker}: "{content}"')

    conversation_history = '\n'.join(history_lines)

    # Build target utterance
    target_speaker = f"Speaker_{current_row['speaker']}"
    target_content = current_row['content']
    target_utterance = f'{target_speaker}: "{target_content}"'

    return conversation_history, target_utterance


def get_audio_features_for_context(df, current_idx, num_utterances=3):
    """
    Get audio feature descriptions for the last N utterances.

    Args:
        df: DataFrame with VAD data
        current_idx: Index of the target utterance
        num_utterances: Number of utterances to get features for

    Returns:
        String describing audio features
    """
    # For now, return a placeholder since audio features need to be loaded separately
    # This can be enhanced to load from the audio features CSV
    return "Audio features not available for this sample."


def create_openai_messages_zero_shot(conversation_history, target_utterance, audio_features):
    """
    Create messages for OpenAI API call using zero-shot VAD prompt.
    """
    developer_message = """Now you are expert of evaluating emotions for dialogues based on Valence-Arousal-Dominance (VAD) Model. Please rate the emotion on a scale of 1-5 (only give integer) for each dimension."""

    user_message = f"""You will be analyzing a dialogue to evaluate the emotion expressed in a specific target utterance using the Valence-Arousal-Dominance (VAD) Model. The VAD model measures emotion across three dimensions:
* Valence: How positive or negative the emotion is (1 = very negative, 5 = very positive)
* Arousal: The intensity or energy level of the emotion (1 = very calm/passive, 5 = very excited/active)
* Dominance: The degree of control or power expressed (1 = very submissive/controlled, 5 = very dominant/in-control)

Input:
The conversation history leading up to the target utterance:
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

    messages = [
        {"role": "developer", "content": developer_message},
        {"role": "user", "content": user_message}
    ]

    return messages


def create_openai_messages_few_shot(conversation_history, target_utterance, audio_features):
    """
    Create messages for OpenAI API call using few-shot VAD prompt with examples.
    """
    developer_message = """Now you are expert of evaluating emotions for dialogues based on Valence-Arousal-Dominance (VAD) Model. Please rate the emotion on a scale of 1-5 (only give integer) for each dimension."""

    # Build few-shot examples
    examples = get_few_shot_examples()
    examples_text = "\n\nHere are some examples to guide your analysis:\n"

    for i, example in enumerate(examples, 1):
        examples_text += f"\n--- Example {i} ---\n"
        examples_text += f"Conversation history:\n{example['conversation_history']}\n"
        examples_text += f"Target utterance: {example['target_utterance']}\n"
        examples_text += f"Audio features: {example['audio_features']}\n"
        examples_text += f"<scratchpad>\n{example['scratchpad']}\n</scratchpad>\n"
        examples_text += f"<answer>\n{json.dumps(example['answer'])}\n</answer>\n"

    user_message = f"""You will be analyzing a dialogue to evaluate the emotion expressed in a specific target utterance using the Valence-Arousal-Dominance (VAD) Model. The VAD model measures emotion across three dimensions:
* Valence: How positive or negative the emotion is (1 = very negative, 5 = very positive)
* Arousal: The intensity or energy level of the emotion (1 = very calm/passive, 5 = very excited/active)
* Dominance: The degree of control or power expressed (1 = very submissive/controlled, 5 = very dominant/in-control)
{examples_text}

--- Now analyze this dialogue ---

Input:
The conversation history leading up to the target utterance:
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

    messages = [
        {"role": "developer", "content": developer_message},
        {"role": "user", "content": user_message}
    ]

    return messages


def extract_vad_from_output(output):
    """
    Extract VAD values from model output with multiple fallback strategies.

    Args:
        output: Raw model output string

    Returns:
        Tuple of (v_value, a_value, d_value) as floats, or (None, None, None) if extraction fails
    """
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

    # Strategy 2: Find JSON object anywhere in output
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

    # Strategy 3: Regex patterns for individual values
    v_match = re.search(r'"?v_value"?\s*:\s*"?(\d)"?', output, re.IGNORECASE)
    a_match = re.search(r'"?a_value"?\s*:\s*"?(\d)"?', output, re.IGNORECASE)
    d_match = re.search(r'"?d_value"?\s*:\s*"?(\d)"?', output, re.IGNORECASE)

    if v_match and a_match and d_match:
        try:
            v = int(v_match.group(1))
            a = int(a_match.group(1))
            d = int(d_match.group(1))
            return float(v), float(a), float(d)
        except ValueError:
            pass

    # Strategy 4: Look for patterns like "Valence: 3, Arousal: 4, Dominance: 2"
    val_match = re.search(r'valence[:\s]+(\d)', output, re.IGNORECASE)
    aro_match = re.search(r'arousal[:\s]+(\d)', output, re.IGNORECASE)
    dom_match = re.search(r'dominance[:\s]+(\d)', output, re.IGNORECASE)

    if val_match and aro_match and dom_match:
        try:
            v = int(val_match.group(1))
            a = int(aro_match.group(1))
            d = int(dom_match.group(1))
            return float(v), float(a), float(d)
        except ValueError:
            pass

    return None, None, None


def run_openai_inference(client, model_name, messages, max_retries=3):
    """
    Run inference using OpenAI API with retry logic.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=800,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API call failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"API call failed after {max_retries} attempts: {e}")
                return None


def main():
    parser = argparse.ArgumentParser(description='OpenAI API inference for VAD prediction')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to VAD data file (CSV, pickle, or JSONL)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save results')
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini',
                        help='OpenAI model name')
    parser.add_argument('--experiments_setting', type=str, default='zero_shot',
                        choices=['zero_shot', 'few_shot'],
                        help='Experiment setting')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key')
    parser.add_argument('--data_percent', type=float, default=1.0,
                        help='Percentage of data to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--window_size', type=int, default=12,
                        help='Conversation history window size')
    parser.add_argument('--batch_delay', type=float, default=0.1,
                        help='Delay between API calls')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize OpenAI client
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key required")

    client = OpenAI(api_key=api_key)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read data
    df = read_vad_data(args.data_file, percent=args.data_percent, random_seed=args.seed)
    print(f"Loaded {len(df)} samples")

    # Select prompt function
    if args.experiments_setting == 'few_shot':
        create_messages_fn = create_openai_messages_few_shot
        print("Using few-shot prompting")
    else:
        create_messages_fn = create_openai_messages_zero_shot
        print("Using zero-shot prompting")

    # Run inference
    all_outputs = []
    preds_v, preds_a, preds_d = [], [], []
    golds_v, golds_a, golds_d = [], [], []
    failed_cases = []

    print(f"\n***** Running VAD inference with {args.model_name} ({args.experiments_setting}) *****\n")

    for idx in tqdm(range(len(df)), desc="Processing"):
        row = df.iloc[idx]

        # Build conversation history
        conversation_history, target_utterance = build_conversation_history(df, idx, args.window_size)
        audio_features = get_audio_features_for_context(df, idx)

        # Create messages
        messages = create_messages_fn(conversation_history, target_utterance, audio_features)

        # Run inference
        output = run_openai_inference(client, args.model_name, messages)

        if output is None:
            output = ""

        # Extract VAD values
        v, a, d = extract_vad_from_output(output)

        # Get ground truth
        gold_v = float(row['valence'])
        gold_a = float(row['arousal'])
        gold_d = float(row['dominance'])

        # Handle extraction failures - use default value of 3
        if v is None or a is None or d is None:
            v = v or 3.0
            a = a or 3.0
            d = d or 3.0
            failed_cases.append(idx)

        # Clamp values to valid range
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
            "index": idx,
            "dialogue_id": row['dialogue_id'],
            "turn_id": row['turn_id'],
            "content": row['content'],
            "output": output,
            "pred_v": v,
            "pred_a": a,
            "pred_d": d,
            "gold_v": gold_v,
            "gold_a": gold_a,
            "gold_d": gold_d,
        })

        # Log progress
        if (idx + 1) % 50 == 0 or idx == 0:
            current_metrics = compute_vad_metrics(
                golds_v, preds_v, golds_a, preds_a, golds_d, preds_d
            )
            print(f"\n{'='*80}")
            print(f"Sample {idx + 1}/{len(df)}")
            print(f"Running Avg CCC: {current_metrics['avg_ccc']:.4f}")
            print(f"V_CCC: {current_metrics['valence_ccc']:.4f}, A_CCC: {current_metrics['arousal_ccc']:.4f}, D_CCC: {current_metrics['dominance_ccc']:.4f}")
            print(f"{'='*80}\n")

        if args.batch_delay > 0:
            time.sleep(args.batch_delay)

    # Compute final metrics
    metrics = compute_vad_metrics(golds_v, preds_v, golds_a, preds_a, golds_d, preds_d)

    # Save results
    preds_path = os.path.join(args.output_dir, "preds_for_eval.text")
    with open(preds_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("VAD PREDICTION RESULTS\n")
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

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save experiment config
    config_path = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "model_name": args.model_name,
            "experiments_setting": args.experiments_setting,
            "data_file": args.data_file,
            "data_percent": args.data_percent,
            "window_size": args.window_size,
            "seed": args.seed,
            "num_samples": len(df),
            "num_failed": len(failed_cases)
        }, f, indent=2)

    # Print results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nValence  - CCC: {metrics['valence_ccc']:.4f}, RMSE: {metrics['valence_rmse']:.4f}, MAE: {metrics['valence_mae']:.4f}")
    print(f"Arousal  - CCC: {metrics['arousal_ccc']:.4f}, RMSE: {metrics['arousal_rmse']:.4f}, MAE: {metrics['arousal_mae']:.4f}")
    print(f"Dominance- CCC: {metrics['dominance_ccc']:.4f}, RMSE: {metrics['dominance_rmse']:.4f}, MAE: {metrics['dominance_mae']:.4f}")
    print(f"\nAverage CCC: {metrics['avg_ccc']:.4f}")
    print(f"Overall RMSE: {metrics['overall_rmse']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
