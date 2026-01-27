"""
OpenAI API inference script for emotion recognition in conversations.
Supports GPT-4o-mini, GPT-4o, and other OpenAI models.
Supports both zero-shot and few-shot modes.
"""

import os
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
import re
import time
from dotenv import load_dotenv

# Load environment variables from .env file (override=True to override existing env vars)
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
    """
    Returns few-shot examples for emotion classification.
    These examples help clarify distinctions between similar emotions.
    """
    examples = [
        # Example 1: EXCITED vs HAPPY
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
        # Example 2: FRUSTRATED vs SAD
        {
            "input": """Speaker_0: "How's the job search going?"
Speaker_1: "Terrible. I've applied to 50 places and no one even calls back. The whole system is rigged if you don't know someone."

Target speech characteristics: moderate volume with high variation, moderate pitch with moderate variation, moderate speaking rate.

For Speaker_1: "Terrible. I've applied to 50 places and no one even calls back. The whole system is rigged if you don't know someone." """,
            "output": {
                "emotion_label": "frustrated",
                "reasoning": "The content expresses complaint about an unfair situation rather than loss or grief. The moderate-to-high audio variation indicates controlled but emphatic delivery characteristic of exasperation. This is frustrated (irritation at obstacles) rather than sad (dejection or sorrow)."
            }
        },
        # Example 3: NEUTRAL (true neutral)
        {
            "input": """Speaker_0: "What time is the meeting?"
Speaker_1: "It's at 3pm in conference room B."

Target speech characteristics: moderate volume with low variation, moderate pitch with low variation, moderate speaking rate.

For Speaker_1: "It's at 3pm in conference room B." """,
            "output": {
                "emotion_label": "neutral",
                "reasoning": "The content is purely informational with no emotional valence. The audio features show moderate, stable delivery with low variation, indicating matter-of-fact information sharing. This is truly neutral - not low-arousal happiness or subdued sadness, but absence of emotion."
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
    """Calculate the editing distance between two strings"""
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
    """Return the label with minimum edit distance to output"""
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
    Extract emotion label from JSON-formatted output with multiple fallback strategies.
    """
    if not output or output.strip() == '':
        return None

    output_lower = output.lower()
    label_set_lower = [l.lower() for l in label_set]

    # Strategy 1: Try to parse as valid JSON
    try:
        # Find JSON object in the output
        json_match = re.search(r'\{[^{}]*\}', output)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            # Check for emotion_label field
            for key in ['emotion_label', 'detected_emotion_label', 'emotion']:
                if key in parsed:
                    detected_label = parsed[key].strip().lower()
                    if detected_label in label_set_lower:
                        return detected_label
                    # Try to find closest match
                    for label in label_set_lower:
                        if label in detected_label or detected_label in label:
                            return label
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass

    # Strategy 2: Use regex to extract from JSON-like structure
    regex_patterns = [
        r'"emotion_label"\s*:\s*"([^"]+)"',
        r'"detected_emotion_label"\s*:\s*"([^"]+)"',
        r'"emotion"\s*:\s*"([^"]+)"',
        r'emotion_label["\s:]+([a-zA-Z]+)',
    ]

    for pattern in regex_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            detected_label = match.group(1).strip().lower()
            if detected_label in label_set_lower:
                return detected_label
            for label in label_set_lower:
                if label in detected_label or detected_label in label:
                    return label

    # Strategy 3: Search for any valid emotion label appearing first in output
    label_positions = []
    for label in label_set_lower:
        pos = output_lower.find(label)
        if pos != -1:
            label_positions.append((pos, label))

    if label_positions:
        label_positions.sort(key=lambda x: x[0])
        return label_positions[0][1]

    # Strategy 4: Fall back to edit distance
    return optimize_output(output_lower, label_set_lower)


def read_data(file_path, percent=1.0, random_seed=42):
    """Read data from JSON file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    if percent < 1.0:
        random.seed(random_seed)
        n_samples = int(len(data) * percent)
        data = random.sample(data, n_samples)

    return data


def parse_input_for_openai(input_text, dataset):
    """
    Parse the processed input text to extract conversation_history, target_utterance, and audio_features
    for the OpenAI prompt format.

    Input format example:
    "Now you are expert... speakers. ### \t Speaker_0:"text" ### Target speech characteristics: ... For Speaker_0:"text", ..."
    """
    # Extract the conversation between ### ### markers
    # Note: The text contains "'### ###'" as description, so we need to find the ACTUAL markers
    # The actual conversation starts after "speakers." followed by ### and Speaker_
    conv_match = re.search(r'speakers\.\s*###(.+?)###', input_text, re.DOTALL)
    if not conv_match:
        # Fallback: find ### followed by Speaker content
        conv_match = re.search(r'###\s*(\t?\s*Speaker_.+?)###', input_text, re.DOTALL)
    conversation_history = ""
    if conv_match:
        conversation_history = conv_match.group(1).strip()

    # Extract target utterance (from "For Speaker_X:..." pattern)
    target_match = re.search(r'For\s+(Speaker_\d+:["\'][^"\']+["\'])', input_text)
    target_utterance = ""
    if target_match:
        target_utterance = target_match.group(1).replace('<', '').replace('>', '')
    else:
        # Try to get the last utterance from conversation
        speaker_utterances = re.findall(r'Speaker_\d+:["\'][^"\']+["\']', conversation_history)
        if speaker_utterances:
            target_utterance = speaker_utterances[-1]

    # Extract audio features - try multiple patterns
    audio_features = ""

    # Pattern 1: "Target speech characteristics: ..."
    audio_match = re.search(r'Target speech characteristics:\s*([^.]+\.)', input_text, re.IGNORECASE)
    if audio_match:
        audio_features = audio_match.group(1).strip()

    # Pattern 2: "Audio description of target utterance: ..."
    if not audio_features:
        audio_match = re.search(r'Audio description of target utterance:\s*([^.]+\.)', input_text, re.IGNORECASE)
        if audio_match:
            audio_features = audio_match.group(1).strip()

    # Pattern 3: Check for features between ### and "For" or newline
    if not audio_features:
        after_conv_match = re.search(r'###\s*([^#]+?)(?:\n\n|For\s+Speaker)', input_text, re.DOTALL)
        if after_conv_match:
            potential_features = after_conv_match.group(1).strip()
            # Check if it contains audio-related keywords
            if any(kw in potential_features.lower() for kw in ['volume', 'pitch', 'speaking rate', 'variation']):
                audio_features = potential_features

    # Pattern 4: Check for audio features in parentheses after utterances
    if not audio_features:
        audio_in_parens = re.findall(r'\(([^)]*(?:pitch|volume|speaking rate)[^)]*)\)', input_text, re.IGNORECASE)
        if audio_in_parens:
            audio_features = "; ".join(audio_in_parens)

    return conversation_history, target_utterance, audio_features


def create_openai_messages_zero_shot(conversation_history, target_utterance, audio_features, label_str):
    """
    Create the messages for OpenAI API call using zero-shot prompt format.
    """
    developer_message = f"""Now you are expert of emotional analysis for dialogues. Please select one emotion label word from [{label_str}] and always respond in strict json format."""

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

    messages = [
        {"role": "developer", "content": developer_message},
        {"role": "user", "content": user_message}
    ]

    return messages


def create_openai_messages_few_shot(conversation_history, target_utterance, audio_features, label_str):
    """
    Create the messages for OpenAI API call using few-shot prompt format with examples.
    """
    developer_message = f"""Now you are expert of emotional analysis for dialogues. Please select one emotion label word from [{label_str}] and always respond in strict json format."""

    # Build few-shot examples string
    examples = get_few_shot_examples()
    examples_text = "\n\nHere are some examples to guide your analysis:\n"
    for i, example in enumerate(examples, 1):
        examples_text += f"\n--- Example {i} ---\n"
        examples_text += f"Input:\n{example['input']}\n"
        examples_text += f"Output:\n{json.dumps(example['output'], indent=2)}\n"

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

    messages = [
        {"role": "developer", "content": developer_message},
        {"role": "user", "content": user_message}
    ]

    return messages


def run_openai_inference(client, model_name, messages, max_retries=3):
    """
    Run inference using OpenAI API with retry logic.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,  # Low temperature for more deterministic output
                max_tokens=300,
                response_format={"type": "json_object"}  # Force JSON output
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API call failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"API call failed after {max_retries} attempts: {e}")
                return None


def main():
    parser = argparse.ArgumentParser(description='OpenAI API inference for emotion recognition')
    parser.add_argument('--dataset', type=str, required=True, choices=['iemocap', 'meld', 'EmoryNLP'],
                        help='Dataset to evaluate')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save results')
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini',
                        help='OpenAI model name (e.g., gpt-4o-mini, gpt-4o, gpt-4-turbo)')
    parser.add_argument('--experiments_setting', type=str, default='zero_shot',
                        choices=['zero_shot', 'few_shot'],
                        help='Experiment setting: zero_shot or few_shot')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env variable)')
    parser.add_argument('--data_percent', type=float, default=1.0, help='Percentage of data to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_delay', type=float, default=0.1,
                        help='Delay between API calls in seconds (to avoid rate limits)')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize OpenAI client
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get label information
    emotional_label_dict, emotional_label_str = get_labels_attr(args.dataset)
    label_set = list(emotional_label_dict.keys())

    # Read test data
    test_file = os.path.join(args.data_dir, "test.json")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    test_data = read_data(test_file, percent=args.data_percent, random_seed=args.seed)
    print(f"Loaded {len(test_data)} test samples")

    # Select prompt creation function based on experiment setting
    if args.experiments_setting == 'few_shot':
        create_messages_fn = create_openai_messages_few_shot
        print("Using few-shot prompting with examples")
    else:
        create_messages_fn = create_openai_messages_zero_shot
        print("Using zero-shot prompting")

    # Run inference
    all_outputs = []
    preds = []
    golds = []
    confuse_case = []
    failed_extraction_case = []

    print(f"\n***** Running inference with {args.model_name} ({args.experiments_setting}) *****\n")

    for idx, sample in enumerate(tqdm(test_data, desc="Processing")):
        input_text = sample['input']
        target = sample['target'].lower()

        # Parse input for OpenAI format
        conversation_history, target_utterance, audio_features = parse_input_for_openai(input_text, args.dataset)

        # Create messages
        messages = create_messages_fn(conversation_history, target_utterance, audio_features, emotional_label_str)

        # Run inference
        output = run_openai_inference(client, args.model_name, messages)

        if output is None:
            output = ""

        # Extract emotion label
        extracted_label = extract_emotion_from_json(output, label_set)

        if extracted_label is None:
            extracted_label = optimize_output(output, label_set)
            confuse_case.append(idx)
            failed_extraction_case.append(idx)

        # Store results
        golds.append(emotional_label_dict[target])
        preds.append(emotional_label_dict.get(extracted_label.lower(), 0))

        all_outputs.append({
            "index": idx,
            "input": input_text,
            "output": output,
            "target": target,
            "extracted_label": extracted_label
        })

        # Log every 50 samples for real-time monitoring
        if (idx + 1) % 50 == 0 or idx == 0:
            current_acc = sum(1 for g, p in zip(golds, preds) if g == p) / len(golds) * 100
            print(f"\n{'='*80}")
            print(f"Sample {idx + 1}/{len(test_data)} | Running Accuracy: {current_acc:.2f}%")
            print(f"{'='*80}")
            print(f"Target Sentence: {target_utterance}")
            print(f"Target Emotion: {target}")
            print(f"Predicted Emotion: {extracted_label}")
            print(f"LLM Output: {output}")
            print(f"{'='*80}\n")

        # Add delay to avoid rate limits
        if args.batch_delay > 0:
            time.sleep(args.batch_delay)

    # Calculate scores
    score, res_matrix = report_score(dataset=args.dataset, golds=golds, preds=preds)

    # Save results
    preds_for_eval_path = os.path.join(args.output_dir, "preds_for_eval.text")
    with open(preds_for_eval_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(score))
        f.write(f'\n{res_matrix}')
        f.write(f'\nconfuse_case: {confuse_case}\n')
        f.write(f'\nThe num of confuse_case is: {len(confuse_case)}\n')
        f.write(f'\nfailed_extraction_case: {failed_extraction_case}\n')
        f.write(f'\nThe num of failed_extraction_case is: {len(failed_extraction_case)}\n')
        f.write(json.dumps(all_outputs, indent=5, ensure_ascii=False))

    # Save experiment config
    config_path = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": args.model_name,
            "dataset": args.dataset,
            "experiments_setting": args.experiments_setting,
            "data_percent": args.data_percent,
            "seed": args.seed,
            "num_samples": len(test_data)
        }, f, indent=2)

    # Print results
    print("\n***** Results *****")
    print(f"Accuracy: {score['Acc_SA']}%")
    print(f"Weighted F1: {score['F1_SA']}%")
    print(f"\nClassification Report:\n{res_matrix}")
    print(f"\nResults saved to: {preds_for_eval_path}")


if __name__ == "__main__":
    main()
