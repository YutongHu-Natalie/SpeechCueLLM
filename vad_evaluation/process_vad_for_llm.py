"""
Process VAD data for LLM training/inference.

This script prepares the VAD dataset in the format expected by the LLM pipeline,
creating train/test JSON files with prompts and targets similar to the emotion
recognition task.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Optional


def load_audio_features(audio_features_path: str) -> dict:
    """
    Load audio features from CSV and create a lookup dictionary.

    Args:
        audio_features_path: Path to audio features CSV

    Returns:
        Dictionary mapping turn_id to audio feature description
    """
    if not os.path.exists(audio_features_path):
        print(f"Audio features file not found: {audio_features_path}")
        return {}

    df = pd.read_csv(audio_features_path)
    features_dict = {}

    for _, row in df.iterrows():
        turn_id = row.get('utterance_id') or row.get('turn_id') or row.get('id')
        if turn_id is None:
            continue

        # Build feature description based on available columns
        descriptions = []

        # Volume/Intensity
        if 'avg_intensity' in row:
            intensity = row['avg_intensity']
            intensity_var = row.get('intensity_variation', 0)
            intensity_level = describe_level(intensity, [40, 50, 60, 70])
            var_level = describe_level(intensity_var, [5, 10, 15, 20])
            descriptions.append(f"{intensity_level} volume with {var_level} variation")

        # Pitch
        if 'avg_pitch' in row:
            pitch = row['avg_pitch']
            pitch_std = row.get('pitch_std', 0)
            pitch_level = describe_level(pitch, [100, 150, 200, 250])
            pitch_var = describe_level(pitch_std, [20, 40, 60, 80])
            descriptions.append(f"{pitch_level} pitch with {pitch_var} variation")

        # Speaking rate
        if 'articulation_rate' in row:
            rate = row['articulation_rate']
            rate_level = describe_level(rate, [3, 4, 5, 6])
            descriptions.append(f"{rate_level} speaking rate")

        if descriptions:
            features_dict[turn_id] = ", ".join(descriptions)
        else:
            features_dict[turn_id] = "Audio features not available"

    return features_dict


def describe_level(value, thresholds):
    """Convert numeric value to descriptive level."""
    if pd.isna(value):
        return "moderate"

    levels = ["very low", "low", "moderate", "high", "very high"]

    for i, threshold in enumerate(thresholds):
        if value < threshold:
            return levels[i]
    return levels[-1]


def build_conversation_context(
    df: pd.DataFrame,
    current_idx: int,
    window_size: int = 12,
    audio_features: dict = None
) -> tuple:
    """
    Build conversation context for a given utterance.

    Args:
        df: DataFrame with all utterances
        current_idx: Index of target utterance
        window_size: Number of previous utterances to include
        audio_features: Dictionary of audio feature descriptions

    Returns:
        Tuple of (conversation_history, target_utterance, audio_description)
    """
    current_row = df.iloc[current_idx]
    dialogue_id = current_row['dialogue_id']
    turn_index = current_row['turn_index']

    # Get all utterances from same dialogue
    dialogue_df = df[df['dialogue_id'] == dialogue_id].sort_values('turn_index')

    # Get utterances in window
    start_idx = max(0, turn_index - window_size)
    history_df = dialogue_df[
        (dialogue_df['turn_index'] >= start_idx) &
        (dialogue_df['turn_index'] < turn_index)
    ]

    # Build conversation history
    history_lines = []
    for _, row in history_df.iterrows():
        speaker = f"Speaker_{row['speaker']}"
        content = row['content']
        history_lines.append(f'{speaker}: "{content}"')

    conversation_history = '\n'.join(history_lines) if history_lines else "No prior context."

    # Target utterance
    target_speaker = f"Speaker_{current_row['speaker']}"
    target_content = current_row['content']
    target_utterance = f'{target_speaker}: "{target_content}"'

    # Audio features for last 3 utterances
    audio_descriptions = []
    recent_df = dialogue_df[
        (dialogue_df['turn_index'] >= max(0, turn_index - 2)) &
        (dialogue_df['turn_index'] <= turn_index)
    ]

    if audio_features:
        for _, row in recent_df.iterrows():
            turn_id = row['turn_id']
            if turn_id in audio_features:
                speaker = f"Speaker_{row['speaker']}"
                audio_descriptions.append(f"{speaker}: {audio_features[turn_id]}")

    audio_description = "\n".join(audio_descriptions) if audio_descriptions else "Audio features not available."

    return conversation_history, target_utterance, audio_description


def create_vad_prompt(
    conversation_history: str,
    target_utterance: str,
    audio_features: str,
    short_version: bool = False
) -> str:
    """Create the VAD prediction prompt."""

    if short_version:
        prompt = f"""Rate emotion using VAD (1-5):
- Valence: 1=negative, 5=positive
- Arousal: 1=calm, 5=excited
- Dominance: 1=submissive, 5=dominant

Conversation:
{conversation_history}

Target: {target_utterance}
Audio: {audio_features}

JSON output: {{"v_value": "X", "a_value": "X", "d_value": "X"}}
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

Audio features:
{audio_features}

Rate the emotion in the target utterance. Consider both the content and audio features.
Provide your ratings in JSON format inside <answer> tags:
<answer>
{{"v_value": "your integer 1-5", "a_value": "your integer 1-5", "d_value": "your integer 1-5"}}
</answer>"""

    return prompt


def process_vad_dataset(
    input_file: str,
    output_dir: str,
    audio_features_path: Optional[str] = None,
    window_size: int = 12,
    short_prompt: bool = False,
    test_session: str = "05"
):
    """
    Process VAD dataset and create train/test JSON files.

    Args:
        input_file: Path to VAD data (CSV or pickle)
        output_dir: Directory to save processed files
        audio_features_path: Optional path to audio features CSV
        window_size: Conversation history window size
        short_prompt: Use shorter prompts for LoRA training
        test_session: Session to use as test set (leave-one-session-out)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_pickle(input_file)

    print(f"Loaded {len(df)} samples from {input_file}")

    # Load audio features
    audio_features = {}
    if audio_features_path:
        audio_features = load_audio_features(audio_features_path)
        print(f"Loaded {len(audio_features)} audio feature entries")

    # Split by session
    train_df = df[df['session'] != test_session].reset_index(drop=True)
    test_df = df[df['session'] == test_session].reset_index(drop=True)

    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Process each split
    for split_name, split_df in [('train', train_df), ('test', test_df)]:
        records = []

        for idx in range(len(split_df)):
            row = split_df.iloc[idx]

            # Build context
            conversation_history, target_utterance, audio_description = build_conversation_context(
                split_df, idx, window_size, audio_features
            )

            # Create prompt
            prompt = create_vad_prompt(
                conversation_history,
                target_utterance,
                audio_description,
                short_version=short_prompt
            )

            # Create target (the expected output)
            # Round VAD values to nearest integer for discrete prediction
            target = json.dumps({
                "v_value": str(int(round(row['valence']))),
                "a_value": str(int(round(row['arousal']))),
                "d_value": str(int(round(row['dominance'])))
            })

            record = {
                "input": prompt,
                "target": target,
                "path": row['dialogue_id'],  # For compatibility with existing pipeline
                # Additional metadata
                "dialogue_id": row['dialogue_id'],
                "turn_id": row['turn_id'],
                "gold_valence": float(row['valence']),
                "gold_arousal": float(row['arousal']),
                "gold_dominance": float(row['dominance']),
            }
            records.append(record)

        # Save as JSON lines
        output_file = os.path.join(output_dir, f"{split_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"Saved {len(records)} records to {output_file}")

    # Save processing config
    config = {
        "input_file": input_file,
        "audio_features_path": audio_features_path,
        "window_size": window_size,
        "short_prompt": short_prompt,
        "test_session": test_session,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
    }
    with open(os.path.join(output_dir, "processing_config.json"), 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Process VAD data for LLM training')
    parser.add_argument(
        '--input_file',
        type=str,
        default='./data/vad_full_dataset.csv',
        help='Path to VAD data file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./processed_data',
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--audio_features',
        type=str,
        default=None,
        help='Path to audio features CSV (optional)'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=12,
        help='Conversation history window size'
    )
    parser.add_argument(
        '--short_prompt',
        action='store_true',
        help='Use shorter prompts for LoRA training'
    )
    parser.add_argument(
        '--test_session',
        type=str,
        default='05',
        help='Session to use as test set'
    )

    args = parser.parse_args()

    process_vad_dataset(
        input_file=args.input_file,
        output_dir=args.output_dir,
        audio_features_path=args.audio_features,
        window_size=args.window_size,
        short_prompt=args.short_prompt,
        test_session=args.test_session
    )


if __name__ == "__main__":
    main()
