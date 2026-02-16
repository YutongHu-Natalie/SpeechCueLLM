"""
Calculate Krippendorff's Alpha for IEMOCAP Annotations

This script calculates inter-rater reliability using Krippendorff's alpha for:
1. Emotion labels (nominal data)
2. VAD annotations (interval data)
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import krippendorff


def load_all_json_files(json_dir: str) -> List[Dict]:
    """Load all IEMOCAP JSON files."""
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    all_data = []

    for json_file in sorted(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)

    print(f"Loaded {len(all_data)} JSON files from {json_dir}")
    return all_data


def extract_emotion_annotations(all_json_data: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """
    Extract categorical emotion annotations for Krippendorff's alpha calculation.

    Instead of tracking specific annotator IDs, we treat each annotation position
    as a generic "rater" slot, which is appropriate when different annotators
    rate different items.

    When an annotator provides multiple labels (separated by semicolons),
    we select the one that best matches other annotators' choices.

    Returns:
        reliability_data: 2D array where rows are rater positions and columns are items
        turn_ids: List of turn IDs for reference
    """
    turn_data = []
    max_annotators = 0
    multi_label_count = 0

    for dialogue_data in all_json_data:
        for turn in dialogue_data.get("turns", []):
            turn_id = turn.get("turn_id", "")
            categorical_annotations = turn.get("emotion", {}).get("categorical_annotations", [])

            if categorical_annotations:
                # Extract emotion labels, handling multiple labels
                raw_emotions = [ann.get("emotion") for ann in categorical_annotations
                               if ann.get("emotion")]

                if raw_emotions:
                    # Resolve multiple labels by choosing the one most similar to others
                    resolved_emotions = resolve_multiple_labels(raw_emotions)

                    # Count how many had multiple labels
                    multi_label_count += sum(1 for e in raw_emotions if ';' in e)

                    turn_data.append({
                        "turn_id": turn_id,
                        "annotations": resolved_emotions
                    })
                    max_annotators = max(max_annotators, len(resolved_emotions))

    turn_ids = [item["turn_id"] for item in turn_data]

    print(f"\nEmotion Annotations:")
    print(f"  Maximum annotators per utterance: {max_annotators}")
    print(f"  Number of utterances: {len(turn_ids)}")
    print(f"  Annotations with multiple labels resolved: {multi_label_count}")

    # Create reliability matrix (max_annotators x items)
    # Use NaN for missing values
    reliability_matrix = np.full((max_annotators, len(turn_data)), np.nan, dtype=object)

    for item_idx, item in enumerate(turn_data):
        for rater_idx, emotion in enumerate(item["annotations"]):
            reliability_matrix[rater_idx, item_idx] = emotion

    # Calculate statistics
    annotations_per_item = [len(item["annotations"]) for item in turn_data]
    print(f"  Average annotators per utterance: {np.mean(annotations_per_item):.2f}")
    print(f"  Median annotators per utterance: {np.median(annotations_per_item):.0f}")

    return reliability_matrix, turn_ids


def resolve_multiple_labels(emotions: List[str]) -> List[str]:
    """
    When annotators provide multiple labels (separated by semicolons),
    resolve each one by choosing the label most similar to other annotators.

    Args:
        emotions: List of emotion labels, potentially with semicolons

    Returns:
        List of resolved single emotion labels
    """
    # Split all multi-label annotations
    split_emotions = []
    for e in emotions:
        if ';' in e:
            # Split and clean
            labels = [label.strip() for label in e.split(';') if label.strip()]
            split_emotions.append(labels)
        else:
            split_emotions.append([e.strip()])

    # For each annotation, if it has multiple labels, choose the best one
    resolved = []
    for idx, labels in enumerate(split_emotions):
        if len(labels) == 1:
            # Single label, use as-is
            resolved.append(labels[0])
        else:
            # Multiple labels - choose the one that appears most in other annotations
            other_labels = []
            for other_idx, other_split in enumerate(split_emotions):
                if other_idx != idx:
                    other_labels.extend(other_split)

            # Count which of this annotator's labels appears most in others
            best_label = labels[0]  # default to first
            max_count = 0
            for label in labels:
                count = other_labels.count(label)
                if count > max_count:
                    max_count = count
                    best_label = label

            resolved.append(best_label)

    return resolved


def extract_vad_annotations(all_json_data: List[Dict]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extract VAD annotations for Krippendorff's alpha calculation.

    Instead of tracking specific annotator IDs, we treat each annotation position
    as a generic "rater" slot, which is appropriate when different annotators
    rate different items.

    Returns:
        reliability_data: Dictionary with keys 'valence', 'arousal', 'dominance',
                         each containing a 2D array (rater positions x items)
        turn_ids: List of turn IDs for reference
    """
    turn_data = []
    max_annotators = 0

    for dialogue_data in all_json_data:
        for turn in dialogue_data.get("turns", []):
            turn_id = turn.get("turn_id", "")
            vad_annotations = turn.get("emotion", {}).get("vad_annotations", [])

            if vad_annotations:
                # Extract VAD values without tracking specific annotator IDs
                vad_values = []
                for ann in vad_annotations:
                    valence = ann.get("valence")
                    arousal = ann.get("arousal")
                    dominance = ann.get("dominance")

                    if valence is not None:
                        vad_values.append({
                            "valence": valence,
                            "arousal": arousal,
                            "dominance": dominance
                        })

                if vad_values:
                    turn_data.append({
                        "turn_id": turn_id,
                        "annotations": vad_values
                    })
                    max_annotators = max(max_annotators, len(vad_values))

    turn_ids = [item["turn_id"] for item in turn_data]

    print(f"\nVAD Annotations:")
    print(f"  Maximum annotators per utterance: {max_annotators}")
    print(f"  Number of utterances: {len(turn_ids)}")

    # Create reliability matrices for each dimension
    reliability_data = {}
    for dimension in ["valence", "arousal", "dominance"]:
        matrix = np.full((max_annotators, len(turn_data)), np.nan, dtype=float)

        for item_idx, item in enumerate(turn_data):
            for rater_idx, vad_dict in enumerate(item["annotations"]):
                value = vad_dict[dimension]
                if value is not None:
                    matrix[rater_idx, item_idx] = float(value)

        reliability_data[dimension] = matrix

    # Calculate statistics
    annotations_per_item = [len(item["annotations"]) for item in turn_data]
    print(f"  Average annotators per utterance: {np.mean(annotations_per_item):.2f}")
    print(f"  Median annotators per utterance: {np.median(annotations_per_item):.0f}")

    return reliability_data, turn_ids


def calculate_emotion_krippendorff_alpha(reliability_matrix: np.ndarray) -> float:
    """
    Calculate Krippendorff's alpha for nominal emotion data.

    Args:
        reliability_matrix: 2D array (rater positions x items) with emotion labels

    Returns:
        Krippendorff's alpha value
    """
    print(f"\n{'='*60}")
    print("Krippendorff's Alpha for Emotion Labels (Nominal)")
    print(f"{'='*60}")

    # Check data coverage
    total_cells = reliability_matrix.size
    non_missing = np.sum(~pd.isna(reliability_matrix))
    coverage = (non_missing / total_cells) * 100

    print(f"Data coverage: {non_missing}/{total_cells} ({coverage:.2f}%)")
    print(f"Number of rater positions: {reliability_matrix.shape[0]}")
    print(f"Number of items: {reliability_matrix.shape[1]}")

    # Convert string labels to numeric codes for krippendorff.alpha
    # Get all unique labels (excluding NaN)
    flat_data = reliability_matrix.flatten()
    unique_labels = list(set(flat_data[~pd.isna(flat_data)]))
    label_to_code = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    print(f"\nEmotion label mapping (total {len(unique_labels)} unique labels):")
    for label, code in sorted(label_to_code.items()):
        print(f"  {label}: {code}")

    # Create numeric matrix
    numeric_matrix = np.full_like(reliability_matrix, np.nan, dtype=float)
    for i in range(reliability_matrix.shape[0]):
        for j in range(reliability_matrix.shape[1]):
            val = reliability_matrix[i, j]
            if not pd.isna(val):
                numeric_matrix[i, j] = label_to_code[val]

    # Calculate Krippendorff's alpha for nominal data
    alpha = krippendorff.alpha(reliability_data=numeric_matrix, level_of_measurement="nominal")

    print(f"\nKrippendorff's Alpha (Nominal): {alpha:.4f}")

    # Interpretation
    if alpha >= 0.8:
        interpretation = "Excellent reliability"
    elif alpha >= 0.667:
        interpretation = "Acceptable for tentative conclusions"
    elif alpha >= 0.6:
        interpretation = "Marginal reliability"
    else:
        interpretation = "Poor reliability"

    print(f"Interpretation: {interpretation}")

    return alpha


def calculate_vad_krippendorff_alpha(reliability_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Calculate Krippendorff's alpha for interval VAD data.

    Args:
        reliability_data: Dictionary with 'valence', 'arousal', 'dominance' matrices

    Returns:
        Dictionary with alpha values for each dimension
    """
    print(f"\n{'='*60}")
    print("Krippendorff's Alpha for VAD Annotations (Interval)")
    print(f"{'='*60}")

    alphas = {}

    for dimension in ["valence", "arousal", "dominance"]:
        print(f"\n{dimension.upper()}:")
        matrix = reliability_data[dimension]

        # Check data coverage
        total_cells = matrix.size
        non_missing = np.sum(~np.isnan(matrix))
        coverage = (non_missing / total_cells) * 100

        print(f"  Data coverage: {non_missing}/{total_cells} ({coverage:.2f}%)")
        print(f"  Number of rater positions: {matrix.shape[0]}")
        print(f"  Number of items: {matrix.shape[1]}")

        # Calculate Krippendorff's alpha for interval data
        alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="interval")
        alphas[dimension] = alpha

        print(f"  Krippendorff's Alpha (Interval): {alpha:.4f}")

        # Interpretation
        if alpha >= 0.8:
            interpretation = "Excellent reliability"
        elif alpha >= 0.667:
            interpretation = "Acceptable for tentative conclusions"
        elif alpha >= 0.6:
            interpretation = "Marginal reliability"
        else:
            interpretation = "Poor reliability"

        print(f"  Interpretation: {interpretation}")

    # Calculate average alpha across all VAD dimensions
    avg_alpha = np.mean(list(alphas.values()))
    print(f"\nAverage VAD Alpha: {avg_alpha:.4f}")

    return alphas


def analyze_emotion_distribution(reliability_matrix: np.ndarray) -> None:
    """Analyze the distribution of emotion labels."""
    print(f"\n{'='*60}")
    print("Overall Emotion Label Distribution")
    print(f"{'='*60}")

    # Flatten and remove NaN values
    flat_data = reliability_matrix.flatten()
    valid_labels = flat_data[~pd.isna(flat_data)]

    unique, counts = np.unique(valid_labels, return_counts=True)
    total = len(valid_labels)

    print(f"\nTotal annotations: {total}")
    for label, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")


def analyze_vad_statistics(reliability_data: Dict[str, np.ndarray]) -> None:
    """Analyze VAD annotation statistics."""
    print(f"\n{'='*60}")
    print("Overall VAD Annotation Statistics")
    print(f"{'='*60}")

    for dimension in ["valence", "arousal", "dominance"]:
        print(f"\n{dimension.upper()}:")
        matrix = reliability_data[dimension]

        # Flatten and remove NaN values
        flat_data = matrix.flatten()
        valid_data = flat_data[~np.isnan(flat_data)]

        if len(valid_data) > 0:
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)

            print(f"  Total annotations: {len(valid_data)}")
            print(f"  Mean: {mean:.2f}, Std: {std:.2f}")
            print(f"  Min: {min_val}, Max: {max_val}")


def main():
    json_dir = "./IEMOCAP_json_files"

    # Load all JSON files
    all_json_data = load_all_json_files(json_dir)

    # Extract and analyze emotion annotations
    emotion_matrix, emotion_turns = extract_emotion_annotations(all_json_data)
    analyze_emotion_distribution(emotion_matrix)
    emotion_alpha = calculate_emotion_krippendorff_alpha(emotion_matrix)

    # Extract and analyze VAD annotations
    vad_matrices, vad_turns = extract_vad_annotations(all_json_data)
    analyze_vad_statistics(vad_matrices)
    vad_alphas = calculate_vad_krippendorff_alpha(vad_matrices)

    # Save results to file
    results = {
        "emotion": {
            "krippendorff_alpha": float(emotion_alpha),
            "max_raters_per_item": int(emotion_matrix.shape[0]),
            "num_utterances": len(emotion_turns)
        },
        "vad": {
            "valence_alpha": float(vad_alphas["valence"]),
            "arousal_alpha": float(vad_alphas["arousal"]),
            "dominance_alpha": float(vad_alphas["dominance"]),
            "average_alpha": float(np.mean(list(vad_alphas.values()))),
            "max_raters_per_item": int(vad_matrices["valence"].shape[0]),
            "num_utterances": len(vad_turns)
        }
    }

    output_file = "krippendorff_alpha_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nEmotion Labels (Nominal):")
    print(f"  Krippendorff's Alpha: {emotion_alpha:.4f}")
    print(f"\nVAD Annotations (Interval):")
    print(f"  Valence Alpha: {vad_alphas['valence']:.4f}")
    print(f"  Arousal Alpha: {vad_alphas['arousal']:.4f}")
    print(f"  Dominance Alpha: {vad_alphas['dominance']:.4f}")
    print(f"  Average Alpha: {np.mean(list(vad_alphas.values())):.4f}")


if __name__ == "__main__":
    main()
