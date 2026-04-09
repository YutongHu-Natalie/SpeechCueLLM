"""
Calculate Fleiss' Kappa for IEMOCAP Annotations

This script calculates inter-rater reliability using Fleiss' Kappa for:
1. Emotion labels (categorical/nominal data)
2. VAD annotations (discretized into 5 bins to apply kappa)

Fleiss' Kappa extends Cohen's Kappa to the case of multiple raters and
handles variable numbers of raters per item.

Note: For continuous VAD data, Krippendorff's Alpha (interval) is the more
principled measure. The kappa here uses discretized scores as a complement.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


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


# ---------------------------------------------------------------------------
# Shared extraction helpers (mirrors calculate_krippendorff_alpha.py)
# ---------------------------------------------------------------------------

def resolve_multiple_labels(emotions: List[str]) -> List[str]:
    """
    When annotators provide multiple labels (separated by semicolons),
    resolve each one by choosing the label most similar to other annotators.
    """
    split_emotions = []
    for e in emotions:
        if ';' in e:
            labels = [label.strip() for label in e.split(';') if label.strip()]
            split_emotions.append(labels)
        else:
            split_emotions.append([e.strip()])

    resolved = []
    for idx, labels in enumerate(split_emotions):
        if len(labels) == 1:
            resolved.append(labels[0])
        else:
            other_labels = []
            for other_idx, other_split in enumerate(split_emotions):
                if other_idx != idx:
                    other_labels.extend(other_split)

            best_label = labels[0]
            max_count = 0
            for label in labels:
                count = other_labels.count(label)
                if count > max_count:
                    max_count = count
                    best_label = label
            resolved.append(best_label)

    return resolved


def extract_emotion_annotations(all_json_data: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """
    Extract categorical emotion annotations.
    Returns reliability matrix (rater_positions x items) and turn IDs.
    """
    turn_data = []
    max_annotators = 0

    for dialogue_data in all_json_data:
        for turn in dialogue_data.get("turns", []):
            turn_id = turn.get("turn_id", "")
            categorical_annotations = turn.get("emotion", {}).get("categorical_annotations", [])

            if categorical_annotations:
                raw_emotions = [ann.get("emotion") for ann in categorical_annotations
                                if ann.get("emotion")]
                if raw_emotions:
                    resolved_emotions = resolve_multiple_labels(raw_emotions)
                    turn_data.append({"turn_id": turn_id, "annotations": resolved_emotions})
                    max_annotators = max(max_annotators, len(resolved_emotions))

    turn_ids = [item["turn_id"] for item in turn_data]

    print(f"\nEmotion Annotations:")
    print(f"  Maximum annotators per utterance: {max_annotators}")
    print(f"  Number of utterances: {len(turn_ids)}")

    reliability_matrix = np.full((max_annotators, len(turn_data)), np.nan, dtype=object)
    for item_idx, item in enumerate(turn_data):
        for rater_idx, emotion in enumerate(item["annotations"]):
            reliability_matrix[rater_idx, item_idx] = emotion

    annotations_per_item = [len(item["annotations"]) for item in turn_data]
    print(f"  Average annotators per utterance: {np.mean(annotations_per_item):.2f}")
    print(f"  Median annotators per utterance: {np.median(annotations_per_item):.0f}")

    return reliability_matrix, turn_ids


def extract_vad_annotations(all_json_data: List[Dict]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extract VAD annotations.
    Returns dict of reliability matrices (rater_positions x items) and turn IDs.
    """
    turn_data = []
    max_annotators = 0

    for dialogue_data in all_json_data:
        for turn in dialogue_data.get("turns", []):
            turn_id = turn.get("turn_id", "")
            vad_annotations = turn.get("emotion", {}).get("vad_annotations", [])

            if vad_annotations:
                vad_values = []
                for ann in vad_annotations:
                    valence = ann.get("valence")
                    arousal = ann.get("arousal")
                    dominance = ann.get("dominance")
                    if valence is not None:
                        vad_values.append({
                            "valence": valence,
                            "arousal": arousal,
                            "dominance": dominance,
                        })

                if vad_values:
                    turn_data.append({"turn_id": turn_id, "annotations": vad_values})
                    max_annotators = max(max_annotators, len(vad_values))

    turn_ids = [item["turn_id"] for item in turn_data]

    print(f"\nVAD Annotations:")
    print(f"  Maximum annotators per utterance: {max_annotators}")
    print(f"  Number of utterances: {len(turn_ids)}")

    reliability_data = {}
    for dimension in ["valence", "arousal", "dominance"]:
        matrix = np.full((max_annotators, len(turn_data)), np.nan, dtype=float)
        for item_idx, item in enumerate(turn_data):
            for rater_idx, vad_dict in enumerate(item["annotations"]):
                value = vad_dict[dimension]
                if value is not None:
                    matrix[rater_idx, item_idx] = float(value)
        reliability_data[dimension] = matrix

    annotations_per_item = [len(item["annotations"]) for item in turn_data]
    print(f"  Average annotators per utterance: {np.mean(annotations_per_item):.2f}")
    print(f"  Median annotators per utterance: {np.median(annotations_per_item):.0f}")

    return reliability_data, turn_ids


# ---------------------------------------------------------------------------
# Kappa helpers
# ---------------------------------------------------------------------------

def build_count_table(numeric_matrix: np.ndarray, n_categories: int) -> np.ndarray:
    """
    Convert a (raters x items) integer code matrix into a Fleiss count table
    of shape (n_items, n_categories).

    Missing rater slots (NaN) simply contribute 0 to the count for that item.
    Items rated by fewer than 2 raters are kept in the table (count row sums
    to 1); statsmodels' fleiss_kappa handles variable rater counts.
    """
    n_raters, n_items = numeric_matrix.shape
    count_table = np.zeros((n_items, n_categories), dtype=int)

    for item_idx in range(n_items):
        for rater_idx in range(n_raters):
            val = numeric_matrix[rater_idx, item_idx]
            if not np.isnan(val):
                count_table[item_idx, int(val)] += 1

    return count_table


def fleiss_kappa_variable_raters(count_table: np.ndarray) -> float:
    """
    Compute Fleiss' Kappa allowing different numbers of raters per item.

    statsmodels.fleiss_kappa requires a fixed n per item; this implementation
    follows the generalized formula (Fleiss, 1971) where each item may be rated
    by a different number of raters.

    count_table: (n_items, n_categories) integer counts.
    Items with fewer than 2 raters are excluded automatically.
    """
    n_raters_per_item = count_table.sum(axis=1)
    valid_mask = n_raters_per_item >= 2
    count_table = count_table[valid_mask]
    n_raters_per_item = n_raters_per_item[valid_mask]

    n_items = count_table.shape[0]
    if n_items == 0:
        return np.nan

    total_ratings = n_raters_per_item.sum()

    # Marginal category proportions
    p_j = count_table.sum(axis=0) / total_ratings

    # Observed agreement per item: P_i = sum_j n_ij(n_ij-1) / (n_i(n_i-1))
    P_i = np.array([
        np.sum(row * (row - 1)) / (n * (n - 1))
        for row, n in zip(count_table, n_raters_per_item)
    ])

    P_bar = P_i.mean()         # mean observed agreement
    P_e = np.sum(p_j ** 2)    # expected agreement by chance

    if abs(1.0 - P_e) < 1e-10:
        return 1.0

    return float((P_bar - P_e) / (1.0 - P_e))


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) interpretation of kappa."""
    if kappa >= 0.81:
        return "Almost perfect agreement"
    elif kappa >= 0.61:
        return "Substantial agreement"
    elif kappa >= 0.41:
        return "Moderate agreement"
    elif kappa >= 0.21:
        return "Fair agreement"
    elif kappa >= 0.0:
        return "Slight agreement"
    else:
        return "Poor agreement (less than chance)"


# ---------------------------------------------------------------------------
# Kappa calculations
# ---------------------------------------------------------------------------

def calculate_emotion_fleiss_kappa(reliability_matrix: np.ndarray) -> float:
    """
    Calculate Fleiss' Kappa for nominal emotion data with multiple raters.

    Args:
        reliability_matrix: 2D array (rater_positions x items) with emotion labels

    Returns:
        Fleiss' Kappa value
    """
    print(f"\n{'='*60}")
    print("Fleiss' Kappa for Emotion Labels (Nominal)")
    print(f"{'='*60}")

    total_cells = reliability_matrix.size
    non_missing = np.sum(~pd.isna(reliability_matrix))
    coverage = (non_missing / total_cells) * 100
    print(f"Data coverage: {non_missing}/{total_cells} ({coverage:.2f}%)")
    print(f"Number of rater positions: {reliability_matrix.shape[0]}")
    print(f"Number of items: {reliability_matrix.shape[1]}")

    # Map string labels to integer codes
    flat_data = reliability_matrix.flatten()
    unique_labels = sorted(set(v for v in flat_data if not pd.isna(v)))
    label_to_code = {label: idx for idx, label in enumerate(unique_labels)}

    print(f"\nEmotion label mapping ({len(unique_labels)} unique labels):")
    for label, code in sorted(label_to_code.items()):
        print(f"  {label}: {code}")

    # Build numeric matrix (raters x items)
    numeric_matrix = np.full(reliability_matrix.shape, np.nan, dtype=float)
    for i in range(reliability_matrix.shape[0]):
        for j in range(reliability_matrix.shape[1]):
            val = reliability_matrix[i, j]
            if not pd.isna(val):
                numeric_matrix[i, j] = label_to_code[val]

    count_table = build_count_table(numeric_matrix, len(unique_labels))

    # Drop items with fewer than 2 raters (can't contribute to agreement)
    raters_per_item = count_table.sum(axis=1)
    valid_mask = raters_per_item >= 2
    count_table_valid = count_table[valid_mask]
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"\nDropped {n_dropped} items with fewer than 2 raters.")
    print(f"Items used for kappa: {count_table_valid.shape[0]}")

    kappa = fleiss_kappa_variable_raters(count_table_valid)

    print(f"\nFleiss' Kappa (Nominal): {kappa:.4f}")
    print(f"Interpretation: {interpret_kappa(kappa)}")

    return float(kappa)


def calculate_vad_fleiss_kappa(
    reliability_data: Dict[str, np.ndarray],
    n_bins: int = 5,
) -> Dict[str, float]:
    """
    Calculate Fleiss' Kappa for VAD annotations by discretizing scores into bins.

    Continuous VAD ratings are binned into `n_bins` equal-width bins based on
    the observed data range for each dimension, then Fleiss' Kappa is computed
    on the resulting categorical labels.

    Args:
        reliability_data: Dict with 'valence', 'arousal', 'dominance' matrices
                          (rater_positions x items), continuous values
        n_bins: Number of equal-width bins (default 5, matching typical 1–5 scale)

    Returns:
        Dict with kappa values for each dimension
    """
    print(f"\n{'='*60}")
    print(f"Fleiss' Kappa for VAD Annotations (Discretized into {n_bins} bins)")
    print(f"{'='*60}")

    kappas = {}

    for dimension in ["valence", "arousal", "dominance"]:
        print(f"\n{dimension.upper()}:")
        matrix = reliability_data[dimension]

        total_cells = matrix.size
        non_missing = np.sum(~np.isnan(matrix))
        coverage = (non_missing / total_cells) * 100
        print(f"  Data coverage: {non_missing}/{total_cells} ({coverage:.2f}%)")

        # Determine bin edges from valid data
        valid_vals = matrix[~np.isnan(matrix)]
        data_min, data_max = valid_vals.min(), valid_vals.max()
        bin_edges = np.linspace(data_min, data_max, n_bins + 1)
        # Extend edges slightly so the max value falls inside the last bin
        bin_edges[-1] += 1e-8
        print(f"  Value range: [{data_min:.2f}, {data_max:.2f}]")
        print(f"  Bin edges: {np.round(bin_edges, 2).tolist()}")

        # Discretize
        discrete_matrix = np.full(matrix.shape, np.nan, dtype=float)
        valid_mask = ~np.isnan(matrix)
        discrete_matrix[valid_mask] = np.digitize(matrix[valid_mask], bin_edges) - 1
        # Clip to [0, n_bins-1] just in case of floating-point edge cases
        discrete_matrix[valid_mask] = np.clip(discrete_matrix[valid_mask], 0, n_bins - 1)

        count_table = build_count_table(discrete_matrix, n_bins)

        # Drop items with fewer than 2 raters
        raters_per_item = count_table.sum(axis=1)
        valid_items = raters_per_item >= 2
        count_table_valid = count_table[valid_items]
        n_dropped = (~valid_items).sum()
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} items with fewer than 2 raters.")
        print(f"  Items used for kappa: {count_table_valid.shape[0]}")

        kappa = fleiss_kappa_variable_raters(count_table_valid)
        kappas[dimension] = float(kappa)

        print(f"  Fleiss' Kappa ({n_bins}-bin): {kappa:.4f}")
        print(f"  Interpretation: {interpret_kappa(kappa)}")

    avg_kappa = float(np.mean(list(kappas.values())))
    print(f"\nAverage VAD Kappa: {avg_kappa:.4f}")

    return kappas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    json_dir = "./IEMOCAP_json_files"

    all_json_data = load_all_json_files(json_dir)

    # --- Emotion labels ---
    emotion_matrix, emotion_turns = extract_emotion_annotations(all_json_data)
    emotion_kappa = calculate_emotion_fleiss_kappa(emotion_matrix)

    # --- VAD annotations ---
    vad_matrices, vad_turns = extract_vad_annotations(all_json_data)
    vad_kappas = calculate_vad_fleiss_kappa(vad_matrices, n_bins=5)

    # --- Save results ---
    results = {
        "emotion": {
            "fleiss_kappa": emotion_kappa,
            "max_raters_per_item": int(emotion_matrix.shape[0]),
            "num_utterances": len(emotion_turns),
        },
        "vad": {
            "valence_kappa": vad_kappas["valence"],
            "arousal_kappa": vad_kappas["arousal"],
            "dominance_kappa": vad_kappas["dominance"],
            "average_kappa": float(np.mean(list(vad_kappas.values()))),
            "max_raters_per_item": int(vad_matrices["valence"].shape[0]),
            "num_utterances": len(vad_turns),
            "n_bins_for_vad": 5,
        },
    }

    output_file = "kappa_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nEmotion Labels (Nominal):")
    print(f"  Fleiss' Kappa: {emotion_kappa:.4f}  ({interpret_kappa(emotion_kappa)})")
    print(f"\nVAD Annotations (5-bin discretized):")
    for dim in ["valence", "arousal", "dominance"]:
        k = vad_kappas[dim]
        print(f"  {dim.capitalize()} Kappa: {k:.4f}  ({interpret_kappa(k)})")
    print(f"  Average Kappa: {np.mean(list(vad_kappas.values())):.4f}")


if __name__ == "__main__":
    main()
