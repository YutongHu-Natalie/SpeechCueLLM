"""
VAD Data Preparation Script for IEMOCAP Dataset

This script extracts VAD (Valence-Arousal-Dominance) scores from IEMOCAP JSON files
and creates two datasets:
1. Full dataset: All samples with VAD scores
2. Filtered dataset: Samples with low standard deviation (high inter-annotator agreement)

The output does not affect the existing emotion recognition pipeline.
"""

import os
import json
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def load_iemocap_json_files(json_dir: str) -> List[Dict]:
    """
    Load all IEMOCAP JSON files from the specified directory.

    Args:
        json_dir: Path to the IEMOCAP_json_files directory

    Returns:
        List of parsed JSON data from all files
    """
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    all_data = []

    for json_file in sorted(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)

    print(f"Loaded {len(all_data)} JSON files from {json_dir}")
    return all_data


def extract_vad_data(all_json_data: List[Dict]) -> pd.DataFrame:
    """
    Extract VAD data from all IEMOCAP JSON files.

    Args:
        all_json_data: List of parsed JSON data

    Returns:
        DataFrame with VAD data for each utterance
    """
    records = []

    for dialogue_data in all_json_data:
        dialogue_id = dialogue_data.get("dialogue_id", "")
        session = dialogue_data.get("session", "")
        category = dialogue_data.get("category", "")

        for turn in dialogue_data.get("turns", []):
            turn_id = turn.get("turn_id", "")
            turn_index = turn.get("turn_index", 0)
            speaker = turn.get("speaker", "")
            content = turn.get("content", "")

            # Extract time information
            time_info = turn.get("time", {})
            start_time = time_info.get("start", 0.0)
            end_time = time_info.get("end", 0.0)
            duration = time_info.get("duration", 0.0)

            # Extract emotion and VAD information
            emotion_info = turn.get("emotion", {})
            primary_label = emotion_info.get("primary_label", "")

            # Extract averaged VAD scores
            averaged_vad = emotion_info.get("averaged_vad", {})
            valence = averaged_vad.get("valence", None)
            arousal = averaged_vad.get("arousal", None)
            dominance = averaged_vad.get("dominance", None)
            valence_std = averaged_vad.get("valence_std", None)
            arousal_std = averaged_vad.get("arousal_std", None)
            dominance_std = averaged_vad.get("dominance_std", None)

            # Extract individual annotator VAD scores
            vad_annotations = emotion_info.get("vad_annotations", [])
            num_vad_annotators = len(vad_annotations)

            # Store individual annotator scores for reference
            annotator_valences = [ann.get("valence") for ann in vad_annotations if ann.get("valence") is not None]
            annotator_arousals = [ann.get("arousal") for ann in vad_annotations if ann.get("arousal") is not None]
            annotator_dominances = [ann.get("dominance") for ann in vad_annotations if ann.get("dominance") is not None]

            # Calculate combined standard deviation (average of V, A, D stds)
            if valence_std is not None and arousal_std is not None and dominance_std is not None:
                avg_vad_std = (valence_std + arousal_std + dominance_std) / 3
            else:
                avg_vad_std = None

            record = {
                "dialogue_id": dialogue_id,
                "session": session,
                "category": category,
                "turn_id": turn_id,
                "turn_index": turn_index,
                "speaker": speaker,
                "content": content,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "primary_emotion_label": primary_label,
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
                "valence_std": valence_std,
                "arousal_std": arousal_std,
                "dominance_std": dominance_std,
                "avg_vad_std": avg_vad_std,
                "num_vad_annotators": num_vad_annotators,
                "annotator_valences": annotator_valences,
                "annotator_arousals": annotator_arousals,
                "annotator_dominances": annotator_dominances,
            }
            records.append(record)

    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} utterances with VAD data")
    return df


def filter_high_agreement_samples(
    df: pd.DataFrame,
    std_threshold: Optional[float] = None,
    percentile: float = 50.0,
    method: str = "percentile"
) -> pd.DataFrame:
    """
    Filter samples with high inter-annotator agreement (low standard deviation).

    Args:
        df: DataFrame with VAD data
        std_threshold: If method="threshold", use this value as the max allowed std
        percentile: If method="percentile", keep samples below this percentile of avg_vad_std
        method: "threshold" or "percentile"

    Returns:
        Filtered DataFrame with high agreement samples
    """
    # Remove rows with missing VAD data
    df_valid = df.dropna(subset=["avg_vad_std"])

    if method == "threshold":
        if std_threshold is None:
            raise ValueError("std_threshold must be provided when method='threshold'")
        df_filtered = df_valid[df_valid["avg_vad_std"] <= std_threshold]
        print(f"Filtered using threshold={std_threshold}: {len(df_filtered)} samples (out of {len(df_valid)})")

    elif method == "percentile":
        threshold = np.percentile(df_valid["avg_vad_std"], percentile)
        df_filtered = df_valid[df_valid["avg_vad_std"] <= threshold]
        print(f"Filtered using {percentile}th percentile (threshold={threshold:.4f}): {len(df_filtered)} samples (out of {len(df_valid)})")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'threshold' or 'percentile'")

    return df_filtered


def compute_statistics(df: pd.DataFrame, name: str = "Dataset") -> Dict:
    """
    Compute and print statistics for a VAD dataset.

    Args:
        df: DataFrame with VAD data
        name: Name of the dataset for printing

    Returns:
        Dictionary with statistics
    """
    stats = {}

    print(f"\n{'='*60}")
    print(f"Statistics for {name}")
    print(f"{'='*60}")

    print(f"\nTotal samples: {len(df)}")
    stats["total_samples"] = len(df)

    # VAD score statistics
    for dim in ["valence", "arousal", "dominance"]:
        if dim in df.columns:
            valid_data = df[dim].dropna()
            print(f"\n{dim.capitalize()}:")
            print(f"  Mean: {valid_data.mean():.4f}")
            print(f"  Std: {valid_data.std():.4f}")
            print(f"  Min: {valid_data.min():.4f}")
            print(f"  Max: {valid_data.max():.4f}")
            stats[f"{dim}_mean"] = valid_data.mean()
            stats[f"{dim}_std"] = valid_data.std()

    # Standard deviation statistics
    if "avg_vad_std" in df.columns:
        valid_std = df["avg_vad_std"].dropna()
        print(f"\nAverage VAD Std (inter-annotator disagreement):")
        print(f"  Mean: {valid_std.mean():.4f}")
        print(f"  Std: {valid_std.std():.4f}")
        print(f"  Min: {valid_std.min():.4f}")
        print(f"  Max: {valid_std.max():.4f}")
        print(f"  Median: {valid_std.median():.4f}")
        print(f"  25th percentile: {np.percentile(valid_std, 25):.4f}")
        print(f"  75th percentile: {np.percentile(valid_std, 75):.4f}")
        stats["avg_std_mean"] = valid_std.mean()
        stats["avg_std_median"] = valid_std.median()

    # Emotion label distribution
    if "primary_emotion_label" in df.columns:
        print(f"\nEmotion label distribution:")
        emotion_counts = df["primary_emotion_label"].value_counts()
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} ({100*count/len(df):.1f}%)")
        stats["emotion_distribution"] = emotion_counts.to_dict()

    # Session distribution
    if "session" in df.columns:
        print(f"\nSession distribution:")
        session_counts = df["session"].value_counts().sort_index()
        for session, count in session_counts.items():
            print(f"  Session {session}: {count} ({100*count/len(df):.1f}%)")
        stats["session_distribution"] = session_counts.to_dict()

    return stats


def save_datasets(
    df_full: pd.DataFrame,
    df_filtered: pd.DataFrame,
    output_dir: str,
    split_by_session: bool = True
) -> None:
    """
    Save the full and filtered datasets in multiple formats.

    Args:
        df_full: Full VAD dataset
        df_filtered: Filtered high-agreement dataset
        output_dir: Output directory
        split_by_session: Whether to create train/test splits based on session
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save as CSV (for easy inspection)
    df_full.to_csv(os.path.join(output_dir, "vad_full_dataset.csv"), index=False)
    df_filtered.to_csv(os.path.join(output_dir, "vad_filtered_dataset.csv"), index=False)

    # Save as pickle (for later use with all data types preserved)
    df_full.to_pickle(os.path.join(output_dir, "vad_full_dataset.pkl"))
    df_filtered.to_pickle(os.path.join(output_dir, "vad_filtered_dataset.pkl"))

    # Save as JSON lines (for compatibility with LLM training pipeline)
    full_records = df_full.to_dict(orient="records")
    filtered_records = df_filtered.to_dict(orient="records")

    with open(os.path.join(output_dir, "vad_full_dataset.jsonl"), "w") as f:
        for record in full_records:
            # Convert numpy types to native Python types
            clean_record = {}
            for k, v in record.items():
                if isinstance(v, (np.integer, np.floating)):
                    clean_record[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean_record[k] = v.tolist()
                else:
                    clean_record[k] = v
            f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "vad_filtered_dataset.jsonl"), "w") as f:
        for record in filtered_records:
            clean_record = {}
            for k, v in record.items():
                if isinstance(v, (np.integer, np.floating)):
                    clean_record[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean_record[k] = v.tolist()
                else:
                    clean_record[k] = v
            f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

    print(f"\nSaved datasets to {output_dir}")
    print(f"  - vad_full_dataset.csv/pkl/jsonl ({len(df_full)} samples)")
    print(f"  - vad_filtered_dataset.csv/pkl/jsonl ({len(df_filtered)} samples)")

    # Create train/test splits based on session (following IEMOCAP leave-one-session-out protocol)
    if split_by_session:
        splits_dir = os.path.join(output_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)

        sessions = df_full["session"].unique()
        for test_session in sorted(sessions):
            # Full dataset split
            train_full = df_full[df_full["session"] != test_session]
            test_full = df_full[df_full["session"] == test_session]

            # Filtered dataset split
            train_filtered = df_filtered[df_filtered["session"] != test_session]
            test_filtered = df_filtered[df_filtered["session"] == test_session]

            # Save splits
            split_name = f"session_{test_session}_test"
            train_full.to_pickle(os.path.join(splits_dir, f"full_train_{split_name}.pkl"))
            test_full.to_pickle(os.path.join(splits_dir, f"full_test_{split_name}.pkl"))
            train_filtered.to_pickle(os.path.join(splits_dir, f"filtered_train_{split_name}.pkl"))
            test_filtered.to_pickle(os.path.join(splits_dir, f"filtered_test_{split_name}.pkl"))

        print(f"  - Created leave-one-session-out splits in {splits_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare VAD data from IEMOCAP JSON files")
    parser.add_argument(
        "--json_dir",
        type=str,
        default="../IEMOCAP_json_files",
        help="Path to IEMOCAP JSON files directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--filter_method",
        type=str,
        default="percentile",
        choices=["percentile", "threshold"],
        help="Method for filtering high-agreement samples"
    )
    parser.add_argument(
        "--filter_percentile",
        type=float,
        default=50.0,
        help="Percentile for filtering (used when filter_method='percentile')"
    )
    parser.add_argument(
        "--filter_threshold",
        type=float,
        default=0.5,
        help="Threshold for filtering (used when filter_method='threshold')"
    )
    parser.add_argument(
        "--no_splits",
        action="store_true",
        help="Disable creating train/test splits by session"
    )

    args = parser.parse_args()

    # Load JSON files
    all_json_data = load_iemocap_json_files(args.json_dir)

    # Extract VAD data
    df_full = extract_vad_data(all_json_data)

    # Filter high-agreement samples
    if args.filter_method == "percentile":
        df_filtered = filter_high_agreement_samples(
            df_full,
            percentile=args.filter_percentile,
            method="percentile"
        )
    else:
        df_filtered = filter_high_agreement_samples(
            df_full,
            std_threshold=args.filter_threshold,
            method="threshold"
        )

    # Compute and print statistics
    stats_full = compute_statistics(df_full, "Full VAD Dataset")
    stats_filtered = compute_statistics(df_filtered, "Filtered VAD Dataset (High Agreement)")

    # Save datasets
    save_datasets(
        df_full,
        df_filtered,
        args.output_dir,
        split_by_session=not args.no_splits
    )

    # Save statistics
    stats = {
        "full_dataset": stats_full,
        "filtered_dataset": stats_filtered,
        "filter_settings": {
            "method": args.filter_method,
            "percentile": args.filter_percentile if args.filter_method == "percentile" else None,
            "threshold": args.filter_threshold if args.filter_method == "threshold" else None
        }
    }

    with open(os.path.join(args.output_dir, "vad_statistics.json"), "w") as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        json.dump(convert_numpy(stats), f, indent=2, ensure_ascii=False)

    print(f"\nSaved statistics to {os.path.join(args.output_dir, 'vad_statistics.json')}")
    print("\nVAD data preparation complete!")


if __name__ == "__main__":
    main()
