"""
Error Analysis for Emotion Recognition Experiments
Analyzes:
1. Confusion patterns between emotion pairs (happy/excited, frustrated/sad, neutral)
2. Utterance length impact on accuracy
3. Conversation history availability (0, 1, 2, 3+ context utterances)

Usage:
    python error_analysis_emotion.py                    # Analyze with audio features only
    python error_analysis_emotion.py --include-pure-text  # Include pure text experiments
    python error_analysis_emotion.py --pure-text-only   # Only pure text experiments
"""

import json
import re
import os
import argparse
from collections import defaultdict
from pathlib import Path

# Emotion classes
EMOTIONS = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
EMOTION_ABBREV = {'hap': 'happy', 'sad': 'sad', 'neu': 'neutral',
                  'ang': 'angry', 'exc': 'excited', 'fru': 'frustrated'}


def parse_prediction_file(filepath):
    """Parse a prediction file and extract all samples."""
    samples = []
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the JSON array in the file
    match = re.search(r'\[\s*\{', content)
    if not match:
        return samples

    json_start = match.start()
    json_content = content[json_start:]

    try:
        samples = json.loads(json_content)
    except json.JSONDecodeError:
        # Try to parse line by line if array fails
        lines = json_content.strip().split('\n')
        for line in lines:
            line = line.strip().rstrip(',')
            if line.startswith('{') and line.endswith('}'):
                try:
                    samples.append(json.loads(line))
                except:
                    pass

    return samples


def extract_target_utterance(input_text):
    """Extract the target utterance from the input prompt."""
    # Pattern 1: LLaMA format - Select the emotion of < Speaker_X:"..." >
    pattern1 = r'(?:Select the emotion|emotional label) of\s*<\s*Speaker_\d+:\s*"([^"]+)"'
    match = re.search(pattern1, input_text)
    if match:
        return match.group(1)

    # Pattern 2: GPT format - For  Speaker_X:"...", based on
    pattern2 = r'For\s+Speaker_\d+:\s*"([^"]+)"'
    match = re.search(pattern2, input_text)
    if match:
        return match.group(1)

    return ""


def count_context_utterances(input_text):
    """Count the number of context utterances (speaker turns) in the conversation."""
    # Pattern: Speaker_X:"..." - allow any spacing
    pattern = r'Speaker_\d+:\s*"[^"]*"'
    matches = re.findall(pattern, input_text)
    # Subtract 1 because the last one is the target utterance
    return max(0, len(matches) - 1)


def analyze_confusion_matrix(samples):
    """Build and analyze the confusion matrix."""
    confusion = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        target = sample.get('target', '')
        pred = sample.get('extracted_label', sample.get('output', ''))

        # Clean the prediction
        pred = pred.strip().lower().split()[0] if pred else ''
        target = target.strip().lower()

        if target in EMOTIONS and pred in EMOTIONS:
            confusion[target][pred] += 1

    return confusion


def print_confusion_matrix(confusion, title="Confusion Matrix"):
    """Print a formatted confusion matrix."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)

    # Header
    label = "True\\Pred"
    header = f"{label:<12}" + "".join([f"{e:<10}" for e in EMOTIONS])
    print(header)
    print("-" * len(header))

    # Rows
    for true_label in EMOTIONS:
        row = f"{true_label:<12}"
        for pred_label in EMOTIONS:
            count = confusion[true_label][pred_label]
            row += f"{count:<10}"
        print(row)


def analyze_specific_confusions(confusion):
    """Analyze specific confusion pairs of interest."""
    print("\n" + "="*80)
    print("SPECIFIC CONFUSION ANALYSIS")
    print("="*80)

    # happy vs excited
    hap_as_exc = confusion['happy']['excited']
    exc_as_hap = confusion['excited']['happy']
    hap_total = sum(confusion['happy'].values())
    exc_total = sum(confusion['excited'].values())

    print(f"\n1. Happy <-> Excited Confusion:")
    print(f"   Happy predicted as Excited: {hap_as_exc}/{hap_total} ({100*hap_as_exc/hap_total:.1f}%)" if hap_total > 0 else "   No happy samples")
    print(f"   Excited predicted as Happy: {exc_as_hap}/{exc_total} ({100*exc_as_hap/exc_total:.1f}%)" if exc_total > 0 else "   No excited samples")

    # frustrated vs sad
    fru_as_sad = confusion['frustrated']['sad']
    sad_as_fru = confusion['sad']['frustrated']
    fru_total = sum(confusion['frustrated'].values())
    sad_total = sum(confusion['sad'].values())

    print(f"\n2. Frustrated <-> Sad Confusion:")
    print(f"   Frustrated predicted as Sad: {fru_as_sad}/{fru_total} ({100*fru_as_sad/fru_total:.1f}%)" if fru_total > 0 else "   No frustrated samples")
    print(f"   Sad predicted as Frustrated: {sad_as_fru}/{sad_total} ({100*sad_as_fru/sad_total:.1f}%)" if sad_total > 0 else "   No sad samples")

    # frustrated vs angry
    fru_as_ang = confusion['frustrated']['angry']
    ang_as_fru = confusion['angry']['frustrated']
    ang_total = sum(confusion['angry'].values())

    print(f"\n3. Frustrated <-> Angry Confusion:")
    print(f"   Frustrated predicted as Angry: {fru_as_ang}/{fru_total} ({100*fru_as_ang/fru_total:.1f}%)" if fru_total > 0 else "   No frustrated samples")
    print(f"   Angry predicted as Frustrated: {ang_as_fru}/{ang_total} ({100*ang_as_fru/ang_total:.1f}%)" if ang_total > 0 else "   No angry samples")

    # Neutral misclassifications
    neu_total = sum(confusion['neutral'].values())
    print(f"\n4. Neutral Misclassifications:")
    if neu_total > 0:
        for emotion in EMOTIONS:
            if emotion != 'neutral':
                count = confusion['neutral'][emotion]
                if count > 0:
                    print(f"   Neutral predicted as {emotion}: {count}/{neu_total} ({100*count/neu_total:.1f}%)")

    # What gets predicted as neutral
    print(f"\n5. Predictions as Neutral (when it shouldn't be):")
    for emotion in EMOTIONS:
        if emotion != 'neutral':
            total = sum(confusion[emotion].values())
            count = confusion[emotion]['neutral']
            if count > 0 and total > 0:
                print(f"   {emotion} predicted as Neutral: {count}/{total} ({100*count/total:.1f}%)")


def analyze_utterance_length(samples):
    """Analyze accuracy based on utterance length."""
    print("\n" + "="*80)
    print("UTTERANCE LENGTH ANALYSIS")
    print("="*80)

    length_bins = [
        (0, 20, "Very Short (1-20 chars)"),
        (20, 50, "Short (21-50 chars)"),
        (50, 100, "Medium (51-100 chars)"),
        (100, 200, "Long (101-200 chars)"),
        (200, float('inf'), "Very Long (200+ chars)")
    ]

    results = {name: {'correct': 0, 'total': 0, 'samples': []} for _, _, name in length_bins}

    for sample in samples:
        target = sample.get('target', '').strip().lower()
        pred = sample.get('extracted_label', sample.get('output', '')).strip().lower().split()[0] if sample.get('extracted_label') or sample.get('output') else ''
        utterance = extract_target_utterance(sample.get('input', ''))

        if not target or not pred or target not in EMOTIONS or pred not in EMOTIONS:
            continue

        length = len(utterance)

        for min_len, max_len, name in length_bins:
            if min_len < length <= max_len:
                results[name]['total'] += 1
                if target == pred:
                    results[name]['correct'] += 1
                else:
                    results[name]['samples'].append({
                        'utterance': utterance[:50] + "..." if len(utterance) > 50 else utterance,
                        'target': target,
                        'pred': pred
                    })
                break

    print(f"\n{'Length Category':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 55)

    for _, _, name in length_bins:
        data = results[name]
        acc = 100 * data['correct'] / data['total'] if data['total'] > 0 else 0
        print(f"{name:<25} {data['correct']:<10} {data['total']:<10} {acc:.1f}%")

    return results


def analyze_context_history(samples):
    """Analyze accuracy based on number of context utterances."""
    print("\n" + "="*80)
    print("CONVERSATION HISTORY ANALYSIS")
    print("="*80)

    context_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'by_emotion': defaultdict(lambda: {'correct': 0, 'total': 0})})

    for sample in samples:
        target = sample.get('target', '').strip().lower()
        pred = sample.get('extracted_label', sample.get('output', '')).strip().lower().split()[0] if sample.get('extracted_label') or sample.get('output') else ''

        if not target or not pred or target not in EMOTIONS or pred not in EMOTIONS:
            continue

        num_context = count_context_utterances(sample.get('input', ''))

        # Bin context counts: 0, 1, 2, 3+
        context_bin = min(num_context, 3)
        if context_bin >= 3:
            context_key = "3+ utterances"
        else:
            context_key = f"{context_bin} utterances"

        context_results[context_key]['total'] += 1
        context_results[context_key]['by_emotion'][target]['total'] += 1

        if target == pred:
            context_results[context_key]['correct'] += 1
            context_results[context_key]['by_emotion'][target]['correct'] += 1

    print(f"\n{'Context History':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 50)

    for key in ["0 utterances", "1 utterances", "2 utterances", "3+ utterances"]:
        if key in context_results:
            data = context_results[key]
            acc = 100 * data['correct'] / data['total'] if data['total'] > 0 else 0
            print(f"{key:<20} {data['correct']:<10} {data['total']:<10} {acc:.1f}%")

    # Breakdown by emotion for each context level
    print(f"\n{'Context':<15} {'Emotion':<12} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 57)

    for key in ["0 utterances", "1 utterances", "2 utterances", "3+ utterances"]:
        if key in context_results:
            for emotion in EMOTIONS:
                data = context_results[key]['by_emotion'][emotion]
                if data['total'] > 0:
                    acc = 100 * data['correct'] / data['total']
                    print(f"{key:<15} {emotion:<12} {data['correct']:<10} {data['total']:<10} {acc:.1f}%")

    return context_results


def analyze_single_experiment(filepath, experiment_name):
    """Analyze a single experiment file."""
    print("\n" + "#"*80)
    print(f"# EXPERIMENT: {experiment_name}")
    print("#"*80)

    samples = parse_prediction_file(filepath)
    if not samples:
        print(f"No samples found in {filepath}")
        return None

    print(f"Total samples: {len(samples)}")

    # Calculate overall accuracy
    correct = sum(1 for s in samples
                  if s.get('target', '').strip().lower() ==
                  (s.get('extracted_label', s.get('output', '')).strip().lower().split()[0] if s.get('extracted_label') or s.get('output') else ''))
    print(f"Overall Accuracy: {100*correct/len(samples):.2f}%")

    # Confusion matrix
    confusion = analyze_confusion_matrix(samples)
    print_confusion_matrix(confusion)
    analyze_specific_confusions(confusion)

    # Utterance length analysis
    length_results = analyze_utterance_length(samples)

    # Context history analysis
    context_results = analyze_context_history(samples)

    return {
        'samples': samples,
        'confusion': confusion,
        'length_results': length_results,
        'context_results': context_results,
        'accuracy': 100*correct/len(samples)
    }


def find_best_checkpoint(experiment_dir):
    """Find the best checkpoint (highest accuracy) in an experiment directory."""
    best_acc = -1
    best_file = None

    if not os.path.exists(experiment_dir):
        return None, -1

    for f in os.listdir(experiment_dir):
        if f.startswith('preds_for_eval') and f.endswith('.text'):
            filepath = os.path.join(experiment_dir, f)
            with open(filepath, 'r') as file:
                first_line = file.readline()
                try:
                    data = json.loads(first_line)
                    acc = data.get('Acc_SA', 0)
                    if acc > best_acc:
                        best_acc = acc
                        best_file = filepath
                except:
                    pass

    return best_file, best_acc


def find_prediction_file(experiment_dir):
    """Find a single prediction file for zero-shot/few-shot experiments."""
    if not os.path.exists(experiment_dir):
        return None, -1

    for f in os.listdir(experiment_dir):
        if f.startswith('preds_for_eval') and f.endswith('.text'):
            filepath = os.path.join(experiment_dir, f)
            with open(filepath, 'r') as file:
                first_line = file.readline()
                try:
                    data = json.loads(first_line)
                    acc = data.get('Acc_SA', 0)
                    return filepath, acc
                except:
                    pass
    return None, -1


def get_experiments(base_dir, include_pure_text=False, pure_text_only=False):
    """Get all experiment configurations."""
    experiments = {}

    if not pure_text_only:
        # With audio features - LoRA experiments
        experiments.update({
            "LLaMA2-7B LoRA (audio)": (
                base_dir / "experiments/LLaMA2/lora/iemocap/window_12/LR_3e-4_BS_8_per_1.0_des_context_class5_11",
                "lora"
            ),
            "LLaMA3.1-8B LoRA (audio)": (
                base_dir / "experiments/LLaMA3-instruct/lora/iemocap/window_12/LR_3e-4_BS_8_per_1.0_des_context_class5_42",
                "lora"
            ),
            "LLaMA3.3-70B LoRA (audio)": (
                base_dir / "experiments/LLaMA3-instruct-70b/lora/lora/iemocap/window_12/LR_3e-4_BS_8_per_1.0_des_context_class5_11",
                "lora"
            ),
        })

        # GPT zero-shot experiments
        experiments.update({
            "GPT-4o-mini Zero-Shot (audio)": (
                base_dir / "experiments/OpenAI-gpt-4o-mini/zero_shot/iemocap/window_12/per_1.0_des_context_class5_11",
                "single"
            ),
            "GPT-4o-mini Few-Shot (audio)": (
                base_dir / "experiments/OpenAI-gpt-4o-mini/few_shot/iemocap/window_12/per_1.0_des_context_class5_11",
                "single"
            ),
            "GPT-5-mini Zero-Shot (audio)": (
                base_dir / "experiments/OpenAI-gpt-5-mini/zero_shot/iemocap/window_12/per_1.0_des_context_class5_11",
                "single"
            ),
            "GPT-5-mini Few-Shot (audio)": (
                base_dir / "experiments/OpenAI-gpt-5-mini/few_shot/iemocap/window_12/per_1.0_des_context_class5_11",
                "single"
            ),
        })

    # Pure text experiments (without audio features)
    if include_pure_text or pure_text_only:
        experiments.update({
            "LLaMA2-7B LoRA (text only)": (
                base_dir / "experiments_pure_text/LLaMA2/iemocap/window_12/LR_3e-4_BS_8_per_1.0_des_context_class5_11",
                "lora"
            ),
            "LLaMA3.1-8B LoRA (text only)": (
                base_dir / "experiments_pure_text/LLaMA3-instruct/lora/iemocap/window_12/LR_3e-4_BS_8_per_1.0_des_context_class5_11",
                "lora"
            ),
        })

    return experiments


def print_comparative_summary(all_results):
    """Print a comparative summary of all experiments."""
    print("\n" + "="*100)
    print("COMPARATIVE SUMMARY")
    print("="*100)

    # Overall accuracy comparison
    print("\n--- Overall Accuracy ---")
    print(f"{'Experiment':<45} {'Accuracy':<10}")
    print("-" * 55)
    for exp_name in sorted(all_results.keys()):
        acc = all_results[exp_name].get('accuracy', 0)
        print(f"{exp_name:<45} {acc:.2f}%")

    # Compare happy/excited confusion across models
    print("\n--- Happy <-> Excited Confusion Rate ---")
    print(f"{'Experiment':<45} {'Hap→Exc':<12} {'Exc→Hap':<12}")
    print("-" * 69)
    for exp_name in sorted(all_results.keys()):
        confusion = all_results[exp_name]['confusion']
        exc_total = sum(confusion['excited'].values())
        hap_total = sum(confusion['happy'].values())
        exc_as_hap = confusion['excited']['happy']
        hap_as_exc = confusion['happy']['excited']

        hap_exc_rate = f"{100*hap_as_exc/hap_total:.1f}%" if hap_total > 0 else "N/A"
        exc_hap_rate = f"{100*exc_as_hap/exc_total:.1f}%" if exc_total > 0 else "N/A"
        print(f"{exp_name:<45} {hap_exc_rate:<12} {exc_hap_rate:<12}")

    # Compare frustrated/angry confusion across models
    print("\n--- Frustrated <-> Angry Confusion Rate ---")
    print(f"{'Experiment':<45} {'Fru→Ang':<12} {'Ang→Fru':<12}")
    print("-" * 69)
    for exp_name in sorted(all_results.keys()):
        confusion = all_results[exp_name]['confusion']
        fru_total = sum(confusion['frustrated'].values())
        ang_total = sum(confusion['angry'].values())
        fru_as_ang = confusion['frustrated']['angry']
        ang_as_fru = confusion['angry']['frustrated']

        fru_ang_rate = f"{100*fru_as_ang/fru_total:.1f}%" if fru_total > 0 else "N/A"
        ang_fru_rate = f"{100*ang_as_fru/ang_total:.1f}%" if ang_total > 0 else "N/A"
        print(f"{exp_name:<45} {fru_ang_rate:<12} {ang_fru_rate:<12}")

    # Compare sad/frustrated confusion across models
    print("\n--- Sad <-> Frustrated Confusion Rate ---")
    print(f"{'Experiment':<45} {'Sad→Fru':<12} {'Fru→Sad':<12}")
    print("-" * 69)
    for exp_name in sorted(all_results.keys()):
        confusion = all_results[exp_name]['confusion']
        sad_total = sum(confusion['sad'].values())
        fru_total = sum(confusion['frustrated'].values())
        sad_as_fru = confusion['sad']['frustrated']
        fru_as_sad = confusion['frustrated']['sad']

        sad_fru_rate = f"{100*sad_as_fru/sad_total:.1f}%" if sad_total > 0 else "N/A"
        fru_sad_rate = f"{100*fru_as_sad/fru_total:.1f}%" if fru_total > 0 else "N/A"
        print(f"{exp_name:<45} {sad_fru_rate:<12} {fru_sad_rate:<12}")

    # Per-emotion accuracy comparison
    print("\n--- Per-Emotion Accuracy ---")
    header = f"{'Experiment':<40}"
    for emotion in EMOTIONS:
        header += f"{emotion[:3]:<8}"
    print(header)
    print("-" * (40 + 8 * len(EMOTIONS)))

    for exp_name in sorted(all_results.keys()):
        confusion = all_results[exp_name]['confusion']
        row = f"{exp_name:<40}"
        for emotion in EMOTIONS:
            total = sum(confusion[emotion].values())
            correct = confusion[emotion][emotion]
            acc = f"{100*correct/total:.0f}%" if total > 0 else "N/A"
            row += f"{acc:<8}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Error analysis for emotion recognition experiments')
    parser.add_argument('--include-pure-text', action='store_true',
                        help='Include pure text experiments (without audio features)')
    parser.add_argument('--pure-text-only', action='store_true',
                        help='Only analyze pure text experiments')
    args = parser.parse_args()

    base_dir = Path("/Users/nataliehu/Desktop/emory/Honor Thesis/SpeechCueLLM")

    experiments = get_experiments(base_dir, args.include_pure_text, args.pure_text_only)

    all_results = {}

    for exp_name, (exp_dir, exp_type) in experiments.items():
        if exp_type == "lora":
            best_file, best_acc = find_best_checkpoint(exp_dir)
        else:
            best_file, best_acc = find_prediction_file(exp_dir)

        if best_file:
            print(f"\n\nUsing checkpoint: {Path(best_file).name} (Acc: {best_acc:.2f}%)")
            results = analyze_single_experiment(best_file, exp_name)
            if results:
                all_results[exp_name] = results
        else:
            print(f"\n[SKIP] No prediction file found for: {exp_name}")
            print(f"       Directory: {exp_dir}")

    if all_results:
        print_comparative_summary(all_results)


if __name__ == "__main__":
    main()
