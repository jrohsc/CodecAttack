import argparse
import json
import re
from collections import defaultdict


def extract_prefix_number(s: str):
    """Extracts the first integer from the start of a string."""
    if not isinstance(s, str):
        return None
    match = re.match(r'^\d+', s.strip())
    return int(match.group()) if match else None


def get_task_id(task_name: str) -> str:
    """Maps detailed task names to broader categories."""
    if not isinstance(task_name, str):
        return "unknown"
    if 'speech_dialogue' in task_name or 'speech_QA' in task_name:
        return 'speech'
    if 'music' in task_name:
        return 'music'
    if 'sound' in task_name:
        return 'sound'
    if 'speech_and_music' in task_name:
        return 'speech_and_music'
    if 'speech_and_sound' in task_name:
        return 'speech_and_sound'
    return 'other'


def main():
    parser = argparse.ArgumentParser(description="Calculate scores from LLM judge outputs for AIR-Bench Chat.")
    parser.add_argument("--input_file", type=str, default="./logs/Chat_result/Chat_result_prefix_scored.jsonl")
    args = parser.parse_args()

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {args.input_file}")
        return

    stats = defaultdict(lambda: {"total": 0, "wins": 0, "gt_score_sum": 0, "model_score_sum": 0})
    fail_num = 0

    for item in data:
        task_id = get_task_id(item.get('task_name'))
        judge_output = item.get('gen', '').strip()
        scores_str = judge_output.split()

        if len(scores_str) < 2:
            print(f"[WARN] Failed to parse scores from: '{judge_output}' (not enough values)")
            fail_num += 1
            continue

        gt_score = extract_prefix_number(scores_str[0])
        model_score = extract_prefix_number(scores_str[1])

        if gt_score is None or model_score is None or not (0 <= gt_score <= 10 and 0 <= model_score <= 10):
            print(f"[WARN] Failed to parse valid scores (0-10) from: '{judge_output}'")
            fail_num += 1
            continue

        stats[task_id]["total"] += 1
        stats[task_id]["gt_score_sum"] += gt_score
        stats[task_id]["model_score_sum"] += model_score
        if model_score > gt_score:
            stats[task_id]["wins"] += 1

    # --- Print Results ---
    print("\n--- AIR-Bench Chat Evaluation Results ---")
    print(f"Source: {args.input_file}\n")
    header = f"{'Task':<25} | {'Total':>7} | {'Win Rate':>10} | {'GT Avg Score':>14} | {'Model Avg Score':>17}"
    print(header)
    print("-" * len(header))

    total_samples = 0
    total_wins = 0

    # Sort tasks for consistent output order
    sorted_tasks = sorted(stats.keys())

    for task_id in sorted_tasks:
        s = stats[task_id]
        if s['total'] == 0:
            continue
        
        win_rate = s['wins'] / s['total']
        gt_avg = s['gt_score_sum'] / s['total']
        model_avg = s['model_score_sum'] / s['total']
        
        total_samples += s['total']
        total_wins += s['wins']

        print(f"{task_id:<25} | {s['total']:>7} | {win_rate:>10.4f} | {gt_avg:>14.2f} | {model_avg:>17.2f}")

    print("-" * len(header))
    if total_samples > 0:
        overall_win_rate = total_wins / total_samples
        print(f"Overall Win Rate: {overall_win_rate:.4f} ({total_wins}/{total_samples})\n")
    
    if fail_num > 0:
        print(f"Failed to parse scores for {fail_num} items.")

if __name__ == "__main__":
    main()
