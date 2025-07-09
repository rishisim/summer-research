import random
import hotpotqa_agent as hqa # Import our library
import json
import os

def get_processed_indices(output_file_path):
    processed_indices = set()
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for entry in data:
                            idx = entry.get('question_idx')
                            if idx is not None:
                                processed_indices.add(idx)
                    elif isinstance(data, dict):
                        idx = data.get('question_idx')
                        if idx is not None:
                            processed_indices.add(idx)
                except Exception as e:
                    print(f"Warning: Could not parse {output_file_path} as JSON array: {e}")
        except Exception as e:
            print(f"Warning: Could not read {output_file_path}: {e}")
    return processed_indices

# --- Main Configuration ---
NUM_TASKS_TODAY = 20
BASELINE_OUTPUT_FILE = 'HotPotQA_Experiment/react_baseline_results.json'
NEW_FRAMEWORK_OUTPUT_FILE = 'HotPotQA_Experiment/react_cot_synth_results.json'

print("Setting up dataset indices...")
all_indices = list(range(7405))
random.Random(42).shuffle(all_indices)

# Exclude already processed indices
processed_indices = get_processed_indices(BASELINE_OUTPUT_FILE) | get_processed_indices(NEW_FRAMEWORK_OUTPUT_FILE)
remaining_indices = [idx for idx in all_indices if idx not in processed_indices]
indices_for_today = remaining_indices[:NUM_TASKS_TODAY]
print(f"Prepared to run {len(indices_for_today)} new tasks (skipping {len(processed_indices)} already processed).")

# --- Main Execution Loop ---
for i, idx in enumerate(indices_for_today):
    print(f"--- Processing Task {i+1}/{NUM_TASKS_TODAY} (Index: {idx}) ---")

    # 1. Run Baseline ReAct
    print("Running baseline ReAct (num_traces=1)...")
    # Pass the loaded prompt template from hqa module
    _, baseline_info = hqa.webthink(idx=idx, initial_prompt_template=hqa.WEBTHINK_PROMPT_TEMPLATE, to_print=False, num_traces=1)
    hqa.append_to_json(baseline_info, BASELINE_OUTPUT_FILE)
    print(f"  > Baseline results saved to {BASELINE_OUTPUT_FILE}")

    # 2. Run New Framework
    print("Running new framework (num_traces=3)...")
    synthesized_answer, multi_trace_info = hqa.webthink(idx=idx, initial_prompt_template=hqa.WEBTHINK_PROMPT_TEMPLATE, to_print=False, num_traces=3)
    # Unwrap to get the correct environment with get_metrics
    hotpot_env = hqa.env
    while hasattr(hotpot_env, 'env') and not hasattr(hotpot_env, 'get_metrics'):
        hotpot_env = hotpot_env.env
    gt_answer = multi_trace_info[0].get('gt_answer') if multi_trace_info else None
    metrics = hotpot_env.get_metrics({'answer': synthesized_answer}) if synthesized_answer and hasattr(hotpot_env, 'get_metrics') else {'em': 0, 'f1': 0, 'reward': 0}
    new_framework_result = {
        'question_idx': idx,
        'question_text': multi_trace_info[0].get('question_text') if multi_trace_info else None,
        'synthesized_answer': synthesized_answer,
        'ground_truth_answer': gt_answer,
        'em': metrics['em'],
        'reward': metrics['reward'],
        'f1': metrics['f1'],
        'traces': multi_trace_info
    }
    hqa.append_to_json(new_framework_result, NEW_FRAMEWORK_OUTPUT_FILE)
    print(f"  > New framework results saved to {NEW_FRAMEWORK_OUTPUT_FILE}")
    print("-" * 20)

print("\nAll tasks for today completed!")
