import random
import hotpotqa_agent as hqa # Import our library

# --- Main Configuration ---
NUM_TASKS_TODAY = 10
BASELINE_OUTPUT_FILE = 'react_baseline_results.json'
NEW_FRAMEWORK_OUTPUT_FILE = 'react_cot_synth_results.json'

print("Setting up dataset indices...")
all_indices = list(range(7405))
random.Random(42).shuffle(all_indices)

indices_for_today = all_indices[:NUM_TASKS_TODAY]
print(f"Prepared to run {len(indices_for_today)} tasks.")

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
    # Pass the loaded prompt template from hqa module
    synthesized_answer, multi_trace_info = hqa.webthink(idx=idx, initial_prompt_template=hqa.WEBTHINK_PROMPT_TEMPLATE, to_print=False, num_traces=3)
    new_framework_result = {
        'question_idx': idx,
        'question_text': multi_trace_info[0].get('question_text') if multi_trace_info else None,
        'synthesized_answer': synthesized_answer,
        'ground_truth_answer': multi_trace_info[0].get('gt_answer') if multi_trace_info else None,
        'traces': multi_trace_info
    }
    hqa.append_to_json(new_framework_result, NEW_FRAMEWORK_OUTPUT_FILE)
    print(f"  > New framework results saved to {NEW_FRAMEWORK_OUTPUT_FILE}")
    print("-" * 20)

print("\nAll tasks for today completed!")
