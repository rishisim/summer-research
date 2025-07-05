import random
import json
import os
import sys

# Ensure the FEVER_Experiment directory is in the Python path
# to allow importing fever_agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import fever_agent as fa
except ImportError as e:
    print(f"Error importing fever_agent: {e}")
    print("Make sure fever_agent.py is in the FEVER_Experiment directory.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during fever_agent import: {e}")
    sys.exit(1)


# --- Main Configuration ---
# The FEVER dev set (paper_dev.jsonl) has 19998 claims.
# The notebook mentioned 7405, which might be a different split or an older version.
# We'll use the count from the `FeverWrapper` default file if possible, or a safe number.
# For now, let's assume the indices correspond to lines in `paper_dev.jsonl`.
# The `FeverWrapper` handles the actual loading based on index.
MAX_FEVER_DEV_EXAMPLES = 19998 # Based on paper_dev.jsonl line count

NUM_TASKS_TODAY = 3 # Keep small for testing, can be increased later
BASELINE_OUTPUT_FILE = 'react_baseline_results.json' # Path relative to FEVER_Experiment
NEW_FRAMEWORK_OUTPUT_FILE = 'react_multi_trace_results.json' # Path relative to FEVER_Experiment

# Ensure output files are saved within the FEVER_Experiment directory
BASELINE_OUTPUT_FILE_PATH = os.path.join(os.path.dirname(__file__), BASELINE_OUTPUT_FILE)
NEW_FRAMEWORK_OUTPUT_FILE_PATH = os.path.join(os.path.dirname(__file__), NEW_FRAMEWORK_OUTPUT_FILE)


print("Setting up dataset indices for FEVER...")
# Ensure indices are within the valid range for the FEVER dev set
# The FeverWrapper in wikienv.py loads 'data/paper_dev.jsonl' which has 19998 lines (0-19997)
# If your specific FEVER dataset has a different size, adjust MAX_FEVER_DEV_EXAMPLES.
all_indices = list(range(MAX_FEVER_DEV_EXAMPLES)) # Indices from 0 to MAX_FEVER_DEV_EXAMPLES-1
random.Random(42).shuffle(all_indices)

# Ensure NUM_TASKS_TODAY does not exceed available indices
if NUM_TASKS_TODAY > len(all_indices):
    print(f"Warning: NUM_TASKS_TODAY ({NUM_TASKS_TODAY}) is greater than available examples ({len(all_indices)}).")
    print(f"Adjusting NUM_TASKS_TODAY to {len(all_indices)}.")
    NUM_TASKS_TODAY = len(all_indices)

indices_for_today = all_indices[:NUM_TASKS_TODAY]
print(f"Prepared to run {len(indices_for_today)} tasks on FEVER dev set.")
print(f"Selected indices: {indices_for_today}")

# --- Ensure GEMINI_API_KEY is set ---
if not os.getenv('GEMINI_API_KEY'):
    print("CRITICAL: GEMINI_API_KEY environment variable not set.")
    print("Please set this variable before running experiments.")
    sys.exit(1)
else:
    print("GEMINI_API_KEY found.")

# --- Load Prompt Template from Agent ---
# The fever_agent loads its own prompt. We just need to ensure it's accessible.
# The WEBTHINK_PROMPT_TEMPLATE is loaded in fever_agent.py itself.
# We can pass `None` to `webthink` for `initial_prompt_template` to use its default.
# Or, if fever_agent exposes it, we can use that. Let's assume fever_agent handles it.

print(f"Using prompt template defined within fever_agent.py: {fa.WEBTHINK_PROMPT_TEMPLATE[:100]}...") # Print a snippet

# --- Main Execution Loop ---
for i, idx in enumerate(indices_for_today):
    print(f"\n--- Processing Task {i+1}/{NUM_TASKS_TODAY} (FEVER Index: {idx}) ---")

    # 1. Run Baseline ReAct (1 trace)
    print(f"Running baseline ReAct (num_traces=1) for index {idx}...")
    try:
        # Pass None for initial_prompt_template to use the one loaded in fever_agent
        _, baseline_info = fa.webthink(idx=idx, initial_prompt_template=fa.WEBTHINK_PROMPT_TEMPLATE, to_print=True, num_traces=1)

        # Ensure results are JSON serializable (especially float32/64 to float)
        baseline_info_serializable = json.loads(json.dumps(baseline_info, default=lambda o: float(o) if hasattr(o, 'item') else str(o))) # Handle numpy types

        fa.append_to_json(baseline_info_serializable, BASELINE_OUTPUT_FILE_PATH)
        print(f"  > Baseline results for index {idx} saved to {BASELINE_OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"  ERROR during baseline ReAct for index {idx}: {e}")
        error_info = {'question_idx': idx, 'error': str(e), 'details': 'Baseline ReAct failed'}
        fa.append_to_json(error_info, BASELINE_OUTPUT_FILE_PATH)

    # 2. Run New Framework (e.g., 3 traces)
    NUM_TRACES_NEW_FRAMEWORK = 3
    print(f"Running new framework (num_traces={NUM_TRACES_NEW_FRAMEWORK}) for index {idx}...")
    try:
        # Pass None for initial_prompt_template
        synthesized_reward, multi_trace_info = fa.webthink(idx=idx, initial_prompt_template=fa.WEBTHINK_PROMPT_TEMPLATE, to_print=True, num_traces=NUM_TRACES_NEW_FRAMEWORK)

        # The multi_trace_info should already be the synthesized summary if num_traces > 1
        # Ensure it's serializable
        multi_trace_info_serializable = json.loads(json.dumps(multi_trace_info, default=lambda o: float(o) if hasattr(o, 'item') else str(o)))

        # The structure for FEVER's multi-trace info is already good for saving:
        # It includes 'question_idx', 'synthesized_answer', 'gt_answer', 'individual_traces', etc.
        fa.append_to_json(multi_trace_info_serializable, NEW_FRAMEWORK_OUTPUT_FILE_PATH)
        print(f"  > New framework results for index {idx} saved to {NEW_FRAMEWORK_OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"  ERROR during new framework for index {idx}: {e}")
        error_info = {'question_idx': idx, 'error': str(e), 'details': f'New framework ({NUM_TRACES_NEW_FRAMEWORK} traces) failed'}
        fa.append_to_json(error_info, NEW_FRAMEWORK_OUTPUT_FILE_PATH)

    print("-" * 30)

print("\nAll FEVER tasks for today completed!")
print(f"Baseline results are in: {BASELINE_OUTPUT_FILE_PATH}")
print(f"New framework results are in: {NEW_FRAMEWORK_OUTPUT_FILE_PATH}")
