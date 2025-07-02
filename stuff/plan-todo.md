# Implementation Plan


Okay, so CoT_SC was implemented with a synthesized answer at the end basically taking in the best of all three final outputs of the reasoning traces (takes in context as question). ReAct was benchmarked through EM match with the HotPotQA dataset, and question idx 1 seems to be an open ended question and actually the LLM can answer it correctly, but its not an exact match (seen in example below. 

- We need to run the 500 tasks test with typical ReAct (reasoning trace = 1), run it with ReAct + CoT_SC (reasoning traces = 3) and find the exact match metrics.
- Decide what other evaluation metrics to use because its open-ended (perhaps look at what other metrics ReAct used for other datasets, research BertScore, Semantic Similarity, and more).

Example:
```
idxs = list(range(7405)) # Ensure idxs is defined

# Example of running with multiple traces (num_traces > 1)
print("--- Testing webthink with num_traces=3 ---")
synthesized_answer, traces_list = webthink(idx=idxs[1], to_print=True, num_traces=3)

print(f"\n--- Synthesized Answer for idxs[0] (num_traces=3) ---")
print(synthesized_answer)
print("---")

if isinstance(traces_list, list):
    for i, trace_info in enumerate(traces_list):
        print(f"Details for Trace {i+1}:")
        print(f"  Question Index: {trace_info.get('question_idx')}")
        print(f"  Question Text: {trace_info.get('question_text')}")
        print(f"  Trace Answer: {trace_info.get('answer')}") 
        print(f"  Ground Truth: {trace_info.get('gt_answer')}") # ADDED
        print(f"  Trace Reward: {trace_info.get('reward')}")
        print(f"  Trace EM: {trace_info.get('em')}")
        print(f"  Trace F1: {trace_info.get('f1')}")
        print("  ---")
else:
    print("Error: Expected a list of traces as the second part of the tuple.")
    print(f"Received for traces_list: {traces_list}")

# Example of running with a single trace (num_traces = 1)
print("\n--- Testing webthink with num_traces=1 ---")
reward_single, info_single = webthink(idx=idxs[1], to_print=True, num_traces=1)

print(f"\n--- Single Trace Result for idxs[1] (num_traces=1) ---")
print(f"Reward: {reward_single}")
if info_single and isinstance(info_single, dict):
    print(f"  Question Index: {info_single.get('question_idx')}")
    print(f"  Question Text: {info_single.get('question_text')}")
    print(f"  Answer: {info_single.get('answer')}")
    print(f"  Ground Truth: {info_single.get('gt_answer')}") # ADDED
    print(f"  EM: {info_single.get('em')}")
    print(f"  F1: {info_single.get('f1')}")
else:
    print("Error: Received no valid info_single for single trace run.")
```

```
# Cell A: Demonstrate single trace for idxs[0]
print("--- Running webthink with num_traces=1 for idxs[0] ---")
if 'idxs' not in globals():
    print("WARNING: 'idxs' not found globally. Assuming it will be defined by a preceding cell.")
    print("For standalone execution of this cell, ensure 'idxs', 'env', 'llm', and 'webthink' are defined.")
    idxs = list(range(7405))
if 'webthink' in globals() and 'env' in globals() and 'llm' in globals() and 'idxs' in globals():
    print(f"Using idxs[1] which is: {idxs[1]}")
    r_single, info_single = webthink(idx=idxs[1], to_print=True, num_traces=1)
    print("\n--- Single Trace (idxs[0]) Summary ---")
    print(f"Reward: {r_single}")
    if info_single and isinstance(info_single, dict):
        print(f"Answer: {info_single.get('answer')}")
        print(f"EM: {info_single.get('em')}")
        print(f"F1: {info_single.get('f1')}")
        print(f"Num Calls: {info_single.get('n_calls')}")
    else:
        print(f"Error: info_single was not a valid dictionary. Value: {info_single}")
else:
    print("ERROR: One or more required components (idxs, webthink, env, llm) are not defined.")
    print("Please ensure all preceding setup cells are executed.")
```

## Phase 1:
![image](https://github.com/user-attachments/assets/03ef81b6-71ea-4ea5-8c82-3ca4943f8117)

- Determine what LLM to choose
- Determine what environment to work in (do we need access to sol or can we just do it locally/colab)
- Implement ReAct with the LLM
- Determine how to use CoT_SC (exactly what it is and think about practicality (how many reasoning traces, how to determine which to choose, etc.))
- Determine a more detailed framework of how CoT_SC works with ReAct
- Implement CoT_SC with ReAct: Updated ReAct.
