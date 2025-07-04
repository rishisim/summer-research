import os
import json
import time
import requests
from google import genai
from google.genai import types
import wikienv, wrappers

# --- Environment Setup ---
env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

# --- LLM and Helper Functions ---
client = genai.Client() # Assuming API key is in env
# client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"]) # for scripts

def llm(prompt, stop=["\n"], num_traces=1):
  # This delay handles the 15 RPM limit by waiting ~4 seconds per call.
  time.sleep(4.1)

  temperature_setting = 0.0 if num_traces == 1 else 0.7
  response = client.models.generate_content(
    model="gemini-2.5-flash-lite-preview-06-17",
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
        stop_sequences=stop,
        temperature=temperature_setting,
        max_output_tokens=100,
        top_p=1.0
    )
  )
  return response.text

import re

def extract_final_answer_from_trace_string(trace_trajectory_string):
    """
    Extracts the final answer from a ReAct trace trajectory string.
    Looks for the last occurrence of 'Action X: Finish[answer]'.
    """
    pattern = re.compile(r"^Action \d+: Finish\[(.*?)\]\s*$", re.MULTILINE)
    matches = pattern.findall(trace_trajectory_string)

    if matches:
        # The last match in the string is the one we want
        return matches[-1].strip()

    return None

def extract_answers_from_traces(all_traces_info):
    """
    Extracts the final answer from each trace in the all_traces_info list.
    """
    extracted_answers = []
    if not isinstance(all_traces_info, list):
        print(f"Warning: extract_answers_from_traces expected a list, got {type(all_traces_info)}")
        return extracted_answers

    for i, trace_info in enumerate(all_traces_info):
        trajectory = trace_info.get('traj', '')
        answer_from_traj = extract_final_answer_from_trace_string(trajectory)

        if answer_from_traj is not None:
            extracted_answers.append(answer_from_traj)
        else:
            env_answer = trace_info.get('answer')
            if env_answer:
                extracted_answers.append(env_answer)
            else:
                extracted_answers.append(None)
    return [ans for ans in extracted_answers if ans is not None]

def synthesize_answer_with_llm(list_of_answers, question_for_context=""):
    """
    Synthesizes a single best answer from a list of answers using an LLM.
    Includes the original question for better context if provided.
    """
    if not list_of_answers:
        return "Error: No answers provided to synthesize."

    unique_answers = sorted(list(set(str(a).strip() for a in list_of_answers if str(a).strip())))
    if len(unique_answers) == 0:
        return "Error: No valid answers found after filtering to synthesize."
    if len(unique_answers) == 1:
        return unique_answers[0]

    prompt_template = """As an expert analyst, your task is to determine the single best answer from the following list, which was generated in response to the same question.\n{question_context}\nReview all answers, identify the most consistent and factually correct choice, and return that single answer. For fixed-choice questions (like yes/no or numbers), this will be a majority vote. For text-based answers, synthesize the information into the most accurate and complete response. Ignore any clear outliers or factually incorrect statements.\n\nGenerated Answers:\n{formatted_answers}\n\nFinal Answer:"""

    question_context_str = ""
    if question_for_context:
        question_context_str = f"The question asked was: \"{question_for_context}\"\n\n"

    formatted_answers = ""
    for i, ans in enumerate(list_of_answers):
        formatted_answers += f"{i+1}. {ans}\n"
    formatted_answers = formatted_answers.strip()

    synthesizer_prompt = prompt_template.format(
        question_context=question_context_str,
        formatted_answers=formatted_answers
    )

    final_answer = llm(synthesizer_prompt, num_traces=1)
    return final_answer.strip()

def append_to_json(data_dict, json_file_path):
    if os.path.exists(json_file_path) and os.path.getsize(json_file_path) > 0:
        with open(json_file_path, 'r') as f:
            try:
                results_list = json.load(f)
            except json.JSONDecodeError:
                results_list = []
    else:
        results_list = []

    results_list.append(data_dict)

    with open(json_file_path, 'w') as f:
        json.dump(results_list, f, indent=4)

# --- Webthink Agent ---
# Load prompts
prompt_file_path = os.path.join(os.path.dirname(__file__), 'prompts_naive.json')

try:
    with open(prompt_file_path, 'r') as f:
        prompt_dict = json.load(f)
    webthink_examples = prompt_dict['webthink_simple6']
    instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: \n(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.\n(3) Finish[answer], which returns the answer and finishes the task.\nHere are some examples.\n"""
    WEBTHINK_PROMPT_TEMPLATE = instruction + webthink_examples
except FileNotFoundError:
    print(f"ERROR: Prompt file {prompt_file_path} not found. Webthink might not work correctly.")
    WEBTHINK_PROMPT_TEMPLATE = "ERROR_PROMPT_FILE_NOT_FOUND" # Fallback
except KeyError:
    print(f"ERROR: Key 'webthink_simple6' not found in {prompt_file_path}. Webthink might not work correctly.")
    WEBTHINK_PROMPT_TEMPLATE = "ERROR_PROMPT_KEY_NOT_FOUND" # Fallback


def webthink(idx=None, initial_prompt_template=WEBTHINK_PROMPT_TEMPLATE, to_print=True, num_traces=1):
    all_traces_info = []
    question_for_synthesis = "" # Define outside loop to store it

    if num_traces <= 0:
        if to_print:
            print(f"Warning: webthink called with num_traces = {num_traces}. Must be > 0.")
        return "[INVALID_NUM_TRACES]", []

    for trace_num in range(num_traces):
        question = env.reset(idx=idx) # Reset environment for each trace
        if trace_num == 0: # Capture question on first trace for synthesizer
            question_for_synthesis = question

        current_prompt = initial_prompt_template + question + "\n"

        if to_print:
            print(f"--- Trace {trace_num + 1}/{num_traces} ---")
            print(idx, question)

        n_calls, n_badcalls = 0, 0

        for i in range(1, 8): # Max 7 steps per trace
            n_calls += 1
            thought_action = llm(current_prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"], num_traces=1 if num_traces == 1 else 0.7) # Pass num_traces to llm correctly
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
            except:
                n_badcalls += 1
                n_calls += 1 # LLM call for action also counts
                thought = thought_action.strip().split('\n')[0]
                action = llm(current_prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"], num_traces=1 if num_traces == 1 else 0.7).strip() # Pass num_traces

            obs, r, done, info = step(env, action[0].lower() + action[1:])
            obs = obs.replace('\\n', '')

            step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            current_prompt += step_str

            if to_print:
                print(step_str)

            if done:
                break

        if not done: # If loop finished without 'Finish' action
            obs, r, done, info = step(env, "finish[]") # Default finish
            if 'traj' not in info: # Ensure info is a dict
                info = {}
            info.update({'finish_action_obs': obs})


        trace_info_updates = info.copy()
        trace_info_updates.update({
            'n_calls': n_calls,
            'n_badcalls': n_badcalls,
            'traj': current_prompt,
            'question_idx': idx,
            'question_text': question,
            'trace_num': trace_num + 1
        })
        # Ensure all keys from info are preserved, and new ones are added/updated
        # The 'info' from env.step() can overwrite keys if not handled carefully.
        # Let's assume 'info' from env.step() is the base and we update it.
        # However, the original code did info.copy() then updated.
        # Let's stick to the original logic for now: info from step is primary, then we add our stuff.
        # The notebook code was:
        # trace_info = info.copy()
        # trace_info.update({...})
        # This means `info` from the last step (or finish[]) is the base.

        # Let's refine how trace_info is constructed to ensure all original info fields are kept
        # and our specific trace metadata is added.
        final_trace_info = info # Start with the info from the last env.step
        final_trace_info.update(trace_info_updates) # Add/overwrite with our collected data
        all_traces_info.append(final_trace_info)

        if to_print:
            print(f"(Trace {trace_num + 1}) Info: {final_trace_info}\n")
            if num_traces > 1 and trace_num < num_traces - 1:
                print(f"--- End of Trace {trace_num + 1} ---\n")

    if not all_traces_info:
        if to_print:
            print("Warning: No traces were generated despite num_traces > 0.")
        return "[NO_TRACE_GENERATED]", []

    if num_traces == 1:
        # For single trace, the 'reward' is directly from the trace's info.
        # The original notebook code returned info.get('reward', 0.0) which seems problematic if info is not always a dict with 'reward'
        # Let's assume all_traces_info[0] is the dict we need.
        final_r = all_traces_info[0].get('reward', 0.0) # Default to 0.0 if 'reward' not found
        return final_r, all_traces_info[0]

    else: # num_traces > 1
        if to_print:
            print("\n--- Starting Answer Synthesis ---")

        extracted_answers = extract_answers_from_traces(all_traces_info)

        if to_print:
            print(f"Extracted Answers for Synthesis: {extracted_answers}")

        if not extracted_answers:
            if to_print:
                print("Warning: No answers extracted from traces. Cannot synthesize.")
            # Return all trace details even if synthesis fails
            return "[SYNTHESIS_FAILED_NO_EXTRACTED_ANSWERS]", all_traces_info

        synthesized_answer = synthesize_answer_with_llm(extracted_answers, question_for_synthesis)

        if to_print:
            print(f"Synthesized Answer: {synthesized_answer}")
            print("--- End of Answer Synthesis ---\n")

        # For multiple traces, the "final answer" is the synthesized one.
        # The second element returned should be the list of all trace infos.
        return synthesized_answer, all_traces_info
