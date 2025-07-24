import argparse
import json
import os
import random
import re
import sys
import time
import requests
from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# --- Gemini API Configuration ---
try:
    client = genai.Client()
except Exception as e:
    print(f"ERROR: Failed to initialize Gemini client: {e}")
    sys.exit(1)

def llm(prompt, stop=None, num_traces=1):
    if stop is None: stop = ["\n"]
    time.sleep(4.1)
    temperature_setting = 0.0 if num_traces == 1 else 0.7
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                stop_sequences=stop,
                temperature=temperature_setting,
                max_output_tokens=200,
                top_p=1.0
            )
        )
        return response.text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""

# --- WebShop Environment Interaction ---
WEBSHOP_URL = "http://localhost:3000"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html', 'Features': 'features_page.html',
    'Reviews': 'review_page.html', 'Attributes': 'attributes_page.html',
}

def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return element.parent.name not in ignore and not isinstance(element, Comment)

def webshop_text(session, page_type, **kwargs):
    options = kwargs.get('options', {})
    try:
        url_map = {
            'init': f'{WEBSHOP_URL}/{session}',
            'search': f'{WEBSHOP_URL}/search_results/{session}/{kwargs.get("query_string", "")}/{kwargs.get("page_num", 1)}',
            'item': f'{WEBSHOP_URL}/item_page/{session}/{kwargs.get("asin", "")}/{kwargs.get("query_string", "")}/{kwargs.get("page_num", 1)}/{options}',
            'item_sub': f'{WEBSHOP_URL}/item_sub_page/{session}/{kwargs.get("asin", "")}/{kwargs.get("query_string", "")}/{kwargs.get("page_num", 1)}/{kwargs.get("subpage", "")}/{options}',
            'end': f'{WEBSHOP_URL}/done/{session}/{kwargs.get("asin", "")}/{options}'
        }
        url = url_map.get(page_type)
        if not url: raise ValueError(f"Invalid page_type: {page_type}")

        html = requests.get(url).text
        html_obj = BeautifulSoup(html, 'html.parser')
        texts = html_obj.find_all(string=True)
        visible_texts = list(filter(tag_visible, texts))
        
        # Match notebook formatting exactly
        observation = ''
        option_type = ''
        page_options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                else:
                    processed_t = f'[{t}]'
                page_options[str(t)] = option_type
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 3:
                    processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t = '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        
        info = {'asins': asins, 'option_types': page_options}
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
            idx = visible_texts.index('Your score (min 0.0, max 1.0)')
            info['reward'] = float(visible_texts[idx + 1])
            observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info
    except requests.exceptions.RequestException as e:
        return f"Error connecting to WebShop: {e}", {'error': str(e)}

class WebShopEnv:
    def __init__(self):
        self.sessions = {}

    def step(self, session, action):
        done = False
        observation_ = None
        action_type = action.split('[')[0]

        if action_type == 'reset':
            self.sessions[session] = {'session': session, 'page_type': 'init'}  # Match notebook format
        elif action_type == 'think': pass
        elif action_type == 'search':
            assert self.sessions[session]['page_type'] == 'init'
            query = action[7:-1]
            self.sessions[session] = {'session': session, 'page_type': 'search', 'query_string': query, 'page_num': 1}
        elif action_type == 'click':
            button = action[6:-1]
            page_type = self.sessions[session]['page_type']
            if button == 'Buy Now':
                assert page_type == 'item'
                self.sessions[session]['page_type'] = 'end'
                done = True
            elif button == 'Back to Search':
                assert page_type in ['search', 'item_sub', 'item']
                self.sessions[session] = {'session': session, 'page_type': 'init'}
            elif button == '< Prev':
                assert page_type in ['search', 'item_sub', 'item']
                if page_type == 'item_sub': 
                    self.sessions[session]['page_type'] = 'item'
                elif page_type == 'item': 
                    self.sessions[session]['page_type'] = 'search'
                    self.sessions[session]['options'] = {}  # Clear options when going back
            elif button in ACTION_TO_TEMPLATE:
                assert page_type == 'item'  # Only from main item page
                self.sessions[session]['page_type'] = 'item_sub'
                self.sessions[session]['subpage'] = button
            else:
                if page_type == 'search':
                    assert button in self.sessions[session].get('asins', [])  # must be asins
                    self.sessions[session]['page_type'] = 'item'
                    self.sessions[session]['asin'] = button
                elif page_type == 'item':
                    assert 'option_types' in self.sessions[session]
                    assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
                    option_type = self.sessions[session]['option_types'][button]
                    if 'options' not in self.sessions[session]: 
                        self.sessions[session]['options'] = {}
                    self.sessions[session]['options'][option_type] = button
                    observation_ = f'You have clicked {button}.'
        else:
            assert False, f"Invalid action format: {action}"

        observation, info = webshop_text(session=session, **{k:v for k,v in self.sessions[session].items() if k != 'session'})
        if 'error' in info: return observation, 0.0, True
        if observation_: observation = observation_
        self.sessions[session].update(info)
        return observation, info.get('reward', 0.0), done

# --- ReAct Agent Logic ---
FEW_SHOT_PROMPT = """Webshop
Instruction: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[Search]

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation:
[Back to Search] 
Page 1 (Total results: 50)
[B078GWRC1J]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin...
$10.99
Action: think[B078GWRC1J is a bright citrus deodorant that is less then 50 dollars. I can check it.]
Observation: OK.
Action: click[B078GWRC1J]
Observation:
[Back to Search]
scent [assorted scents][bright citrus]
size [3 ounce (pack of 1)]
Price: $10.99
[Buy Now]
Action: think[The item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.
Action: click[bright citrus]
Observation: You have clicked bright citrus.
Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).
Action: click[Buy Now]
"""

def run_single_trace(env, session_id, instruction, to_print=True, max_steps=15):
    """Run a single reasoning trace for WebShop task."""
    action = 'reset'
    initial_prompt = f"{FEW_SHOT_PROMPT}\nInstruction: {instruction}\n[Search]\n"
    prompt_history = ''
    trajectory = []
    n_calls = 0

    for i in range(max_steps):
        try:
            observation, reward, done = env.step(session_id, action)
        except AssertionError:
            observation, reward, done = 'Invalid action!', 0.0, False

        if action.startswith('think['): observation = 'OK.'
        
        trajectory.append({'step': i, 'action': action, 'observation': observation.strip()})
        if to_print:
            print(f"\n--- Step {i+1}/{max_steps} ---")
            print(f"Action: {action}")
            print(f"Observation:\n    " + "\n    ".join(observation.strip().split('\n')))
        
        if done:
            if to_print: print(f"Episode finished with reward: {reward}")
            return reward, trajectory, n_calls

        if i == 0: prompt_history = f"Observation: {observation}\n\nAction:"
        else: prompt_history += f" {action}\nObservation: {observation}\n\nAction:"
        
        full_prompt = initial_prompt + prompt_history[-(6000 - len(initial_prompt)):]
        action = llm(full_prompt, stop=['\n']).strip()
        n_calls += 1
        if not action: break
    
    if to_print: print(f"Max steps reached. Ending episode with reward: {reward}")
    return reward, trajectory, n_calls

def run_single_episode(env, session_id, instruction, to_print=True, max_steps=15):
    """Backward compatibility wrapper for single episode execution."""
    reward, trajectory, _ = run_single_trace(env, session_id, instruction, to_print, max_steps)
    return reward, trajectory

# --- Synthesis Functions for Multi-Trace ---
def extract_trajectories_from_traces(all_traces_info):
    """Extract the full trajectory string from each trace in the all_traces_info list."""
    extracted_trajectories = []
    if not isinstance(all_traces_info, list):
        print("Warning: all_traces_info is not a list")
        return []

    for i, trace_info in enumerate(all_traces_info):
        if isinstance(trace_info, dict) and 'trajectory' in trace_info:
            trajectory = trace_info['trajectory']
            # Convert trajectory to string format
            trajectory_str = ""
            for step in trajectory:
                trajectory_str += f"Action: {step['action']}\nObservation: {step['observation']}\n"
            extracted_trajectories.append(trajectory_str.strip())
        else:
            print(f"Warning: trace_info {i} missing 'trajectory' key or not a dict")
            extracted_trajectories.append("")

    return extracted_trajectories

def synthesize_decision_deterministic(all_traces_info):
    """
    Deterministically selects the best trajectory based on reward and step count.
    Returns the trajectory number (1-indexed) of the best trajectory.
    
    Selection criteria:
    1. Highest reward
    2. If tied, fewest steps (most efficient)
    """
    if not all_traces_info:
        return 1
    
    # Extract reward and step count for each trajectory
    trajectory_metrics = []
    for i, trace_info in enumerate(all_traces_info):
        if isinstance(trace_info, dict):
            reward = trace_info.get('final_reward', 0.0)
            trajectory = trace_info.get('trajectory', [])
            step_count = len(trajectory) if trajectory else float('inf')
            trajectory_metrics.append((reward, step_count, i + 1))  # (reward, steps, trajectory_num)
        else:
            trajectory_metrics.append((0.0, float('inf'), i + 1))
    
    # Sort by reward (descending), then by step count (ascending)
    trajectory_metrics.sort(key=lambda x: (-x[0], x[1]))
    
    return trajectory_metrics[0][2]  # Return trajectory number

# --- Core Webthink Logic ---
def webthink_webshop(env, session_id, instruction, num_traces=1, to_print=True, max_steps=15):
    """
    Main function for WebShop reasoning with support for both single and multi-trace execution.
    
    Args:
        env: WebShop environment instance
        session_id: Session identifier
        instruction: Shopping instruction
        num_traces: Number of reasoning traces to run (1 for standard ReAct, >1 for synthesized)
        to_print: Whether to print trace details
        max_steps: Maximum steps per trace
    
    Returns:
        (reward, info_dict): Final reward and detailed information
    """
    if num_traces <= 0:
        print("Error: num_traces must be positive")
        return 0.0, {'error': 'Invalid num_traces'}

    all_traces_info = []
    
    print(f"Running {num_traces} trace(s) for session {session_id}")
    
    for trace_num in range(num_traces):
        if num_traces > 1:
            print(f"\n=== TRACE {trace_num + 1}/{num_traces} ===")
        
        # Run single trace
        reward, trajectory, n_calls = run_single_trace(env, session_id, instruction, to_print, max_steps)
        
        # Store trace information
        trace_info = {
            'trace_num': trace_num + 1,
            'n_calls': n_calls,
            'trajectory': trajectory,
            'final_reward': reward
        }
        all_traces_info.append(trace_info)
        
        if num_traces > 1:
            print(f"Trace {trace_num + 1} completed with reward: {reward}")

    if not all_traces_info:
        return 0.0, {'error': 'No traces completed'}

    if num_traces == 1:
        # Single trace - return as before
        trace_info = all_traces_info[0]
        return trace_info['final_reward'], {
            'n_calls': trace_info['n_calls'],
            'trajectory': trace_info['trajectory'],
            'final_reward': trace_info['final_reward']
        }
    else:
        # Multi-trace synthesis
        print(f"\n=== SYNTHESIZING {num_traces} TRACES ===")
        
        # Synthesize best decision deterministically
        best_trajectory_num = synthesize_decision_deterministic(all_traces_info)
        
        # Use reward from best trajectory
        best_trace_info = all_traces_info[best_trajectory_num - 1]  # Convert to 0-indexed
        final_reward = best_trace_info['final_reward']
        
        print(f"Synthesis selected trajectory {best_trajectory_num} with reward: {final_reward}")
        
        return final_reward, {
            'num_traces_run': num_traces,
            'synthesized_decision': best_trajectory_num,
            'individual_traces': all_traces_info,
            'final_reward': final_reward,
            'synthesis_reasoning': ""
        }

# --- File Utilities & Experiment Runner ---
def append_to_json(data, filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r+', encoding='utf-8') as f:
            try: file_data = json.load(f)
            except json.JSONDecodeError: file_data = []
            if isinstance(file_data, list): file_data.append(data)
            else: file_data = [data]
            f.seek(0)
            json.dump(file_data, f, indent=2)
            f.truncate()
    else:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([data], f, indent=2)

def get_processed_indices(output_file):
    processed_indices = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for entry in data:
                    if 'session_id_index' in entry:
                        processed_indices.add(entry['session_id_index'])
            except (json.JSONDecodeError, AttributeError): pass
    return processed_indices

def get_output_filename(num_traces):
    """Get appropriate filename based on whether using synthesis or not."""
    if num_traces == 1:
        return 'webshop_trajectories.json'  # Standard ReAct
    else:
        return 'webshop_synthesized_trajectories.json'  # Synthesized ReAct

def run_task_with_both_modes(env, task_index, instruction):
    """Run a single task with both standard and synthesized ReAct modes."""
    session_id = str(task_index)
    results = {}
    
    # Run Standard ReAct (1 trace)
    print(f"\n[STANDARD REACT] Running 1 trace for Session {session_id}")
    print("="*50)
    try:
        reward_std, info_std = webthink_webshop(env, session_id, instruction, num_traces=1, to_print=True)
        results['standard'] = {
            'reward': reward_std,
            'info': info_std,
            'success': True
        }
        print(f"Standard ReAct completed with reward: {reward_std}")
    except Exception as e:
        print(f"Standard ReAct failed: {e}")
        results['standard'] = {
            'reward': 0.0,
            'info': {'trajectory': [{'error': str(e)}]},
            'success': False,
            'error': str(e)
        }
    
    # Run Synthesized ReAct (3 traces)
    print(f"\n[SYNTHESIZED REACT] Running 3 traces for Session {session_id}")
    print("="*50)
    try:
        reward_synth, info_synth = webthink_webshop(env, session_id, instruction, num_traces=3, to_print=True)
        results['synthesized'] = {
            'reward': reward_synth,
            'info': info_synth,
            'success': True
        }
        print(f"Synthesized ReAct completed with reward: {reward_synth}")
    except Exception as e:
        print(f"Synthesized ReAct failed: {e}")
        results['synthesized'] = {
            'reward': 0.0,
            'info': {'individual_traces': [{'error': str(e)}]},
            'success': False,
            'error': str(e)
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run a ReAct agent on the WebShop environment.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of tasks to run (each task will be run with both standard and synthesized ReAct)")
    parser.add_argument("--num_traces", type=int, default=None, help="Override to run only one mode: 1=standard ReAct only, 3=synthesized ReAct only")
    args = parser.parse_args()

    env = WebShopEnv()
    
    # Determine which modes to run
    run_both_modes = args.num_traces is None
    if run_both_modes:
        print("Running BOTH Standard ReAct and Synthesized ReAct for each task")
        standard_file = 'webshop_trajectories.json'
        synthesized_file = 'webshop_synthesized_trajectories.json'
    else:
        print(f"Running only {'Standard' if args.num_traces == 1 else 'Synthesized'} ReAct")
        output_file = get_output_filename(args.num_traces)
    
    MAX_WEBSHOP_TASKS = 300  # Limited to 0-299 range
    all_indices = list(range(MAX_WEBSHOP_TASKS))
    random.Random(42).shuffle(all_indices)

    if run_both_modes:
        # For both modes, find tasks that need to be run in either file
        processed_std = get_processed_indices(standard_file)
        processed_synth = get_processed_indices(synthesized_file)
        processed_indices = processed_std.union(processed_synth)
        print(f"Found {len(processed_std)} standard and {len(processed_synth)} synthesized completed tasks.")
    else:
        processed_indices = get_processed_indices(output_file)
        print(f"Found {len(processed_indices)} already completed tasks in {output_file}.")

    remaining_indices = [idx for idx in all_indices if idx not in processed_indices]
    tasks_to_run = remaining_indices[:args.num_episodes]
    
    print(f"Preparing to run {len(tasks_to_run)} new tasks.")
    if not tasks_to_run:
        print("No new tasks to run. All tasks might be complete.")
        return

    tasks_run_this_session = 0
    for i, task_index in enumerate(tasks_to_run):
        session_id = str(task_index)
        print('\n' + '='*60)
        print(f"RUNNING TASK {i+1}/{len(tasks_to_run)} (Session: {session_id})")
        print('='*60)
        
        try:
            # Get instruction for this task
            obs, info = webshop_text(session=task_index, page_type='init')
            if 'error' in info:
                raise requests.exceptions.RequestException(obs)
            
            instruction_match = re.search(r"Instruction:\s*(.*)", obs, re.DOTALL)
            if not instruction_match:
                print(f"DEBUG - Full observation: {obs}")
                raise ValueError(f"Instruction pattern not found in observation for session {session_id}")
            
            instruction = instruction_match.group(1).strip()
            print(f"Instruction: {instruction}")
            
            if run_both_modes:
                # Run both modes for this task
                results = run_task_with_both_modes(env, task_index, instruction)
                
                # Save standard ReAct results
                if results['standard']['success']:
                    std_episode_data = {
                        'session_id_index': task_index,
                        'episode_human_readable': len(get_processed_indices(standard_file)) + tasks_run_this_session + 1,
                        'instruction': instruction,
                        'final_reward': results['standard']['reward'],
                        'trajectory': results['standard']['info'].get('trajectory', [])
                    }
                else:
                    std_episode_data = {
                        'session_id_index': task_index,
                        'episode_human_readable': len(get_processed_indices(standard_file)) + tasks_run_this_session + 1,
                        'instruction': instruction,
                        'final_reward': 0.0,
                        'trajectory': [{'error': results['standard'].get('error', 'Unknown error')}]
                    }
                append_to_json(std_episode_data, standard_file)
                
                # Save synthesized ReAct results
                if results['synthesized']['success']:
                    synth_episode_data = {
                        'session_id_index': task_index,
                        'episode_human_readable': len(get_processed_indices(synthesized_file)) + tasks_run_this_session + 1,
                        'instruction': instruction,
                        'final_reward': results['synthesized']['reward'],
                        'num_traces_run': results['synthesized']['info'].get('num_traces_run', 3),
                        'synthesized_decision': results['synthesized']['info'].get('synthesized_decision', 1),
                        'individual_traces': results['synthesized']['info'].get('individual_traces', []),
                        'synthesis_reasoning': results['synthesized']['info'].get('synthesis_reasoning', "")
                    }
                else:
                    synth_episode_data = {
                        'session_id_index': task_index,
                        'episode_human_readable': len(get_processed_indices(synthesized_file)) + tasks_run_this_session + 1,
                        'instruction': instruction,
                        'final_reward': 0.0,
                        'num_traces_run': 3,
                        'synthesized_decision': 1,
                        'individual_traces': [{'error': results['synthesized'].get('error', 'Unknown error')}],
                        'synthesis_reasoning': ""
                    }
                append_to_json(synth_episode_data, synthesized_file)
                
                print(f"\nResults saved:")
                print(f"   Standard ReAct -> {standard_file} (reward: {results['standard']['reward']})")
                print(f"   Synthesized ReAct -> {synthesized_file} (reward: {results['synthesized']['reward']})")
                
            else:
                # Run single mode only (original behavior)
                reward, info_dict = webthink_webshop(env, session_id, instruction, 
                                                   num_traces=args.num_traces, to_print=True)
                
                if args.num_traces == 1:
                    episode_data = {
                        'session_id_index': task_index,
                        'episode_human_readable': len(processed_indices) + tasks_run_this_session + 1,
                        'instruction': instruction,
                        'final_reward': reward,
                        'trajectory': info_dict.get('trajectory', [])
                    }
                else:
                    episode_data = {
                        'session_id_index': task_index,
                        'episode_human_readable': len(processed_indices) + tasks_run_this_session + 1,
                        'instruction': instruction,
                        'final_reward': reward,
                        'num_traces_run': info_dict.get('num_traces_run', args.num_traces),
                        'synthesized_decision': info_dict.get('synthesized_decision', 1),
                        'individual_traces': info_dict.get('individual_traces', []),
                        'synthesis_reasoning': info_dict.get('synthesis_reasoning', "")
                    }
                
                append_to_json(episode_data, output_file)
                print(f"Result for session {session_id} saved to {output_file}")
            
            tasks_run_this_session += 1

        except Exception as e:
            print(f"An unrecoverable error occurred for session {session_id}: {e}")
            if run_both_modes:
                # Save error data to both files
                std_error_data = {
                    'session_id_index': task_index,
                    'episode_human_readable': len(get_processed_indices(standard_file)) + tasks_run_this_session + 1,
                    'instruction': 'N/A', 'final_reward': 0.0,
                    'trajectory': [{'error': str(e)}]
                }
                synth_error_data = {
                    'session_id_index': task_index,
                    'episode_human_readable': len(get_processed_indices(synthesized_file)) + tasks_run_this_session + 1,
                    'instruction': 'N/A', 'final_reward': 0.0,
                    'num_traces_run': 3, 'synthesized_decision': 1,
                    'individual_traces': [{'error': str(e)}], 'synthesis_reasoning': ""
                }
                append_to_json(std_error_data, standard_file)
                append_to_json(synth_error_data, synthesized_file)
            else:
                error_data = {
                    'session_id_index': task_index,
                    'episode_human_readable': len(processed_indices) + tasks_run_this_session + 1,
                    'instruction': 'N/A', 'final_reward': 0.0,
                    'trajectory': [{'error': str(e)}] if args.num_traces == 1 else [],
                    'individual_traces': [{'error': str(e)}] if args.num_traces > 1 else []
                }
                append_to_json(error_data, output_file)

    print(f"\nCompleted {tasks_run_this_session} new tasks in this session.")
    if run_both_modes:
        print(f"Results stored in:")
        print(f"   Standard ReAct: {standard_file}")
        print(f"   Synthesized ReAct: {synthesized_file}")
    else:
        print(f"All results are stored in {output_file}")

if __name__ == "__main__":
    main()
