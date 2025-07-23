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

def run_single_episode(env, session_id, instruction, to_print=True, max_steps=15):
    action = 'reset'
    initial_prompt = f"{FEW_SHOT_PROMPT}\nInstruction: {instruction}\n[Search]\n"
    prompt_history = ''
    trajectory = []

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
            print(f"Episode finished with reward: {reward}")
            return reward, trajectory

        if i == 0: prompt_history = f"Observation: {observation}\n\nAction:"
        else: prompt_history += f" {action}\nObservation: {observation}\n\nAction:"
        
        full_prompt = initial_prompt + prompt_history[-(6000 - len(initial_prompt)):]
        action = llm(full_prompt, stop=['\n']).strip()
        if not action: break
    
    print(f"Max steps reached. Ending episode with reward: {reward}")
    return reward, trajectory

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

def main():
    parser = argparse.ArgumentParser(description="Run a ReAct agent on the WebShop environment.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of new tasks to attempt in this session.")
    args = parser.parse_args()

    env = WebShopEnv()
    output_file = 'webshop_trajectories.json'
    
    MAX_WEBSHOP_TASKS = 300  # Limited to 0-299 range
    all_indices = list(range(MAX_WEBSHOP_TASKS))
    random.Random(42).shuffle(all_indices)

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
        session_id = str(task_index)  # Use just the number, not "fixed_" prefix
        print('=================================')
        print(f"RUNNING TASK {i+1}/{len(tasks_to_run)} (Session: {session_id})")
        print('=================================')
        
        try:
            obs, info = webshop_text(session=session_id, page_type='init')
            if 'error' in info:
                raise requests.exceptions.RequestException(obs)
            
            # *** FIX: Safely check for the instruction before trying to access it ***
            instruction_match = re.search(r"Instruction:\s*(.*)", obs, re.DOTALL)
            if not instruction_match:
                print(f"DEBUG - Full observation: {obs}")  # Print full observation if no match
                raise ValueError(f"Instruction pattern not found in observation for session {session_id}")
            
            instruction = instruction_match.group(1).strip()
            print(f"Instruction: {instruction}")
            print("="*50)
            
            reward, trajectory = run_single_episode(env, session_id, instruction)
            
            episode_data = {
                'session_id_index': task_index,
                'episode_human_readable': len(processed_indices) + tasks_run_this_session + 1,
                'instruction': instruction,
                'final_reward': reward,
                'trajectory': trajectory
            }
            append_to_json(episode_data, output_file)
            print(f"Result for session {session_id} saved to {output_file}")
            tasks_run_this_session += 1

        except Exception as e:
            print(f"An unrecoverable error occurred for session {session_id}: {e}")
            error_data = {
                'session_id_index': task_index,
                'episode_human_readable': len(processed_indices) + tasks_run_this_session + 1,
                'instruction': 'N/A', 'final_reward': 0.0,
                'trajectory': [{'error': str(e)}]
            }
            append_to_json(error_data, output_file)

    print(f"\nCompleted {tasks_run_this_session} new tasks in this session.")
    print(f"All results are stored in {output_file}")

if __name__ == "__main__":
    main()
