import argparse
import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# --- Gemini API Configuration (Corrected based on documentation) ---
try:
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()
except Exception as e:
    print(f"ERROR: Failed to initialize Gemini client: {e}")
    print("Please ensure your GEMINI_API_KEY is set in a .env file or as an environment variable.")
    exit(1)


def llm(prompt, stop=["\n"], num_traces=1):
  # This delay handles the 15 RPM limit by waiting ~4 seconds per call.
  time.sleep(4.1) # Adjusted sleep time if necessary based on API limits

  temperature_setting = 0.0 if num_traces == 1 else 0.7
  response = client.models.generate_content(
    model="gemini-2.5-flash-lite", # Or other appropriate Gemini model
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
        stop_sequences=stop,
        temperature=temperature_setting,
        max_output_tokens=100, # As in original FEVER
        top_p=1.0
    )
  )
  return response.text

# --- WebShop Environment Interaction ---
WEBSHOP_URL = "http://localhost:3000"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}


def clean_str(p):
    """Clean string for LLM output."""
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    """Filter for visible text elements in HTML."""
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options=None, subpage=''):
    """
    Get the simplified text representation of a WebShop page.
    This function is adapted from the reference notebook.
    """
    if options is None:
        options = {}

    options_str = str(options) if options else ''

    if page_type == 'init':
        url = f'{WEBSHOP_URL}/{session}'
    elif page_type == 'search':
        url = f'{WEBSHOP_URL}/search_results/{session}/{query_string}/{page_num}'
    elif page_type == 'item':
        url = f'{WEBSHOP_URL}/item_page/{session}/{asin}/{query_string}/{page_num}/{options_str}'
    elif page_type == 'item_sub':
        url = f'{WEBSHOP_URL}/item_sub_page/{session}/{asin}/{query_string}/{page_num}/{subpage}/{options_str}'
    elif page_type == 'end':
        url = f'{WEBSHOP_URL}/done/{session}/{asin}/{options_str}'
    else:
        raise ValueError(f"Invalid page_type: {page_type}")

    try:
        html = requests.get(url).text
        html_obj = BeautifulSoup(html, 'html.parser')
        texts = html_obj.findAll(text=True)
        visible_texts = list(filter(tag_visible, texts))

        observation = ''
        # Simplified parsing logic based on the notebook
        for t in visible_texts:
            if t == '\n':
                continue
            if t.parent.name == 'button':
                observation += f'[{t}] '
            elif t.parent.name == 'label':
                observation += f'[{t}] '
            elif t.parent.get('class') == ["product-link"]:
                observation += f'[{t}] '
            else:
                observation += str(t).strip() + ' '

        # Extract other useful info
        info = {}
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
            try:
                idx = visible_texts.index('Your score (min 0.0, max 1.0)')
                info['reward'] = float(visible_texts[idx + 1])
                observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
            except (ValueError, IndexError):
                pass


        return clean_str(observation), info

    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to WebShop server at {WEBSHOP_URL}. Please ensure it is running.", {'error': str(e)}


class WebShopEnv:
    """A class to manage the state and actions within the WebShop environment."""

    def __init__(self):
        self.sessions = {}

    def step(self, session, action, instruction_keywords=None):
        done = False
        observation_ = None
        reward = 0.0

        if action == 'reset':
            self.sessions[session] = {'session': session, 'page_type': 'init'}
        elif action.startswith('think['):
            pass # No state change for think
        elif action.startswith('search['):
            if self.sessions[session]['page_type'] != 'init':
                return "Invalid action: Cannot search from this page.", 0.0, False
            query = action[7:-1]
            self.sessions[session].update({'page_type': 'search', 'query_string': query, 'page_num': 1, 'instruction_keywords': instruction_keywords})
        elif action.startswith('click['):
            button = action[6:-1]
            page_type = self.sessions[session].get('page_type')

            if button == 'Buy Now':
                if page_type != 'item':
                    return "Invalid action: Can only buy from an item page.", 0.0, False
                self.sessions[session]['page_type'] = 'end'
                done = True
            elif button == 'Back to Search':
                self.sessions[session] = {'session': session, 'page_type': 'init'}
            elif button in ACTION_TO_TEMPLATE:
                if page_type != 'item':
                    return "Invalid action: Can only view subpages from an item page.", 0.0, False
                self.sessions[session].update({'page_type': 'item_sub', 'subpage': button})
            else: # Could be an item link or an option
                if page_type == 'search':
                    # This is the corrected logic, matching the notebook.
                    # It uses the keywords from the original instruction for the item page URL.
                    self.sessions[session]['page_type'] = 'item'
                    self.sessions[session]['asin'] = button
                    self.sessions[session]['query_string'] = self.sessions[session]['instruction_keywords']
                elif page_type == 'item':
                    if 'options' not in self.sessions[session]:
                        self.sessions[session]['options'] = {}
                    # Simplified option handling
                    self.sessions[session]['options'][button] = button
                    observation_ = f'You have clicked {button}.'
                else:
                    return f"Invalid action: Cannot click '{button}' on this page.", 0.0, False
        else:
            return "Invalid action format.", 0.0, False

        observation, info = webshop_text(**self.sessions[session])
        if observation_:
            observation = observation_
        
        if 'error' in info:
             return observation, 0.0, True # End episode on connection error

        self.sessions[session].update(info)
        reward = info.get('reward', 0.0)
        return observation, reward, done


# --- ReAct Agent Logic ---
FEW_SHOT_PROMPT = """Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""

REACT_PROMPT_TEMPLATE = """{few_shot_prompt}

Instruction: {instruction}
{history}Action:"""

def run_episode(session_id, instruction, env, max_steps=15):
    """Runs a single episode of the ReAct agent."""
    # Extract keywords from the instruction for URL generation, as the server expects this.
    instruction_keywords = str(re.findall(r"i would like a (.*)", instruction))

    action = 'reset'
    history = ''
    
    for i in range(max_steps):
        print(f"\n--- Step {i+1} ---")
        
        try:
            # Pass the instruction_keywords to the step function
            observation, reward, done = env.step(session_id, action, instruction_keywords)
        except Exception as e:
            observation = f'Invalid action! Error: {e}'
            reward = 0
            done = True # End episode on error

        if action.startswith('think['):
            observation = 'OK.'
        
        print(f"Action: {action}")
        print(f"Observation: {observation}")

        if done:
            print(f"Episode finished with reward: {reward}")
            return reward

        # Build the prompt
        if i == 0:
             history += f"Observation: {observation}\n"
        else:
             history += f"Action: {action}\nObservation: {observation}\n"

        prompt = REACT_PROMPT_TEMPLATE.format(
            few_shot_prompt=FEW_SHOT_PROMPT,
            instruction=instruction, 
            history=history
        )

        # Get next action from LLM
        llm_action = llm(prompt, stop=['\n']).strip()
        
        # Basic action validation
        if not (llm_action.startswith('search[') or llm_action.startswith('click[') or llm_action.startswith('think[')):
            print(f"LLM generated invalid action format: '{llm_action}'. Ending episode.")
            action = 'think[Invalid action format from LLM. Trying to recover.]' # Try to recover
        else:
            action = llm_action


    print("Max steps reached. Ending episode.")
    # Final check for reward if max steps reached
    _, reward, _ = env.step(session_id, 'click[Buy Now]') # Attempt to buy at the end
    return reward


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run a ReAct agent on the WebShop environment.")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of WebShop tasks to attempt.")
    args = parser.parse_args()

    env = WebShopEnv()
    results = []

    for i in range(args.num_episodes):
        session_id = f"session_{i}"
        print(f"\n=================\nEpisode {i+1}/{args.num_episodes}\n=================")
        
        # Fetch a new instruction for each episode
        try:
            obs, _ = webshop_text(session=session_id, page_type='init')
            instruction_match = re.search(r"Instruction:\s*(.*)", obs)
            if not instruction_match:
                print("Could not find instruction in initial observation.")
                break
            instruction = instruction_match.group(1).strip()
            print(f"Instruction: {instruction}")
        except (requests.exceptions.RequestException, IndexError) as e:
            print(f"Failed to get new instruction: {e}")
            print("Please make sure the WebShop server is running and accessible.")
            break

        reward = run_episode(session_id, instruction, env)
        results.append({'episode': i + 1, 'instruction': instruction, 'final_reward': reward})

    # Save results
    output_file = 'webshop_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
