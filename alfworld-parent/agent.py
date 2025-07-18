import os
import sys
import json
import time
import yaml
import alfworld
from prompts import ALFWORLD_PROMPTS

def get_prompt(task_type, prompt_type='react'):
    if prompt_type == 'react':
        return 'Interact with a household to solve a task. Here are two examples.\n' + ALFWORLD_PROMPTS[f'react_{task_type}_1'] + ALFWORLD_PROMPTS[f'react_{task_type}_0'] + '\nHere is the task.\n'
    else:
        # For proposed react, we will just use the standard prompt for now
        return 'Interact with a household to solve a task. Here are two examples.\n' + ALFWORLD_PROMPTS[f'react_{task_type}_1'] + ALFWORLD_PROMPTS[f'react_{task_type}_0'] + '\nHere is the task.\n'

# --- LLM Configuration and Interaction ---
from google import genai # Ensuring this matches the notebook
from google.genai import types # Ensuring this matches the notebook

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client() # Assuming API key is in env

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

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob

class AlfWorldAgent:
    def __init__(self, prompt_type='react', llm=llm):
        self.prompt_type = prompt_type
        self.llm = llm
        with open('base_config.yaml') as reader:
            self.config = yaml.safe_load(reader)
        
        # Try to use the documented alfworld environment setup
        try:
            from alfworld.agents.environment import get_environment
            env_type = self.config['env']['type']
            self.env = get_environment(env_type)(self.config, train_eval="eval_out_of_distribution")
            self.env = self.env.init_env(batch_size=1)
        except ImportError:
            # Fallback approach using standard alfworld
            import alfworld.agents.modules.generic as generic
            self.env = generic.get_environment(self.config['env']['type'])(self.config, train_eval="eval_out_of_distribution")
            self.env = self.env.init_env(batch_size=1)

    def run(self, task_info, num_traces=1, to_print=True):
        if num_traces == 1:
            return self.run_react(task_info, to_print)
        else:
            return self.run_proposed_react(task_info, num_traces, to_print)

    def run_react(self, task_info, to_print=True):
        run_id = f"alfworld_{task_info['task_id'].replace('/', '_')}_react_prompt_1"
        agent_type = "Typical ReAct"

        ob, info = self.env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        
        # Get the appropriate prompt for this task type
        prompt_prefix = get_prompt(task_info['task_type'], 'react')
        full_prompt = prompt_prefix + ob + '\n>'

        trajectory = []
        for i in range(1, 50):
            action = self.llm(full_prompt, stop=['\n']).strip()
            observation, reward, done, info = self.env.step([action])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]

            if action.startswith('think:'):
                observation = 'OK.'

            trajectory.append({
                "step": i,
                "thought": action if action.startswith('think:') else None,
                "action": action if not action.startswith('think:') else None,
                "observation": observation
            })

            if to_print:
                print(f'Action {i}: {action}\nObservation {i}: {observation}')

            full_prompt += f' {action}\n{observation}\n>'

            if done:
                break

        return {
            "run_id": run_id,
            "agent_type": agent_type,
            "task_info": task_info,
            "is_success": reward,
            "trajectory": trajectory
        }

    def run_proposed_react(self, task_info, num_traces, to_print=True):
        run_id = f"alfworld_{task_info['task_id'].replace('/', '_')}_proposed_react_prompt_1"
        agent_type = "Proposed ReAct"

        traces = []
        for _ in range(num_traces):
            ob, info = self.env.reset()
            ob = '\n'.join(ob[0].split('\n\n')[1:])
            
            # Get the appropriate prompt for this task type
            prompt_prefix = get_prompt(task_info['task_type'], 'react')
            full_prompt = prompt_prefix + ob + '\n>'

            trajectory = []
            for i in range(1, 50):
                action = self.llm(full_prompt, stop=['\n']).strip()
                observation, reward, done, info = self.env.step([action])
                observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]

                if action.startswith('think:'):
                    observation = 'OK.'

                trajectory.append({
                    "step": i,
                    "thought": action if action.startswith('think:') else None,
                    "action": action if not action.startswith('think:') else None,
                    "observation": observation
                })

                if to_print:
                    print(f'Action {i}: {action}\nObservation {i}: {observation}')

                full_prompt += f' {action}\n{observation}\n>'

                if done:
                    break

            traces.append({"trajectory": trajectory, "is_success": reward})

        successful_trace = next((trace for trace in traces if trace["is_success"]), None)

        if successful_trace:
            return {
                "run_id": run_id,
                "agent_type": agent_type,
                "task_info": task_info,
                "is_success": True,
                "trajectory": successful_trace["trajectory"]
            }
        else:
            # If no trace was successful, return the first one
            return {
                "run_id": run_id,
                "agent_type": agent_type,
                "task_info": task_info,
                "is_success": False,
                "trajectory": traces[0]["trajectory"]
            }
