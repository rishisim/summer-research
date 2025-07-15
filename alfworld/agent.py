import os
import sys
import json
import time
import yaml
import alfworld
import alfworld.agents.environment
from prompts import ALFWORLD_PROMPTS

from google import genai # Ensuring this matches the notebook
from google.genai import types # Ensuring this matches the notebook

# --- LLM and Helper Functions ---
# The client gets the API key from the environment variable `API_KEY`.
client = genai.Client()

def llm(prompt, stop=["\n"], num_traces=1):
    time.sleep(4.1)
    temperature_setting = 0.0 if num_traces == 1 else 0.7
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite-preview-06-17",
                contents=prompt,
                generation_config=types.GenerationConfig(
                    stop_sequences=stop,
                    temperature=temperature_setting,
                    max_output_tokens=100,
                    top_p=1.0
                )
            )
            if response and response.text:
                return response.text
            time.sleep(2)
        except Exception as e:
            print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise
    return "think: I need to finish now.\nfinish"

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
        self.env = getattr(alfworld.agents.environment, self.config["env"]["type"])(self.config, train_eval="eval_out_of_distribution")
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

        trajectory = []
        for i in range(1, 50):
            action = self.llm(ob + '\n>', stop=['\n']).strip()
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

            ob += f'\n> {action}\n{observation}'

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

            trajectory = []
            for i in range(1, 50):
                action = self.llm(ob + '\n>', stop=['\n']).strip()
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

                ob += f'\n> {action}\n{observation}'

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
