import os
import json
import random
from itertools import permutations
from agent import AlfWorldAgent
from prompts import ALFWORLD_PROMPTS

def run_experiment(num_tasks=1):
    agent = AlfWorldAgent()
    results = []

    prefixes = {
        'pick_and_place': 'put',
        'pick_clean_then_place': 'clean',
        'pick_heat_then_place': 'heat',
        'pick_cool_then_place': 'cool',
        'look_at_obj': 'examine',
        'pick_two_obj': 'puttwo'
    }

    for i in range(num_tasks):
        ob, info = agent.env.reset()
        task_id = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        task_type_key = [k for k, v in prefixes.items() if task_id.startswith(k)][0]
        task_type = prefixes[task_type_key]
        goal = ob[0].split('\n\n')[1].replace('Your task is to: ', '')

        task_info = {
            "task_id": task_id,
            "task_type": task_type,
            "task_goal": goal
        }

        # Run typical ReAct
        react_result = agent.run_react(task_info, to_print=False)
        results.append(react_result)

        # Run proposed ReAct
        proposed_react_result = agent.run_proposed_react(task_info, num_traces=3, to_print=False)
        results.append(proposed_react_result)

    with open('alfworld_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ALFWorld experiments')
    parser.add_argument('--num-tasks', type=int, default=1,
                      help='Number of tasks to run (default: 1)')
    
    args = parser.parse_args()
    run_experiment(num_tasks=args.num_tasks)
