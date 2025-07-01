
# Brainstorming
![image](https://github.com/user-attachments/assets/6f30922d-9652-43fd-81c5-5be56f86b1c2)

## Unified Agent Structure:
1. Combines 4 frameworks (Plan & Solve, Self Reflexion, CoT-SC, and ReAct)
2. Basically the idea is to address the pitfalls of each of the frameworks by integrating them together and thus improve the overall benchmarks against datasets like WebShop, HotPotQA, etc.
3. Plan & Solve (Planner Layer)
    1. Given a complex task, the first LLM call is to the Planner. Generates a high-level, multi-step plan.
4. Self Reflexion (Context or ST/LT Mem)
    1. Takes the context of the actions so far which consists of the current trajectory of step i, and any notable experiences long term throughout the entire round. Also includes current step and overall task.
    2. Gets updated for every interaction with environment.
5. LLM Evaluator
    1. Evaluates the LLM. If step success, moves to next step. If first iteration, error, or indefinite loop, itsends everything to the CoT-SC
6. Worker (CoT-SC)
    1. Based on context, LLM proposes multiple reasoning and actions, and then chooses the best or the most frequent one.
7. Execute action on Env.
8. Send Observation to Context
9. Loops until LLM Evaluator deems task as success.

## Phase 1
![image](https://github.com/user-attachments/assets/e61f4820-c07d-46a6-8355-8d3dff1faa6d)

## Phase 2
![image](https://github.com/user-attachments/assets/244ba479-c1b2-4945-9a2d-f6493a421fab)

## Phase 3: Final
![image](https://github.com/user-attachments/assets/6f30922d-9652-43fd-81c5-5be56f86b1c2)

# References
**The Github Standard Workflow:**

- Use one main branch, typically called main or develop, as your single source of truth.
- When you want to add a feature (e.g., "implement the CoT-SC worker"), you create a feature branch from main (e.g., git checkout -b feature/cot-sc-worker).
- You work on that feature in its own branch. Once it's complete and tested, you merge it back into main.
- To mark the completion of a major phase, you use a Git Tag. So when Phase 1 is done, you'll tag that commit: git tag -a v1.0 -m "Phase 1: Updated ReAct complete".

## Project Creation/Development Notes
When I created it, I first git cloned the repository with the ReAct stuff inside it (git clone [https link]) so now its in the local folder. Then I created an env, and activated the environment ```.\venv\Scripts\activate```, so that I can install dependencies within the environment.
