
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

## Git Branch Workflow Summary

Here is a complete summary of the standard Git workflow, from starting a new task to merging it back into the main project. This guide assumes your main branch is called `main`.

---

### The Complete Git Workflow: A Step-by-Step Guide

#### Step 1: Start Fresh (Get the Latest Code)

Always begin by making sure your local repository is up-to-date with the remote server (GitHub).

1.  **Switch to your main branch.**
    ```bash
    git checkout main
    ```

2.  **Pull the latest changes.** This fetches updates from the remote and merges them into your local `main` branch.
    ```bash
    git pull
    ```

#### Step 2: Create Your Feature Branch

Never work directly on the `main` branch. Create a new, descriptively named branch for every new feature, bugfix, or task. **Use hyphens for spaces.**

1.  **Create and switch to your new branch.** The `checkout -b` command does both in one step.
    ```bash
    # Example for a new feature
    git checkout -b feature/user-authentication

    # Example for a bugfix
    git checkout -b fix/login-button-bug
    ```
    You are now in a safe, isolated environment and can start working.

#### Step 3: Do the Work (The Add-Commit-Push Loop)

As you work, save your progress in small, logical chunks called commits.

1.  **Make your changes:** Write code, add files, etc.

2.  **Stage your changes:** Tell Git which files to include in the next commit.
    ```bash
    # Stage all changed files in the current directory
    git add .

    # Or stage a specific file
    git add path/to/your/file.js
    ```

3.  **Commit your changes:** Save the staged files with a clear, descriptive message.
    ```bash
    git commit -m "feat: Add email and password fields to login form"
    ```

4.  **(Optional but Recommended) Push your branch to the remote:** This backs up your work and is necessary for collaboration. The first time you push a new branch, you need to publish it and set its "upstream" tracking link.
    ```bash
    git push -u origin feature/user-authentication
    ```
    After this initial push, you only need to run `git push` for subsequent commits on this branch.

#### Step 4: Finish and Merge Your Feature

Once your feature is complete and tested, you need to merge it back into the `main` branch.

1.  **Update `main` again:** Before merging, ensure `main` is still up-to-date, as others may have merged their own work.
    ```bash
    git checkout main
    git pull
    ```

2.  **Update your feature branch with the latest from `main`:** Switch back to your feature branch and **rebase** it on top of the latest `main`. This keeps the project history clean and linear.
    ```bash
    git checkout feature/user-authentication
    git rebase main
    ```
    *If you have conflicts, Git will pause and ask you to resolve them. Fix the files, then run `git add .` and `git rebase --continue`.*

3.  **Merge your feature branch into `main`:** Now that your branch is up-to-date, you can safely merge it.
    ```bash
    git checkout main
    git merge feature/user-authentication
    ```

4.  **Push the updated `main` branch:**
    ```bash
    git push
    ```

#### Step 5: Clean Up

After your branch has been successfully merged, it's good practice to delete it to keep the repository tidy.

1.  **Delete the local branch:**
    ```bash
    git branch -d feature/user-authentication
    ```

2.  **Delete the remote branch on GitHub:**
    ```bash
    git push origin --delete feature/user-authentication
    ```

### **Git Workflow: Merging to an Upstream Branch and Tagging**

This guide covers merging a feature branch into a long-running upstream branch (like `develop`) and then applying a version tag to mark the progress.

-----

#### **Step 1: Start Fresh**

Always begin by syncing with the remote upstream branch you will eventually merge into.

1. **Switch to your upstream branch.**
    ```bash
    # Substitute 'develop' with your actual branch name
    git checkout develop
    ```

2. **Pull the latest changes.**
    ```bash
    git pull
    ```

-----

#### **Step 2: Create Your Feature Branch**

Create your new branch from the up-to-date upstream branch.

1. **Create and switch to your new branch.**
    ```bash
    # This new branch is based on the current state of 'develop'
    git checkout -b feature/new-module
    ```

-----

#### **Step 3: Do the Work (The Add-Commit-Push Loop)**

Work on your feature, committing and pushing as you go.

1. **Make changes**, `git add .`, and `git commit`.
    ```bash
    git add .
    git commit -m "feat: Implement new data processing module"
    ```

2. **Push your feature branch to the remote.**
    ```bash
    # The -u flag sets the upstream tracking link for your feature branch
    git push -u origin feature/new-module
    ```

-----

#### **Step 4: Finish and Merge Your Feature into the Upstream Branch**

Once your work is complete, merge it into the target `develop` branch.

1. **Update the target `develop` branch.**
    ```bash
    git checkout develop
    git pull
    ```

2. **Update your feature branch.** Rebase it on top of the latest `develop`.
    ```bash
    git checkout feature/new-module
    git rebase develop
    ```
    > If you have conflicts, resolve them, then run `git add .` and `git rebase --continue`.

3. **Merge your feature branch.**
    ```bash
    git checkout develop
    git merge feature/new-module
    ```

4. **Push the updated `develop` branch.**
    ```bash
    git push
    ```

-----

#### **Step 5: Create a Version Tag**

Tag the new state of the `develop` branch.

1. **Ensure you're on the `develop` branch.**
    ```bash
    git checkout develop
    ```

2. **Create an annotated tag.**
    ```bash
    # Example for a pre-release version
    git tag -a v1.1.0-alpha.1 -m "Add new data processing module"
    ```

3. **Push the new tag to the remote.**
    ```bash
    # Push the specific tag
    git push origin v1.1.0-alpha.1

    # Or push all local tags
    git push --tags
    ```

-----

#### **Step 6: Clean Up**

Delete your now-merged feature branch.

1. **Delete the local branch.**
    ```bash
    git branch -d feature/new-module
    ```

2. **Delete the remote branch.**
    ```bash
    git push origin --delete feature/new-module
    ```


## Project Creation/Development Notes
When I created it, I first git cloned the repository with the ReAct stuff inside it (git clone [https link]) so now its in the local folder. Then I created an env, and activated the environment ```.\venv\Scripts\activate```, so that I can install dependencies within the environment.

For gemini ```.\gemini-agent-env\Scripts\Activate.ps1``` and on wsl to run webshop: ```./run_dev.sh```.

on powershell to run webshop tasks:
```python run_agent.py --num_episodes 5```

Run (venv) PS C:\Users\rishi\Documents\summer-research\react-research> $env:GOOGLE_API_KEY = "API_KEY_HERE" to execute the python scripts.


#### For Linux:

```rishi@rishisim-slim:/mnt/c/Users/Rishi/Documents/summer-research/react-research$ source /mnt/c/Users/Rishi/Documents/summer-research/react-research/venv-linux/bin/activate```

```rishi@rishisim-slim:/mnt/c/Users/Rishi/Documents/summer-research/react-research$ export GOOGLE_API_KEY="API_KEY_HERE"```

