# "Research"
We identified three distinct failure modes in modern LLM agents. We show that existing frameworks like Plan-and-Solve, ReAct, and Reflexion each primarily address only one of these. Our proposed Unified Agent Structure is the first to explicitly design components to counteract all three failure modes within a single, coherent architecture. We demonstrate through rigorous ablation studies that removing any one of our core components (Planner, CoT-SC Worker, or Evaluator) leads to a measurable drop in performance, proving that it is the synergistic combination of these elements that drives the superior performance.

We argue that these components must operate within a tightly-coupled, hierarchical control loop to be truly effective. Our contribution is the design of such a architecture, where a Planner provides high-level strategic direction, a Worker ensures reasoning robustness via CoT-SC, and an LLM Evaluator acts as the crucial link between them, monitoring progress and governing a Reflection process that continuously refines the agent's contextual understanding.

## More about the LLM Evaluator
**Role:** Defined it as a controller or dispatcher. Its primary job is to check the state of the overall sub-task progress.

**Outputs/Decisions:**
* **[Task Complete]:** The sub-task is finished. Stop the current loop and signal success to the main controller.
* **[In Progress, No Error]:** The task is not finished, but things are proceeding.
* **[Stuck / Infinite Loop]:** The agent is not making progress and is repeating actions or thoughts.

**Action Flow:**
* If **[Task Complete]**, the flow stops.
* If **[In Progress, No Error]**, the output is sent to Self-Reflection for summarization (ST Memory update), and the loop continues to the Worker.
* If **[Stuck / Infinite Loop]**, this is an error condition that needs to be handled. You mentioned stopping and moving on, which is one strategy.

The Evaluator isn't judging the correctness of a single action, but rather the momentum of the entire sub-task loop.

