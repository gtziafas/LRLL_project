policy_gen_prompt_template = """### Python robot control code.
import numpy as np
{API}
{examples}
## Instruction: "{user_input}"."""

policy_gen_history_prompt_template = """### Python robot control code.
import numpy as np
{API}
{examples}
{history}
## Instruction: "{user_input}"."""


success_gen_prompt_template = '''### Python robot simulation environment
import numpy as np
class TabletopEnv:
    """ A tabletop simulation environment where a robot can take actions """
    def __init__(self):
        pass
    def reset(self):
        pass
    def get_obj_pos(self, obj_name):
        "Returns the xyz position of an object based on its name."
        pass
    def is_ee_close(self, target_point):
        "Checks whether the end-effector is close to given point."
        pass
    def is_obj_in_gripper(self, obj_name):
        "Checks whether an object is currently grasped by robot."
        pass
    def get_region_xyz(self, region):
        "Returns the xyz position of a table region specified by name (middle, left side, top left corner etc.)"
        pass
    def get_init_ee_pos(self):
        "Returns the initial end-effector position."
        pass
    def denormalize_xy(self):
        "Convert normalized coordinates to actual tabletop coordinates."
        pass
        
env = TabletopEnv()
_ = env.reset()
{examples}
## Instruction: "{user_input}".'''


example_template_var = """Here's an example:
---
REFERENCE:
 - {demo_task}

State: objects = {state_input}
RESPONSE:
Reasoning: {reason}
{proposed_tasks}
---"""

example_template_comp = """Here's an example input-response:
---
Hints: {hints}
Completed tasks so far: {completed}
Failed tasks so far: {failed}

RESPONSE:
Reasoning: {reason}
{proposed_tasks}
---"""


exploration_prompt_template_var = """You are a helpful assistant that suggests next tasks to complete in a robot simulation environment. Given a reference task instruction, you should propose new instructions that introduce variations in the included concepts.
CONCEPTS:
- object: block, bowl etc.
- color: blue, red, yellow etc. (use any color)
- region: middle, top corner, top left corner, left side etc.
- relation: left/right, west/east, north/south, etc.
- location: leftmost, topmost etc.
- magnitude: a little, a lot 
- ordinality: first, second etc.
- movement direction: towards, away
- continuous value: 3, 1.21, 0.19 etc.
- ... (other)

You should only respond in the format as described below:
State: ... (A list of object names that appear in the scene, this will be given)

RESPONSE:
Reasoning: ... (Based on the concepts present in the reference task and the state information, do reasoning about what variations of concepts you will introduce.)
Task 1: [task_1]
... (repeat proposing N tasks)
Task N: [task_N]

{example}

Begin:
---
REFERENCE:
{demos}

State: {state_input}. Use only these names for the proposed tasks.
RESPONSE:"""


exploration_prompt_template_comp = """You are a helpful assistant that suggests next tasks to complete in a robot simulation environment. Given a set of completed and failed tasks and some general hints, you should propose new task instructions..

I will give you the following information in this specific format:
Hints: ... (A message giving general instructions about what the proposed tasks should be)
Completed tasks so far: ... (A list of successful task instructions that you have proposed so far)
Failed tasks so far: ... (A list of failed task instructions that you have proposed so far)

You must folllow the following criteria:
1. You should act as a mentor and guide me to the next tasks based on my current learning progress.
2. The proposed tasks must NOT be the same as the completed tasks so far.
3. Use the list of failed tasks so far to understand what tasks you should NOT propose.
4. The proposed tasks should follow the advice offered by the Hints. For example, if the hints say: "[...] by the end you should be able to [X1],[X2],...", then [X1],[X2] are general descriptions of the type of tasks you should propose.
5. Try to propose task instructions that combine concepts and skills from the tasks completed so far. For example, if you have completed: "[...] towards the top left corner", "[...] cm left of the red block", the proposed tasks should include: "[...] cm left of the top left corner", "[...] towards the red block" etc.

You should respond strictly with the format described below:
RESPONSE FORMAT:
---
Reasoning: ... (Based on the general criteria, the hints, the state information and the completed tasks so far, do reasoning about what variations of concepts and combinations of skills you will introduce)
Task 1: [task_1]
... (repeat proposing N tasks)
Task N: [task_N]
---

{example}

Start!

Hints: {hints}
Completted tasks so far: {completed}
Failed tasks so far: {failed}

RESPONSE:
---"""

example0_var = """Here's an example:
---
REFERENCE:
 - go 3 cm left from (0.32, 0.69)

State: ['gray block', 'pink block', 'red block', 'orange bowl', 'blue bowl']
RESPONSE:
Reasoning: The reference task includes the concepts: 3, (0.32, 0.69), left. I will modify the continuous values with random values, and the spatial relation 'left' with concepts: 'right','above','below'. I could also use the movement direction concepts ('towards', 'away') to replace the relation concept. I will finally combine the concepts of direction and magnitude ('a little', 'a lot') to replace the relation concept.
Task 1: go 1 cm right from (-0.1, 0.55)
Task 2: go 1 cm east of (-0.1, 0.55)
Task 3: move 2 cm north of (0.32, 0.69)
Task 4: move 2.6 cm south of (-0.2, 0.33)
Task 5: go 4 cm left from (0.41, 0.11)
Task 6: go 1.8 cm west of (0.41, 0.11)
Task 7: go 3.2 cm towards (0.19, 0.31)
Task 8: move 2.1 cm away from (-0.2, 0.37)
Task 9: go a little towards (0.22, 0.01)
Task 10: move a lot away from (0.1, 0.19)
---"""

example0_comp = """Here's an example input-response:
---
Hints: You should learn basic arm functionalities such as moving to a specified point, opening and closing the figners. By the end you should be able to combine moving to location and opening/closing.
Completed tasks so far: ["go to (-0.13, 0.29)", "open fingers", "close the gripper"]
Failed tasks so far: []

RESPONSE:
Reasoning: Based on the hints, I will combine moving to point location and activate gripper.
Task 1: "go to (0.3, 0.67)"
Task 2: "open fingers at (-0.1, 0.42)"
Task 3: "close at (0.5, 0.76)"
---"""


fdef_prompt_template = """You have to define a Python function out of examples. You will be given examples in the following format:

from utils import [func_1], [func_2], ...
### Example 1
## Instruction: [instruction_str]
[code_str]
...
### Example N
## Instruction : [instruction_str]
[code_str]

and then you have to extract the common logic of the examples to define a function. Don't use function calls in this new function that are outside the ones given in the imports. Also provide a docstring. Finally, write an evaluation of the examples according to the new function.

Use the following examples to understand what you have to do:
{context_examples}

Now begin to define a function from examples:
---
{API}
{input_examples}"""


fabs_prompt_template = """You have to define a series of Python function that abstract the logic of a given set of examples. You will be given examples in the following format:

from utils import [func_1], [func_2], ...
### Example 1
## Instruction: [instruction_str]
[code_str]
...
### Example N
## Instruction : [instruction_str]
[code_str]

Don't use function calls in this new function that are outside the ones given in the imports. Provide a docstring for each generated function.

Use the following examples to understand what you have to do:
{context_examples}

Now begin to define a function from examples:
---
{API}
{input_examples}"""


example0_fdef = '''import numpy as np
from utils import detect, look, get_name, say
### Example 1
## Instruction: "tell me what is inside the top shelf".
def example_1():
    top_shelf = detect('top shelf')[0]
    things_in_top_shelf = look(top_shelf, "inside")
    things_in_top_shelf_names = [get_name(thing) for thing in things_in_top_shelf]
    say(things_in_top_shelf)
### Example 2
## Instruction: "tell me what is on top of the black countertop".
def example_2():
    black_countertop = detect('black countertop')[0]
    things_on_black_countertop= look(black_countertop, "on")
    things_on_black_countertop_names = [get_name(thing) for thing in things_on_black_countertop]
    say(things_on_black_countertop_names)
### Function definition:
def tell_names_with_relation_to_container(container: str, relation: str):
    """
    Find all objects that have a certain relation to a given container, and say their names
    
    Args:
    - container (str): The name of the target container
    - relation (str): The name of the relation to the container the target objects must have

    Returns:
    """
    container = detect(container)[0]
    things_rel_to_container = look(container, relation)
    things_rel_to_container_names = [get_name(thing) for thing in things_rel_to_container]
    say(things_rel_to_container_names)
# Example 1: tell_names_with_relation_to_container("top shelf", "inside")
# Example 2: tell_names_with_relation_to_container("black countertop", "on")'''
