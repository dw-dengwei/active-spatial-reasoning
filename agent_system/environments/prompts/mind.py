# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- Mind --------------------- #


MIND_TEMPLATE = f"""[Task]
We provide you a cognitive map (image and text) that shows the general layout with a agent for a scene and a question that need to spatially reason on the cognitive map.
Your task is to analyze the cognitive map and question, then answer the question.
[Question]
<question>

[Valid Actions]
At each conversation, you are required to use exactly one of the following actions:
1. Answer the question
2. Retrieve view information

[Action 1: Answer the question]
If you think the information you gained is enough to answer the question, you should output the answer.
Please do step by step reasoning and translate the spatial reasoning into a python assertion code.
[IMPORTANT] The code should return a boolean value because I will execute the code to check if the reasoning is correct.
Your final output should formatted like the following xml-string format:
<response>
  <step>
    <think>According to the cognitive map, from the view of image 4 (view 4), object 1 ("display shelves and window") is to the left of the coke can.</think>
    <code>obj1 in obj0.left(view=v4)</code>
  </step>
  <step>
    <think>Therefore, the answer should be B. Display shelves and window</think>
    <code>agent.answer("B. Display shelves and window")</code>
  </step>
</response>
You can call the following reasoning functions at each spatial reasoning step:
- turn_right(face): Calculate the direction (left, up, right, or down) after turning right from the given direction (`face`). Returns the new direction as a string.
- turn_left(face): Calculate the direction (left, up, right, or down) after turning left from the given direction (`face`). Returns the new direction as a string.
- turn_back(face): Calculate the direction (left, up, right, or down) after turning 180 degrees from the given direction (`face`). Returns the new direction as a string.
- go_straight(face): Calculate the movement vector (dx, dy) for moving straight in the given direction (`face`). Returns a tuple of (x_offset, y_offset).
- move_forward(face): Same as go_straight(face).
- src.front(view=view_object): Get all objects (and views) that are in front of the src object (or src view) from the view object's perspective. Returns a list of objects (and views). `src` is the variable name for an object (`obj1`, `obj2`, `obj3`, etc.) or a view (`v1`, `v2`, `v3`, etc.).
- src.right(view=view_object): Get all objects (and views) that are to the right of the src object (or src view) from the view object's perspective. Returns a list of objects (and views). `src` is the variable name for an object (`obj1`, `obj2`, `obj3`, etc.) or a view (`v1`, `v2`, `v3`, etc.).
- src.left(view=view_object): Get all objects (and views) that are to the left of the src object (or src view) from the view object's perspective. Returns a list of objects (and views). `src` is the variable name for an object (`obj1`, `obj2`, `obj3`, etc.) or a view (`v1`, `v2`, `v3`, etc.).
- src.opposite(view=view_object): Get all objects (and views) that are opposite to the src object (or src view) from the view object's perspective. Returns a list of objects (and views). `src` is the variable name for an object (`obj1`, `obj2`, `obj3`, etc.) or a view (`v1`, `v2`, `v3`, etc.).
- src.back(view=view_object): Get all objects (and views) that are behind the src object (or src view) from the view object's perspective. Returns a list of objects (and views). `src` is the variable name for an object (`obj1`, `obj2`, `obj3`, etc.) or a view (`v1`, `v2`, `v3`, etc.).
- src.behind(view=view_object): Same as src.back(view=view_object).
You can answer the question by calling the following function:
- agent.answer(answer): let the agent to answer the question with the answer string.

[Action 2: Retrieve view information]
If you think the information in the cognitive map is not enough to answer the question, you should retrieve the information from a view to enrich the cognitive map to help you to answer the question.
You should provide step-by-step reasoning and code for each step.
Your final output should formatted like the following xml-string format:
<response>
  <step>
    <think>To answer the question, I need the information of the right side of object 1 from view 2. Therefore, I need to retrieve the information from the view that opposite to the right side of object 1, i.e., left side of object 1 from the perspective of view 2.</think>
    <code></code>
  </step>
  <step>
    <think>According to the definitions of the views, if I need to retrieve left side of view x. I need to retrive ((x+1) mod total number of views). Therefore, view 3 is the left side of view 2.</think>
    <code></code>
  </step>
  <step>
    <think>Therefore, I need to retrieve the information from view 3.</think>
    <code>agent.retrieve(3)</code>
  </step>
</response>
You do not need to call the reasoning functions at each step.
The retrieval function that you should call at last:
- agent.retrieve(view_number): retrieve the information from the view with the view_number. 

[IMPORTANT: valid views you can retrieve]
<valid_view_desc>


[Cognitive Map Definition]
- The map uses a 10x10 grid where [0,0] is at the top-left corner and [9,9] is at the bottom-right corner
- Objects are positioned at the center of their respective grid cells and are rendered as colored circles.
- The objects are labeled with a number as follows: <cogmap_description>
- The direction of the objects are marked as red arrows, red circles with a cross (x) inside and red circles with a dot (•) inside:
  * up = a red arrow towards the top of the grid (decreasing y-value)
  * right = a red arrow towards the right of the grid (increasing x-value)
  * down = a red arrow towards the bottom of the grid (increasing y-value)
  * left = a red arrow towards the left of the grid (decreasing x-value)
  * inner = straight into the 2D map, red circles with a cross (x) inside, perpendicular to the grid, pointing away from you
  * outer = straight out of the 2D map, red circles with a dot (•) inside, perpendicular to the grid, pointing towards you

[Textual Cognitive Map]
<free_form_cogmap>
[Image Cognitive Map]
<image>

The input image and textual cognitive map are different representations of the cognitive map of the scene. Please use the image and the question to answer the question or control the agent or retrieve view information."""  # noqa: E501


COGMAP_CONSTRUCTION_TEMPLATE = """[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the
provided image, which shows the scene from view <view_index>. You will then complete a
detailed cognitive map based on the given partial cognitive map, representing the scene using a 10x10 grid coordinate system.
[Rules]
1. Focus ONLY on these objects in the scene: <object_categories>
2. Create a cognitive map with the following structure in the bird's view:
- A 10x10 grid where [0,0] is at the top-left corner and [9,9] is at the bottom-right corner
- up = towards the top of the grid (decreasing y)
- right = towards the right of the grid (increasing x)
- down = towards the bottom of the grid (increasing y)
- left = towards the left of the grid (decreasing x)
- inner = straight into the 2D map (perpendicular to the grid, pointing away from you)
- outer = straight out of the 2D map (perpendicular to the grid, pointing towards you)
- Include positions of all objects from the specified categories
- Estimate the center location (coordinates [x, y]) of each instance within provided categories
- If a category contains multiple instances, include all of them
- Each object's estimated location should accurately reflect its real position in the scene, preserving the relative spatial relationships among all objects
- Combine and merge information from the image since it is pointing to the same scene, calibrating the object locations accordingly
- Include camera positions and directions for the view
3. Carefully integrate information from the view to create a single coherent spatial representation.
[Output Format]
Please do step by step reasoning and finally output the complete cognitive map with a python dictionary format.
Your final output should formatted like the following xml-string format:
<response>
  <step>
    <think>In the input scene image, I can see black sneaker in front of the light purple sofa.</think>
    <code></code>
  </step>
  <step>
    <think>The viewpoint in the input scene image is taking a 90-degree clockwise rotation from view 2.</think>
    <code></code>
  </step>
  <step>
    <think>Now, I can complete the partial cognitive map by the information I gained from the input scene image. I will call `agent.update_cogmap` with a absolutely correct JSON string format parameter.</think>
    <code>agent.update_cogmap('{
  "objects": [
    {
      "name": "black sneaker",
      "position": [
        5,
        5
      ],
      "facing": "left"
    },
    {
      "name": "light purple sofa",
      "position": [
        5,
        8
      ],
      "facing": "down"
    },
    {
      "name": "wooden dining table",
      "position": [
        8,
        5
      ]
    }
  ],
  "views": [
    {
      "name": "Image 2",
      "position": [
        4,
        5
      ],
      "facing": "right"
    },
    {
      "name": "Image 3",
      "position": [
        5,
        4
      ],
      "facing": "down"
    }
  ]
}')</code>
  </step>
</response>
You can call the following reasoning functions at each spatial reasoning step:
- agent.update_cogmap(cogmap: dict): let the agent to update the cognitive map with the cognitive map json string.

[IMPORTANT]
You should focus ONLY on these objects in the scene: <object_categories>

[Partial Cognitive Map]
<free_form_cogmap>
[Scene Image]
<image>
[Relation]
The viewpoint in the input scene image is: <view_relation>.

The input image is a rendering of the partial cognitive map. Please use the scene image to complete the partial cognitive map."""  # noqa: E501

# [Partial Image Cognitive Map]
# <image>
# The first input image is a rendering of the partial cognitive map and the second input image is the scene image. Please use the scene image to complete the partial cognitive map."""  # noqa: E501

NO_RETRIEVE_TEMPLATE = """[Task]
We provide you a cognitive map (image and text) that shows the general layout with a agent for a scene and a question that need to spatially reason on the cognitive map.
Your task is to analyze the cognitive map and question, then answer the question.
[Question]
<question>

[Instruction]
Please do step by step reasoning and translate the spatial reasoning into a python assertion code.
[IMPORTANT] The code should return a boolean value because I will execute the code to check if the reasoning is correct.
Your final output should formatted like the following xml-string format:
<response>
  <step>
    <think>According to the cognitive map, from the view of image 4 (view 4), object 1 ("display shelves and window") is to the left of the coke can.</think>
    <code>obj1 in obj0.left(view=v4)</code>
  </step>
  <step>
    <think>Therefore, the answer should be B. Display shelves and window</think>
    <code>agent.answer("B. Display shelves and window")</code>
  </step>
</response>
You can call the following reasoning functions at each spatial reasoning step:
- turn_right(face): Calculate the direction (left, up, right, or down) after turning right from the given direction (`face`). Returns the new direction as a string.
- turn_left(face): Calculate the direction (left, up, right, or down) after turning left from the given direction (`face`). Returns the new direction as a string.
- turn_back(face): Calculate the direction (left, up, right, or down) after turning 180 degrees from the given direction (`face`). Returns the new direction as a string.
- go_straight(face): Calculate the movement vector (dx, dy) for moving straight in the given direction (`face`). Returns a tuple of (x_offset, y_offset).
- move_forward(face): Same as go_straight(face).
- src.front(view=view_object): Get all objects (and views) that are in front of the src object (or src view) from the view object's perspective. Returns a list of objects (and views). `src` is the variable name for an object (`obj1`, `obj2`, `obj3`, etc.) or a view (`v1`, `v2`, `v3`, etc.).
- src.right(view=view_object): Get all objects (and views) that are to the right of the src object (or src view) from the view object's perspective. Returns a list of objects (and views). `src` is the variable name for an object (`obj1`, `obj2`, `obj3`, etc.) or a view (`v1`, `v2`, `v3`, etc.).
- src.left(view=view_object): Get all objects (and views) that are to the left of the src object (or src view) from the view object's perspective. Returns a list of objects (and views). `src` is the variable name for an object (`obj1`, `obj2`, `obj3`, etc.) or a view (`v1`, `v2`, `v3`, etc.).
- src.opposite(view=view_object): Get all objects (and views) that are opposite to the src object (or src view) from the view object's perspective. Returns a list of objects (and views). `src` is the variable name for an object (`obj1`, `obj2`, `obj3`, etc.) or a view (`v1`, `v2`, `v3`, etc.).
- src.back(view=view_object): Get all objects (and views) that are behind the src object (or src view) from the view object's perspective. Returns a list of objects (and views). `src` is the variable name for an object (`obj1`, `obj2`, `obj3`, etc.) or a view (`v1`, `v2`, `v3`, etc.).
- src.behind(view=view_object): Same as src.back(view=view_object).
You can answer the question by calling the following function:
- agent.answer(answer): let the agent to answer the question with the answer string.

[Textual Cognitive Map]
<free_form_cogmap>
[Image Cognitive Map]
<image>

The input image and textual cognitive map are different representations of the cognitive map of the scene. Please use the image and the question to answer the question or control the agent or retrieve view information."""  # noqa: E501
