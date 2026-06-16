import argparse
from rich import print
from rich.pretty import pprint
import os
import base64
import json
import pickle
import re
from html import escape
from io import BytesIO
from typing import Optional
from PIL import Image
from agent_system.environments.env_package.mind.projection import mind_projection
from tqdm.auto import tqdm
from agent_system.environments.env_package.mind.envs import CogMapEnv, correct_list_check, evaluate_constant_boolean_expression
from agent_system.environments.env_package.mind.envs import AnnotationType, could_answer, get_meta_objects
from copy import deepcopy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('result_base', type=str, help='Path to the result directory')
parser.add_argument('--output_html', action='store_true', help='Output HTML visualization')
args = parser.parse_args()

result_base = args.result_base
output_html = args.output_html

if not os.path.isdir(result_base):
    print(f"Result base {result_base} is not a directory")
    exit(1)

with open(os.path.join(result_base, 'output_list.pkl'), 'rb') as f:
    data = pickle.load(f)
print(len(data))

dataset = {}
with open('data/scaffold/all/MindCube_tinybench.jsonl') as f:
    for line in f.readlines():
        line = json.loads(line)
        dataset[line['id']] = line

def find_gt(_id: str) -> str | None:
    if _id not in dataset:
        return None
    return dataset[_id]['gt_answer']

def check_progress(step_codes: list[str], env, question, gt_answer, meta_objects, verbose):
    """
    step_codes format: [str, str, str, ...]
    """
    step_rewards = [0.0] * len(step_codes)
    non_empty_steps = 0

    # 优化：在循环外生成一次 context，避免每次迭代都调用 export_context()
    # 由于这是检查历史步骤，环境状态在循环中不会改变
    base_context = env.export_context()

    for i, code in enumerate(step_codes):
        # 跳过空步骤，但我们会在后面统计非空步骤数量
        if not code or code.strip() == '':
            continue

        non_empty_steps += 1
        # 为每个步骤创建独立的 context 副本，避免步骤间相互影响
        local_context = deepcopy(base_context)
        global_context = dict(local_context)
        global_context['__builtins__'] = __builtins__
        
        try:
            # console.log(f"Running code: {code} with local context: {local_context}. objects: {self.env.objects}, views: {self.env.views}", style="green")
            code = correct_list_check(code)
            constant_result = evaluate_constant_boolean_expression(code)
            if constant_result is not None:
                raise ValueError("Constant result")
            else:
                result = eval(code, global_context, local_context)
        except Exception as e:
            result = False


    cogmap_reward = 0.0
    # check if the cogmap is correct
    if env.gt_cogmap_dict is not None:
        for object in env.objects.objects:
            if object.name.lower().strip() not in [o['name'].lower().strip() for o in env.gt_cogmap_dict['objects']]:
                cogmap_reward -= 0.5
                if verbose:
                    pprint(f"{object.name} not in {[o['name'].lower().strip() for o in env.gt_cogmap_dict['objects']]}")
            else:
                # cogmap_reward.append(1.0)
                target_object = [o for o in env.gt_cogmap_dict['objects'] if o['name'].lower().strip() == object.name.lower().strip()][0]
                # print(target_object, object)
                if object.position != tuple(target_object['position']):
                    cogmap_reward -= 0.5
                    if verbose:
                        pprint(f"{object.name} position not in {[o['position'] for o in env.gt_cogmap_dict['objects']]}")
                else:
                    pass
                    # cogmap_reward.append(1.0)
        for view in env.views.views:
            if view.name.lower().strip().split(" ")[-1].strip() not in [v['name'].lower().strip().split(" ")[-1].strip() for v in env.gt_cogmap_dict['views']]:
                cogmap_reward -= 0.5
                if verbose:
                    pprint(f"{view.name} not in {[v['name'].lower().strip().split(" ")[-1].strip() for v in env.gt_cogmap_dict['views']]}")
            else:
                # cogmap_reward.append(1.0)
                target_view = [v for v in env.gt_cogmap_dict['views'] if v['name'].lower().strip().split(" ")[-1].strip() == view.name.lower().strip().split(" ")[-1].strip()][0]
                # print(target_view, view)
                if tuple(view.position) != tuple(target_view['position']) or view.facing != target_view['facing']:
                    cogmap_reward -= 0.5
                    if verbose:
                        pprint(f"{view.position} or {view.facing} not in {target_view['position']} or {target_view['facing']}")
                else:
                    pass
                    # cogmap_reward.append(1.0)
    
    retrieve_relative_reward = 0
    _, cogmap_img_info = env.render_mind_map(with_agent=True)
    free_form_cogmap = env.free_form_cgmap(cogmap_img_info)
    answer_type = could_answer(question, gt_answer, meta_objects, free_form_cogmap)
    if answer_type == AnnotationType.COMPLETE:
        pass
        # retrieve_relative_reward += 1.0
    else:
        retrieve_relative_reward += -0.5
    
    return non_empty_steps, retrieve_relative_reward, cogmap_reward

def extract_answer(text: str) -> Optional[str]:
    """
    Extract the answer from model response text using regular expressions.
    Returns the last occurrence of the letter of the answer (A, B, C, D, or E)
    based on pattern priority - tries higher priority patterns first.
    
    Args:
        text: The model response text
        
    Returns:
        The last answer letter found by the highest priority matching pattern,
        or None if not found
    """
    if not text:
        return None
    
    # First, try to match simple answer format: A., B., C., D., E. with highest priority
    simple_pattern_matches = list(re.finditer(r'([A-E])\.', text))
    if simple_pattern_matches:
        return simple_pattern_matches[-1].group(1)
    
    # Then check if <Answer> tag exists and extract content after it
    answer_section_match = re.search(r'<Answer>(.*?)(?:<|$)', text, re.DOTALL)
    if answer_section_match:
        answer_section = answer_section_match.group(1)
        # Check for specific patterns in the answer section
        for pattern in [
            r'[Mm]y answer is ([A-E])',
            r'[Mm]y answer is ([A-E])\.',
            r'[Tt]he answer is ([A-E])',
            r'(?:Answer: )?([A-E])\.',
            r'\b([A-E])\b'
        ]:
            matches = list(re.finditer(pattern, answer_section))
            if matches:
                return matches[-1].group(1)
    
    # If no matches found after <Answer> tag, proceed with regular priority patterns
    patterns = [
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'",]+(?=(?:\n|$|\.|"))',  # Full answer with description
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'"]+',  # Answer with partial description
        r'(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)',  # Answer at line beginning
        r'[\*\"]([A-E])[\*\"]',  # Answer in quotes or asterisks
        r'\bAnswer:?\s*([A-E])\b',  # Answer following "Answer:"
        r'[Mm]y answer is ([A-E])',  # Added pattern for "My answer is X"
        r'[Mm]y answer is ([A-E])\.',  # Added pattern for "My answer is X."
        r'answer is ([A-E])',  # Added pattern for phrases like "The answer is X"
    ]
    
    # Try each pattern in order of priority
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Return the last match found by this pattern
            return matches[-1].group(1)
    
    # If none of the priority patterns match, try line-by-line parsing
    # First, try the more specific pattern on each line
    lines = text.split('\n')
    line_matches = []
    
    for i, line in enumerate(lines):
        # Look for full answer pattern in each line
        match = re.search(r'([A-E])\. [A-Za-z0-9 \-\(\)\'",]+', line)
        if match:
            line_matches.append((i, match.group(1)))
    
    if line_matches:
        # Return the answer from the last line that matched
        return line_matches[-1][1]
    
    # Finally, try the most general pattern on each line
    for i in reversed(range(len(lines))):  # Start from bottom
        line = lines[i]
        match = re.search(r'\b([A-E])\b', line)
        if match:
            return match.group(1)
    
    return None  # No answer found

def image_to_base64(image):
    # concat images horizontally with PIL
    if isinstance(image, list):
        new_image = Image.new('RGB', (sum(i.width for i in image), image[0].height))
        for i, img in enumerate(image):
            new_image.paste(img, (i * img.width, 0))
        image = new_image
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def highlight_section_headers(text):
    """
    Highlight section headers wrapped in square brackets while avoiding coordinates.
    Section headers are typically words/phrases, not numbers or coordinates.
    """
    import re
    
    # Pattern to match section headers: [word or phrase] but not [number,number] or [number]
    # This will match things like [Task], [Referenced Question], [Instruction] etc.
    # but avoid [1,1], [5, 5], [0,0] etc.
    pattern = r'\[([A-Za-z][A-Za-z\s]*[A-Za-z]|[A-Za-z])\]'
    
    def replace_header(match):
        header_text = match.group(1)
        return f'<span style="background-color: #f9e79f; color: #333; font-weight: bold; padding: 2px 6px; border-radius: 3px; border: 1px solid #fbc02d;">[{header_text}]</span>'
    
    return re.sub(pattern, replace_header, text)

class Agent:
    def goto(self, x, y):
        pass

    def face(self, direction):
        pass

    def answer(self, ans):
        return ans

agent = Agent()

all_html_display = []
results = []
correct_cnt = 0
style_html = """
<style>
</style>
"""
cnt = 0
for idx in tqdm(range(len(data))):
    _id = data[idx].non_tensor_batch['id']

    if output_html:
        html_display = """
        <div style="display: flex; flex-direction: column; align-items: flex-start; margin-bottom: 20px; border: 5px solid #555; padding: 10px; border-radius: 5px;">
        {content}
        </div>
        """
        content = ""
        content += f"<h2 style='text-align: center; margin-bottom: 30px; color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px;'>ID: {_id}</h2>"
    traj_len = len(data[idx].non_tensor_batch['response'])
    item = {"id": _id, "traj_len": traj_len}

    data_item = dataset[_id]
    view_idx_to_image_path = {i + 1: os.path.join('./data', image_path) for i, image_path in enumerate(data_item['images']) if image_path is not None}
    cogmap_env = CogMapEnv(objects=[], question="", gt_answer=find_gt(_id), gt_total_views=len(view_idx_to_image_path), view_idx_to_image_path=view_idx_to_image_path, views=[], gt_cogmap_dict=json.loads(data_item['grounded_cogmap']), split='test', retrieve_strategy='adaptive')

    for i in range(traj_len):
        obs = data[idx].non_tensor_batch['obs'][i]['image']
        prompt = data[idx].non_tensor_batch['obs'][i]['text']
        if output_html:
            safe_prompt = escape(prompt)
            # Apply highlighting to section headers in the prompt
            safe_prompt = highlight_section_headers(safe_prompt)
            safe_response = escape(data[idx].non_tensor_batch['response'][i].replace('<|im_end|>', '').replace('<|endoftext|>', ''))
            safe_response = (
                safe_response
            )
            content += f"""
            <div style="display: flex; align-items: flex-start; gap: 30px; margin-bottom: 20px; padding: 20px; background-color: #f9f9f9; border-radius: 8px;">
                <div style="flex: 0 0 auto; text-align: center;">
                    <h3 style="margin: 0 0 10px 0; color: #2c3e50; font-size: 16px;">Observation</h3>
                    <img src="data:image/png;base64,{image_to_base64(obs)}" width="400" height="400" style="border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                </div>
                <div style="flex: 1; text-align: center;">
                    <h3 style="margin: 0 0 10px 0; color: #2c3e50; font-size: 16px;">Prompt</h3>
                    <div style="background-color: white; padding: 15px; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: left; max-height: 500px; overflow-y: auto;">
                        {safe_prompt}
                    </div>
                </div>
            </div>
            <div style="width: 100%; display: flex; justify-content: center; margin: 0 0 10px 0;">
                <svg width="40" height="50" style="display: block;" xmlns="http://www.w3.org/2000/svg">
                    <line x1="20" y1="0" x2="20" y2="35" stroke="#888" stroke-width="4" />
                    <polygon points="10,35 30,35 20,48" fill="#888"/>
                </svg>
            </div>
            <div style="margin-bottom: 15px;">
                <div style="background-color: #f0f8ff; padding: 15px; border-radius: 4px; border-left: 4px solid #3498db;">
                <h3 style="margin: 0 0 10px 0; color: #2c3e50; font-size: 16px; text-align: center;">Response</h3>
                    {safe_response}
                </div>
            </div>
            """

        resposne = data[idx].non_tensor_batch['response'][i]
        action, valids = mind_projection([resposne])
        action = action[0]
        valids = valids[0]
        obs = cogmap_env.execute(action)
        
    # last_response = data[idx].non_tensor_batch['response'][-1]
    # last_action, valids = mind_projection([last_response])

    non_empty_steps, retrieve_relative_reward, cogmap_reward = check_progress(cogmap_env.step_codes, cogmap_env, data_item['question'], find_gt(_id), get_meta_objects(data_item['meta_info'], _id), verbose=False)
    # pprint({'retrieve_relative_reward': retrieve_relative_reward, 'cogmap_reward': cogmap_reward})
    # exit(1)

    answer = cogmap_env.pred_answer
    # if valids[0] == 1:
    # if True:
    #     last_action = last_action[0]
    #     try:
    #         last_action = eval(last_action[-1]['code'])
    #         answer = extract_answer(last_response)
    #     except AttributeError:
    #         answer = None
    #     except Exception as e:
    #         answer = None

    item["answer"] = answer
    item["gt"] = find_gt(_id)
    item["retrieve_relative_reward"] = retrieve_relative_reward
    item["cogmap_reward"] = cogmap_reward
    # if retrieve_relative_reward < 0 or cogmap_reward < 0:
    #     break
    if item["gt"] is not None:
        item["correct"] = item['gt'] == item['answer']
    else:
        item["correct"] = False
    if item["correct"]:
        correct_cnt += 1
    
    
    if output_html:
        content += f"""
            <div style="width: 100%; display: flex; justify-content: center; margin: 0 0 10px 0;">
                <svg width="40" height="50" style="display: block;" xmlns="http://www.w3.org/2000/svg">
                    <line x1="20" y1="0" x2="20" y2="35" stroke="#888" stroke-width="4" />
                    <polygon points="10,35 30,35 20,48" fill="#888"/>
                </svg>
            </div>
            <div style="margin-bottom: 15px; display: flex; justify-content: center;">
                <div style="background-color: {'#e8f5e8' if item['correct'] else '#fff5f5'}; padding: 20px; border-radius: 8px; border-left: 4px solid {'#27ae60' if item['correct'] else '#e74c3c'}; text-align: center; max-width: 600px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 15px 0; color: #27ae60; font-size: 18px; text-align: center; font-weight: bold;">Final Result</h3>
                    <table style="width: 100%; border-collapse: collapse; margin: 0 auto; font-size: 16px;">
                        <tr>
                            <td style="padding: 12px; border: 2px solid #27ae60; color: #2c3e50; font-weight: bold; text-align: center; background-color: #f8f9fa;">Answer</td>
                            <td style="padding: 12px; border: 2px solid #27ae60; color: #2c3e50; font-weight: bold; text-align: center; background-color: #f8f9fa;">GT</td>
                            <td style="padding: 12px; border: 2px solid #27ae60; color: #2c3e50; font-weight: bold; text-align: center; background-color: #f8f9fa;">Correct</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 2px solid #27ae60; color: #e74c3c; font-weight: bold; text-align: center; background-color: #fff5f5;">{answer}</td>
                            <td style="padding: 12px; border: 2px solid #27ae60; color: #3498db; font-weight: bold; text-align: center; background-color: #f0f8ff;">{item["gt"]}</td>
                            <td style="padding: 12px; border: 2px solid #27ae60; color: {'#27ae60' if item['correct'] else '#e74c3c'}; font-weight: bold; text-align: center; background-color: {'#f0fff0' if item['correct'] else '#fff5f5'}; font-size: 20px;">{'✅ True' if item['correct'] else '❌ False'}</td>
                        </tr>
                    </table>
                </div>
            </div>
        """
    cnt += 1
    # if cnt > 100:
    #     break
    results.append(item)
    if output_html:
        all_html_display.append(html_display.format(content=content))

# "".join(all_html_display)
if output_html:
    with open(os.path.join(result_base, "all_html_display.html"), "w") as f:
        f.write(style_html + "".join(all_html_display))

def detect_setting(id: str) -> str:
    if 'around' in id:
        return 'around'
    elif 'among' in id:
        return 'among'
    elif 'translation' in id:
        return 'translation'
    elif 'rotation' in id:
        return 'rotation'
    else:
        return 'unknown'

detail_results = {}
retrieve_relative_rewards = []
cogmap_rewards = []
correct_cogmap_score = []
incorrect_cogmap_score = []
correct_retrieve_score = []
incorrect_retrieve_score = []
with open(os.path.join(result_base, "results.jsonl"), "w") as f:
    for result in results:
        setting = detect_setting(result['id'])
        if setting not in detail_results:
            detail_results[setting] = {'correct': 0, 'incorrect': 0}
        if result['correct']:
            detail_results[setting]['correct'] += 1
            correct_cogmap_score.append(result['cogmap_reward'])
            correct_retrieve_score.append(result['retrieve_relative_reward'])
        else:
            detail_results[setting]['incorrect'] += 1
            incorrect_cogmap_score.append(result['cogmap_reward'])
            incorrect_retrieve_score.append(result['retrieve_relative_reward'])
        retrieve_relative_rewards.append(result['retrieve_relative_reward'])
        cogmap_rewards.append(result['cogmap_reward'])
        f.write(json.dumps(result) + "\n")
    
with open(os.path.join(result_base, "acc.txt"), "w") as f:
    print(
        f"Overall: {correct_cnt} / {len(data)} = {correct_cnt / len(data)}, retrieve_relative_reward: {np.mean(retrieve_relative_rewards)}, cogmap_reward: {np.mean(cogmap_rewards)}, correct_cogmap_score: {np.mean(correct_cogmap_score)}, incorrect_cogmap_score: {np.mean(incorrect_cogmap_score)}, correct_retrieve_score: {np.mean(correct_retrieve_score)}, incorrect_retrieve_score: {np.mean(incorrect_retrieve_score)}"
    )
    f.write(
        f"Overall: {correct_cnt} / {len(data)} = {correct_cnt / len(data)}, retrieve_relative_reward: {np.mean(retrieve_relative_rewards)}, cogmap_reward: {np.mean(cogmap_rewards)}, correct_cogmap_score: {np.mean(correct_cogmap_score)}, incorrect_cogmap_score: {np.mean(incorrect_cogmap_score)}, correct_retrieve_score: {np.mean(correct_retrieve_score)}, incorrect_retrieve_score: {np.mean(incorrect_retrieve_score)}\n"
    )
    for setting, data in detail_results.items():
        print(f"{setting}: {data['correct']} / {data['correct'] + data['incorrect']} = {data['correct'] / (data['correct'] + data['incorrect'])}")
        f.write(f"{setting}: {data['correct']} / {data['correct'] + data['incorrect']} = {data['correct'] / (data['correct'] + data['incorrect'])}\n")
