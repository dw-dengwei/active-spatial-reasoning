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

from __future__ import annotations
import ast
import os
import ray
import time
import json
import random

import re
from dataclasses import dataclass, asdict
import logging
from agent_system.environments.env_package.mind.context import ObjectContext, ViewContext, turn_right, turn_left, turn_back, go_straight, move_forward
from copy import deepcopy
from pydantic import BaseModel, field_validator, ValidationError
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def correct_list_check(code_string: str) -> str:
    pattern = r'(\[.*?\])\s+in\s+(.*)'
    replacement_template = r'all(x in \2 for x in \1)'
    corrected_code = re.sub(pattern, replacement_template, code_string)
    return corrected_code


def evaluate_constant_boolean_expression(code_string: str) -> bool | None:
    """
    返回布尔表达式是否是常量表达式，如果是则返回对应的布尔值，否则返回 None。
    """
    stripped = code_string.strip()
    if not stripped:
        return None

    try:
        tree = ast.parse(stripped, mode="eval")
    except SyntaxError:
        return None

    def is_trivial_self_comparison(expr: ast.expr) -> bool:
        for node in ast.walk(expr):
            if isinstance(node, ast.Compare):
                if len(node.ops) != 1 or len(node.comparators) != 1:
                    continue
                if not isinstance(node.ops[0], (ast.Eq, ast.Is)):
                    continue
                left_dump = ast.dump(node.left, annotate_fields=True, include_attributes=False)
                right_dump = ast.dump(node.comparators[0], annotate_fields=True, include_attributes=False)
                if left_dump == right_dump:
                    return True
        return False

    if isinstance(tree, ast.Expression) and is_trivial_self_comparison(tree.body):
        return True

    forbidden_nodes = (
        ast.Name,
        ast.Attribute,
        ast.Call,
        ast.Subscript,
        ast.Lambda,
        ast.GeneratorExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.IfExp,
        ast.NamedExpr,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
    )

    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            return None

    try:
        value = eval(compile(tree, filename="<constant_boolean_check>", mode="eval"), {}, {})
    except Exception:
        return None

    return value if isinstance(value, bool) else None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Pillow is required to render the mind map. Install with: pip install pillow"
    ) from exc


class OptionType:
    OBJECT = 1
    OTHER = 2

class AnnotationType:
    COMPLETE = 1
    MISSING_QUESTION_OBJECT = 2
    MISSING_QUESTION_VIEW = 3

english_to_number = {
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
}

def get_meta_objects(meta_info: list, id) -> list:
    def generate_around_cogmap(meta_info):
        return meta_info[1][1]
    def generate_among_cogmap(meta_info):
        return meta_info[0]
    def generate_translation_cogmap(meta_info):
        return meta_info[1:]
    def generate_rotation_cogmap(meta_info):
        return meta_info

    getter = {
        'around': generate_around_cogmap,
        'among': generate_among_cogmap,
        'translation': generate_translation_cogmap,
        'rotation': generate_rotation_cogmap
    }
    def detect_setting(_id: str) -> str:
        for setting in getter.keys():
            if setting in _id:
                return setting
        return None

    return getter[detect_setting(id)](meta_info)

def parse_question(full_question: str, gt_answer: str, meta_objects: list) -> tuple[dict, dict]:
    try:
        question_txt = full_question.split(':')[-1].split('?')[0].strip()
        options_str = full_question.split(':')[-1].split('?')[1].strip()
    except IndexError:
        question_txt = full_question.split(':')[-1].split('.')[0].strip()
        options_str = full_question.split(':')[-1].split('.')[1].strip()
    options = re.split(r'(?=[A-Z]\.\s+)', options_str)
    options = [opt[2:].strip().lower() for opt in options if opt.strip()]

    question_view = []
    question_view.extend(list(map(lambda s: int(s.replace('image ', '')), re.findall(r'image [1-9]', question_txt))))
    question_view.extend(list(map(lambda s: english_to_number[s.replace(' view', '')], re.findall(r'first|second|third|fourth view', question_txt))))

    question_structure = {'txt': question_txt, 'view': question_view, 'objects': [], 'full_question': full_question}
    if gt_answer.lower() == 'a':
        gt_answer = options[0]
    elif gt_answer.lower() == 'b':
        gt_answer = options[1]
    elif gt_answer.lower() == 'c':
        gt_answer = options[2]
    elif gt_answer.lower() == 'd':
        gt_answer = options[3]
    options_structure = {'options': options, 'option_type': None, 'gt_answer': gt_answer}
    for m_obj in meta_objects:
        if m_obj.strip().lower() in question_txt.strip().lower():
            question_structure['objects'].append(m_obj.lower().strip())
        if m_obj.strip().lower() in options:
            options_structure['option_type'] = OptionType.OBJECT
    return question_structure, options_structure
            
def could_answer(full_question: str, gt_answer: str, meta_objects: list, current_cgmap_info: str) -> AnnotationType:
    question_structure, options_structure = parse_question(full_question, gt_answer, meta_objects)

    if options_structure['option_type'] == OptionType.OBJECT:
        for question_view in question_structure['view']:
            if f"view {question_view}" not in current_cgmap_info.strip().lower():
                return AnnotationType.MISSING_QUESTION_VIEW
        for question_objects in question_structure['objects']:
            if question_objects.strip().lower() not in current_cgmap_info.strip().lower():
                return AnnotationType.MISSING_QUESTION_OBJECT
        if options_structure['gt_answer'].strip().lower() not in current_cgmap_info.strip().lower():
            return False
    elif options_structure['option_type'] == OptionType.OTHER:
        for question_view in question_structure['view']:
            if f"view {question_view}" not in current_cgmap_info.strip().lower():
                return AnnotationType.MISSING_QUESTION_VIEW
        for question_objects in question_structure['objects']:
            if question_objects.strip().lower() not in current_cgmap_info.strip().lower():
                return AnnotationType.MISSING_QUESTION_OBJECT
    
    return AnnotationType.COMPLETE


def fix_malformed_json_string(s: str) -> str:
    """
    通过字符串处理来修正一个特定类型的无效JSON字符串。
    这种无效JSON的特征是，本应是列表的地方被错误地写成了没有键的对象。
    例如： "key": { { "item": 1 }, { "item": 2 } }
    修正后： "key": [ { "item": 1 }, { "item": 2 } ]
    """
    
    s_list = list(s)
    
    pattern = r'":\s*{\s*(?={)'
    
    for match in reversed(list(re.finditer(pattern, s))):
        
        # 定位到需要被替换的 '{' 的索引
        start_index = s.rfind('{', 0, match.end())

        # 从这个位置开始，使用计数器来寻找与之匹配的 '}'
        brace_level = 1
        end_index = -1
        for i in range(start_index + 1, len(s_list)):
            char = s_list[i]
            if char == '{':
                brace_level += 1
            elif char == '}':
                brace_level -= 1
            
            if brace_level == 0:
                end_index = i
                break
        
        # 如果成功找到了匹配的括号，就进行替换
        if end_index != -1:
            s_list[start_index] = '['
            s_list[end_index] = ']'

    return "".join(s_list)

class AgentSyntaxError(Exception):
    """Agent调用链语法错误基类"""
    pass


class InvalidMethodNameError(AgentSyntaxError):
    """无效的方法名错误"""
    def __init__(self, method_name: str):
        self.method_name = method_name
        super().__init__(f"无效的方法名: '{method_name}'")


class MalformedParameterError(AgentSyntaxError):
    """格式错误的参数错误"""
    def __init__(self, param_str: str, reason: str = ""):
        self.param_str = param_str
        self.reason = reason
        super().__init__(f"格式错误的参数: {param_str} {reason}")


class UnmatchedParenthesesError(AgentSyntaxError):
    """括号不匹配错误"""
    def __init__(self, action: str):
        self.action = action
        super().__init__(f"括号不匹配: '{action}'")


class InvalidActionFormatError(AgentSyntaxError):
    """无效的动作格式错误"""
    def __init__(self, action: str, reason: str = ""):
        self.action = action
        self.reason = reason
        super().__init__(f"无效的动作格式: '{action}' {reason}")

def extract_answer(text: str) -> str | None:
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

@dataclass(frozen=True)
class CogMapObject:
    name: str
    position: tuple[int, int]
    facing: str = None

@dataclass(frozen=True)
class CogMap:
    objects: list[CogMapObject]

@dataclass(frozen=True)
class AgentStatus:
    position: tuple[int, int]
    facing: str

@dataclass(frozen=True)
class Observation:
    gt_objects: list[str]
    image: Image.Image
    free_form_cogmap: str
    question: str
    total_views: int
    retrieved_image_path: dict
    cogmap_dict: dict
    cnt: dict
    pred_answer: str = None

@dataclass(frozen=True)
class View:
    name: str
    position: tuple[int, int]
    facing: str

@dataclass(frozen=True)
class Views:
    views: list[View]

class CogMapEnv:
    GridSize = tuple[int, int]
    Point = tuple[int, int]
    def __init__(self, objects: list[dict[str, any]], question: str, gt_answer: str, gt_total_views: int, view_idx_to_image_path: dict[int, str], views: list[dict[str, any]] | None = None, gt_cogmap_dict: dict = None, split='train', retrieve_strategy="adaptive", gt_objects: list[str] = None):
        self.split = split
        self.retrieve_strategy = retrieve_strategy
        self.objects = CogMap(objects=[CogMapObject(**obj) for obj in objects])
        self.gt_objects = gt_objects
        self.question = question
        self.agent_status = None
        # AgentStatus(position=(0, 0), facing='up')
        self.views = Views(views=[View(**view) for view in views])
        self.gt_total_views = gt_total_views
        self.view_idx_to_image_path = view_idx_to_image_path
        self.gt_answer = gt_answer
        self.pred_answer = None
        self.is_done = False
        self.action_length = 0
        self.step_codes = []
        self.retrieved_image_path = None
        self.invalid_reason = None
        self.invalid_cnt = 0
        self.last_action = None
        self.invalid_retrieve_cnt = 0
        self.invalid_update_cogmap_cnt = 0
        self.invalid_answer_cnt = 0
        self.invalid_control_cnt = 0
        self.invalid_goto_cnt = 0
        self.invalid_face_cnt = 0
        self.cnt_retrieve = 0
        self.cnt_update_cogmap = 0
        self.cnt_answer = 0
        self.cnt_control = 0
        self.stage = None
        self.repeat_retrieve_cnt = 0
        self.gt_cogmap_dict = gt_cogmap_dict
        self.function_mapping = {
            'answer': self.answer,
            'invalid': self.invalid,
            'retrieve': self.retrieve,
            'update_cogmap': self.update_cogmap
        }

    def export_context(self):
        context = {}
        for idx, obj in enumerate(self.objects.objects):
            context[f"obj{idx}"] = ObjectContext(obj.name, obj.position, obj.facing)
        for idx, view in enumerate(self.views.views):
            context[f"v{idx+1}"] = ViewContext(view.name, view.position, view.facing)
        # context['agent'] = AgentContext('agent', self.agent_status.position, self.agent_status.facing)
        for name, obj in context.items():
            obj.setup_environment(list(context.values()))
        for func in [turn_right, turn_left, turn_back, go_straight, move_forward]:
            context[func.__name__] = deepcopy(func)
        return deepcopy(context)

    def _name_to_color(self, name: str) -> tuple[int, int, int]:
        """Derive a consistent, visually distinct RGB color from a name string.

        Uses a simple hash so the same name gets the same color across renders.
        """
        h = 0
        for ch in name:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        # Map hash segments to RGB with some clamping for brightness
        r = 80 + (h & 0x7F)
        g = 80 + ((h >> 7) & 0x7F)
        b = 80 + ((h >> 14) & 0x7F)
        return int(r), int(g), int(b)

    def _direction_vector(self, direction: str) -> tuple[int, int, int]:
        """Map a facing direction to a 2D or pseudo-3D vector.

        Returns (dx, dy, dz). dz encodes inner/outer for optional styling.
        """
        try:
            direction = direction.lower().strip()
        except:
            direction = None

        if direction == "up":
            return (0, -1, 0)
        if direction == "down":
            return (0, 1, 0)
        if direction == "left":
            return (-1, 0, 0)
        if direction == "right":
            return (1, 0, 0)
        if direction == "inner":
            return (0, 0, 1)
        if direction == "outer":
            return (0, 0, -1)
        return (0, 0, 0)
    
    def render_mind_map(
        self,
        grid_size: GridSize = (10, 10),
        cell_size: int = 32,
        margin: int = 24,
        grid_color: tuple[int, int, int] = (0, 0, 0),
        background_color: tuple[int, int, int] = (255, 255, 255),
        show_indices: bool = True,
        save_path: str | None = None,
        with_agent: bool = True,
        with_views: bool = True,
    ) -> [Image.Image, dict]:
        """Render a 10x10 grid scene with objects into an image.

        Args:
            objects: Iterable of dicts with keys: "name" (str), "position" ([x, y]),
                optionally "facing" (str: up/right/down/left/inner/outer).
            grid_size: Tuple of (width, height) in cells.
            cell_size: Size of each cell in pixels.
            margin: Outer margin around the grid in pixels.
            grid_color: RGB for grid lines.
            background_color: RGB for background.
            show_indices: If True, draw x/y indices along top/left.
            save_path: If provided, save the image to this path.
            with_agent: If true, render agent on the image.
            with_views: If true, render views on the image.

        Returns:
            PIL Image instance of the rendered map.
        """
        grid_w, grid_h = grid_size
        img_w = margin * 2 + grid_w * cell_size
        img_h = margin * 2 + grid_h * cell_size

        image = Image.new("RGB", (img_w, img_h), color=background_color)
        draw = ImageDraw.Draw(image)
        cogmap_img_info = {'objects': {}}

        # Draw grid
        for x in range(grid_w + 1):
            px = margin + x * cell_size
            draw.line([(px, margin), (px, margin + grid_h * cell_size)], fill=grid_color)
        for y in range(grid_h + 1):
            py = margin + y * cell_size
            draw.line([(margin, py), (margin + grid_w * cell_size, py)], fill=grid_color)

        def _grid_pos_to_coord(pos: tuple[int, int]) -> tuple[int, int]:
            if isinstance(pos, list | tuple) and len(pos) >= 2:
                x, y = pos
            else:
                x, y = 0, 0
            
            x = max(0, min(grid_w - 1, x))
            y = max(0, min(grid_h - 1, y))

            cx = margin + x * cell_size + cell_size // 2
            cy = margin + y * cell_size + cell_size // 2
            return cx, cy


        # Try to load a default font for labels
        try:
            font = ImageFont.load_default(size=15)
        except Exception:
            font = None  # type: ignore

        # Draw indices
        if show_indices and font is not None:
            for x in range(grid_w):
                px = margin + x * cell_size + cell_size // 2
                draw.text((px - 3, margin - 16), str(x), fill=(0, 0, 0), font=font)
            for y in range(grid_h):
                py = margin + y * cell_size + cell_size // 2
                draw.text((margin - 16, py - 6), str(y), fill=(0, 0, 0), font=font)

        # Draw objects
        marker_radius = cell_size * 0.5
        arrow_len = int(cell_size * 0.8)
        for i, obj in enumerate(self.objects.objects):
            cogmap_img_info['objects'][i] = obj
            name = obj.name
            pos_raw = obj.position
            cx, cy = _grid_pos_to_coord(pos_raw)

            color = self._name_to_color(name)

            # Marker (circle)
            draw.ellipse(
                [
                    (cx - marker_radius, cy - marker_radius),
                    (cx + marker_radius, cy + marker_radius),
                ],
                fill=color,
                outline=(30, 30, 30),
                width=2,
            )

            # Facing arrow if present
            facing = obj.facing
            if facing is not None and facing != '':
                facing = facing.strip()
                dx, dy, dz = self._direction_vector(facing)
                arrow_color = (255, 0, 0)

                if dz == 0:
                    end_x = cx + dx * arrow_len
                    end_y = cy + dy * arrow_len
                    draw.line([(cx, cy), (end_x, end_y)], fill=arrow_color, width=4)
                    # Simple arrow head
                    ahx = end_x - dx * 6 + (-dy) * 6
                    ahy = end_y - dy * 6 + (dx) * 6
                    bhx = end_x - dx * 6 - (-dy) * 6
                    bhy = end_y - dy * 6 - (dx) * 6
                    draw.polygon([(end_x, end_y), (ahx, ahy), (bhx, bhy)], fill=arrow_color)
                else:
                    # Physics convention: into page = cross, out of page = dot
                    sym_x = cx
                    sym_y = cy
                    radius = cell_size * 0.5
                    # Outer circle
                    draw.ellipse(
                        [(sym_x - radius, sym_y - radius), (sym_x + radius, sym_y + radius)],
                        outline=arrow_color,
                        width=4,
                    )
                    if dz > 0:
                        # inner (into page): draw a cross inside the circle
                        cross_r = int(radius * 0.6)
                        draw.line(
                            [(sym_x - cross_r, sym_y - cross_r), (sym_x + cross_r, sym_y + cross_r)],
                            fill=arrow_color,
                            width=4,
                        )
                        draw.line(
                            [(sym_x - cross_r, sym_y + cross_r), (sym_x + cross_r, sym_y - cross_r)],
                            fill=arrow_color,
                            width=4,
                        )
                    else:
                        # outer (out of page): draw a filled dot inside the circle
                        dot_r = int(radius * 0.5)
                        draw.ellipse(
                            [(sym_x - dot_r, sym_y - dot_r), (sym_x + dot_r, sym_y + dot_r)],
                            fill=arrow_color,
                        )


        if with_views:
            cogmap_img_info['views'] = {}
            view_color = (0, 0, 0)
            for i, view in enumerate(self.views.views):
                view_cx, view_cy = _grid_pos_to_coord(view.position)
                if view.facing is None or view.facing == '':
                    continue
                dx, dy, dz = self._direction_vector(view.facing)
                draw.polygon([(view_cx, view_cy - cell_size // 2.3), (view_cx + cell_size // 2.3, view_cy + cell_size // 2.3), (view_cx - cell_size // 2.3, view_cy + cell_size // 2.3)], outline=view_color, fill=(255, 255, 255))
                end_x = view_cx + dx * cell_size // 2
                end_y = view_cy + dy * cell_size // 2
                draw.line([(view_cx, view_cy), (end_x, end_y)], fill=view_color, width=3)
                ahx = end_x - dx * 5 + (-dy) * 5
                ahy = end_y - dy * 5 + (dx) * 5
                bhx = end_x - dx * 5 - (-dy) * 5
                bhy = end_y - dy * 5 - (dx) * 5
                draw.polygon([(end_x, end_y), (ahx, ahy), (bhx, bhy)], fill=view_color)

                if font is not None:
                    view_idx = view.name.split(' ')[-1]
                    text = 'v' + view_idx
                    bbox = draw.textbbox((view_cx, view_cy), text, font=font, anchor="mm")
                    draw.rectangle(bbox, fill=(255, 255, 255))
                    draw.text((view_cx, view_cy), text, fill=(0, 0, 0), font=font, anchor="mm")
                cogmap_img_info['views']['v' + view_idx] = view

        if with_agent and self.agent_status is not None:
            agent_color = '#fa8c35'
            cogmap_img_info['agent'] = self.agent_status
            agent_pos = self.agent_status.position
            agent_facing = self.agent_status.facing
            agent_cx, agent_cy = _grid_pos_to_coord(agent_pos)
            # draw a triangle centered at agent_cx, agent_cy
            draw.polygon([(agent_cx, agent_cy - cell_size // 2), (agent_cx + cell_size // 2, agent_cy + cell_size // 2), (agent_cx - cell_size // 2, agent_cy + cell_size // 2)], fill=agent_color)

            dx, dy, dz = self._direction_vector(agent_facing)
            end_x = agent_cx + dx * arrow_len
            end_y = agent_cy + dy * arrow_len
            draw.line([(agent_cx, agent_cy), (end_x, end_y)], fill=agent_color, width=4)
            ahx = end_x - dx * 6 + (-dy) * 6
            ahy = end_y - dy * 6 + (dx) * 6
            bhx = end_x - dx * 6 - (-dy) * 6
            bhy = end_y - dy * 6 - (dx) * 6
            draw.polygon([(end_x, end_y), (ahx, ahy), (bhx, bhy)], fill=agent_color)

        for i, obj in enumerate(self.objects.objects):
            cogmap_img_info['objects'][i] = obj
            name = obj.name
            pos_raw = obj.position
            cx, cy = _grid_pos_to_coord(pos_raw)
            if font is not None:
                text = str(i)
                bbox = draw.textbbox((cx, cy), text, font=font, anchor="mm")
                draw.rectangle(bbox, fill=(255, 255, 255))
                draw.text((cx, cy), text, fill=(0, 0, 0), font=font, anchor="mm")
        

        if save_path:
            image.save(save_path)

        return image, cogmap_img_info

    def free_form_cgmap(self, cogmap_img_info: dict) -> str:
        result = ""
        for k, v in cogmap_img_info['objects'].items():
            if v.facing:
                result += f'"{v.name}" is at `{v.position}`, facing "{v.facing}", marked as a circle, which is labeled with "{k}", in the image. '
            else:
                result += f'"{v.name}" is at `{v.position}`, marked as a circle, which is labeled with "{k}", in the image. '
        
        if 'views' in cogmap_img_info:
            for k, v in cogmap_img_info['views'].items():
                view_name = v.name.replace('Image ', 'View ')
                result += f'View of "{view_name}" is at `{v.position}`, facing "{v.facing}", marked as a white triangle, which is labeled with "{k}", in the image. '
        
        if 'agent' in cogmap_img_info:
            agent = cogmap_img_info['agent']
            result += f'The agent is at `{agent.position}`, facing "{agent.facing}", marked as an orange triangle in the image. '
        
        if result.strip() == '':
            result = 'The cognitive map is empty.'
        return result

    def parse(self, action):
        """
        解析agent调用链并返回执行步骤
        
        Args:
            action: 字符串形式的agent调用链，如 "agent.goto(1,2).face('left')"
            
        Returns:
            list: 解析后的步骤列表，每个步骤包含方法名和参数
            
        Raises:
            InvalidActionFormatError: 当输入格式无效时
            UnmatchedParenthesesError: 当括号不匹配时
            InvalidMethodNameError: 当方法名无效时
            MalformedParameterError: 当参数格式错误时
        """
        import re
        
        # 输入验证
        if not isinstance(action, str):
            raise InvalidActionFormatError(str(action), "输入必须是字符串")
        
        if not action.strip():
            raise InvalidActionFormatError(action, "输入不能为空")
            
        # 检查括号匹配
        if not self._check_parentheses(action):
            raise UnmatchedParenthesesError(action)
            
        # 移除开头的 "agent." 前缀
        if action.startswith("agent."):
            action = action[5:]  # 移除 "agent."
        elif not action.startswith("agent."):
            raise InvalidActionFormatError(action, "action 必须以 agent. 开头")
        elif not action.strip():
            raise InvalidActionFormatError(action, "移除agent前缀后不能为空")
        
        # 验证基本格式：应该包含方法调用
        if not re.search(r'\w+\(', action):
            raise InvalidActionFormatError(action, "必须包含方法调用")
        
        # 解析方法调用链的正则表达式
        # 匹配方法名和参数，支持字符串和数字参数
        pattern = r'(\.?\w+)\(([^)]*)\)'
        matches = re.findall(pattern, action)
        
        if not matches:
            raise InvalidActionFormatError(action, "未找到有效的方法调用")
        
        steps = []
        answer_cnt = 0
        goto_cnt = 0
        face_cnt = 0
        retrieve_cnt = 0
        update_cogmap_cnt = 0
        for method_name, params_str in matches:
            # 验证方法调用格式
            is_dotted = method_name.startswith('.')
            if not is_dotted:
                raise InvalidActionFormatError(action, "方法调用必须以 . 开头")

            if is_dotted:
                method_name = method_name[1:]

            # 验证方法名
            if not self._is_valid_method_name(method_name):
                raise InvalidMethodNameError(method_name)
            
            # 解析参数
            try:
                params = self._parse_parameters(params_str)
            except Exception as e:
                raise ValueError(f"解析失败: {str(e)}") from e
            
            if method_name == 'answer':
                answer_cnt += 1
                if len(params) != 1:
                    raise MalformedParameterError(params_str, "answer 方法需要一个参数")
                answer = params[0]
                extracted_answer = extract_answer(answer)
                if not extracted_answer:
                    answer = params_str.replace('"','').replace("'",'').strip()
                    extracted_answer = extract_answer(answer)
                    if not extracted_answer:
                        raise MalformedParameterError(params_str, "answer is invalid")
                    params = [answer]
            elif method_name == 'retrieve':
                retrieve_cnt += 1
                view = params[0]
                view = int(view)
                if view not in range(1, self.gt_total_views + 1):
                    if view < 1:
                        view = 1
                    else:
                        view = self.gt_total_views
                params = [view]

            elif method_name == 'update_cogmap':
                update_cogmap_cnt += 1
                if len(params) != 1:
                    raise ValueError("update_cogmap 方法需要一个参数")
                class ObjectModel(BaseModel):
                    name: str
                    position: tuple[int, int]
                    facing: Optional[str] = None


                    @field_validator('facing')
                    def check_facing(cls, facing: Optional[str]) -> Optional[str]:
                        return facing

                class ViewModel(BaseModel):
                    name: str
                    position: tuple[int, int]

                class CogMapModel(BaseModel):
                    objects: list[ObjectModel]
                    views: list[ViewModel]

                if not isinstance(params[0], str):
                    params[0] = json.dumps(params[0])
                try:
                    CogMapModel.model_validate_json(params[0])
                except ValidationError:
                    data = json.loads(params[0])
                    for idx, obj in enumerate(data['objects']):
                        if isinstance(data['objects'][idx]['position'], list):
                            if all([isinstance(pos, list) for pos in data['objects'][idx]['position']]):
                                sum_pos = [0, 0]
                                for pos in data['objects'][idx]['position']:
                                    sum_pos[0] += pos[0]
                                    sum_pos[1] += pos[1]
                                data['objects'][idx]['position'] = (sum_pos[0] / len(data['objects'][idx]['position']), sum_pos[1] / len(data['objects'][idx]['position']))
                    
                    for idx, view in enumerate(data['views']):
                        if isinstance(data['views'][idx]['position'], list):
                            if all([isinstance(pos, list) for pos in data['views'][idx]['position']]):
                                sum_pos = [0, 0]
                                for pos in data['views'][idx]['position']:
                                    sum_pos[0] += pos[0]
                                    sum_pos[1] += pos[1]
                                data['views'][idx]['position'] = (sum_pos[0] / len(data['views'][idx]['position']), sum_pos[1] / len(data['views'][idx]['position']))
                    params[0] = json.dumps(data)
                    CogMapModel.model_validate_json(params[0])
                except Exception:
                    params[0] = fix_malformed_json_string(params[0])
                    CogMapModel.model_validate_json(params[0])

            steps.append({
                'method': method_name,
                'params': params
            })

        if goto_cnt > 0 or face_cnt > 0:
            if goto_cnt == 1 and face_cnt == 1:
                control_cnt = 1
            else:
                raise InvalidActionFormatError(action, "goto 和 face 方法必须成对调用")
        else:
            control_cnt = 0
        
        if control_cnt + answer_cnt + retrieve_cnt + update_cogmap_cnt != 1:
            raise InvalidActionFormatError(action, "action 只能调用一次，且只能调用一次goto().face(), answer, retrieve, update_cogmap方法")
        
        if control_cnt > 0:
            self.cnt_control += 1
        if answer_cnt > 0:
            self.cnt_answer += 1
        if update_cogmap_cnt > 0:
            self.cnt_update_cogmap += 1
        if retrieve_cnt > 0:
            self.cnt_retrieve += 1
        return steps

    def execute(self, action):
        step_codes = [step['code'].strip() for step in action if step['code'].strip() != '' and 'agent' not in step['code'].strip()]
        self.step_codes.extend(step_codes)

        valid_reasoning = True

        if valid_reasoning:
            try:
                action_code = [step['code'].strip() for step in action if 'agent' in step['code'].strip()]
                action_code = action_code[-1]
                steps = self.parse(action_code)
                self.action_length += len(steps)
            except Exception as e:
                print(f"Error parsing action: {e} | {action_code} | {action}")
                steps = [{'method': 'invalid', 'params': [f'In parsing action: {e} | {action_code} | {action}']}]
        else:
            steps = [{'method': 'invalid', 'params': ['invalid']}]

        for step in steps:
            # console.log(f"Executing step: {step}", style="green")
            try:
                # print(f"Executing step: {step['method']} | {step['params']}")
                self.function_mapping[step['method']](*step['params'])
            except Exception as e:
                print(f"Error executing step: {e} | {step['method']} | {step['params']}")
                self.function_mapping['invalid']([f'In executing step: {e} | {step['method']} | {step['params']}'])
                break

        image, cogmap_img_info = self.render_mind_map(with_agent=True)
        free_form_cogmap = self.free_form_cgmap(cogmap_img_info)
        cogmap_dict = {
            'objects': [asdict(obj) for obj in self.objects.objects],
            'views': [asdict(view) for view in self.views.views]
        }
        return Observation(
            gt_objects=self.gt_objects,
            image=image,
            free_form_cogmap=free_form_cogmap,
            pred_answer=self.pred_answer,
            question=self.question,
            total_views=self.gt_total_views,
            retrieved_image_path=self.retrieved_image_path,
            cogmap_dict=cogmap_dict,
            cnt={"cnt_retrieve": self.cnt_retrieve, "cnt_update_cogmap": self.cnt_update_cogmap, "cnt_answer": self.cnt_answer, "cnt_control": self.cnt_control},
        )
    
    def answer(self, answer: str):
        self.last_action = "answer"
        self.stage = 'answer'
        self.is_done = True
        extracted_answer = extract_answer(answer)
        self.pred_answer = extracted_answer

    def retrieve(self, view: int):
        self.stage = 'retrieve'
        self.last_action = "retrieve"
        self.is_done = False
        if self.retrieve_strategy == "random":
            all_views = list(range(1, self.gt_total_views + 1))
            known_views = []
            for cog_view in self.views.views:
                known_views.append(int(cog_view.name.split(" ")[-1].strip()))
            remaining_views = [view for view in all_views if view not in known_views]
            try:
                view = random.choice(remaining_views)
            except Exception:
                view = 1
        elif self.retrieve_strategy == "greedy":
            all_views = list(range(1, self.gt_total_views + 1))
            known_views = []
            for cog_view in self.views.views:
                known_views.append(int(cog_view.name.split(" ")[-1].strip()))
            remaining_views = [view for view in all_views if view not in known_views]
            view = remaining_views[0]
        else:
            known_views = []
            for cog_view in self.views.views:
                known_views.append(int(cog_view.name.split(" ")[-1].strip()))
                if str(view) in cog_view.name:
                    self.repeat_retrieve_cnt += 1
                    break

        self.retrieved_image_path = {view: self.view_idx_to_image_path[view]}

    def update_cogmap(self, cogmap: str | dict):
        self.stage = 'update_cogmap'
        self.last_action = "update_cogmap"
        if isinstance(cogmap, str):
            cogmap = json.loads(cogmap)
        self.is_done = False
        def clip(val, min, max):
            if val < min:
                return min
            if val > max:
                return max
            return val
        for idx, obj in enumerate(cogmap['objects']):
            cogmap['objects'][idx]['position'] = (clip(obj['position'][0], 0, 9), clip(obj['position'][1], 0, 9))
        for idx, view in enumerate(cogmap['views']):
            cogmap['views'][idx]['position'] = (clip(view['position'][0], 0, 9), clip(view['position'][1], 0, 9))

        # merge the original objects and views with the updated objects and views
        for obj in cogmap['objects']:
            if obj['name'] not in [o.name for o in self.objects.objects]:
                if 'facing' in obj:
                    if obj['facing'] is not None and 'back' in obj['facing']:
                        obj['facing'] = 'down'
                self.objects.objects.append(CogMapObject(**obj))
        for view in cogmap['views']:
            if view['name'] not in [v.name for v in self.views.views]:
                try:
                    view_idx = int(view['name'].split(" ")[-1].strip())
                except Exception:
                    view_idx_group = re.search(r'(\d+)', view['name'])
                    if view_idx_group is not None:
                        view_idx = view_idx_group.group(1)
                    else:
                        continue
                view['name'] = f"Image {view_idx}"
                if 'facing' in view:
                    if view['facing'] is not None and 'back' in view['facing']:
                        view['facing'] = 'down'
                self.views.views.append(View(**view))
        self.retrieved_image_path = None

    def invalid(self, reason: str):
        self.stage = 'invalid'
        self.is_done = True
        self.pred_answer = None
        self.invalid_reason = reason
        self.invalid_cnt += 1
        self.last_action = "invalid"
        if 'retrieve' in reason:
            self.invalid_retrieve_cnt += 1
        if 'update_cogmap' in reason:
            self.invalid_update_cogmap_cnt += 1
        if 'answer' in reason:
            self.invalid_answer_cnt += 1
        if 'control' in reason:
            self.invalid_control_cnt += 1

    def _check_parentheses(self, action: str) -> bool:
        """
        检查括号是否匹配
        
        Args:
            action: 要检查的字符串
            
        Returns:
            bool: 括号是否匹配
        """
        count = 0
        for char in action:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
                if count < 0:  # 右括号多于左括号
                    return False
        return count == 0  # 左括号和右括号数量相等
    
    def _is_valid_method_name(self, method_name: str) -> bool:
        """
        验证方法名是否有效
        
        Args:
            method_name: 方法名
            
        Returns:
            bool: 方法名是否有效
        """
        # 定义有效的方法名列表
        valid_methods = {
            'goto', 'face', 'answer', 'retrieve', 'update_cogmap'
        }
        
        # 检查方法名是否在有效列表中
        return method_name in valid_methods
    
    def _parse_parameters(self, params_str: str):
        """
        解析方法参数字符串
        
        Args:
            params_str: 参数字符串，如 "1,2" 或 "'left'" 或 "'A. turn right'"
            
        Returns:
            list: 解析后的参数列表
            
        Raises:
            MalformedParameterError: 当参数格式错误时
        """
        if not params_str:
            return []
        # 使用 ast.literal_eval 安全地评估字符串
        try:
        # if True:
            evaluated_param = ast.literal_eval(params_str)
        except Exception as e:
            first_ = params_str.find('{')
            last_ = params_str.rfind('}')
            params_str = params_str[first_:last_ + 1]
            evaluated_param = params_str
        if isinstance(evaluated_param, list):
            return evaluated_param
        elif isinstance(evaluated_param, tuple):
            return list(evaluated_param)
        return [evaluated_param]
    
    def _convert_parameter(self, param_str):
        """
        转换参数字符串为适当的数据类型
        
        Args:
            param_str: 参数字符串
            
        Returns:
            转换后的参数值
            
        Raises:
            MalformedParameterError: 当参数格式无效时
        """
        param_str = param_str.strip()
        
        if not param_str:
            raise MalformedParameterError(param_str, "参数不能为空")
        
        # 字符串参数（带引号）
        if (param_str.startswith("'") and param_str.endswith("'")) or \
           (param_str.startswith('"') and param_str.endswith('"')):
            if len(param_str) < 2:
                raise MalformedParameterError(param_str, "字符串参数格式错误")
            return param_str[1:-1]  # 移除引号
        
        # 数字参数
        try:
            if '.' in param_str:
                raise MalformedParameterError(param_str, "float is not supported")
            else:
                return int(param_str)
        except ValueError:
            # 如果无法转换为数字，检查是否是有效的标识符
            if param_str.replace('_', '').isalnum():
                return param_str
            else:
                raise MalformedParameterError(param_str, "无效的参数格式")

    def task_completed(self):
        return self.is_done
    
    def evaluate(self):
        if not self.task_completed() or self.pred_answer is None:
            return False
        
        extracted_answer = extract_answer(self.pred_answer)
        is_correct = extracted_answer == self.gt_answer if extracted_answer else False
        return is_correct

    def close(self):
        pass
    
    def reset(self):
        self.stage = None
        self.is_done = False
        self.pred_answer = None
        self.invalid_reason = None
        self.agent_status = None
        # AgentStatus(position=(0, 0), facing='up')
        self.action_length = 0
        self.step_codes = []
        self.retrieved_image_path = None
        self.cnt_retrieve = 0
        self.cnt_update_cogmap = 0
        self.cnt_answer = 0
        self.cnt_control = 0
        self.repeat_retrieve_cnt = 0
        self.invalid_retrieve_cnt = 0
        self.invalid_cnt = 0
        self.last_action = None
        self.invalid_update_cogmap_cnt = 0
        self.invalid_answer_cnt = 0
        self.invalid_control_cnt = 0
        self.invalid_goto_cnt = 0
        self.invalid_face_cnt = 0

def generate_around_cogmap(item) -> dict[str, Any]:
    """
    """
    meta_info = item.get("meta_info", [])

    objects = meta_info[1][1]
    return objects


def generate_among_cogmap(item) -> dict[str, Any]:
    """
    """
    objects = item.get("meta_info", [])[0]

    return objects


def generate_translation_cogmap(item) -> list[str]:
    """
    """
    meta_info = item.get("meta_info", [])
    objects = meta_info[1:]

    return objects


def generate_rotation_cogmap(item) -> list[str]:
    """
    """
    objects = item.get("meta_info", [])
    return objects

getter = {
    'around': generate_around_cogmap,
    'among': generate_among_cogmap,
    'translation': generate_translation_cogmap,
    'rotation': generate_rotation_cogmap
}

def detect_setting(_id: str) -> str:
    for setting in getter.keys():
        if setting in _id:
            return setting
    return None

class MindWorker:
    """
    Ray Actor that holds an instance of Mind and operates the environment
    based on method calls from the main process.
    """

    def __init__(self, max_interactions, data_item, orm_weight=1.0, prm_weight=0.5, success_step_reward=1.0, failure_step_penalty=0.5, mindcube_data_base='./data', split='train', use_sac=True, use_cogmap_reward=True, use_retrieve_reward=True, retrieve_strategy="adaptive"):
        cogmap, self.question, self.gt_answer, self._id = data_item['grounded_cogmap'], data_item['question'], data_item['gt_answer'], data_item['id']
        cogmap = json.loads(cogmap)
        self.gt_cogmap_dict = cogmap
        self.objects, self.views = cogmap['objects'], cogmap['views']
        self.view_idx_to_image_path = {i + 1: os.path.join(mindcube_data_base, image_path) for i, image_path in enumerate[Any](data_item['images']) if image_path is not None}
        setting = detect_setting(self._id)
        if setting is None:
            raise ValueError(f"Unsupported setting: {self._id}")
        self.all_object_categories = getter[setting](data_item)
        self.meta_info = data_item['meta_info']
        self.current_step_count = 0
        self.max_interactions = max_interactions
        self.env = None
        self.split = split
        self.use_sac = use_sac
        self.use_cogmap_reward = use_cogmap_reward
        self.use_retrieve_reward = use_retrieve_reward
        self.w_orm = orm_weight
        if use_sac:
            self.w_prm = prm_weight
        else:
            self.w_prm = 0
        self.success_step_reward = success_step_reward
        self.failure_step_penalty = failure_step_penalty
        self.retrieve_strategy = retrieve_strategy
        
    def reset(self):
        """Reset the environment with a new task."""
        if self.env is not None:
            self.env.close()
            time.sleep(2)

        self.current_step_count = 0

        self.env = CogMapEnv(
            objects=[],
            question=self.question,
            gt_answer=self.gt_answer,
            gt_total_views=len(self.view_idx_to_image_path),
            view_idx_to_image_path=self.view_idx_to_image_path,
            views=[],
            gt_cogmap_dict=self.gt_cogmap_dict,
            split=self.split,
            retrieve_strategy=self.retrieve_strategy,
            gt_objects=[o["name"] for o in self.gt_cogmap_dict["objects"]],
        )
        image, info = self.env.render_mind_map(with_agent=False)
        free_form_cogmap = self.env.free_form_cgmap(info)

        obs = {'question': self.question, 'image': image, 'free_form_cogmap': free_form_cogmap, 'id': self._id, 'total_views': len(self.view_idx_to_image_path), 'gt_objects': [o['name'] for o in self.gt_cogmap_dict['objects']]}
        return obs

    def check_progress(self, step_codes: list[str]):
        """
        step_codes format: [str, str, str, ...]
        """
        step_rewards = [0.0] * len(step_codes)
        non_empty_steps = 0

        base_context = self.env.export_context()

        for i, code in enumerate(step_codes):
            if not code or code.strip() == '':
                continue

            non_empty_steps += 1
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
                # print(f"Error executing code: {e} | {code} | {local_context.keys()}")
                # print(f"Local context: {local_context}")
                # print(f"Global context: {global_context}")
                # print('-' * 100)
                result = False

            # 核心修改：不再使用-1.0的严厉惩罚
            if isinstance(result, bool) and result:
                step_rewards[i] = self.success_step_reward
            else:
                step_rewards[i] = -self.failure_step_penalty

        progress_reward = sum(step_rewards)

        length_reward = 0.0

        cogmap_reward = []
        # check if the cogmap is correct
        if self.env.gt_cogmap_dict is not None:
            for object in self.env.objects.objects:
                if object.name.lower().strip() not in [o['name'].lower().strip() for o in self.env.gt_cogmap_dict['objects']]:
                    cogmap_reward.append(-0.5)
                else:
                    cogmap_reward.append(1.0)
                    target_object = [o for o in self.env.gt_cogmap_dict['objects'] if o['name'].lower().strip() == object.name.lower().strip()][0]
                    # print(target_object, object)
                    if object.position != tuple(target_object['position']):
                        cogmap_reward.append(-0.5)
                    else:
                        cogmap_reward.append(1.0)
            for view in self.env.views.views:
                if view.name.lower().strip().split(" ")[-1].strip() not in [v['name'].lower().strip().split(" ")[-1].strip() for v in self.env.gt_cogmap_dict['views']]:
                    cogmap_reward.append(-0.5)
                else:
                    cogmap_reward.append(1.0)
                    target_view = [v for v in self.env.gt_cogmap_dict['views'] if v['name'].lower().strip().split(" ")[-1].strip() == view.name.lower().strip().split(" ")[-1].strip()][0]
                    # print(target_view, view)
                    if view.position != tuple(target_view['position']) or view.facing != target_view['facing']:
                        cogmap_reward.append(-0.5)
                    else:
                        cogmap_reward.append(1.0)
        
        retrieve_relative_reward = 0
        _, cogmap_img_info = self.env.render_mind_map(with_agent=True)
        free_form_cogmap = self.env.free_form_cgmap(cogmap_img_info)
        answer_type = could_answer(self.question, self.gt_answer, self.all_object_categories, free_form_cogmap)
        if answer_type == AnnotationType.COMPLETE:
            retrieve_relative_reward += 1.0
        else:
            retrieve_relative_reward += -0.5
        
        cogmap_reward = sum(cogmap_reward) #  / len(cogmap_reward) if len(cogmap_reward) > 0 else 0.0

        return progress_reward, length_reward, step_rewards, step_codes, non_empty_steps, retrieve_relative_reward, cogmap_reward

    def step(self, action):
        """Execute one step in the environment."""
        if self.env is None:
            raise RuntimeError("Environment not reset before step. Please call reset() first.")

        self.current_step_count += 1

        obs = self.env.execute(action)
        current_stage = self.env.stage

        done = self.env.task_completed() or (self.current_step_count >= self.max_interactions)

        progress_reward, _, step_rewards, step_codes, non_empty_steps, retrieve_relative_reward, cogmap_reward = self.check_progress(self.env.step_codes)
        if done:
            is_success = self.env.evaluate()
            result_reward = 5.0 if is_success else 0.0

            if is_success:
                reward = self.w_orm * result_reward + self.w_prm * (progress_reward + 1.0 * cogmap_reward + 1.0 * retrieve_relative_reward)
            else:
                reward = 0


            info = {
                "last_action": self.env.last_action,
                "is_done": done,
                "won": is_success,
                "step_count": self.current_step_count,
                "id": self._id,
                "step_reward": step_rewards,
                "step_code": step_codes,
                "no_code_cnt": len(step_codes) - non_empty_steps,
                "log_metric_invalid_cnt": self.env.invalid_cnt,
                "log_metric_cnt_retrieve": self.env.cnt_retrieve,
                "log_metric_cnt_update_cogmap": self.env.cnt_update_cogmap,
                "log_metric_cnt_answer": self.env.cnt_answer,
                "log_metric_cnt_control": self.env.cnt_control,
                "log_metric_repeat_retrieve_cnt": self.env.repeat_retrieve_cnt,
                "log_metric_cogmap_reward": cogmap_reward,
                "log_metric_retrieve_relative_reward": retrieve_relative_reward,
                "log_metric_progress_reward": progress_reward,
                "log_metric_invalid_retrieve_cnt": self.env.invalid_retrieve_cnt,
                "log_metric_invalid_update_cogmap_cnt": self.env.invalid_update_cogmap_cnt,
                "log_metric_invalid_answer_cnt": self.env.invalid_answer_cnt,
                "log_metric_invalid_control_cnt": self.env.invalid_control_cnt,
                "log_metric_invalid_goto_cnt": self.env.invalid_goto_cnt,
                "log_metric_invalid_face_cnt": self.env.invalid_face_cnt,
                "log_metric_length_reward": 0,
                "log_metric_base_reward": 0,
                "log_metric_quality_bonus": 0,
                "invalid_reason": self.env.invalid_reason,
                "progress_retrieve": retrieve_relative_reward,
                "progress_cogmap": cogmap_reward,
                "progress_answer": progress_reward,
            }

        else:
            reward = 0.0
            info = {
                "last_action": self.env.last_action,
                "is_done": done,
                "won": False,
                "step_count": self.current_step_count,
                "id": self._id,
                "step_reward": [0.0],
                "step_code": [],
                "no_code_cnt": 0,
                "log_metric_cnt_retrieve": self.env.cnt_retrieve,
                "log_metric_invalid_cnt": self.env.invalid_cnt,
                "log_metric_cnt_update_cogmap": self.env.cnt_update_cogmap,
                "log_metric_cnt_answer": self.env.cnt_answer,
                "log_metric_cnt_control": self.env.cnt_control,
                "log_metric_repeat_retrieve_cnt": self.env.repeat_retrieve_cnt,
                "log_metric_cogmap_reward": 0.0,
                "log_metric_retrieve_relative_reward": 0.0,
                "log_metric_progress_reward": 0.0,
                "log_metric_invalid_retrieve_cnt": self.env.invalid_retrieve_cnt,
                "log_metric_invalid_update_cogmap_cnt": self.env.invalid_update_cogmap_cnt,
                "log_metric_invalid_answer_cnt": self.env.invalid_answer_cnt,
                "log_metric_invalid_control_cnt": self.env.invalid_control_cnt,
                "log_metric_invalid_goto_cnt": self.env.invalid_goto_cnt,
                "log_metric_invalid_face_cnt": self.env.invalid_face_cnt,
                "invalid_reason": self.env.invalid_reason,
                "stage": current_stage,
            }
            if current_stage == 'retrieve':
                key = 'progress_retrieve'
                info[key] = retrieve_relative_reward
                info['progress_cogmap'] = None
                info['progress_answer'] = None
            elif current_stage == 'update_cogmap':
                key = 'progress_cogmap'
                info[key] = cogmap_reward
                info['progress_retrieve'] = None
                info['progress_answer'] = None
            elif current_stage == 'answer':
                key = 'progress_answer'
                info[key] = progress_reward
                info['progress_retrieve'] = None
                info['progress_cogmap'] = None
            else:
                pass

        return obs, reward, done, info

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()


class MindEnvs:
    """
    A Ray-based distributed wrapper for Mind.
    - Creates multiple Ray actors, each holding a separate AppWorld instance.
    - Implements Gym-style interfaces such as step() / reset() / close().
    """

    def __init__(self, dataset_path, max_interactions, seed, env_num, group_n, start_server_id, resources_per_worker, data_size=None, port_file="mind_ports.ports", split="train", use_sac=True, use_cogmap_reward=True, use_retrieve_reward=True, retrieve_strategy="adaptive"):
        super().__init__()
        self.use_cogmap_reward = use_cogmap_reward
        self.use_retrieve_reward = use_retrieve_reward
        self.dataset_path = dataset_path
        self.max_interactions = max_interactions
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.data = []
        self.data_idx = 0
        self.resources_per_worker = resources_per_worker
        with open(dataset_path) as f:
            for idx, line in enumerate(f.readlines()):
                try:
                    data = json.loads(line)
                except Exception as e:
                    print(f"Error loading data: {e}, {idx}")
                    raise e
                self.data.append(data)

        if data_size:
            self.data = self.data[:int(data_size)]

        random.seed(seed)
        self.split = split
        self.use_sac = use_sac
        self.retrieve_strategy = retrieve_strategy
        
        if self.env_num > len(self.data):
            raise ValueError(f"Env_num ({self.env_num}) exceeds available task_ids in '{self.dataset_path}' ({len(self.data)}). Please reducing env_num to {len(self.data)}.")
            
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(logging_config=ray.LoggingConfig(encoding="TEXT", log_level="INFO", additional_log_standard_attrs=['name']))


    def fetch_data(self):
        yield self.data[self.data_idx]
        self.data_idx = (self.data_idx + 1) % len(self.data)

    def step(self, actions):
        """
        actions: Must be a list with length equal to self.num_processes,
        each sent to the corresponding worker.
        
        Return format follows Gym's step() convention:
            observations, rewards, dones, infos
        """
        futures = []
        action_len = len(actions)
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i % len(actions)])
            futures.append(future)

        # Collect results
        results = ray.get(futures)

        # truncate
        results = results[:action_len]
        
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self, index):
        """
        Reset all worker environments simultaneously,
        returning each environment's initial observation and info.
        """
        # Create Ray actors (workers)
        env_worker = ray.remote(**self.resources_per_worker)(MindWorker)
        self.workers = []
        for idx in index:
            data_item = self.data[idx]
            worker = env_worker.remote(
                max_interactions=self.max_interactions,
                data_item=data_item,
                split=self.split,
                use_sac=self.use_sac,
                use_cogmap_reward=self.use_cogmap_reward,
                use_retrieve_reward=self.use_retrieve_reward,
                retrieve_strategy=self.retrieve_strategy,
            )
            self.workers.append(worker)

        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        
        obs_list = []

        for obs in results:
            obs_list.append(obs)

        return obs_list

    def close(self):
        """Close all workers."""
        # Send close commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.close.remote()
            futures.append(future)
        
        # Wait for all workers to close
        ray.get(futures)
        
        # Shutdown Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def render(self):
        """Implement this if visualization is needed."""
        pass


def build_mind_envs(dataset_path,
                        max_interactions=10,
                        seed=0,
                        env_num=1,
                        group_n=1,
                        start_server_id=0,
                        data_size=None,
                        resources_per_worker={"num_cpus": 0.1},
                        split='train',
                        use_sac=True,
                        use_cogmap_reward=True,
                        use_retrieve_reward=True,
                        retrieve_strategy="adaptive",
                        ):

    return MindEnvs(
        dataset_path=dataset_path,
        max_interactions=max_interactions,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        start_server_id=start_server_id,
        data_size=data_size,
        resources_per_worker=resources_per_worker,
        split=split,
        use_sac=use_sac,
        use_cogmap_reward=use_cogmap_reward,
        use_retrieve_reward=use_retrieve_reward,
        retrieve_strategy=retrieve_strategy,
    )


if __name__ == "__main__":
    env = CogMapEnv([], '', '', 0, {}, [])
    action_str = """agent.answer('D')"""

    steps = env.parse(action_str)
    print(steps)
