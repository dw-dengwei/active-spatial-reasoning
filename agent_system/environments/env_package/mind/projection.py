import re
import copy
from rich.pretty import pprint
import xmltodict
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Optional, Union

class Step(BaseModel):
    model_config = {"extra": "forbid"} 
    think: str
    code: Optional[str] = None

class ResponseData(BaseModel):
    model_config = {"extra": "forbid"}
    step: List[Step]

class Root(BaseModel):
    model_config = {"extra": "forbid"}
    response: ResponseData

def answer_preprocess(code_str: str) -> str:
    uppper_code_str = code_str.upper()
    if 'A.' in uppper_code_str or '"A"' in uppper_code_str or "'A'" in uppper_code_str:
        return 'agent.answer("A.")'
    elif 'B.' in uppper_code_str or '"B"' in uppper_code_str or "'B'" in uppper_code_str:
        return 'agent.answer("B.")'
    elif 'C.' in uppper_code_str or '"C"' in uppper_code_str or "'C'" in uppper_code_str:
        return 'agent.answer("C.")'
    elif 'D.' in uppper_code_str or '"D"' in uppper_code_str or "'D'" in uppper_code_str:
        return 'agent.answer("D.")'
    return code_str


def parse_model_rollout(xml_str: str) -> tuple[Union[List[Dict], None], int]:
    """
    解析模型输出的 XML。
    策略：Masking -> Strict Parse -> Unmasking -> Check Logic
    
    Returns:
        steps: List[Dict] 或 None
        valid: 1 (Strict Valid) 或 0 (Invalid/Fallback)
    """
    
    # 用于暂存提取出来的原始内容
    think_storage = []
    code_storage = []

    def mask_content(text, tag, storage_list):
        """
        将指定标签内的内容替换为占位符，并将原始内容存入 list
        """
        pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
        
        def replacer(match):
            content = match.group(1)
            # 存入列表
            index = len(storage_list)
            storage_list.append(content)
            # 返回合法的 XML 占位符
            return f"<{tag}>__{tag.upper()}_PLACEHOLDER_{index}__</{tag}>"
        
        return pattern.sub(replacer, text)

    try:
        masked_xml = mask_content(xml_str, "code", code_storage)
        masked_xml = mask_content(masked_xml, "think", think_storage)
        
        raw_dict = xmltodict.parse(masked_xml, force_list={'step'})
        validated_data = Root(**raw_dict)

        response_obj = validated_data.model_dump()['response']
        steps = []
        
        for step in response_obj['step']:
            # --- 还原 Think ---
            think_raw = step['think']
            if "__THINK_PLACEHOLDER_" in think_raw:
                # 提取 index
                idx_match = re.search(r"__THINK_PLACEHOLDER_(\d+)__", think_raw)
                if idx_match:
                    idx = int(idx_match.group(1))
                    if 0 <= idx < len(think_storage):
                        think_raw = think_storage[idx]
            
            # --- 还原 Code ---
            code_raw = step['code']
            if "__CODE_PLACEHOLDER_" in code_raw:
                idx_match = re.search(r"__CODE_PLACEHOLDER_(\d+)__", code_raw)
                if idx_match:
                    idx = int(idx_match.group(1))
                    if 0 <= idx < len(code_storage):
                        code_raw = code_storage[idx]

            steps.append({
                "think": think_raw,
                "code": code_raw
            })
            
        # 4. 业务逻辑校验
        if not steps:
            return None, 0
            
        if 'agent' not in steps[-1]['code']:
            return None, 0 
            
        return steps, 1

    except (Exception, ValidationError) as e:
        # print(f"Strict parsing failed: {e}") # Debug 用
        pass

    try:
        code_pattern = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
        code_contents = code_pattern.findall(xml_str)
        
        if not code_contents:
            return None, 0

        fallback_result = []
        for content in code_contents:
            content = content.strip()
            if content == "":
                continue
            fallback_result.append({
                "think": "", # Fallback 模式下放弃提取 think
                "code": content
            })
            
        if not fallback_result:
            return None, 0
            
        if 'agent' not in fallback_result[-1]['code']:
            return None, 0
            
        return fallback_result, 0

    except Exception:
        return None, 0

def mind_projection(actions: list[str]):
    """
    处理 actions 的主函数。
    """
    actions = copy.deepcopy(actions)
    valids = [0] * len(actions)

    for i in range(len(actions)):
        try:
            xml_str = actions[i]
            
            parsed_steps, valid = parse_model_rollout(xml_str)
            
            if not parsed_steps:
                actions[i] = [{"think": "", "code": "", "invalid_response": xml_str}]
                valids[i] = 0 # 确保是 0
            else:
                if 'answer' in parsed_steps[-1]['code']:
                    parsed_steps[-1]['code'] = answer_preprocess(parsed_steps[-1]['code'])

                actions[i] = parsed_steps
                valids[i] = valid

        except Exception as e:
            print(f"Critical Error: {e}")
            actions[i] = [{"think": f"Error: {str(e)}", "code": ""}]
            valids[i] = 0

    return actions, valids
