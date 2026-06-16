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

from typing import Any
from functools import partial
import os
import re
from agent_system.environments.prompts import MIND_TEMPLATE, COGMAP_CONSTRUCTION_TEMPLATE, NO_RETRIEVE_TEMPLATE
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
from omegaconf import OmegaConf
from PIL import Image
from rich.console import Console
import json
console = Console()

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class MindEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.retrieve_strategy = config.data.retrieve_strategy
        self.max_retrieve = config.data.max_retrieve
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        obs = self.envs.reset(index=kwargs['index'])
        self.memory.reset(batch_size = len(obs))
        self.pre_text_obs = obs

        full_text_obs, full_img_obs = self.build_obs(obs, init=True)

        # console.log(full_text_obs[0], full_img_obs[0], obs[0], [i['id'] for i in obs][0])
        return {'text': full_text_obs, 'image': full_img_obs, 'anchor': obs, 'id': [i['id'] for i in obs]}, {}

    def step(self, text_actions: list[str]):
        actions, valid = self.projection_f(text_actions)
        obs, rewards, dones, infos = self.envs.step(actions)

        # self.memory.store({'obs': obs, 'action': actions})
        self.pre_text_obs = obs
        full_text_obs, full_img_obs = self.build_obs(obs)

        # add action_valid to infos
        for i, info in enumerate[Any](infos):
            info['is_action_valid'] = to_numpy(valid[i])

        next_observations = {'text': full_text_obs, 'image': full_img_obs, 'anchor': obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        # console.log({k: v[0] for k, v in next_observations.items()}, rewards[0], dones[0], infos[0])
        return next_observations, rewards, dones, infos

    def build_obs(self, obs: list, init: bool = False) -> list[str]:
        text_obs = []
        img_obs = []
        for i in range(len(obs)):
            current_image_obs = []
            o = obs[i]
            if init:
                text_obs.append(
                    MIND_TEMPLATE.replace("<question>", o["question"])
                    .replace("<free_form_cogmap>", o["free_form_cogmap"])
                    .replace("<valid_view_desc>", f"Because you have not retrieved any view information yet, you can retrieve the information from the following views :{list(range(1, o['total_views'] + 1))}.")
                )
                current_image_obs.append(o['image'])
            else:
                if o.retrieved_image_path is None:
                    if self.retrieve_strategy == "greedy":
                        max_retrieve = o.total_views
                    else:
                        max_retrieve = self.max_retrieve
                    if o.cnt['cnt_retrieve'] >= max_retrieve:
                        text_obs.append(NO_RETRIEVE_TEMPLATE.replace('<question>', o.question).replace('<free_form_cogmap>', o.free_form_cogmap))
                    else:
                        known_views = []
                        for view in o.cogmap_dict["views"]:
                            try:
                                known_views.append(int(view['name'].split(" ")[-1].strip()))
                            except Exception:
                                # use regex to find the view index
                                view_idx_group = re.search(r'(\d+)', view['name'])
                                if view_idx_group is not None:
                                    known_views.append(int(view_idx_group.group(1)))
                                else:
                                    continue
                        all_views = list(range(1, o.total_views + 1))
                        remaining_views = [view for view in all_views if view not in known_views]
                        text_obs.append(
                            MIND_TEMPLATE.replace("<question>", o.question)
                            .replace("<free_form_cogmap>", o.free_form_cogmap)
                            .replace("<valid_view_desc>", f"Because you have already known the information for views {known_views}, you should not retrieve the information for these views. You can only retrieve the information for the remaining views: {remaining_views}.")
                        )
                    current_image_obs.append(o.image)
                else:
                    view_idx = list[Any](o.retrieved_image_path.keys())[0]
                    if not os.path.exists(o.retrieved_image_path[int(view_idx)]):
                        console.log(f"Retrieved image path {o.retrieved_image_path[int(view_idx)]} does not exist", style="red")
                    if o.total_views == 2:
                        view_relation = 'The relationship between the first and the second views is unknown.'
                    elif o.total_views == 3:
                        question = o.question
                        if '(front, left, and right)' in question:
                            view_relation = 'The three views (view 1, 2, and 3) capture the scene from different viewpoints (front, left, and right)'
                        elif '(back, left, and right)' in question:
                            view_relation = 'The three views (view 1, 2, and 3) capture the scene from different viewpoints (back, left, and right)'
                        elif '(clockwise)' in question:
                            view_relation = """These three images (image 1, 2, and 3) show the same scene from three different viewpoints. The image 2 was taken after turning the camera 90 degrees to the right (clockwise) from the position of image 1. For image 3, the camera was turned another 90 degrees right, so it's basically facing the opposite direction of image 1."""  # noqa: E501
                        else:
                            raise ValueError(f"Unsupported question: {question}")

                    elif o.total_views == 4:
                        view_relation = 'The four views (view 1, 2, 3, and 4) capture the scene from different viewpoints (front, left, back, and right), with each camera aligned with room walls and partially capturing the surroundings'
                    else:
                        raise ValueError(f"Unsupported total views: {o.total_views}")

                    text_obs.append(COGMAP_CONSTRUCTION_TEMPLATE.replace('<view_index>', str(view_idx)).replace('<free_form_cogmap>', json.dumps(o.cogmap_dict, indent=2)).replace('<view_relation>', view_relation).replace('<object_categories>', ', '.join(list(map(lambda s: f'"{s}"', o.gt_objects)))))
                    # current_image_obs.append(o.image)
                    current_image_obs.append(Image.open(o.retrieved_image_path[int(view_idx)]).convert('RGB'))

            img_obs.append(current_image_obs)

        return text_obs, img_obs


def make_envs(config):
    """
    Create enviroments
    """
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "wondermind" in config.env.env_name.lower():
        from agent_system.environments.env_package.mind import build_mind_envs, mind_projection
        _envs = build_mind_envs(
            dataset_path=config.data.train_jsonl_path,
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            data_size=config.data.train_data_size,
            group_n=group_n,
            start_server_id=0,
            resources_per_worker=resources_per_worker,
            split='train',
            use_sac=config.data.use_sac,
            use_cogmap_reward=config.data.use_cogmap_reward,
            use_retrieve_reward=config.data.use_retrieve_reward,
            retrieve_strategy=config.data.retrieve_strategy
        )
        _val_envs = build_mind_envs(
            dataset_path=config.data.val_jsonl_path,
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            data_size=config.data.val_data_size,
            group_n=1,
            start_server_id=config.data.train_batch_size*group_n,
            resources_per_worker=resources_per_worker,
            split='test',
            use_sac=config.data.use_sac,
            use_cogmap_reward=config.data.use_cogmap_reward,
            use_retrieve_reward=config.data.use_retrieve_reward,
            retrieve_strategy=config.data.retrieve_strategy
        )
        projection_f = partial(mind_projection)
        envs = MindEnvironmentManager(_envs, projection_f, config)
        val_envs = MindEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
