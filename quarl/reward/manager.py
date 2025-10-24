# Copyright 2024 Quark RL Team
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

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional

import torch
from joblib import Parallel, delayed
from ray.util.joblib import register_ray
from verl import DataProto

from quarl.interface import RewardFuncInfo

register_ray()


class QuarkRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        reward_fns: Dict[str, RewardFuncInfo],
        reward_fn_key="data_source",
        save_records=True,
        save_path=None,
    ) -> None:
        self.tokenizer = tokenizer
        # the number of batches of decoded responses to print to the console
        self.num_examine = num_examine
        self.reward_fns = reward_fns
        self.reward_fn_key = reward_fn_key
        self.save_records = save_records

        
        if save_records:
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.base_save_dir = f"quark_reward_records_{timestamp}"
            else:
                self.base_save_dir = save_path

            
            os.makedirs(self.base_save_dir, exist_ok=True)

            self.sampling_counters = {}

    def process_data(self, data_item, index):
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum().long()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum().long()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch[self.reward_fn_key]

        # decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        label = extra_info.get("label", "unknown")

        return {
            "idx": index,
            "prompt_str": prompt_str,
            "data_source": data_source,
            "response_str": response_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "valid_response_length": valid_response_length,
            "label": label,
        }

    def __call__(self, data: DataProto, return_dict=False):
        batch_size = len(data)
        rule_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        rule_reward_tensor_multiply = torch.ones_like(data.batch["responses"], dtype=torch.float32)

        reward_extra_info = {}

        all_data = [self.process_data(data_item, i) for i, data_item in enumerate(data)]  # list of data dict

        reward_fn_data_map = self.dispatch_data(all_data)
        if len(reward_fn_data_map) != 0:
            
            all_fn_results = Parallel(backend="ray", n_jobs=len(reward_fn_data_map), verbose=1)(
                delayed(self.wrap_compute_score_task)(reward_fn_name, batch_data)
                for reward_fn_name, batch_data in reward_fn_data_map.items()
            )

            for fn_result in all_fn_results:
                reward_fn_name = fn_result["reward_fn_name"]
                score_results = fn_result["results"]
                batch_data = reward_fn_data_map[reward_fn_name]
                reward_fn_info = self.reward_fns[reward_fn_name]

                if not len(score_results) > 0:
                    continue

                for i, score in enumerate(score_results):
                    idx = score["idx"]
                    valid_response_length = batch_data[i]["valid_response_length"]
                    reward = score["score"] 

                
                    if self.save_records:
                        data_item = batch_data[i]

                    
                        split_name = (
                            data_item["extra_info"].get("split", "unknown") if data_item["extra_info"] else "unknown"
                        )

                    
                        if "train" in split_name.lower():
                            filename = "train.jsonl"
                        elif "test" in split_name.lower() or "eval" in split_name.lower():
                            filename = "test.jsonl"
                        else:
                            filename = "unknown.jsonl"

                        file_path = os.path.join(self.base_save_dir, filename)

                        
                        prompt_hash = hashlib.md5(data_item["prompt_str"].encode("utf-8")).hexdigest()[:8]
                        sampling_key = f"{data_item['label']}_{prompt_hash}"

                        if sampling_key not in self.sampling_counters:
                            self.sampling_counters[sampling_key] = 0
                        self.sampling_counters[sampling_key] += 1

                        sampling_id = self.sampling_counters[sampling_key]

                        record = {
                            "index": idx,
                            "prompt": data_item["prompt_str"],
                            "response": data_item["response_str"],
                            "ground_truth": data_item["ground_truth"],
                            "reward_fn_name": reward_fn_name,
                            "split": split_name,
                            "extra_info": data_item["extra_info"],
                            "score": score,
                            "valid_response_length": data_item["valid_response_length"],
                            "sampling_id": sampling_id,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }

                        with open(file_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    for key, value in score.items():
                        if key == "idx":
                            continue
                        new_key = f"{reward_fn_name}_{key}"
                        if new_key not in reward_extra_info.keys():
                            reward_extra_info[new_key] = [
                                None for _ in range(batch_size)
                            ]  # TODO: what default value should we set here?
                        reward_extra_info[new_key][idx] = value

                    # print(f"[quarl.worker.reward_manager] rule {reward_fn_name} returned score {reward}")

                    if reward_fn_info.integration == "sum":
                    
                        rule_reward_tensor[idx, valid_response_length - 1] += reward
                    elif reward_fn_info.integration == "multiply":
                        rule_reward_tensor_multiply[idx, valid_response_length - 1] *= reward
                    else:
                        raise NotImplementedError

        if "rm_scores" in data.batch.keys():
            rm_scores = data.batch["rm_scores"]
            assert rm_scores.shape == rule_reward_tensor.shape

            reward_tensor = (rm_scores + rule_reward_tensor) * rule_reward_tensor_multiply
        else:
            reward_tensor = rule_reward_tensor * rule_reward_tensor_multiply

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def dispatch_data(self, data_dict_list):
        reward_fn_data_map = {}
        for item in data_dict_list:
            for reward_fn_name, reward_fn_info in self.reward_fns.items():
                if item["label"] in reward_fn_info.labels:
                    if reward_fn_name not in reward_fn_data_map:
                        reward_fn_data_map[reward_fn_name] = [item]
                    else:
                        reward_fn_data_map[reward_fn_name].append(item)
        return reward_fn_data_map

    def wrap_compute_score_task(self, reward_fn_name, batch_data):
        print(f"[quarl.reward.manager] wrap_compute_score_task: {reward_fn_name} data num : {len(batch_data)}")
        results = self.reward_fns[reward_fn_name].reward_fn(batch_data)
        print(f"[quarl.reward.manager] wrap_compute_score_task: {reward_fn_name} finished")
        return {"reward_fn_name": reward_fn_name, "results": results}


class NaiveRewardManagerComputeWithPrompt:
    """The reward manager."""

    def __init__(
        self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", save_records=True, save_path=None
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.save_records = save_records

        if save_records:
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.base_save_dir = f"reward_records_{timestamp}"
            else:
                self.base_save_dir = save_path

            os.makedirs(self.base_save_dir, exist_ok=True)

            self.sampling_counters = {}

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum().long()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum().long()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                prompt_str=prompt_str,  # with prompt for context
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            
            if self.save_records:
            
                split_name = extra_info.get("split", "unknown") if extra_info else "unknown"

            
                if "train" in split_name.lower():
                    filename = "train.jsonl"
                elif "test" in split_name.lower() or "eval" in split_name.lower():
                    filename = "test.jsonl"
                else:
                    filename = "unknown.jsonl"

                file_path = os.path.join(self.base_save_dir, filename)

                
                prompt_hash = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()[:8]
                sampling_key = f"{data_source}_{prompt_hash}"

                
                if sampling_key not in self.sampling_counters:
                    self.sampling_counters[sampling_key] = 0
                self.sampling_counters[sampling_key] += 1

                sampling_id = self.sampling_counters[sampling_key]

                record = {
                    "index": i,
                    "prompt": prompt_str,
                    "response": response_str,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "split": split_name,
                    "extra_info": extra_info,
                    "score": score if isinstance(score, dict) else {"score": score},
                    "valid_response_length": int(valid_response_length),
                    "sampling_id": sampling_id,  
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
