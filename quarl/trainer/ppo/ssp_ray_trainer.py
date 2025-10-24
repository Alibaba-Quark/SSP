"""
Self-Play Ray PPO Trainer for QUARL.

This trainer implements the self-play algorithm where a model acts as both
problem generator and problem solver in alternating phases within each global step.

Self-play flow:
1. Problem Generation Phase: Model generates problems from seeded data
2. Problem Extraction: Extract questions from generation trajectories
3. Problem Solving Phase: Model solves the extracted problems
4. Dual Training: Update model twice with rewards from both phases
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    process_validation_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tracking import Tracking

from quarl.utils.ssp_data_manager import SelfPlayPhase, SSPDataManager
from quarl.utils.problem_extraction import ProblemExtractor, extract_problems_batch

logger = logging.getLogger(__name__)

# Enable debug logging for self-play
SELF_PLAY_DEBUG = os.environ.get("SELF_PLAY_DEBUG", "False").lower() == "true"
if SELF_PLAY_DEBUG:
    logger.setLevel(logging.DEBUG)


class SSPRayPPOTrainer(RayPPOTrainer):
    """
    Self-Play Ray PPO Trainer that implements the dual-phase training process.

    This trainer orchestrates the self-play loop:
    - Phase 1: Problem generation and extraction
    - Phase 2: Problem solving
    - Dual model updates with separate reward computations
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls=None,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
    ):
        self.sp_config = config.self_play

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            reward_fn,
            val_reward_fn,
            train_dataset,
            val_dataset,
            collate_fn,
            train_sampler,
            device_name,
        )

        self.current_phase = SelfPlayPhase.PROBLEM_GENERATION

        self.sp_data_manager = SSPDataManager.remote()
        self.problem_extractor = ProblemExtractor(
            lang=self.config.self_play.lang,
            use_rag_filter=self.sp_config.use_rag_filter,
            use_search_terms_filter=self.sp_config.use_search_terms_filter,
            noisy_rag_materials=self.sp_config.get("noisy_RAG_materials", 0),
            answer_pattern=self.sp_config.get("answer_pattern", "answer"),
        )

        self._load_problem_generation_data()

    def _validate_config(self):
        val = self.sp_config.get("validate_config", True)
        if isinstance(val, str):
            val = val.lower() in ("1", "true", "yes", "y")
        if val:
            super()._validate_config()
        else:
            print("Skipping config validation as per self_play.validate_config=False")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, extra_fields=None):
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        if extra_fields:
            for k, v in extra_fields.items():
                if len(v) == n:
                    base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _validate(self):
        data_source_lst = []
        reward_model_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in tqdm(self.val_dataloader, desc="Validation Progress"):
            test_batch = DataProto.from_single_dict(test_data)

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]

            if input_ids.dtype != torch.long:
                print(f"WARNING: Converting input_ids from {input_ids.dtype} to torch.long")
                input_ids = input_ids.long()

            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            output_ids = test_output_gen_batch.batch["responses"]

            print(f"DEBUG: Validation batch #{len(data_source_lst) + 1}")
            print(f"DEBUG: output_ids type: {type(output_ids)}")
            print(f"DEBUG: output_ids dtype: {output_ids.dtype}")
            print(f"DEBUG: output_ids shape: {output_ids.shape}")
            print(f"DEBUG: output_ids device: {output_ids.device}")
            print(f"DEBUG: test_gen_batch meta_info: {test_gen_batch.meta_info}")

            if output_ids.dtype != torch.long:
                print(f"ERROR: Found float32 output_ids!")
                print(f"DEBUG: output_ids min/max: {output_ids.min().item()} / {output_ids.max().item()}")
                print(f"DEBUG: Contains non-integer values: {not torch.all(output_ids == output_ids.long())}")
                print(f"DEBUG: First few values: {output_ids[0, :100]}")

                is_integer_valued = torch.all(output_ids == output_ids.round())
                print(f"DEBUG: All values are integer-valued: {is_integer_valued}")

                print(f"WARNING: Converting output_ids from {output_ids.dtype} to torch.long")
                if is_integer_valued:
                    output_ids = output_ids.long()
                else:
                    print("ERROR: Cannot safely convert non-integer floats to long!")
                    output_ids = output_ids.round().long()

            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            for key in test_batch.batch.keys():
                if key not in ["old_log_probs", "ref_log_prob"]:
                    test_batch.batch[key] = test_batch.batch[key].long()

            test_batch.meta_info["validate"] = True

            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            reward_model_lst.append(
                test_batch.non_tensor_batch.get("reward_model", ["unknown"] * reward_tensor.shape[0])
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            extra_fields = {}
            data_sources = np.concatenate(data_source_lst, axis=0) if data_source_lst else None
            reward_models = np.concatenate(reward_model_lst, axis=0) if reward_model_lst else None

            if data_sources is not None:
                extra_fields["data_source"] = data_sources
            if reward_models is not None:
                extra_fields["reward_model"] = reward_models
            extra_fields = extra_fields if extra_fields else None

            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                extra_fields=extra_fields,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _load_problem_generation_data(self):
        try:
            generation_data_path = self.sp_config.get("generation_data_path")
            if generation_data_path and os.path.exists(generation_data_path):
                with open(generation_data_path, "r") as f:
                    if generation_data_path.endswith(".jsonl"):
                        generation_data = [json.loads(line) for line in f]
                    else:
                        generation_data = json.load(f)

                ray.get(self.sp_data_manager.set_problem_generation_data.remote(generation_data))
                logger.info(f"Loaded {len(generation_data)} problem generation samples")
            else:
                logger.warning("No generation data path specified or file not found")
                dummy_data = self._create_dummy_generation_data()
                ray.get(self.sp_data_manager.set_problem_generation_data.remote(dummy_data))

        except Exception as e:
            logger.error(f"Error loading problem generation data: {e}")
            raise

    def _create_dummy_generation_data(self) -> List[Dict[str, Any]]:
        dummy_data = [
            {
                "prompt": [{"role": "user", "content": "What are the main causes of the Great Depression, and how did government policies influence its course?"}],
                "data_source": "dummy_generation",
                "reward_model": {"style": "rule"},
                "extra_info": {"type": "qa_search"},
            },
            {
                "prompt": [{"role": "user", "content": "How does CRISPR gene editing work, and what are its potential applications in medicine?"}],
                "data_source": "dummy_generation",
                "reward_model": {"style": "rule"},
                "extra_info": {"type": "qa_search"},
            },
        ]
        return dummy_data

    def fit(self):
        from verl.trainer.ppo.metric_utils import (
            compute_throughout_metrics,
            compute_timing_metrics,
        )

        from quarl.utils.patch.metric_patch import quarl_compute_data_metrics

        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name + "_selfplay",
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                tracking_logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Self-Play Training")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        reward_dynamic_sampling_enabled = self.sp_config.get("reward_dynamic_sampling", {}).get("enable", False)

        if reward_dynamic_sampling_enabled:
            accumulated_proposer_batch = None
            accumulated_solver_batch = None
            last_step_proposer_batch = None
            last_step_solver_batch = None
            num_prompt_in_batch = 0
            num_gen_batches = 0
            target_prompt_batch_size = self.config.data.train_batch_size
            max_num_gen_batches = self.sp_config.get("reward_dynamic_sampling", {}).get("max_num_gen_batches", 20)

        for epoch in range(self.config.trainer.total_epochs):
            dataloader_iter = iter(self.train_dataloader)
            print("Epoch:", epoch)
            while True:
                try:
                    timing_raw = {}
                    all_metrics = {}

                    is_last_step = self.global_steps >= self.total_training_steps

                    with marked_timer("self_play_step", timing_raw):

                        step_metrics = {}

                        with marked_timer("proposer_generation", timing_raw):
                            proposer_gen_batch, solver_batch, sampling_metrics = (
                                self._generate_proposer_with_dynamic_sampling(dataloader_iter, timing_raw)
                            )

                            if proposer_gen_batch is None and solver_batch is None:
                                print(
                                    "No valid problems extracted after dynamic sampling, dataloader exhausted, breaking to next epoch"
                                )
                                break

                            step_metrics["self_play/proposer_samples"] = len(proposer_gen_batch.batch)
                            step_metrics["self_play/extracted_problems"] = len(solver_batch.batch)
                            step_metrics.update(sampling_metrics)
                            step_metrics.update(solver_batch.meta_info.get("metrics", {}))
                    mini_epoch = 0
                    while mini_epoch < self.sp_config.mini_epochs:

                        mini_epoch += 1

                        if mini_epoch == 1:
                            proposer_gen_batch_copy = deepcopy(proposer_gen_batch)
                            solver_batch_copy = deepcopy(solver_batch)
                        else:
                            proposer_gen_batch = deepcopy(proposer_gen_batch_copy)
                            solver_batch = deepcopy(solver_batch_copy)

                        with marked_timer("solver_generation", timing_raw):
                            solver_gen_batch = self._generate_solver_trajectories(solver_batch, timing_raw)
                            step_metrics["self_play/solver_samples"] = len(solver_gen_batch.batch)

                        with marked_timer("reward_calculation", timing_raw):
                            proposer_rewards, solver_rewards = self._calculate_self_play_rewards(
                                proposer_gen_batch, solver_gen_batch, timing_raw
                            )

                        proposer_gen_batch.batch["token_level_scores"] = proposer_rewards
                        solver_gen_batch.batch["token_level_scores"] = solver_rewards

                        if reward_dynamic_sampling_enabled:
                            filtered_proposer_batch, filtered_solver_batch = self._apply_reward_dynamic_sampling(
                                proposer_gen_batch, solver_gen_batch, timing_raw
                            )

                            unique_prompts = set(filtered_proposer_batch.non_tensor_batch["uid"])
                            num_prompt_in_batch += len(unique_prompts)
                            num_gen_batches += 1

                            if last_step_proposer_batch is not None and accumulated_proposer_batch is None:
                                saved_unique_prompts = set(last_step_proposer_batch.non_tensor_batch["uid"])
                                num_prompt_in_batch += len(saved_unique_prompts)
                                print(f"Added {len(saved_unique_prompts)} prompts from saved data")

                            if accumulated_proposer_batch is None:
                                if last_step_proposer_batch is not None:
                                    print(
                                        f"Using {len(last_step_proposer_batch.batch)} saved proposer trajectories from last step"
                                    )
                                    accumulated_proposer_batch = DataProto.concat(
                                        [last_step_proposer_batch, filtered_proposer_batch]
                                    )
                                else:
                                    accumulated_proposer_batch = filtered_proposer_batch

                                if last_step_solver_batch is not None:
                                    print(
                                        f"Using {len(last_step_solver_batch.batch)} saved solver trajectories from last step"
                                    )
                                    accumulated_solver_batch = DataProto.concat(
                                        [last_step_solver_batch, filtered_solver_batch]
                                    )
                                else:
                                    accumulated_solver_batch = filtered_solver_batch
                            else:
                                accumulated_proposer_batch = DataProto.concat(
                                    [accumulated_proposer_batch, filtered_proposer_batch]
                                )
                                accumulated_solver_batch = DataProto.concat(
                                    [accumulated_solver_batch, filtered_solver_batch]
                                )

                            print(
                                f"Reward filtering: {len(unique_prompts)} prompts kept, total: {num_prompt_in_batch}/{target_prompt_batch_size}"
                            )

                            # Check if we have enough prompts
                            if num_prompt_in_batch < target_prompt_batch_size:
                                print(f"Need more prompts: {num_prompt_in_batch} < {target_prompt_batch_size}")
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f"Batch {num_gen_batches}/{max_num_gen_batches}. Continue generating...")
                                    progress_bar.update(1)
                                    self.global_steps += 1
                                    continue
                                else:
                                    print(
                                        f"Max batches reached ({num_gen_batches}/{max_num_gen_batches}). Using accumulated data."
                                    )

                            # Use accumulated batches for training
                            proposer_gen_batch = accumulated_proposer_batch
                            solver_gen_batch = accumulated_solver_batch

                            # Truncate to target size if we have more than needed
                            proposer_n = self.sp_config.get("proposer", {}).get(
                                "n", self.config.actor_rollout_ref.rollout.n
                            )
                            target_traj_size = target_prompt_batch_size * proposer_n

                            if len(proposer_gen_batch.batch) > target_traj_size:
                                print(
                                    f"Truncating proposer batch from {len(proposer_gen_batch.batch)} to {target_traj_size}"
                                )
                                last_step_proposer_batch = proposer_gen_batch[target_traj_size:]
                                proposer_gen_batch = proposer_gen_batch[:target_traj_size]
                                print(
                                    f"Saved {len(last_step_proposer_batch.batch)} proposer trajectories for next round"
                                )
                            else:
                                last_step_proposer_batch = None

                            solver_n = self.config.actor_rollout_ref.rollout.n
                            target_solver_size = target_prompt_batch_size * proposer_n * solver_n

                            if len(solver_gen_batch.batch) > target_solver_size:
                                print(
                                    f"Truncating solver batch from {len(solver_gen_batch.batch)} to {target_solver_size}"
                                )
                                last_step_solver_batch = solver_gen_batch[target_solver_size:]
                                solver_gen_batch = solver_gen_batch[:target_solver_size]
                                print(f"Saved {len(last_step_solver_batch.batch)} solver trajectories for next round")
                            else:
                                last_step_solver_batch = None

                            print(
                                f"Final batch sizes: proposer={len(proposer_gen_batch.batch)}, solver={len(solver_gen_batch.batch)}"
                            )

                            step_metrics["self_play/reward_dynamic_sampling/batches_used"] = num_gen_batches
                            step_metrics["self_play/reward_dynamic_sampling/final_prompt_count"] = num_prompt_in_batch
                            step_metrics["self_play/reward_dynamic_sampling/target_prompt_count"] = (
                                target_prompt_batch_size
                            )

                            step_metrics["self_play/reward_dynamic_sampling/final_proposer_trajectories"] = len(
                                proposer_gen_batch.batch
                            )
                            step_metrics["self_play/reward_dynamic_sampling/final_solver_trajectories"] = len(
                                solver_gen_batch.batch
                            )

                            if last_step_proposer_batch is not None:
                                step_metrics["self_play/reward_dynamic_sampling/saved_proposer_trajectories"] = len(
                                    last_step_proposer_batch.batch
                                )
                            if last_step_solver_batch is not None:
                                step_metrics["self_play/reward_dynamic_sampling/saved_solver_trajectories"] = len(
                                    last_step_solver_batch.batch
                                )

                        if self.sp_config.save_freq > 0 and (
                            self.global_steps % self.sp_config.save_freq == 0
                            or self.sp_config.reward_dynamic_sampling.enable
                        ):
                            try:
                                inputs = self.tokenizer.batch_decode(
                                    proposer_gen_batch.batch["prompts"], skip_special_tokens=True
                                )
                                outputs = self.tokenizer.batch_decode(
                                    proposer_gen_batch.batch["responses"], skip_special_tokens=True
                                )

                                reward_extra_infos_dict = {}
                                if (
                                    hasattr(proposer_gen_batch, "non_tensor_batch")
                                    and proposer_gen_batch.non_tensor_batch
                                ):
                                    for key in ["data_source", "reward_model", "extra_info", "uid"]:
                                        if key in proposer_gen_batch.non_tensor_batch:
                                            reward_extra_infos_dict[key] = proposer_gen_batch.non_tensor_batch[
                                                key
                                            ].tolist()

                                self._dump_trajectories(
                                    inputs=inputs,
                                    outputs=outputs,
                                    scores=proposer_gen_batch.batch["token_level_scores"].sum(-1).cpu().tolist(),
                                    role="proposer",
                                    step=self.global_steps,
                                    reward_extra_infos_dict=(
                                        reward_extra_infos_dict if reward_extra_infos_dict else None
                                    ),
                                )
                            except Exception as e:
                                logger.warning(f"Failed to dump proposer trajectories: {e}")

                            try:
                                inputs = self.tokenizer.batch_decode(
                                    solver_gen_batch.batch["prompts"], skip_special_tokens=True
                                )
                                outputs = self.tokenizer.batch_decode(
                                    solver_gen_batch.batch["responses"], skip_special_tokens=True
                                )

                                reward_extra_infos_dict = {}
                                if hasattr(solver_gen_batch, "non_tensor_batch") and solver_gen_batch.non_tensor_batch:
                                    for key in ["data_source", "reward_model", "extra_info", "uid"]:
                                        if key in solver_gen_batch.non_tensor_batch:
                                            reward_extra_infos_dict[key] = solver_gen_batch.non_tensor_batch[
                                                key
                                            ].tolist()

                                self._dump_trajectories(
                                    inputs=inputs,
                                    outputs=outputs,
                                    scores=solver_gen_batch.batch["token_level_scores"].sum(-1).cpu().tolist(),
                                    role="solver",
                                    step=self.global_steps,
                                    reward_extra_infos_dict=(
                                        reward_extra_infos_dict if reward_extra_infos_dict else None
                                    ),
                                )
                            except Exception as e:
                                logger.warning(f"Failed to dump proposer trajectories: {e}")

                        with marked_timer("model_updates", timing_raw):

                            representative_batch = solver_gen_batch

                        combine_updates = getattr(self.sp_config, "combine_update", False)

                        if combine_updates:
                            batches_to_combine = []
                            if self.sp_config.proposer.enable and proposer_gen_batch is not None:
                                batches_to_combine.append(proposer_gen_batch)
                            if (
                                self.sp_config.solver.enable
                                and solver_gen_batch is not None
                                and (
                                    self.global_steps > self.sp_config.proposer.warm_up_steps
                                    or not self.sp_config.proposer.enable
                                )
                            ):
                                batches_to_combine.append(solver_gen_batch)

                            if len(batches_to_combine) > 1:
                                for batch in batches_to_combine:
                                    for key in [
                                        "extracted_question",
                                        "formatted_prompt",
                                        "problem_type",
                                        "trajectory_index",
                                        "success_rate",
                                        "generation_step",
                                    ]:
                                        if key in batch.non_tensor_batch:
                                            del batch.non_tensor_batch[key]

                                combined_batch = DataProto.concat(batches_to_combine)

                                combined_metrics = self._update_on_trajectories(combined_batch, "combined", timing_raw)
                                step_metrics.update({f"combined/{k}": v for k, v in combined_metrics.items()})
                                representative_batch = combined_batch
                            else:
                                if self.sp_config.proposer.enable and proposer_gen_batch is not None:
                                    proposer_metrics = self._update_on_trajectories(
                                        proposer_gen_batch, "proposer", timing_raw
                                    )
                                    step_metrics.update({f"proposer/{k}": v for k, v in proposer_metrics.items()})
                                    representative_batch = proposer_gen_batch

                                if (
                                    self.sp_config.solver.enable
                                    and solver_gen_batch is not None
                                    and (
                                        self.global_steps > self.sp_config.proposer.warm_up_steps
                                        or not self.sp_config.proposer.enable
                                    )
                                ):
                                    solver_metrics = self._update_on_trajectories(
                                        solver_gen_batch, "solver", timing_raw
                                    )
                                    step_metrics.update({f"solver/{k}": v for k, v in solver_metrics.items()})
                                    representative_batch = solver_gen_batch

                        else:
                            if self.sp_config.proposer.enable:
                                proposer_metrics = self._update_on_trajectories(
                                    proposer_gen_batch, "proposer", timing_raw
                                )
                                step_metrics.update({f"proposer/{k}": v for k, v in proposer_metrics.items()})
                                representative_batch = proposer_gen_batch

                            if self.sp_config.solver.enable and (
                                self.global_steps > self.sp_config.proposer.warm_up_steps
                                or not self.sp_config.proposer.enable
                            ):
                                solver_metrics = self._update_on_trajectories(solver_gen_batch, "solver", timing_raw)
                                step_metrics.update({f"solver/{k}": v for k, v in solver_metrics.items()})
                                representative_batch = solver_gen_batch

                    print(f"Representative batch size: {len(representative_batch.batch)}")
                    all_metrics.update(step_metrics)

                    ray.get(self.sp_data_manager.update_global_step.remote(self.global_steps))

                    all_metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )

                    sp_stats = ray.get(self.sp_data_manager.get_statistics.remote())
                    for key, value in sp_stats.items():
                        if isinstance(value, (int, float)):
                            all_metrics[f"self_play/stats/{key}"] = value

                    if hasattr(solver_gen_batch, "meta_info") and solver_gen_batch.meta_info:
                        solver_metrics = solver_gen_batch.meta_info.get("metrics", {})
                        for key, value in solver_metrics.items():
                            if key.startswith("dynamic_sampling/") and isinstance(value, (int, float, bool)):
                                all_metrics[f"self_play/{key}"] = value
                            if key.startswith("reward_dynamic_sampling/") and isinstance(value, (int, float, bool)):
                                all_metrics[f"self_play/{key}"] = value

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                            or self.sp_config.reward_dynamic_sampling.enable
                        )
                    ):
                        with marked_timer("testing", timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        all_metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                        or self.sp_config.reward_dynamic_sampling.enable
                    ):

                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                            

                    steps_duration = timing_raw["self_play_step"]
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                    print(f"[sp_ray_trainer] compute data metrics")
                    all_metrics.update(
                        quarl_compute_data_metrics(
                            tokenizer=self.tokenizer, batch=representative_batch, use_critic=self.use_critic
                        )
                    )

                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    adapted_timing_raw = timing_raw.copy()

                    if "self_play_step" in adapted_timing_raw and "step" not in adapted_timing_raw:
                        adapted_timing_raw["step"] = adapted_timing_raw["self_play_step"]

                    timer_mapping = {
                        "solver_generation": "gen",
                        "solver_ref": "ref",
                        "solver_values": "values",
                        "solver_adv": "adv",
                        "solver_update_critic": "update_critic",
                        "solver_update_actor": "update_actor",
                    }

                    for our_name, expected_name in timer_mapping.items():
                        if our_name in adapted_timing_raw and expected_name not in adapted_timing_raw:
                            adapted_timing_raw[expected_name] = adapted_timing_raw[our_name]

                    all_metrics.update(
                        compute_timing_metrics(batch=representative_batch, timing_raw=adapted_timing_raw)
                    )
                    all_metrics.update(
                        compute_throughout_metrics(
                            batch=representative_batch, timing_raw=adapted_timing_raw, n_gpus=n_gpus
                        )
                    )

                    # Log metrics
                    tracking_logger.log(data=all_metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1

                    if self.sp_config.extraction_failure.strategy == "reuse":
                        if self.global_steps % self.sp_config.extraction_failure.pool_clear_interval == 0:

                            ray.get(
                                self.sp_data_manager.clear_solving_pool.remote(
                                    keep_ratio=self.sp_config.extraction_failure.keep_ratio
                                )
                            )

                            print(f"Cleared solving pool")

                            has_existing = ray.get(self.sp_data_manager.has_existing_problems.remote())
                            print(f"Has existing problems: {has_existing}")

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return

                    if reward_dynamic_sampling_enabled:
                        accumulated_proposer_batch = None
                        accumulated_solver_batch = None
                        num_prompt_in_batch = 0
                        num_gen_batches = 0

                except StopIteration:
                    break

    def _generate_proposer_with_dynamic_sampling(
        self, dataloader_iter, timing_raw: Dict
    ) -> Tuple[DataProto, Optional[DataProto], Dict[str, Any]]:
        dynamic_sampling_config = self.sp_config.get("dynamic_sampling", {})
        enable_dynamic_sampling = dynamic_sampling_config.get("enable", False)

        try:
            batch_dict = next(dataloader_iter)
        except StopIteration:
            return None, None, {}

        batch = DataProto.from_single_dict(batch_dict)

        if not enable_dynamic_sampling:
            proposer_gen_batch = self._generate_proposer_trajectories(batch, timing_raw)
            solver_batch = self._extract_and_assemble_solver_data(proposer_gen_batch, timing_raw)
            return proposer_gen_batch, solver_batch, {}

        max_retry_attempts = dynamic_sampling_config.get("max_retry_attempts", 3)
        min_valid_ratio = dynamic_sampling_config.get("min_valid_ratio", 1.0)
        target_batch_size = len(batch.batch)

        print(
            f"=== Dynamic Sampling Enabled: max_batches={max_retry_attempts + 1}, min_valid_ratio={min_valid_ratio}, target_size={target_batch_size} ==="
        )

        all_valid_problems = []
        all_proposer_batches = []
        sampling_metrics = {
            "dynamic_sampling/batches_used": 0,
            "dynamic_sampling/total_trajectories": 0,
            "dynamic_sampling/total_valid_problems": 0,
            "dynamic_sampling/final_valid_count": 0,
        }

        current_batch = batch
        batch_count = 0

        while batch_count <= max_retry_attempts:
            batch_count += 1
            sampling_metrics["dynamic_sampling/batches_used"] = batch_count

            print(f"Dynamic sampling batch {batch_count}/{max_retry_attempts + 1}")

            proposer_gen_batch = self._generate_proposer_trajectories(current_batch, timing_raw)

            trajectories = self._extract_trajectories_from_batch(proposer_gen_batch)
            extracted_problems = self._extract_and_process_problems(trajectories)

            valid_problems_this_batch = len(extracted_problems)
            total_trajectories_this_batch = len(trajectories)

            sampling_metrics["dynamic_sampling/total_trajectories"] += total_trajectories_this_batch
            sampling_metrics["dynamic_sampling/total_valid_problems"] += valid_problems_this_batch

            print(f"Batch {batch_count}: {valid_problems_this_batch}/{total_trajectories_this_batch} valid problems")

            if extracted_problems:
                valid_trajectory_indices = [problem["trajectory_index"] for problem in extracted_problems]
                valid_proposer_batch = self._filter_proposer_batch_by_indices(
                    proposer_gen_batch, valid_trajectory_indices
                )
                all_proposer_batches.append(valid_proposer_batch)

            all_valid_problems.extend(extracted_problems)

            current_valid_count = len(all_valid_problems)
            required_valid_count = int(target_batch_size * min_valid_ratio)

            if current_valid_count >= required_valid_count:
                print(
                    f"✅ Dynamic sampling succeeded! Collected {current_valid_count} valid problems (required: {required_valid_count})"
                )
                break

            if batch_count <= max_retry_attempts:
                try:
                    batch_dict = next(dataloader_iter)
                    current_batch = DataProto.from_single_dict(batch_dict)
                    print(
                        f"❌ Need more valid problems: {current_valid_count}/{required_valid_count}, using next batch..."
                    )
                except StopIteration:
                    print(f"❌ No more batches available, collected {current_valid_count} valid problems")
                    break
            else:
                print(f"❌ Max batches reached, collected {current_valid_count} valid problems")

        if len(all_valid_problems) == 0 or len(all_proposer_batches) == 0:
            print("❌ No valid problems found from any batch")
            return None, None, sampling_metrics

        final_proposer_batch = self._combine_proposer_batches(all_proposer_batches)

        final_batch_size = len(all_valid_problems)
        aligned_problems = self._align_problems_with_proposer_batch(all_valid_problems, final_batch_size)

        replicated_count = 0
        if final_batch_size < target_batch_size:
            print(f"🔄 Replicating {final_batch_size} valid problems to reach target size {target_batch_size}")

            needed_count = target_batch_size - final_batch_size
            replicated_count = needed_count

            import random

            replicated_problems = []
            replicated_proposer_trajectories = []

            for i in range(needed_count):
                source_idx = random.randint(0, final_batch_size - 1)

                replicated_problem = deepcopy(aligned_problems[source_idx])
                replicated_problem["trajectory_index"] = final_batch_size + i
                replicated_problems.append(replicated_problem)

                source_proposer_data = final_proposer_batch[source_idx : source_idx + 1]
                replicated_proposer_trajectories.append(source_proposer_data)

            aligned_problems.extend(replicated_problems)

            for replicated_traj in replicated_proposer_trajectories:
                final_proposer_batch = DataProto.concat([final_proposer_batch, replicated_traj])

            print(f"🔄 Replicated {replicated_count} problems, now have {len(aligned_problems)} total")

        else:
            if final_batch_size > target_batch_size:
                print(f"✂️ Truncating {final_batch_size} problems to target size {target_batch_size}")
                aligned_problems = aligned_problems[:target_batch_size]
                final_proposer_batch = final_proposer_batch[:target_batch_size]
                print(f"✂️ Truncated to {len(aligned_problems)} problems")
        self.generate_problem = aligned_problems

        solver_batch = self._prepare_solving_batch_from_data(aligned_problems)

        sampling_metrics["dynamic_sampling/original_valid_count"] = final_batch_size
        sampling_metrics["dynamic_sampling/final_valid_count"] = len(aligned_problems)
        sampling_metrics["dynamic_sampling/replicated_count"] = replicated_count

        if solver_batch is not None:
            solver_batch.meta_info.setdefault("metrics", {}).update(sampling_metrics)

        return final_proposer_batch, solver_batch, sampling_metrics

    def _apply_reward_dynamic_sampling(
        self, proposer_gen_batch: DataProto, solver_gen_batch: DataProto, timing_raw: Dict
    ) -> Tuple[DataProto, DataProto]:
        print("=== Applying Reward-based Dynamic Sampling ===")

        reward_config = self.sp_config.get("reward_dynamic_sampling", {})
        metric_name = reward_config.get("metric", "seq_final_reward")

        if metric_name == "seq_final_reward":
            metric_values = proposer_gen_batch.batch["token_level_scores"].sum(dim=-1).numpy()
        elif metric_name == "seq_reward":
            metric_values = proposer_gen_batch.batch["token_level_scores"].sum(dim=-1).numpy()
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        proposer_gen_batch.non_tensor_batch[metric_name] = metric_values

        uid2metric_vals = defaultdict(list)
        uid2traj_indices = defaultdict(list)

        for idx, (uid, metric_val) in enumerate(
            zip(proposer_gen_batch.non_tensor_batch["uid"], proposer_gen_batch.non_tensor_batch[metric_name])
        ):
            uid2metric_vals[uid].append(metric_val)
            uid2traj_indices[uid].append(idx)

        uid2variance = {}
        for uid, metric_vals in uid2metric_vals.items():
            uid2variance[uid] = np.std(metric_vals)

        kept_uids = [uid for uid, variance in uid2variance.items() if variance > 0 or len(uid2metric_vals[uid]) == 1]

        kept_traj_indices = []
        for uid in kept_uids:
            kept_traj_indices.extend(uid2traj_indices[uid])

        kept_traj_indices.sort()

        print(
            f"Reward filtering: {len(uid2metric_vals)} groups, {len(kept_uids)} kept, {len(kept_traj_indices)}/{len(proposer_gen_batch.batch)} trajectories"
        )

        filtered_proposer_batch = proposer_gen_batch[kept_traj_indices]

        solver_n = self.config.actor_rollout_ref.rollout.n
        kept_solver_indices = []

        for proposer_idx in kept_traj_indices:
            solver_start_idx = proposer_idx * solver_n
            solver_end_idx = solver_start_idx + solver_n
            kept_solver_indices.extend(range(solver_start_idx, solver_end_idx))

        filtered_solver_batch = solver_gen_batch[kept_solver_indices]

        print(f"Solver filtering: {len(kept_solver_indices)}/{len(solver_gen_batch.batch)} trajectories kept")

        filtering_metrics = {
            "reward_dynamic_sampling/original_proposer_groups": len(uid2metric_vals),
            "reward_dynamic_sampling/kept_proposer_groups": len(kept_uids),
            "reward_dynamic_sampling/original_proposer_trajectories": len(proposer_gen_batch.batch),
            "reward_dynamic_sampling/kept_proposer_trajectories": len(filtered_proposer_batch.batch),
            "reward_dynamic_sampling/original_solver_trajectories": len(solver_gen_batch.batch),
            "reward_dynamic_sampling/kept_solver_trajectories": len(filtered_solver_batch.batch),
            "reward_dynamic_sampling/filtering_ratio": (
                len(kept_traj_indices) / len(proposer_gen_batch.batch) if len(proposer_gen_batch.batch) > 0 else 0.0
            ),
        }

        pprint(filtering_metrics)

        return filtered_proposer_batch, filtered_solver_batch

    def _combine_proposer_batches(self, proposer_batches: List[DataProto]) -> DataProto:
        if len(proposer_batches) == 1:
            return proposer_batches[0]

        combined_batch = proposer_batches[0]
        for batch in proposer_batches[1:]:
            combined_batch = DataProto.concat([combined_batch, batch])

        return combined_batch

    def _filter_proposer_batch_by_indices(self, proposer_batch: DataProto, valid_indices: List[int]) -> DataProto:
        if not valid_indices:
            return None

        filtered_batch = proposer_batch[valid_indices]

        return filtered_batch

    def _align_problems_with_proposer_batch(
        self, all_valid_problems: List[Dict[str, Any]], target_size: int
    ) -> List[Dict[str, Any]]:
        aligned_problems = []

        for i, problem in enumerate(all_valid_problems[:target_size]):
            aligned_problem = problem.copy()
            aligned_problem["trajectory_index"] = i
            aligned_problems.append(aligned_problem)

        return aligned_problems

    def _generate_proposer_trajectories(self, batch: DataProto, timing_raw: Dict) -> DataProto:
        print(f"=== Step 1: Proposer Generation at step {self.global_steps} ===")

        ray.get(self.sp_data_manager.switch_to_phase.remote(SelfPlayPhase.PROBLEM_GENERATION))

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "index" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("index")
        if "agent_name" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("agent_name")

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch.meta_info["phase"] = "proposer"
        gen_batch.meta_info["do_sample"] = self.sp_config.get("proposer", {}).get("do_sample", True)
        gen_batch.meta_info["temperature"] = self.sp_config.get("proposer", {}).get("temperature", 0.8)

        self.rollout_n = self.config.actor_rollout_ref.rollout.n
        proposer_n = self.sp_config.get("proposer", {}).get("n", self.rollout_n)

        if SELF_PLAY_DEBUG:
            logger.debug(f"Proposer gen_batch size before repeat: {len(gen_batch)}, proposer_n: {proposer_n}")

        gen_batch = gen_batch.repeat(repeat_times=proposer_n, interleave=True)

        if SELF_PLAY_DEBUG:
            logger.debug(f"Proposer gen_batch size after repeat: {len(gen_batch)}")

        if not self.async_rollout_mode:
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        else:
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
        gen_batch_output.meta_info.pop("timing", None)

        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        batch = batch.repeat(repeat_times=proposer_n, interleave=True)

        batch = batch.union(gen_batch_output)

        for key in batch.batch.keys():
            if key not in ["old_log_probs", "ref_log_prob"]:
                batch.batch[key] = batch.batch[key].long()

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        print(f"Generated {len(batch.batch)} proposer trajectories")

        return batch

    def _extract_and_assemble_solver_data(self, proposer_gen_batch: DataProto, timing_raw: Dict) -> Optional[DataProto]:
        print(f"=== Step 2: Problem Extraction and Assembly at step {self.global_steps} ===")

        trajectories = self._extract_trajectories_from_batch(proposer_gen_batch)
        batch_size = len(trajectories)

        extracted_problems = self._extract_and_process_problems(trajectories)

        extraction_success_mask = [False] * batch_size

        problem_to_trajectory_map = {}
        for problem in extracted_problems:
            traj_idx = problem.get("trajectory_index", -1)
            if 0 <= traj_idx < batch_size:
                extraction_success_mask[traj_idx] = True
                problem_to_trajectory_map[traj_idx] = problem

        solving_data = []
        failed_extraction_count = 0
        reused_problem_count = 0
        metrics = {}

        for i in range(batch_size):
            if i in problem_to_trajectory_map:
                solving_data.append(problem_to_trajectory_map[i])
            else:
                failed_extraction_count += 1
                fallback_problem, reused = self._create_fallback_problem_for_failed_extraction(trajectories[i], i)
                solving_data.append(fallback_problem)
                if reused:
                    reused_problem_count += 1

        print(
            f"Failed extractions: {failed_extraction_count} (reused {reused_problem_count} existing problems, created {failed_extraction_count - reused_problem_count} dummy problems)"
        )
        metrics["self_play/reused_problem_count"] = reused_problem_count
        metrics["self_play/dummy_problem_count"] = failed_extraction_count - reused_problem_count

        print(f"Extracted {len(extracted_problems)} valid problems, padded to {len(solving_data)} total")
        print(
            f"Extraction success rate: {sum(extraction_success_mask)}/{batch_size} = {sum(extraction_success_mask) / batch_size:.2%}"
        )
        metrics["self_play/extraction_success_rate"] = sum(extraction_success_mask) / batch_size

        self.current_extraction_success_mask = extraction_success_mask

        solver_batch = self._prepare_solving_batch_from_data(solving_data)

        self.generate_problem = solving_data

        solver_batch.meta_info.setdefault("metrics", {}).update(metrics)

        return solver_batch

    def _generate_solver_trajectories(self, solver_batch: DataProto, timing_raw: Dict) -> DataProto:
        print(f"=== Step 3: Solver Generation at step {self.global_steps} ===")

        if SELF_PLAY_DEBUG:
            logger.debug(f"Input solver_batch size: {len(solver_batch)}")

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "index" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("index")
        if "agent_name" in solver_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("agent_name")

        gen_batch = solver_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch.meta_info["phase"] = "solver"
        gen_batch.meta_info["do_sample"] = self.sp_config.get("solver", {}).get("do_sample", True)
        gen_batch.meta_info["temperature"] = self.sp_config.get("solver", {}).get("temperature", 0.7)

        self.rollout_n = self.config.actor_rollout_ref.rollout.n

        if SELF_PLAY_DEBUG:
            logger.debug(f"gen_batch size before repeat: {len(gen_batch)}, rollout_n: {self.rollout_n}")

        gen_batch = gen_batch.repeat(repeat_times=self.rollout_n, interleave=True)

        if SELF_PLAY_DEBUG:
            logger.debug(f"gen_batch size after repeat: {len(gen_batch)}")

        if not self.async_rollout_mode:
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        else:
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
        gen_batch_output.meta_info.pop("timing", None)

        solver_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(solver_batch.batch))], dtype=object
        )
        solver_batch = solver_batch.repeat(repeat_times=self.rollout_n, interleave=True)
        solver_batch = solver_batch.union(gen_batch_output)

        for key in solver_batch.batch.keys():
            if key not in ["old_log_probs", "ref_log_prob"]:
                solver_batch.batch[key] = solver_batch.batch[key].long()

        if "response_mask" not in solver_batch.batch.keys():
            solver_batch.batch["response_mask"] = compute_response_mask(solver_batch)

        print(f"Generated {len(solver_batch.batch)} solver trajectories")

        return solver_batch

    def _calculate_self_play_rewards(
        self, proposer_gen_batch: DataProto, solver_gen_batch: DataProto, timing_raw: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print(f"=== Step 4: Reward Calculation at step {self.global_steps} ===")

        with marked_timer("solver_reward_computation", timing_raw, color="yellow"):
            if self.config.reward_model.launch_reward_fn_async:
                future_solver_reward = compute_reward_async.remote(data=solver_gen_batch, reward_fn=self.reward_fn)
                solver_rewards, solver_extra_infos = ray.get(future_solver_reward)
            else:
                solver_rewards, solver_extra_infos = compute_reward(solver_gen_batch, self.reward_fn)

        with marked_timer("proposer_reward_computation", timing_raw, color="orange"):
            proposer_rewards = self._compute_proposer_rewards(proposer_gen_batch, solver_rewards, solver_extra_infos)

        print(f"Computed rewards: proposer_mean={proposer_rewards.mean():.3f}, solver_mean={solver_rewards.mean():.3f}")

        return proposer_rewards, solver_rewards

    def _compute_proposer_rewards(
        self, proposer_gen_batch: DataProto, solver_rewards: torch.Tensor, solver_extra_infos: Dict
    ) -> torch.Tensor:
        proposer_batch_size = len(proposer_gen_batch.batch)
        response_length = proposer_gen_batch.batch["responses"].size(1)

        solver_n_attempts = self.rollout_n
        solver_batch_size = solver_rewards.size(0)
        assert (
            solver_batch_size == proposer_batch_size * solver_n_attempts
        ), f"Expected {proposer_batch_size * solver_n_attempts} solver samples, got {solver_batch_size}"

        solver_rewards_per_attempt = solver_rewards.sum(-1)

        solver_rewards_grouped = solver_rewards_per_attempt.view(proposer_batch_size, solver_n_attempts)

        success_rates = (solver_rewards_grouped > 0).float().mean(dim=1)

        reward_type = self.sp_config.get("proposer", {}).get("reward_type", "1-acc")

        if reward_type == "intermediate_difficulty":
            proposer_reward_values = torch.zeros_like(success_rates)

            proposer_right = self.sp_config.proposer.get("right", 1.0)
            proposer_left = self.sp_config.proposer.get("left", 0.0)
            intermediate_mask = (success_rates > 0) & (success_rates < 1)
            proposer_reward_values[intermediate_mask] = (
                4.0
                * (proposer_left + success_rates[intermediate_mask])
                * (proposer_right - success_rates[intermediate_mask])
            )
        elif reward_type == "format_only":
            proposer_reward_values = torch.ones_like(success_rates)
        else:
            proposer_reward_values = 1.0 - success_rates

        extraction_failure_penalty = self.sp_config.proposer.format_penalty

        if hasattr(self, "current_extraction_success_mask"):
            for i, extraction_success in enumerate(self.current_extraction_success_mask):
                if not extraction_success:
                    proposer_reward_values[i] = extraction_failure_penalty
                    if SELF_PLAY_DEBUG:
                        logger.debug(
                            f"Applying extraction failure penalty {extraction_failure_penalty} to proposer generation {i}"
                        )
        elif hasattr(self, "generate_problem") and self.generate_problem:
            assert len(self.generate_problem) == len(
                proposer_reward_values
            ), "generate_problem and proposer_reward_values must have the same length"
            for i, problem in enumerate(self.generate_problem):
                if i < len(proposer_reward_values):
                    data_source = problem.get("data_source", "")
                    if data_source == "dummy":
                        proposer_reward_values[i] = extraction_failure_penalty
                        if SELF_PLAY_DEBUG:
                            logger.debug(
                                f"Applying extraction failure penalty {extraction_failure_penalty} to proposer generation {i} (dummy data_source)"
                            )

        proposer_rewards = torch.zeros(
            (proposer_batch_size, response_length),
            dtype=torch.float32,
            device=solver_rewards.device,
        )

        if "response_mask" in proposer_gen_batch.batch:
            response_mask = proposer_gen_batch.batch["response_mask"]
            last_token_positions = response_mask.sum(-1) - 1
            for i in range(proposer_batch_size):
                last_pos = last_token_positions[i].item()
                if 0 <= last_pos < response_length:
                    proposer_rewards[i, last_pos] = proposer_reward_values[i].item()
        else:
            proposer_rewards[:, -1] = proposer_reward_values

        extraction_failures = 0
        if hasattr(self, "current_extraction_success_mask"):
            extraction_failures = sum(1 for x in self.current_extraction_success_mask if not x)

        print(f"Proposer reward computation (using formula: {reward_type}):")
        print(f"  - Success rates: mean={success_rates.mean():.3f}, std={success_rates.std():.3f}")
        print(
            f"  - Proposer rewards: mean={proposer_reward_values.mean():.3f}, std={proposer_reward_values.std():.3f}, max={proposer_reward_values.max():.3f}"
        )
        print(f"  - Total problems: {proposer_batch_size}")
        print(f"  - Extraction failures: {extraction_failures}/{proposer_batch_size}")

        print(f"  - Reward range: [{proposer_reward_values.min():.3f}, {proposer_reward_values.max():.3f}]")

        if reward_type == "intermediate_difficulty":
            zero_success = (success_rates == 0).sum().item()
            full_success = (success_rates == 1).sum().item()
            intermediate_count = ((success_rates > 0) & (success_rates < 1)).sum().item()
            print(f"  - Problems with success_rate=0 (zero reward): {zero_success}")
            print(f"  - Problems with success_rate=1 (zero reward): {full_success}")
            print(f"  - Problems with intermediate difficulty (non-zero reward): {intermediate_count}")
            if intermediate_count > 0:
                intermediate_rates = success_rates[(success_rates > 0) & (success_rates < 1)]
                intermediate_rewards = proposer_reward_values[(success_rates > 0) & (success_rates < 1)]
                print(
                    f"  - Intermediate problems success_rate range: [{intermediate_rates.min():.3f}, {intermediate_rates.max():.3f}]"
                )
                print(
                    f"  - Intermediate problems reward range: [{intermediate_rewards.min():.3f}, {intermediate_rewards.max():.3f}]"
                )
        else:
            print(f"  - Problems with success_rate=0 (max reward): {(success_rates == 0).sum().item()}")
            print(f"  - Problems with success_rate=1 (min reward): {(success_rates == 1).sum().item()}")
            print(f"  - Problems with 0<success_rate<1: {((success_rates > 0) & (success_rates < 1)).sum().item()}")

        if hasattr(self, "generate_problem") and self.generate_problem:
            valid_problems_with_success_rate = []
            assert len(self.generate_problem) == len(
                success_rates
            ), f"generate_problem length {len(self.generate_problem)} != success_rates length {len(success_rates)}"

            for i, problem in enumerate(self.generate_problem):
                if i < len(success_rates):
                    data_source = problem.get("data_source", "")
                    if data_source != "dummy":
                        problem_with_rate = problem.copy()
                        problem_with_rate["success_rate"] = success_rates[i].item()
                        problem_with_rate["generation_step"] = self.global_steps
                        valid_problems_with_success_rate.append(problem_with_rate)

            if valid_problems_with_success_rate:
                ray.get(
                    self.sp_data_manager.add_generated_problems_with_success_rate.remote(
                        valid_problems_with_success_rate, generation_step=self.global_steps
                    )
                )
                print(f"Added {len(valid_problems_with_success_rate)} valid problems with success_rate to data manager")

        return proposer_rewards

    def _update_on_trajectories(self, gen_batch: DataProto, role: str, timing_raw: Dict) -> Dict[str, Any]:
        print(f"=== Step 5: {role.title()} Update at step {self.global_steps} ===")

        metrics = {}

        if self.config.trainer.balance_batch:
            self._balance_batch(gen_batch, metrics=metrics)

        gen_batch.meta_info["global_token_num"] = torch.sum(gen_batch.batch["attention_mask"], dim=-1).tolist()

        for key in gen_batch.batch.keys():
            if key not in [
                "old_log_probs",
                "ref_log_prob",
                "token_level_scores",
                "token_level_rewards",
                "advantages",
                "returns",
                "values",
            ]:
                gen_batch.batch[key] = gen_batch.batch[key].long()

        with marked_timer(f"{role}_old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(gen_batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = gen_batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics[f"{role}/actor/entropy"] = entropy_agg.detach().item()
            old_log_prob.batch.pop("entropys")
            gen_batch = gen_batch.union(old_log_prob)

        if self.use_reference_policy:
            with marked_timer(f"{role}_ref", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(gen_batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(gen_batch)
                gen_batch = gen_batch.union(ref_log_prob)

        if self.use_critic:
            with marked_timer(f"{role}_values", timing_raw, color="cyan"):
                values = self.critic_wg.compute_values(gen_batch)
                gen_batch = gen_batch.union(values)

        with marked_timer(f"{role}_adv", timing_raw, color="brown"):
            if self.config.algorithm.use_kl_in_reward:
                gen_batch, kl_metrics = apply_kl_penalty(
                    gen_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update({f"{role}_{k}": v for k, v in kl_metrics.items()})
            else:
                gen_batch.batch["token_level_rewards"] = gen_batch.batch["token_level_scores"]

            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            if role == "solver":
                num_repeat = self.rollout_n
                adv_estimator = self.config.algorithm.adv_estimator
            else:
                num_repeat = self.sp_config.proposer.n
                if num_repeat == 1:
                    adv_estimator = self.sp_config.proposer.adv_estimator
                else:
                    adv_estimator = self.config.algorithm.adv_estimator
            gen_batch = compute_advantage(
                gen_batch,
                adv_estimator=adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )

        if self.use_critic:
            with marked_timer(f"{role}_update_critic", timing_raw, color="pink"):
                critic_output = self.critic_wg.update_critic(gen_batch)
            critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update({f"{role}_critic_{k}": v for k, v in critic_metrics.items()})

        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer(f"{role}_update_actor", timing_raw, color="red"):
                gen_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(gen_batch)
            actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update({f"{role}_actor_{k}": v for k, v in actor_metrics.items()})

        scores = gen_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
        metrics.update(
            {
                f"{role}_reward_mean": np.mean(scores),
                f"{role}_reward_std": np.std(scores),
                f"{role}_reward_max": np.max(scores),
                f"{role}_reward_min": np.min(scores),
            }
        )

        return metrics

    def _dump_trajectories(self, inputs, outputs, scores, role: str, step: int, reward_extra_infos_dict=None):
        try:
            import os

            base_dir = "/primus_oss/output"
            role_dir = os.path.join(base_dir, role)
            os.makedirs(role_dir, exist_ok=True)

            filename = f"{role}_step_{step}.jsonl"
            filepath = os.path.join(role_dir, filename)

            n = len(inputs)
            base_data = {
                "input": inputs,
                "output": outputs,
                "score": scores,
                "step": [step] * n,
                "role": [role] * n,
                "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * n,
            }

            if reward_extra_infos_dict:
                for k, v in reward_extra_infos_dict.items():
                    if len(v) == n:
                        base_data[k] = v

            lines = []
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                lines.append(json.dumps(entry, ensure_ascii=False, default=str))

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            print(f"Dumped {len(inputs)} {role} trajectories to {filepath}")

        except Exception as e:
            logger.warning(f"Failed to dump {role} trajectories for step {step}: {e}")

    def _prepare_solving_batch_from_data(self, solving_data: List[Dict]) -> DataProto:
        assert solving_data is not None, "No solving data available"

        import os
        import tempfile

        import pandas as pd

        df = pd.DataFrame(solving_data)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_file = f.name
            df.to_parquet(temp_file, index=False)

        try:
            from verl.utils.dataset.rl_dataset import RLHFDataset

            if SELF_PLAY_DEBUG:
                logger.debug(f"Input solving_data length: {len(solving_data)}")

            temp_config = deepcopy(self.config.data)
            temp_config.filter_overlong_prompts = False

            temp_dataset = RLHFDataset(
                data_files=[temp_file], tokenizer=self.tokenizer, config=temp_config, processor=self.processor
            )

            if SELF_PLAY_DEBUG:
                logger.debug(f"RLHFDataset length after creation: {len(temp_dataset)}")

            from torchdata.stateful_dataloader import StatefulDataLoader
            from verl.utils.dataset.rl_dataset import collate_fn

            temp_dataloader = StatefulDataLoader(
                dataset=temp_dataset,
                batch_size=len(solving_data),
                num_workers=0,
                drop_last=False,
                collate_fn=collate_fn,
                shuffle=False,
            )

            batch_dict = next(iter(temp_dataloader))
            batch = DataProto.from_single_dict(batch_dict)

            if SELF_PLAY_DEBUG:
                logger.debug(f"Final batch size after processing: {len(batch)}")

            for key in batch.batch.keys():
                if key not in [
                    "old_log_probs",
                    "ref_log_prob",
                    "token_level_scores",
                    "token_level_rewards",
                    "advantages",
                    "returns",
                    "values",
                ]:
                    batch.batch[key] = batch.batch[key].long()

            return batch

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _extract_trajectories_from_batch(self, batch: DataProto) -> List[Dict[str, Any]]:
        trajectories = []

        assert "prompts" in batch.batch and "responses" in batch.batch, "prompts and responses must be in batch"

        if SELF_PLAY_DEBUG:
            logger.debug(f"Extracting trajectories from batch with {len(batch.batch['prompts'])} samples")
            logger.debug(f"Batch keys: {list(batch.batch.keys())}")
            if hasattr(batch, "non_tensor_batch") and batch.non_tensor_batch:
                logger.debug(f"Non-tensor batch keys: {list(batch.non_tensor_batch.keys())}")

        input_ids = batch.batch["prompts"]
        output_ids = batch.batch["responses"]

        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for i, (input_text, output_text) in enumerate(zip(input_texts, output_texts)):
            metadata = {"trajectory_index": i, "global_step": self.global_steps, "phase": self.current_phase}

            if hasattr(batch, "non_tensor_batch") and batch.non_tensor_batch is not None:
                if "reward_model" in batch.non_tensor_batch and i < len(batch.non_tensor_batch["reward_model"]):
                    metadata["reward_model"] = batch.non_tensor_batch["reward_model"][i]
                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Trajectory {i}: reward_model = {metadata['reward_model']}")

                if "data_source" in batch.non_tensor_batch and i < len(batch.non_tensor_batch["data_source"]):
                    metadata["data_source"] = batch.non_tensor_batch["data_source"][i]
                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Trajectory {i}: data_source = {metadata['data_source']}")

                if "extra_info" in batch.non_tensor_batch and i < len(batch.non_tensor_batch["extra_info"]):
                    metadata["extra_info"] = batch.non_tensor_batch["extra_info"][i]
                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Trajectory {i}: extra_info = {metadata['extra_info']}")

            trajectory = {
                "input": input_text,
                "output": output_text,
                "metadata": metadata,
            }
            trajectories.append(trajectory)

            if SELF_PLAY_DEBUG:
                logger.debug(f"Trajectory {i}: input[:100] = {input_text[:100]}...")
                logger.debug(f"Trajectory {i}: output[:200] = {output_text[:200]}...")
                logger.debug(f"Trajectory {i}: metadata = {metadata}")

        if SELF_PLAY_DEBUG:
            logger.debug(f"Extracted {len(trajectories)} trajectories total")

        return trajectories

    def _extract_and_process_problems(self, trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if SELF_PLAY_DEBUG:
            logger.debug(f"Starting problem extraction from {len(trajectories)} trajectories")
            for i, traj in enumerate(trajectories[:3]):
                logger.debug(f"Trajectory {i} output for extraction: {traj['output'][:500]}...")

        ray.get(self.sp_data_manager.reset_current_step_stats.remote())

        start_time = time.time()
        extracted_problems, extraction_stats = extract_problems_batch(trajectories, self.problem_extractor)
        end_time = time.time()
        if SELF_PLAY_DEBUG:
            logger.debug(f"Extracted {len(extracted_problems)} problems in {end_time - start_time} seconds")
            logger.debug(f"Extraction statistics: {extraction_stats}")
            for i, problem in enumerate(extracted_problems[:3]):
                logger.debug(f"Raw problem {i}: {problem}")

        valid_problems = []
        for i, problem in enumerate(extracted_problems):
            if self._validate_extracted_problem(problem):
                valid_problems.append(problem)
                if SELF_PLAY_DEBUG:
                    logger.debug(f"Problem {i} is VALID: {problem.get('extracted_question', '')[:100]}...")
            else:
                if SELF_PLAY_DEBUG:
                    logger.debug(f"Problem {i} is INVALID: {problem}")

        print(f"Extracted {len(extracted_problems)} problems, {len(valid_problems)} valid")

        ray.get(
            self.sp_data_manager.record_extraction_stats.remote(
                trajectories_count=extraction_stats["trajectories_count"],
                answer_matches_count=extraction_stats["answer_matches_count"],
                valid_questions_count=extraction_stats["valid_questions_count"],
                format_error_count=extraction_stats["format_error_count"],
                successful_problems_count=len(valid_problems),
            )
        )

        if SELF_PLAY_DEBUG:
            logger.debug(f"Final valid problems count: {len(valid_problems)}")
            for i, problem in enumerate(valid_problems[:2]):  # Debug first 2 valid problems
                logger.debug(f"Valid problem {i}: {problem}")

        return valid_problems

    def _create_fallback_problem_for_failed_extraction(
        self, trajectory: Dict[str, Any], trajectory_index: int
    ) -> Tuple[Dict[str, Any], bool]:
        extraction_strategy = self.sp_config.get("extraction_failure", {}).get("strategy", "dummy")
        fallback_to_dummy = self.sp_config.get("extraction_failure", {}).get("fallback_to_dummy", True)

        if SELF_PLAY_DEBUG:
            logger.debug(
                f"Handling extraction failure for trajectory {trajectory_index} with strategy: {extraction_strategy}"
                )

        if extraction_strategy == "reuse":
            has_existing = ray.get(self.sp_data_manager.has_existing_problems.remote())

            if has_existing:
                reuse_success_rate_threshold = self.sp_config.extraction_failure.reuse_success_rate_threshold
                existing_problems = ray.get(
                    self.sp_data_manager.get_problems_by_success_rate.remote(1, reuse_success_rate_threshold)
                )

                if existing_problems:
                    reused_problem = self._adapt_existing_problem_for_trajectory(
                        existing_problems[0], trajectory, trajectory_index
                    )
                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Reused existing problem for trajectory {trajectory_index}")
                    return reused_problem, True

            if fallback_to_dummy:
                if SELF_PLAY_DEBUG:
                    logger.debug(
                        f"No existing problems available, falling back to dummy for trajectory {trajectory_index}"
                    )
                return self._create_dummy_problem_for_failed_extraction(trajectory, trajectory_index), False
            else:
                raise RuntimeError(f"Cannot create fallback problem: no existing problems and dummy fallback disabled")

        return self._create_dummy_problem_for_failed_extraction(trajectory, trajectory_index), False

    def _adapt_existing_problem_for_trajectory(
        self, existing_problem: Dict[str, Any], trajectory: Dict[str, Any], trajectory_index: int
    ) -> Dict[str, Any]:
        adapted_problem = deepcopy(existing_problem)

        adapted_problem["trajectory_index"] = trajectory_index
        adapted_problem["data_source"] = f"{adapted_problem.get('data_source', 'unknown')}"

        return adapted_problem

    def _create_dummy_problem_for_failed_extraction(
        self, trajectory: Dict[str, Any], trajectory_index: int
    ) -> Dict[str, Any]:
        metadata = trajectory.get("metadata", {})

        dummy_prompt = [
            {"role": "system", "content": "You are a helpful and harmless assistant."},
            {
                "role": "user",
                "content": "This is a dummy problem. Please directly output </answer>",
            },
        ]

        dummy_reward_model = metadata.get("reward_model", {"ground_truth": {"target": "dummy_answer", "style": "rule"}})

        dummy_problem = {
            "data_source": "dummy",
            "prompt": dummy_prompt,
            "ability": "fact-reasoning",
            "reward_model": dummy_reward_model,
            "extra_info": {
                "question": "This is a dummy problem. Please directly output </answer>",
                "need_tools_kwargs": True,
                "split": "train",
                "tools_kwargs": {
                    "search": {
                        "create_kwargs": {
                            "data_source": "dummy",
                            "question": "This is a dummy problem. Please directly output </answer>",
                            "ground_truth": dummy_reward_model.get("ground_truth"),
                        }
                    }
                },
                "extraction_failed": True,
            },
            "metadata": None,
            "extracted_question": "This is a dummy problem. Please directly output </answer>",
            "formatted_prompt": dummy_prompt,
            "problem_type": "search",
            "trajectory_index": trajectory_index,
        }

        if SELF_PLAY_DEBUG:
            logger.debug(f"Created dummy problem for failed extraction at trajectory {trajectory_index}")

        return dummy_problem

    def _validate_extracted_problem(self, problem: Dict[str, Any]) -> bool:
        question = problem["extracted_question"]
        if len(question.strip()) < 10:
            return False

        answer = problem.get("reward_model", {}).get("ground_truth", {}).get("target", None)
        if answer in question:
            return False

        return True

    def _balance_batch(self, batch: DataProto, metrics):
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size

        from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix="global_seqlen"
        )
        metrics.update(global_balance_stats)

