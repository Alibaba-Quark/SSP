import re
from functools import partial
from typing import Any, Dict

import numpy as np
import torch
import verl
from verl import DataProto
from verl.trainer.ppo.metric_utils import _compute_response_info, compute_data_metrics

from quarl.tool.search_tool import USE_WIKI
from quarl.utils.data_utils import extract_thought_and_answer


def extract_think_content(text):
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return ""


def quarl_compute_data_metrics(tokenizer, batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    print("[quarl_compute_data_metrics] enter patch compute_data_metrics successfully")
    all_response_str = []
    think_length = []
    answer_length = []

    think_total_lengths = []

    for i in range(len(batch)):
        data_item = batch[i]  # DataProtoItem

        prompt_ids = data_item.batch["prompts"]

        prompt_length = prompt_ids.shape[-1]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum().long()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        sequences_str = tokenizer.decode(valid_response_ids)
        think_and_answer = extract_thought_and_answer(sequences_str)
        if think_and_answer is not None:
            think_length.append(len(think_and_answer[0]))
            answer_length.append(len(think_and_answer[1]))
        else:
            think_length.append(0)
            answer_length.append(0)

        think_content = extract_think_content(sequences_str)
        think_total_lengths.append(len(think_content))

        all_response_str.append(sequences_str)

    batch.batch["think_length"] = torch.tensor(think_length, dtype=torch.long)
    batch.batch["answer_length"] = torch.tensor(answer_length, dtype=torch.long)

    # todo: how to avoid duplicate calc in compute_data_metrics
    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    think_length = batch.batch["think_length"].float()
    answer_length = batch.batch["answer_length"].float()
    metrics = compute_data_metrics(batch, use_critic=use_critic)

    
    if think_total_lengths:
        metrics["search_stats/think_total_length/mean"] = np.mean(think_total_lengths)

    if hasattr(batch, "meta_info") and batch.meta_info:
        timing_info = batch.meta_info
        if "avg_turns" in timing_info:
            metrics["search_stats/search_turns_avg"] = timing_info["avg_turns"]
        if "max_turns" in timing_info:
            metrics["search_stats/search_turns_max"] = timing_info["max_turns"]
        if "min_turns" in timing_info:
            metrics["search_stats/search_turns_min"] = timing_info["min_turns"]

    return metrics


def apply_metric_patch(tokenizer):
    verl.trainer.ppo.metric_utils.compute_data_metrics = partial(quarl_compute_data_metrics, tokenizer=tokenizer)
    print("[metric_patch] metric patch applied!!!")
