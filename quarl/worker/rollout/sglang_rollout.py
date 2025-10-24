import asyncio
import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from json import JSONDecodeError
from types import MethodType
from typing import Any, Dict, List, Optional, Type
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.openai_api.protocol import Tool
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin
from verl import DataProto
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionToolCall,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout, _pre_process_inputs
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

from quarl.utils.chat_template import QuarkChatTemplate as QSCT
from quarl.utils.llm_as_a_judge import get_global_judge
from quarl.worker.rollout.schema import QuarkAsyncRolloutRequest

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


QUARK_SEARCH_DETECTOR_NAME = "quark-search"


class QuarkSearchDetector(Qwen25Detector):

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()

        print("[QuarkSearchDetector] QuarkChatTemplate._current_template =", QSCT._current_template)

        self.bosch_token = QSCT.search_prefix
        self.eosch_token = QSCT.search_suffix

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the text contains a Quark format tool call.
        """
        # print(f"[Debug SearchDetector]: check text = {text}")
        # print(f"[Debug SearchDetector]: check has_tool_call = {self.bot_token in text or self.bosch_token in text}")
        return self.bot_token in text or self.bosch_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        non_search_results = super().detect_and_parse(text=text, tools=tools)
        # print(f"[Debug SearchDetector.detect_and_parse]: check non_search_results = {non_search_results}")

        normal_text, calls = non_search_results.normal_text, non_search_results.calls

        idx = normal_text.find(self.bosch_token)
        normal_text = normal_text[:idx] if idx != -1 else text
        if self.bosch_token not in text:
            return non_search_results

        # Find all <tool_call>\n...\n</tool_call> blocks
        pattern = rf"{re.escape(self.bosch_token)}(.*?){re.escape(self.eosch_token)}"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        # print(f"[Debug SearchDetector.detect_and_parse]: check regex match = {match_result_list}")
        for match_result in match_result_list:
            try:
                match_result_jsons = json.dumps({"name": "search", "arguments": {"query_list": [match_result.strip()]}})
                parsed_call = json.loads(match_result_jsons)
                calls.extend(self.parse_base_json(parsed_call, tools))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON part: {match_result_jsons}, JSON parse error: {str(e)}")
                continue
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Streaming incremental parsing for Qwen 2.5 tool calls.
        Uses base class implementation with buffering to handle partial end tokens.
        """
        # result = super().parse_streaming_increment(new_text, tools)
        result = BaseFormatDetector.parse_streaming_increment(self, new_text, tools)

        # Handle partial end tokens that are streamed character by character
        if result.normal_text:
            self._normal_text_buffer += result.normal_text

            # Check if buffer contains complete end token (without leading newline)
            end_token_without_newline = self.eot_token[1:]  # "</tool_call>"
            end_sch_token_without_newline = self.eosch_token[1:]
            if (
                end_token_without_newline in self._normal_text_buffer
                or end_sch_token_without_newline in self._normal_text_buffer
            ):
                cleaned_text = self._normal_text_buffer.replace(end_token_without_newline, "").replace(
                    end_sch_token_without_newline, ""
                )
                self._normal_text_buffer = ""
                result.normal_text = cleaned_text
            else:
                # Check if buffer might contain partial end token at the end
                partial_match_len = self._ends_with_partial_token(self._normal_text_buffer, end_token_without_newline)
                partial_sch_match_len = self._ends_with_partial_token(
                    self._normal_text_buffer, end_sch_token_without_newline
                )

                partial_match_len = max(partial_match_len, partial_sch_match_len)

                if partial_match_len:
                    # Keep potential partial match in buffer, return the rest
                    result.normal_text = self._normal_text_buffer[:-partial_match_len]
                    self._normal_text_buffer = self._normal_text_buffer[-partial_match_len:]
                else:
                    # No partial match, return all buffered text
                    result.normal_text = self._normal_text_buffer
                    self._normal_text_buffer = ""

        return result


class QuarkFunctionCallParser(FunctionCallParser):
    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        **FunctionCallParser.ToolCallParserEnum,
        QUARK_SEARCH_DETECTOR_NAME: QuarkSearchDetector,
    }


class QuarkSGLangRollout(SGLangRollout):
    def _init_sampling_params(self, **kwargs):
        super()._init_sampling_params(**kwargs)

        deepsearch_stopwords = [QSCT.search_suffix, QSCT.answer_suffix, QSCT.user_prefix, QSCT.eos, "<|im_end|>"]

        if "stop" in self.sampling_params:
            if isinstance(self.sampling_params["stop"], list):
                self.sampling_params["stop"].extend(deepsearch_stopwords)
            elif isinstance(self.sampling_params["stop"], str):
                deepsearch_stopwords.append(self.sampling_params["stop"])
                self.sampling_params["stop"] = deepsearch_stopwords
            else:
                raise ValueError(f"""Unsupported type of stop params {self.sampling_params["stop"]}""")
        else:
            self.sampling_params["stop"] = deepsearch_stopwords

        print(f"[QuarkSGLangRollout]: check init sampling params {self.sampling_params}")

    def _initialize_tools(self, config, processing_class):
        """Initialize tools from configuration.

        Args:
            config: Configuration object containing tool-related settings,
                    specifically `config.multi_turn.tool_config_path`.
            tokenizer: The tokenizer instance used for parsing tool calls from
                       the model's generated text.

        Returns:
            tuple: A tuple containing:
                - tool_schemas (list[dict]): OpenAI-formatted JSON schemas
                  defining each tool's capabilities.
                - tool_map (dict[str, BaseTool]): A dictionary mapping tool
                  names to their executable `BaseTool` objects.
                - tool_call_parser_type (str): The identifier for the specific
                  parser type (e.g., 'json_mode', 'tool_code') used to extract
                  tool calls.
                - sgl_tools (list[sglang.srt.openai_api.protocol.Tool]): Tool
                  definitions optimized for SGLang's internal engine.
                - function_call_parser (sglang.srt.function_call_parser.FunctionCallParser):
                  The active parser instance responsible for extracting
                  structured tool calls from model outputs.
        """
        tool_schemas, tool_map, tool_call_parser_type, sgl_tools, _ = super()._initialize_tools(
            config=config, processing_class=processing_class
        )

        # quark tool call parser
        print(
            f"[QuarkSGLangRollout]: switch tool_call_parser from `{tool_call_parser_type}` to `{QUARK_SEARCH_DETECTOR_NAME}`."
        )
        tool_call_parser_type = QUARK_SEARCH_DETECTOR_NAME

        function_call_parser = QuarkFunctionCallParser(
            sgl_tools,
            tool_call_parser_type,
        )

        return (
            tool_schemas,
            tool_map,
            tool_call_parser_type,
            sgl_tools,
            function_call_parser,
        )

    async def _handle_engine_call(
        self, _req: AsyncRolloutRequest, sampling_params: dict, image_data: Optional[list[Any]] = None
    ) -> dict:
        output = await super()._handle_engine_call(_req, sampling_params, image_data=image_data)

        # deal with stop words
        patterns = {QSCT.search_prefix: QSCT.search_suffix, QSCT.answer_prefix: QSCT.answer_suffix}

        content = output.get("text", "")
        stopped_suffix = None
        for i in reversed(range(len(content))):
            for prefix, suffix in patterns.items():
                if prefix in content[i:]:
                    stopped_suffix = suffix
                    break
            if stopped_suffix is not None:
                break

        if stopped_suffix is not None:
            output["text"] = output["text"] + stopped_suffix
        return output

    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n: int = 1) -> list[AsyncRolloutRequest]:
        assert (
            "raw_prompt" in prompts.non_tensor_batch
        ), "need data.return_raw_chat=True, due to no official way do parse_messages"
        logger.info(
            "n is deprecated for SGLang rollout since ray ppo trainer will repeat the prompts for rollout.n times"
        )
        req_list = []
        multi_modal_data_list = prompts.non_tensor_batch.get(
            "multi_modal_data", [None] * len(prompts.non_tensor_batch["raw_prompt"])
        )

        for data_idx, (raw_prompt, multi_modal_data) in enumerate(
            zip(prompts.non_tensor_batch["raw_prompt"], multi_modal_data_list)
        ):
            if self._tool_schemas:
                _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx]
                _tool_schemas = [self._tool_map[k].get_openai_tool_schema() for k in _tools_kwargs.keys()]
                _input_ids = None
                _attention_mask = None
            else:
                _input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][data_idx])
                _attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][data_idx])
                _tools_kwargs = {}
                _tool_schemas = None

            if self.interaction_map:
                _interaction_kwargs = prompts.non_tensor_batch["interaction_kwargs"][data_idx]
            else:
                _interaction_kwargs = {}

            if "tools_kwargs" in prompts.non_tensor_batch:
                ground_truth = (
                    prompts.non_tensor_batch["tools_kwargs"][data_idx]
                    .get("search", {})
                    .get("create_kwargs", {})
                    .get("ground_truth", None)
                )
            else:
                ground_truth = None
            req = QuarkAsyncRolloutRequest(
                batch_data_id=data_idx,
                rollout_offset=0,
                request_id=str(uuid4()),
                state=AsyncRolloutRequestStateEnum.PENDING,
                messages=raw_prompt.tolist() if not isinstance(raw_prompt, list) else raw_prompt,
                multi_modal_data=multi_modal_data,
                tool_schemas=_tool_schemas,
                tools_kwargs=_tools_kwargs,
                interaction_kwargs=_interaction_kwargs,
                input_ids=_input_ids,
                response_ids=None,
                attention_mask=_attention_mask,
                response_attention_mask=None,
                response_position_ids=None,
                response_loss_mask=None,
                reward_scores={},
                max_prompt_len=self.config.prompt_length,
                max_response_len=self.config.response_length,
                max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
                use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
                tokenization_sanity_check_mode=self.config.multi_turn.tokenization_sanity_check_mode,
                processing_class=self.processing_class,
                ground_truth=ground_truth,
                last_generation_output=None,
            )
            error_message = f"""Request {req.request_id} has mismatched lengths:
            input_ids={req.input_ids.shape[-1]},
            attention_mask={req.attention_mask.shape[-1]},
            position_ids={req.position_ids.shape[-1]},
            loss_mask={req.loss_mask.shape[-1]}"""
            assert (
                req.input_ids.shape[-1]
                == req.attention_mask.shape[-1]
                == req.position_ids.shape[-1]
                == req.loss_mask.shape[-1]
            ), error_message
            req_list.append(req)

        return req_list

    async def _async_rollout_a_request(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> AsyncRolloutRequest:
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None
        output = None

        # Add timing tracking
        vllm_time = 0.0
        tool_time = 0.0
        start_time = time.time()

        current_turns = 0
        user_turns = 0
        user_turn_rewards = []

        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": self.config.response_length,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        # Update with any additional kwargs
        request_sampling_params.update(kwargs)

        while current_turns < self.config.multi_turn.max_assistant_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls

                    for tool_call in parsed_tool_calls:
                        if tool_call.function.name == "search":
                            data_source = (
                                _req.tools_kwargs.get("search", {}).get("create_kwargs", {}).get("data_source")
                            )
                            # print("[DebugSearchDataSource] data_source:", data_source)
                            tool_call.function.arguments["data_source"] = data_source
                    # Track tool execution time
                    tool_start = time.time()
                    tool_call_results = await asyncio.gather(
                        *[
                            self._tool_map[tool_call.function.name].execute(
                                _req.request_id,
                                tool_call.function.arguments,
                                **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {}),
                            )
                            for tool_call in parsed_tool_calls
                        ]
                    )
                    tool_time += time.time() - tool_start

                    _req.add_tool_response_messages(self.processing_class, [resp for resp, _, _ in tool_call_results])
                    for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results):
                        _req.update_metrics(metrics, tool_call.function.name)
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                # Only continue the conversation if the prompt length is not greater than max_model_len - 1,
                # since SGLang raises an error when max_new_tokens + 1 is greater to max_model_len (the extra
                # token accounts for the EOS token).
                if len(_req.get_generation_prompt_ids(self.processing_class)) + 1 >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.LENGTH
                    break

                # Video support is not implemented yet
                image_data = (
                    _req.multi_modal_data["image"]
                    if _req.multi_modal_data and "image" in _req.multi_modal_data
                    else None
                )
                video_data = (
                    _req.multi_modal_data["video"]
                    if _req.multi_modal_data and "video" in _req.multi_modal_data
                    else None
                )
                if video_data:
                    logger.warning(
                        "video support is not implemented yet, current length of video data is %d", len(video_data)
                    )
                # Track vLLM generation time
                vllm_start = time.time()
                output = await self._handle_engine_call(_req, request_sampling_params, image_data=image_data)
                vllm_time += time.time() - vllm_start

                content = output["text"]
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.processing_class, content)
                    break
                else:
                    if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                        try:
                            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                        except JSONDecodeError:
                            normed_content = content
                            tool_calls = []
                        except AttributeError:
                            normed_content = content
                            tool_calls = []
                        parsed_tool_calls = []
                        for tool_call in tool_calls:
                            function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                                OpenAIFunctionParsedSchema(
                                    name=tool_call.name,
                                    arguments=tool_call.parameters,
                                )
                            )
                            # Drop the tool call if its arguments has decode error
                            if has_decode_error:
                                continue
                            parsed_tool_calls.append(
                                OpenAIFunctionToolCall(
                                    id=str(tool_call.tool_index),
                                    function=function,
                                )
                            )
                        if len(parsed_tool_calls) > 0:
                            _req.add_assistant_message(
                                self.processing_class, normed_content, tool_calls=parsed_tool_calls
                            )

                        else:
                            _req.add_assistant_message(self.processing_class, content)
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
                    else:
                        _req.add_assistant_message(
                            self.processing_class,
                            content,
                        )

                        if (
                            _req.interaction_kwargs
                            and self.interaction_map
                            and user_turns < self.config.multi_turn.max_user_turns
                            and current_turns < self.config.multi_turn.max_assistant_turns
                        ):
                            _req.state = AsyncRolloutRequestStateEnum.INTERACTING
                        else:
                            break
            elif _req.state == AsyncRolloutRequestStateEnum.INTERACTING:
                user_turns += 1
                messages = [{"role": x.role, "content": x.content} for x in _req.messages]

                # Get interaction by name from interaction_kwargs
                interaction_name = _req.interaction_kwargs.get(
                    "name", "gsm8k"
                )  # Default to gsm8k for backward compatibility
                if interaction_name not in self.interaction_map:
                    raise ValueError(
                        f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                        f"{list(self.interaction_map.keys())}"
                    )

                interaction = self.interaction_map[interaction_name]
                should_terminate_sequence, content, reward, metrics = await interaction.generate_response(
                    _req.request_id, messages, **_req.interaction_kwargs
                )
                user_turn_rewards.append(reward)
                if should_terminate_sequence:
                    finish_reason_type = FinishReasonTypeEnum.STOP
                    _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                    break
                else:
                    _req.add_user_message(self.processing_class, content)
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                    else:
                        _req.state = AsyncRolloutRequestStateEnum.RUNNING

        if current_turns >= self.config.multi_turn.max_assistant_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP

        # Calculate the reward for each tool
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        all_rewards = {**tool_reward_scores, **{"user_turn_rewards": user_turn_rewards}}
        _req.finalize(self.processing_class, all_rewards, finish_reason_type)
        # Add timing information to request
        total_time = time.time() - start_time
        timing_info = {
            "vllm_time": vllm_time,
            "tool_time": tool_time,
            "total_time": total_time,
            "other_time": total_time - vllm_time - tool_time,
            "turns": current_turns,
            "tool_time_per_turn": tool_time / current_turns if current_turns > 0 else 1,
        }
        _req.timing_info = timing_info

        return _req

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generates multi-turn sequences for a batch of prompts.
        For multi-turn generation, each prompt is processed separately via
        `_req_level_generate_sequences` for better tool calling control.
        Note that in multi-turn generation, we repeat the prompts for rollout.n times in ray_trainer.
        Thus we do not need to repeat the prompts here and set the sampling parameter n to 1.
        """
        # Async rollout with tools support
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device
        if self._tp_rank == 0:
            req_list = self._preprocess_prompt_to_async_rollout_requests(
                prompts,
            )
            loop = asyncio.get_event_loop()
            output_req_list = loop.run_until_complete(
                asyncio.gather(
                    *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],
                )
            )
            sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))
        else:
            sorted_output_req_list = None

        dist.barrier()
        [sorted_output_req_list] = broadcast_pyobj(
            data=[sorted_output_req_list],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )

        # Collect timing information from all requests
        timing_info = {
            "total_vllm_time": 0.0,
            "total_tool_time": 0.0,
            "total_generation_time": 0.0,
            "avg_vllm_time": 0.0,
            "avg_tool_time": 0.0,
            "max_vllm_time": 0.0,
            "max_tool_time": 0.0,
            "avg_turns": 0.0,
            "max_turns": 0,
            "min_turns": 0,
            "num_requests": len(sorted_output_req_list) if sorted_output_req_list else 0,
        }

        if sorted_output_req_list:
            vllm_times = []
            tool_times = []
            total_times = []
            turns_list = []
            tool_time_per_turn_list = []

            print(f"length of sorted_output_req_list: {len(sorted_output_req_list)}")

            for req in sorted_output_req_list:
                if hasattr(req, "timing_info") and req.timing_info is not None:
                    vllm_times.append(req.timing_info["vllm_time"])
                    tool_times.append(req.timing_info["tool_time"])
                    total_times.append(req.timing_info["total_time"])
                    turns_list.append(req.timing_info["turns"])
                    tool_time_per_turn_list.append(req.timing_info["tool_time_per_turn"])
            if vllm_times:
                timing_info.update(
                    {
                        "total_vllm_time": sum(vllm_times),
                        "total_tool_time": sum(tool_times),
                        "total_generation_time": sum(total_times),
                        "avg_vllm_time": sum(vllm_times) / len(vllm_times),
                        "avg_tool_time": sum(tool_times) / len(tool_times),
                        "avg_tool_time_per_turn": sum(tool_time_per_turn_list) / len(tool_time_per_turn_list),
                        "max_vllm_time": max(vllm_times),
                        "max_tool_time": max(tool_times),
                    }
                )

            if turns_list:
                timing_info.update(
                    {
                        "avg_turns": sum(turns_list) / len(turns_list),
                        "max_turns": max(turns_list),
                        "min_turns": min(turns_list),
                    }
                )

        # Construct the batch data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        messages = []
        reward_scores = []
        multi_modal_inputs = []
        request_ids = []

        for req in sorted_output_req_list:
            assert req.state == AsyncRolloutRequestStateEnum.COMPLETED, f"Request {req.request_id} is not completed"
            assert (
                req.input_ids.shape[-1]
                == req.attention_mask.shape[-1]
                == req.position_ids.shape[-1]
                == req.loss_mask.shape[-1]
            ), f"""Request {req.request_id} has different length of
                {req.input_ids.shape[-1]=}, {req.attention_mask.shape[-1]=},
                {req.position_ids.shape[-1]=}, {req.loss_mask.shape[-1]=}"""
            error_message_lines = [
                f"""Request {req.request_id} has input_ids length {req.input_ids.shape[-1]}
                    greater than max_model_len {self.config.max_model_len}""",
                f"Decoded input_ids: {self.processing_class.decode(req.input_ids.squeeze(0).long())}",
                f"Decoded prompt_ids: {self.processing_class.decode(req.prompt_ids.squeeze(0).long())}",
                f"Decoded response_ids: {self.processing_class.decode(req.response_ids.squeeze(0).long())}",
                f"Messages: {req.messages}",
                f"Max model length: {req.max_model_len}",
            ]
            error_message = "\n".join(error_message_lines)
            assert req.input_ids.shape[-1] <= self.config.max_model_len, error_message

            prompt_ids.append(req.prompt_ids.to(tgt_device).squeeze(0))
            response_ids.append(req.response_ids.to(tgt_device).squeeze(0))
            if req.response_ids.shape[-1] > self.config.response_length:
                logger.warning(
                    f"""{req.request_id=} has response_ids length {req.response_ids.shape[-1]}
                    greater than max_response_len {self.config.response_length},\n{req=}"""
                )
            prompt_attention_mask.append(req.prompt_attention_mask.to(tgt_device).squeeze(0))
            response_attention_mask.append(req.response_attention_mask.to(tgt_device).squeeze(0))
            prompt_position_ids.append(req.prompt_position_ids.to(tgt_device).squeeze(0))
            response_position_ids.append(req.response_position_ids.to(tgt_device).squeeze(0))
            prompt_loss_mask.append(req.prompt_loss_mask.to(tgt_device).squeeze(0))
            response_loss_mask.append(req.response_loss_mask.to(tgt_device).squeeze(0))
            messages.append({"messages": req.messages})
            reward_scores.append(req.reward_scores)
            multi_modal_inputs.append(req.multi_modal_inputs)
            request_ids.append(req.request_id)

        prompt_ids = pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        if prompt_ids.shape[-1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[-1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        prompt_attention_mask = pad_sequence(
            prompt_attention_mask,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        if prompt_attention_mask.shape[-1] < self.config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(
                prompt_attention_mask, self.config.prompt_length, 0, left_pad=True
            )
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[-1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)

        # padding prompt_position_ids
        if prompt_position_ids[0].dim() == 2:
            # if prompt_position_ids is a 2D tensor
            # e.g. from qwen2vl, prompt_position_ids.shape = (3, seq_len)
            transposed_prompt_position_ids = [p.transpose(0, 1) for p in prompt_position_ids]
            prompt_position_ids = pad_sequence(
                transposed_prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
            prompt_position_ids = prompt_position_ids.transpose(1, 2)
        else:
            prompt_position_ids = pad_sequence(
                prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
        if prompt_position_ids.shape[-1] < self.config.prompt_length:
            prompt_position_ids = pad_sequence_to_length(
                prompt_position_ids, self.config.prompt_length, 0, left_pad=True
            )

        # padding response_position_ids
        if response_position_ids[0].dim() == 2:
            # if response_position_ids is a 2D tensor
            # e.g. from qwen2vl, response_position_ids.shape = (3, seq_len)
            transposed_response_position_ids = [p.transpose(0, 1) for p in response_position_ids]
            response_position_ids = pad_sequence(
                transposed_response_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
            response_position_ids = response_position_ids.transpose(1, 2)
        else:
            response_position_ids = pad_sequence(response_position_ids, batch_first=True, padding_value=0)
        if response_position_ids.shape[-1] < self.config.response_length:
            response_position_ids = pad_sequence_to_length(response_position_ids, self.config.response_length, 0)

        prompt_loss_mask = pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
        if prompt_loss_mask.shape[1] < self.config.prompt_length:
            prompt_loss_mask = pad_sequence_to_length(prompt_loss_mask, self.config.prompt_length, 0, left_pad=True)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)

        # Construct the batch data
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_loss_mask,
                "input_ids": input_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(sorted_output_req_list),
        )

        # free cache engine
        if self._engine is not None and self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())

        non_tensor_batch = {
            "messages": np.array(messages),
            "reward_scores": np.array(reward_scores),
            "request_id": np.array(request_ids),
        }

        is_multimodal = isinstance(self.processing_class, ProcessorMixin) and (
            hasattr(self.processing_class, "image_processor") or hasattr(self.model_hf_config, "vision_config")
        )

        if is_multimodal:
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

        result = DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
        )

        # Add timing information to meta_info
        result.meta_info.update(timing_info)

        return result
