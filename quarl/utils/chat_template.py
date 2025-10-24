import json
import os

from pydantic import BaseModel


class QuarkChatTemplate:

    TEMPLATES = {
        "default": {
            "system_prefix": "<|im_start|>system\n",
            "assist_prefix": "<|im_start|>assistant\n",
            "user_prefix": "<|im_start|>user\n",
            "tool_prefix": "<|im_start|>function\n",
            "bos": "",
            "eos": "<|im_end|>",
            "think_prefix": "<think>",
            "think_suffix": "</think>",
            "search_prefix": "<search>",
            "search_suffix": "</search>",
            "answer_prefix": "<answer>",
            "answer_suffix": "</answer>",
            "search_result_prefix": "\n<information>\n",
            "search_result_suffix": "</information>\n\n",
            "add_think_after_tool": True,  # whether to add think_prefix after tool call
        },
        "qwen2p5": {
            "system_prefix": "<|im_start|>system\n",
            "assist_prefix": "<|im_start|>assistant\n",
            "user_prefix": "<|im_start|>user\n",
            "tool_prefix": "<|im_start|>function\n",
            "bos": "",
            "eos": "<|im_end|>",
            "think_prefix": "<think>",
            "think_suffix": "</think>",
            "search_prefix": "<search>",
            "search_suffix": "</search>",
            "answer_prefix": "<answer>",
            "answer_suffix": "</answer>",
            "search_result_prefix": "\n<information>\n",
            "search_result_suffix": "</information>\n\n",
            "add_think_after_tool": True,  
        },
        "R-Search": {
            "system_prefix": "<|im_start|>system\n",
            "assist_prefix": "<|im_start|>assistant\n",
            "user_prefix": "<|im_start|>user\n",
            "tool_prefix": "<|im_start|>function\n",
            "bos": "",
            "eos": "<|im_end|>",
            "think_prefix": "<think>",
            "think_suffix": "</think>",
            "search_prefix": "<search>",
            "search_suffix": "</search>",
            "answer_prefix": "<answer>",
            "answer_suffix": "</answer>",
            "search_result_prefix": "\n<observation>",
            "search_result_suffix": "</observation>\n",
            "add_think_after_tool": True,  
        },
        "llama3p1": {
            "system_prefix": "<|start_header_id|>system<|end_header_id|>\n\n",
            "assist_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "user_prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
            "tool_prefix": "<|start_header_id|>tool<|end_header_id|>\n\n",
            "bos": "<|begin_of_text|>",
            "eos": "<|eot_id|>",
            "think_prefix": "<think>",
            "think_suffix": "</think>",
            "search_prefix": "<search>",
            "search_suffix": "</search>",
            "answer_prefix": "<answer>",
            "answer_suffix": "</answer>",
            "search_result_prefix": "\n<information>\n",
            "search_result_suffix": "</information>\n\n",
            "add_think_after_tool": True,  
        },
    }

    # The current template type is obtained from the environment variable CHAT_TEMPLATE
    _current_template = os.getenv("QUARK_SEARCH_CHAT_TEMPLATE", "default")

    @classmethod
    def set_template(cls, template_type="default"):
        if template_type in cls.TEMPLATES:
            cls._current_template = template_type
            cls._update_class_attributes()
            print(f"[QuarkChatTemplate] Switched to template: {template_type}")
            return True
        else:
            print(
                f"[QuarkChatTemplate] Template '{template_type}' not found. Available templates: {list(cls.TEMPLATES.keys())}"
            )
            return False

    @classmethod
    def _update_class_attributes(cls):
        config = cls.TEMPLATES[cls._current_template]
        cls.system_prefix = config["system_prefix"]
        cls.assist_prefix = config["assist_prefix"]
        cls.user_prefix = config["user_prefix"]
        cls.tool_prefix = config["tool_prefix"]
        cls.bos = config["bos"]
        cls.eos = config["eos"]
        cls.think_prefix = config["think_prefix"]
        cls.think_suffix = config["think_suffix"]
        cls.search_prefix = config["search_prefix"]
        cls.search_suffix = config["search_suffix"]
        cls.answer_prefix = config["answer_prefix"]
        cls.answer_suffix = config["answer_suffix"]
        cls.search_result_prefix = config["search_result_prefix"]
        cls.search_result_suffix = config["search_result_suffix"]
        cls.add_think_after_tool = config["add_think_after_tool"]

    @classmethod
    def get_available_templates(cls):
        return list(cls.TEMPLATES.keys())

    @classmethod
    def get_current_template(cls):
        return cls._current_template

    @classmethod
    def apply_chat_template(cls, messages, tools=None, add_generation_prompt=False, **kwargs):
        prompt = cls.bos
        messages = [message.model_dump() if isinstance(message, BaseModel) else message for message in messages]

        for i, msg in enumerate(messages):

            if msg["role"] == "system":
                prompt += cls.system_prefix + msg["content"]

            elif msg["role"] == "function":
                prompt += cls.tool_prefix + msg["content"]

            elif msg["role"] == "user":
                prompt += cls.user_prefix + msg["content"]

            elif msg["role"] == "assistant":
                if i - 1 >= 0 and not messages[i - 1]["role"] == "user":
                    prompt += msg["content"]
                else:
                    prompt += cls.assist_prefix + msg["content"]

                if isinstance(msg.get("tool_calls", None), list) and len(msg["tool_calls"]) > 0:
                    search_calls = [
                        tool_call
                        for tool_call in msg["tool_calls"]
                        if tool_call["function"].get("name", None) == "search"
                    ]
                    # non_search_calls = [
                    #     tool_call for tool_call in msg["tool_calls"] if not tool_call['function'].get("name", None) == "search"
                    # ]
                    for search_call in search_calls:
                        search_query = search_call["function"]["arguments"].get("query_list", "")
                        if isinstance(search_query, list):
                            search_query = "\n".join(search_query)

                        prompt += cls.search_prefix + str(search_query) + cls.search_suffix
                    # ToDO: non_search_calls not implemented

            elif msg["role"] == "tool":
                result = json.loads(msg["content"])
                prompt += cls.search_result_prefix + result.get("result", "") + cls.search_result_suffix

            else:
                raise ValueError("no such role!!!", msg["role"])
        if add_generation_prompt and messages[-1]["role"] == "user":
            prompt = prompt + cls.assist_prefix

        return prompt

QuarkChatTemplate._update_class_attributes()


class QwenChatTemplate(QuarkChatTemplate):
    system_prefix = "<｜System｜>"
    assist_prefix = "<｜Assistant｜>"
    user_prefix = "<｜User｜>"
    tool_prefix = "<｜Function｜>"
    bos = "<｜begin▁of▁sentence｜>"
    eos = "<｜end▁of▁sentence｜>"
    think_prefix = "<think>"
    think_suffix = "</think>"
    search_prefix = "<search>"
    search_suffix = "</search>"
    answer_prefix = "<answer>"
    answer_suffix = "</answer>"
    search_result_prefix = "\n<information>\n"
    search_result_suffix = "</information>\n\n"
