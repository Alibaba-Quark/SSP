from typing import Any, Callable, Dict, List, Optional, Union

from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import PaddingStrategy, TensorType


def custom_tokenizer_apply_chat_template(
    self,
    conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
    apply_chat_template_fn=None,  # customized chat template function
    tools: Optional[List[Union[Dict, Callable]]] = None,
    documents: Optional[List[Dict[str, str]]] = None,
    chat_template: Optional[str] = None,
    add_generation_prompt: bool = False,
    continue_final_message: bool = False,
    tokenize: bool = True,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: bool = False,
    max_length: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_dict: bool = False,
    return_assistant_tokens_mask: bool = False,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Union[str, List[int], List[str], List[List[int]], BatchEncoding]:
    """
    Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
    ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to
    determine the format and control tokens to use when converting.

    Args:
        conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]): A list of dicts
            with "role" and "content" keys, representing the chat history so far.
        tools (`List[Union[Dict, Callable]]`, *optional*):
            A list of tools (callable functions) that will be accessible to the model. If the template does not
            support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
            giving the name, description and argument types for the tool. See our
            [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
            for more information.
        documents (`List[Dict[str, str]]`, *optional*):
            A list of dicts representing documents that will be accessible to the model if it is performing RAG
            (retrieval-augmented generation). If the template does not support RAG, this argument will have no
            effect. We recommend that each document should be a dict containing "title" and "text" keys. Please
            see the RAG section of the [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG)
            for examples of passing documents with chat templates.
        chat_template (`str`, *optional*):
            A Jinja template to use for this conversion. It is usually not necessary to pass anything to this
            argument, as the model's template will be used by default.
        add_generation_prompt (bool, *optional*):
            If this is set, a prompt with the token(s) that indicate
            the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
            Note that this argument will be passed to the chat template, and so it must be supported in the
            template for this argument to have any effect.
        continue_final_message (bool, *optional*):
            If this is set, the chat will be formatted so that the final
            message in the chat is open-ended, without any EOS tokens. The model will continue this message
            rather than starting a new one. This allows you to "prefill" part of
            the model's response for it. Cannot be used at the same time as `add_generation_prompt`.
        tokenize (`bool`, defaults to `True`):
            Whether to tokenize the output. If `False`, the output will be a string.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
             Select a strategy to pad the returned sequences (according to the model's padding side and padding
             index) among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        truncation (`bool`, defaults to `False`):
            Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
        max_length (`int`, *optional*):
            Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
            not specified, the tokenizer's `max_length` attribute will be used as a default.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
            values are:
            - `'tf'`: Return TensorFlow `tf.Tensor` objects.
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
            - `'jax'`: Return JAX `jnp.ndarray` objects.
        return_dict (`bool`, defaults to `False`):
            Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
        tokenizer_kwargs (`Dict[str: Any]`, *optional*): Additional kwargs to pass to the tokenizer.
        return_assistant_tokens_mask (`bool`, defaults to `False`):
            Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
            the mask will contain 1. For user and system tokens, the mask will contain 0.
            This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
        **kwargs: Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

    Returns:
        `Union[List[int], Dict]`: A list of token ids representing the tokenized chat so far, including control tokens. This
        output is ready to pass to the model, either directly or via methods like `generate()`. If `return_dict` is
        set, will return a dict of tokenizer outputs instead.
    """

    if return_dict and not tokenize:
        raise ValueError(
            "`return_dict=True` is incompatible with `tokenize=False`, because there is no dict "
            "of tokenizer outputs to return."
        )

    if return_assistant_tokens_mask and not return_dict:
        raise ValueError("`return_assistant_tokens_mask=True` is incompatible with `return_dict=False`")

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    if isinstance(conversation, (list, tuple)) and (
        isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
    ):
        conversations = conversation
        is_batched = True
    else:
        conversations = [conversation]
        is_batched = False

    if continue_final_message:
        if add_generation_prompt:
            raise ValueError(
                "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
            )
        if return_assistant_tokens_mask:
            raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

    template_kwargs = {**self.special_tokens_map, **kwargs}  # kwargs overwrite special tokens if both are present

    # chat_template = self.get_chat_template(chat_template, tools)
    # rendered_chat, generation_indices = render_jinja_template(
    #     conversations=conversations,
    #     tools=tools,
    #     documents=documents,
    #     chat_template=chat_template,
    #     return_assistant_tokens_mask=return_assistant_tokens_mask,
    #     continue_final_message=continue_final_message,
    #     add_generation_prompt=add_generation_prompt,
    #     **template_kwargs,
    # )

    rendered_chat = [
        apply_chat_template_fn(
            messages=conversation, add_generation_prompt=add_generation_prompt, tools=tools, **template_kwargs
        )
        for conversation in conversations
    ]
    # generation_indices = None

    if not is_batched:
        rendered_chat = rendered_chat[0]

    if tokenize:
        out = self(
            rendered_chat,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=False,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )
        if return_dict:
            if return_assistant_tokens_mask:
                raise NotImplementedError
            # assistant_masks = []
            # if is_batched or return_tensors:
            #     input_ids = out["input_ids"]
            # else:
            #     input_ids = [out["input_ids"]]
            # for i in range(len(input_ids)):
            #     current_mask = [0] * len(input_ids[i])
            #     for assistant_start_char, assistant_end_char in generation_indices[i]:
            #         start_token = out.char_to_token(i, assistant_start_char)
            #         end_token = out.char_to_token(i, assistant_end_char - 1)
            #         if start_token is None:
            #             # start_token is out of bounds maybe due to truncation.
            #             break
            #         for token_id in range(start_token, end_token + 1 if end_token else len(input_ids[i])):
            #             current_mask[token_id] = 1
            #             assistant_masks.append(current_mask)

            # if not is_batched and not return_tensors:
            #     assistant_masks = assistant_masks[0]

            # out["assistant_masks"] = assistant_masks

            # if return_tensors:
            #     out.convert_to_tensors(tensor_type=return_tensors)

            return out
        else:
            return out["input_ids"]
    else:
        return rendered_chat
