from functools import partial
from types import MethodType

from transformers import PreTrainedTokenizerBase, ProcessorMixin

from quarl.utils.hf_processor import custom_processor_apply_chat_template
from quarl.utils.hf_tokenizer import custom_tokenizer_apply_chat_template


def switch_apply_chat_template(instance, apply_chat_template_fn):
    if isinstance(instance, ProcessorMixin):
        _custom_apply_chat_template = custom_processor_apply_chat_template
    elif isinstance(instance, PreTrainedTokenizerBase):
        _custom_apply_chat_template = custom_tokenizer_apply_chat_template
    elif instance is None:
        return None
    else:
        raise NotImplementedError

    instance.apply_chat_template = MethodType(
        partial(_custom_apply_chat_template, apply_chat_template_fn=apply_chat_template_fn), instance
    )
    return instance
