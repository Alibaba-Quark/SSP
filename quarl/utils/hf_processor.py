import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from transformers.audio_utils import load_audio
from transformers.image_utils import load_image
from transformers.processing_utils import AllKwargsForChatTemplate
from transformers.video_utils import load_video
from typing_extensions import Unpack

logger = logging.getLogger(__name__)


def custom_processor_apply_chat_template(
    self,
    conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
    apply_chat_template_fn=None,  # customized chat template function
    chat_template: Optional[str] = None,
    **kwargs: Unpack[AllKwargsForChatTemplate],
) -> str:
    """
    Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
    conversations to turn them into a single tokenizable string.

    The input is expected to be in the following format, where each message content is a list consisting of text and
    optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
    `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                {"type": "text", "text": "Please describe this image in detail."},
            ],
        },
    ]

    Args:
        conversation (`Union[List[Dict, [str, str]], List[List[Dict[str, str]]]]`):
            The conversation to format.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
            chat template is used.
    """

    if chat_template is None:
        if isinstance(self.chat_template, dict) and "default" in self.chat_template:
            chat_template = self.chat_template["default"]
        elif isinstance(self.chat_template, dict):
            raise ValueError(
                'The processor has multiple chat templates but none of them are named "default". You need to specify'
                " which one to use by passing the `chat_template` argument. Available templates are: "
                f"{', '.join(self.chat_template.keys())}"
            )
        elif self.chat_template is not None:
            chat_template = self.chat_template
        else:
            raise ValueError("Cannot use apply_chat_template because this processor does not have a chat template.")
    else:
        if isinstance(self.chat_template, dict) and chat_template in self.chat_template:
            # It's the name of a template, not a full template string
            chat_template = self.chat_template[chat_template]
        else:
            # It's a template string, render it directly
            chat_template = chat_template

    if kwargs.get("continue_final_message", False):
        if kwargs.get("add_generation_prompt", False):
            raise ValueError(
                "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
            )
        if kwargs.get("return_assistant_tokens_mask", False):
            raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

    # Fill sets of kwargs that should be used by different parts of template
    processed_kwargs = {
        "mm_load_kwargs": {},
        "template_kwargs": {},
    }

    for kwarg_type in processed_kwargs:
        for key in AllKwargsForChatTemplate.__annotations__[kwarg_type].__annotations__.keys():
            kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__[kwarg_type]
            default_value = getattr(kwarg_type_defaults, key, None)
            value = kwargs.pop(key, default_value)
            if value is not None and not isinstance(value, dict):
                processed_kwargs[kwarg_type][key] = value

    # Pass unprocessed custom kwargs
    processed_kwargs["template_kwargs"].update(kwargs)

    if isinstance(conversation, (list, tuple)) and (
        isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
    ):
        is_batched = True
        conversations = conversation
    else:
        is_batched = False
        conversations = [conversation]

    tokenize = processed_kwargs["template_kwargs"].pop("tokenize", False)
    return_dict = processed_kwargs["template_kwargs"].pop("return_dict", False)
    mm_load_kwargs = processed_kwargs["mm_load_kwargs"]

    if tokenize:
        batch_images, batch_videos = [], []
        batch_audios = []
        batch_video_metadata = []
        for conversation in conversations:
            images, videos = [], []
            video_metadata = []
            for message in conversation:
                visuals = [content for content in message["content"] if content["type"] in ["image", "video"]]
                audio_fnames = [
                    content[key]
                    for content in message["content"]
                    for key in ["audio", "url", "path"]
                    if key in content and content["type"] == "audio"
                ]
                image_fnames = [
                    vision_info[key]
                    for vision_info in visuals
                    for key in ["image", "url", "path", "base64"]
                    if key in vision_info and vision_info["type"] == "image"
                ]
                video_fnames = [
                    vision_info[key]
                    for vision_info in visuals
                    for key in ["video", "url", "path"]
                    if key in vision_info and vision_info["type"] == "video"
                ]

                for fname in image_fnames:
                    images.append(load_image(fname))

                # Audio models do not accept nested list of audios (yet!) so we construct a flat input audio list
                if not mm_load_kwargs["load_audio_from_video"]:
                    for fname in audio_fnames:
                        batch_audios.append(load_audio(fname, sampling_rate=mm_load_kwargs["sampling_rate"]))
                else:
                    for fname in video_fnames:
                        batch_audios.append(load_audio(fname, sampling_rate=mm_load_kwargs["sampling_rate"]))

                for fname in video_fnames:
                    if isinstance(fname, (list, tuple)) and isinstance(fname[0], str):
                        video = [np.array(load_image(image_fname)) for image_fname in fname]
                        # create a 4D video because `load_video` always returns a 4D array
                        video = np.stack(video)
                        metadata = None
                        logger.warning(
                            "When loading the video from list of images, we cannot infer metadata such as `fps` or `duration`. "
                            "If your model uses this metadata during processing, please load the whole video and let the model sample frames instead."
                        )
                    else:
                        # TODO: raushan, should be `self.video_processor.load_video_for_model` when API is added
                        video, metadata = self._load_video_for_model(
                            fname,
                            num_frames=mm_load_kwargs.get("num_frames", None),
                            fps=mm_load_kwargs.get("video_fps", None),
                            backend=mm_load_kwargs["video_load_backend"],
                            **kwargs,
                        )
                        videos.append(video)
                        video_metadata.append(metadata)

            # Currently all processors can accept nested list of batches, but not flat list of visuals
            # So we'll make a batched list of images and let the processor handle it
            if images:
                batch_images.append(images)
            if videos:
                batch_videos.append(videos)
                batch_video_metadata.append(video_metadata)

        # Process conversation with video/image information if needed. Then convert into a prompt using Jinja template
        conversations = self._process_messages_for_chat_template(
            conversations,
            batch_images=batch_images,
            batch_videos=batch_videos,
            batch_video_metadata=batch_video_metadata,
            **processed_kwargs["mm_load_kwargs"],
        )

    # prompt, generation_indices = render_jinja_template(
    #     conversations=conversations,
    #     chat_template=chat_template,
    #     **processed_kwargs["template_kwargs"],  # different flags such as `return_assistant_mask`
    #     **self.tokenizer.special_tokens_map,  # tokenizer special tokens are used by some templates
    # )
    prompt = [
        apply_chat_template_fn(
            messages=conversation,
            **processed_kwargs["template_kwargs"],  # different flags such as `return_assistant_mask`
            **self.tokenizer.special_tokens_map,  # tokenizer special tokens are used by some templates
        )
        for conversation in conversations
    ]

    if not is_batched:
        prompt = prompt[0]

    if tokenize:
        # Tokenizer's `apply_chat_template` never adds special tokens when tokenizing
        # But processor's `apply_chat_template` didn't have an option to tokenize, so users had to format the prompt
        # and pass it to the processor. Users thus never worried about special tokens relying on processor handling
        # everything internally. The below line is to keep BC for that and be able to work with model that have
        # special tokens in the template (consistent with tokenizers). We dont want to raise warning, it will flood command line
        # without actionable solution for users
        single_prompt = prompt[0] if is_batched else prompt
        if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
            kwargs["add_special_tokens"] = False

        out = self(
            text=prompt,
            images=batch_images if batch_images else None,
            videos=batch_videos if batch_videos else None,
            audio=batch_audios if batch_audios else None,
            **kwargs,
        )
        if return_dict:
            if processed_kwargs["template_kwargs"].get("return_assistant_tokens_mask", False):
                raise NotImplementedError
                # assistant_masks = []
                # input_ids = out["input_ids"]
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
                #             out["assistant_masks"] = assistant_masks
                #             out.convert_to_tensors(tensor_type=kwargs.get("return_tensors", None))
            return out
        else:
            return out["input_ids"]
    return prompt
