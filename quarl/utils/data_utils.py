import copy
import json
import re


def remove_prefix(content, prefix):
    for i in reversed(range(min(len(content), len(prefix)))):
        if content[: i + 1] == prefix[-(i + 1) :]:
            content = content[i + 1 :]
            break
    return content


def extract_thought_and_answer(text):
    pattern = r"<think>\n(.*?)\n</think>(.*)"

    if text.count("<think>") == 1 and text.count("</think>") == 1:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            thought_content = match.group(1)
            answer_content = match.group(2)
            return [thought_content, answer_content]
        else:
            return None
    else:
        return None
