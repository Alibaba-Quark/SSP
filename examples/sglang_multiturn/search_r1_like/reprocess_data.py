import argparse

import pandas as pd

R_SEARCH_SYSTEM_CONTENT = """You are a helpful assistant that can solve the given question step by step. For each step, start by explaining your thought process. If additional information is needed, provide a specific query enclosed in <search> and </search>. The system will return the top search results within <observation> and </observation>. You can perform multiple searches as needed. When you know the final answer, use <original_evidence> and </original_evidence> to provide all potentially relevant original information from the observations. Ensure the information is complete and preserves the original wording without modification. If no searches were conducted or observations were made, omit the evidence section. Finally, provide the final answer within <answer> and </answer> tags.
"""


def extract_question(text):
    """Extract the question from the text after 'Question: '."""
    question_start = text.find("Question: ")
    if question_start == -1:
        return text.strip()
    question_start += len("Question: ")
    return text[question_start:].strip()


def process_prompt(row, chat_template):
    prompt = row["prompt"]
    new_prompt = []

    for d in prompt:
        role = d.get("role")
        if chat_template == "R-Search":
            if role == "system":
                d2 = d.copy()
                d2["content"] = R_SEARCH_SYSTEM_CONTENT
                new_prompt.append(d2)
            elif role == "user":
                d2 = d.copy()
                d2["content"] = extract_question(d["content"])
                new_prompt.append(d2)
            else:
                new_prompt.append(d)
        else:
            new_prompt.append(d)

    row["prompt"] = new_prompt
    return row


def process_parquet(input_file, output_file, chat_template):
    df = pd.read_parquet(input_file)
    df = df.apply(lambda row: process_prompt(row, chat_template), axis=1)
    df.to_parquet(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reprocess search_r1_like parquet prompts with different chat templates."
    )
    parser.add_argument("-i", "--input", default="input.parquet", help="Input parquet file path")
    parser.add_argument("-o", "--output", default="output.parquet", help="Output parquet file path")
    parser.add_argument(
        "-t",
        "--template",
        default="R-Search",
        help="Chat template to apply (e.g. R-Search)",
    )
    args = parser.parse_args()

    process_parquet(args.input, args.output, args.template)
