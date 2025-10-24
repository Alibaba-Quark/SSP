system_prompt = """You are an expert question creator. Your primary task is to reverse-engineer a challenging question from a given answer. The question you create must require a solver to perform 'n' sequential searches to solve it. I will provide you with the target answer and the required number of searches, 'n'.

Your Creation Process & Tools:
1. Analyze Scope and Target: Begin by analyzing the provided 'Answer' (your target) and the required number of searches, 'n' (the path's length). This establishes your final destination and the complexity of the logical chain you need to construct.
2. Build the Question by Working Backwards:
    This is the core of the process. You will start from the destination and work your way back to the starting point, step by step.
    2.1. The Crucial First Step: Connection and Discovery
        Start with the final 'Answer', but do not search for the answer itself directly.
        Instead, first analyze the 'Answer'. Brainstorm and identify a closely related yet distinct 'Associated Concept' (e.g., a related historical event, a key figure, a geographic location, a unique attribute, its parent category, etc.).
        Perform an exploratory search with the goal of finding the 'bridging information' that connects this 'Associated Concept' to your final 'Answer'.
        From your search results, extract a unique, verifiable 'preceding fact'. This is a piece of information that, when searched, would logically lead a user to your final 'Answer'. This 'preceding fact' becomes the answer to Search #n-1.
    2.2. Iterate Backwards
        Now, treat this newly found 'preceding fact' as your new target.
        From this point on, you can search for this new target directly to find the preceding piece of information that leads to it. This becomes the answer to the next search in the backward chain (Search #n-2).
    2.3. Construct the Full Chain
        Continuously repeat the iterative process from Step 2.2, using each new fact as the target for the next backward search, until you have constructed a complete logical chain of 'n' links.
        The very first piece of information you uncover in this process (the one at the start of the chain) will become the initial clue for your question.
3. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <question> and </question> without detailed illustrations. For example, <question> xxx </question>.

Here are three example questions:
{}

Critical Rules:
1. Strictly Fact-Based: You must not create questions based on assumptions. The entire logical path to the solution must be grounded in the information you find through searching.
2. No Spoilers: The question must not contain any direct clues that reveal the answer or the intermediate steps.
3. Search is Mandatory: The question must be impossible to answer from general knowledge alone. It must necessitate the search process you have designed.
4. Adhere to Search Count: The number of searches required to solve the question must precisely match the specified 'search count'.
5. Unique Answer: The designed question must be deterministic, leading to a single, unambiguous final answer. The clues at each step must be precise enough to prevent a solver from reasonably arriving at a different, valid conclusion."""

user_prompt = """The answer I provided is: {}
You need to create a question that requires {} searches.
When you have enough information to construct a question, please first check whether the constructed question meets all requirements, especially whether the question is too simple. After checking that all conditions are met, you need to provide the final constructed question in your final response, placing the final question between <question> and </question> tags, for example: <question> During the period from the founding of Tsinghua University to the establishment of Hupan University, which surname was most common among the successive presidents of Central University of Finance and Economics? </question>."""

data = {
    "data_source": "quark_selfplay_en",
    "prompt": [{"content": "", "role": "system"}, {"content": "", "role": "user"}],
    "ability": "fact-reasoning",
    "reward_model": {"ground_truth": {"target": ["dummy"]}, "style": "rule"},
    "extra_info": {
        "index": 0,
        "need_tools_kwargs": True,
        "question": "dummy",
        "split": "train",
        "tools_kwargs": {
            "search": {
                "create_kwargs": {
                    "data_source": "quark_selfplay_en",
                    "ground_truth": {"target": ["dummy"]},
                    "question": "dummy",
                }
            }
        },
    },
    "metadata": None,
}

if __name__ == "__main__":
    """Read the source jsonl (train_answers_random_hop_50000_proposer.jsonl),
    populate `data['prompt']` and `data['extra_info']['index']` for each line,
    and write a new processed jsonl file next to the source.

    Filling rules:
    - `prompt[0]['content']` <- the `system_prompt` string formatted with the
       line's `sys_question_example` field
    - `prompt[1]['content']` <- the `user_prompt` string formatted with
       the line's `ground_truth` and `search_turns` fields
    - `extra_info.index` <- zero-based line index
    Additionally update `reward_model.ground_truth.target` to include the
    line's `ground_truth` for easier downstream use.
    """

    import argparse
    import copy
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Preprocess training jsonl into prompts")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="train_answers_random_hop_50000_proposer.jsonl",
        help="Input jsonl filename (in the same directory as this script)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output jsonl filename (defaults to input filename with _processed suffix)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    src = (Path(args.input) if Path(args.input).is_absolute() else script_dir / args.input).resolve()
    if args.output:
        out = (Path(args.output) if Path(args.output).is_absolute() else script_dir / args.output).resolve()
    else:
        out = src.with_name(src.stem + "_processed" + src.suffix)

    if not src.exists():
        raise FileNotFoundError(f"source file not found: {src}")

    with src.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as e:
                # skip malformed lines but continue
                print(f"skipping line {idx} due to parse error: {e}")
                continue

            processed = copy.deepcopy(data)

            # fill system prompt
            sys_example = item.get("sys_question_example", "")
            sys_text = system_prompt.format(sys_example)
            processed["prompt"][0]["content"] = sys_text

            # prepare user prompt using provided template
            gt = item.get("ground_truth", "")
            turns = item.get("search_turns", "")
            # ensure string formatting (user_prompt expects two placeholders)
            user_text = user_prompt.format(gt, turns)

            processed["prompt"][1]["content"] = user_text

            # set index
            processed["extra_info"]["index"] = idx

            # update reward_model ground truth target for convenience
            try:
                processed["reward_model"]["ground_truth"]["target"] = [gt]
                processed["extra_info"]["tools_kwargs"]["search"]["create_kwargs"]["ground_truth"]["target"] = [gt]
            except Exception:
                pass

            # write as single-line json (utf-8)
            fout.write(json.dumps(processed, ensure_ascii=False) + "\n")

    print(f"Wrote processed records to: {out}")
