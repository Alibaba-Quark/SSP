import os
import random
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor

from quarl.utils.llm_as_a_judge import get_global_judge


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    # If there are 0  matches, return None
    if len(matches) < 1:
        return None
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_math_solution(solution_str):
    """Extract the answer from the solution string within <answer></answer> tags,
    returning boxed content if present, otherwise the raw content.
    If no <answer> tags are present, extract and return the last boxed content."""
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if len(answer_matches) < 1:
        
        boxed_pattern = r"\\boxed\{(.*?)\}"
        boxed_matches = list(re.finditer(boxed_pattern, solution_str, re.DOTALL))
        if boxed_matches:
            return boxed_matches[-1].group(1).strip()
        else:
            return None

    answer_content = answer_matches[-1].group(1).strip()

    boxed_pattern = r"\\boxed\{(.*?)\}"
    boxed_match = re.search(boxed_pattern, answer_content)

    if boxed_match:
        return boxed_match.group(1).strip()
    else:
        return answer_content


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    prompt_str=None,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    if not os.getenv("QUARK_BASE_URL") or not os.getenv("QUARK_MODEL"):
        raise ValueError("QUARK_BASE_URL or QUARK_MODEL is not set")

    judge = get_global_judge(
        base_url=os.getenv("QUARK_BASE_URL"), api_key="dummy_api_key", model=os.getenv("QUARK_MODEL")
    )
    if data_source is not None and data_source in [
        "Algebra",
        "Intermediate Algebra",
        "Prealgebra",
        "Number Theory",
        "Geometry",
        "Precalculus",
        "Counting & Probability",
    ]:
        answer = extract_math_solution(solution_str=solution_str)
    else:
        answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    score = 1.0

    if answer is None:  
        print(f"[QuarkScore]: No answer extracted, data_source: {data_source}")
        return 0.0
    elif em_check(answer, ground_truth["target"]):  
        print(
            f"[QuarkScore]: Answer is correct by em_check, answer: {answer}, ground_truth: {ground_truth['target']}, data_source: {data_source}"
        )
        if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
            score = score / 4
            return score
        return score
    else:
        question = ""
        if prompt_str:
            if "Question: " in prompt_str:
                question = prompt_str.split("Question: ", 1)[1]
            elif "<｜User｜>" in prompt_str:
                question = prompt_str.replace("<｜User｜>", "")
            else:
                question = prompt_str
        import time

        start_time = time.time()
        if judge.model_based_match(question=question, golden_answer=ground_truth["target"], model_answer=answer):
            print(
                f"[QuarkScore]: Answer is correct by judge.model_based_match, answer: {answer}, ground_truth: {ground_truth['target']}, time: {time.time() - start_time}s, data_source: {data_source}"
            )
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                return score
            return score
        else:
            # print(f"[QuarkScore]: Answer is wrong, answer: {answer}, ground_truth: {ground_truth['target']}")
            return 0


def compute_score_batch(batch_data, score_coef: float = 1.0):
    results = [None] * len(batch_data)

    def worker(batch_idx, data):
        solution_str = data["response_str"]
        ground_truth = data["ground_truth"]
        score = compute_score(
            data_source=data.get("data_source", ""),
            solution_str=solution_str,
            ground_truth=ground_truth,
            prompt_str=data.get("prompt_str", None),
            extra_info=data.get("extra_info", {}),
            sandbox_fusion_url=data.get("sandbox_fusion_url", None),
            concurrent_semaphore=data.get("concurrent_semaphore", None),
        )
        results[batch_idx] = {"score": score * score_coef, "idx": data["idx"]}

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=50) as executor:
        for batch_idx, data in enumerate(batch_data):
            executor.submit(worker, batch_idx, data)
    print(f"[QuarkScore]: compute_score_batch time: {time.time() - start_time}s")
    return results
