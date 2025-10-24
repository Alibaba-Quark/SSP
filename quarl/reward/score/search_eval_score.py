from collections import Counter

from quarl.reward.score.qa_em import compute_score as qa_em_fn
from quarl.reward.score.quark_score import compute_score as llm_as_judge_fn
from quarl.reward.score.quark_score import extract_solution, normalize_answer


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def F1_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    score = 0.0
    for ground_truth in golden_answers:
        score = max(score, qa_f1_score(prediction, ground_truth))
    return round(score, 2)


def f1_fn(solution_str, ground_truth):
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0.0
    else:
        return F1_check(answer, ground_truth["target"])


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    prompt_str=None,
    extra_info=None,
):
    return {
        "score": llm_as_judge_fn(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            prompt_str=prompt_str,
            extra_info=extra_info,
        ),
        "EM": qa_em_fn(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        ),
        "F1": f1_fn(
            solution_str=solution_str,
            ground_truth=ground_truth,
        ),
    }
