from verl.utils.reward_score import search_r1_like_qa_em


def compute_score(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


def compute_score_batch(batch_data, score_coef: float = 1.0):
    results = []
    for batch_idx, data in enumerate(batch_data):
        solution_str = data["response_str"]
        ground_truth = data["ground_truth"]
        score = compute_score(
            data_source=data.get("data_source", ""),
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=data.get("extra_info", {}),
            sandbox_fusion_url=data.get("sandbox_fusion_url", None),
            concurrent_semaphore=data.get("concurrent_semaphore", None),
        )
        results.append({"score": score * score_coef, "idx": batch_data[batch_idx]["idx"]})
    return results
