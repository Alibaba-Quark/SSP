"""
Problem extraction utilities for self-play training.

This module provides utilities to extract search-based problems from
proposer generation trajectories and format them for the solver phase.
"""

import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from quarl.reward.score.quark_score import em_check
from quarl.utils.llm_as_a_judge import get_global_judge

logger = logging.getLogger(__name__)

# Enable debug logging for self-play
SELF_PLAY_DEBUG = os.environ.get("SELF_PLAY_DEBUG", "False").lower() == "true"
if SELF_PLAY_DEBUG:
    logger.setLevel(logging.DEBUG)


class ProblemExtractor:
    """
    Extracts search-based problems from proposer generation trajectories.

    This extractor specifically handles problems that require search capabilities,
    extracting questions from <answer></answer> tags and formatting them for solver phase.
    """

    def __init__(
        self,
        lang="zh",
        use_rag_filter=False,
        use_search_terms_filter=False,
        noisy_rag_materials=0,
        answer_pattern="answer",
    ):
        """Initialize the ProblemExtractor for search-based problems."""
        if answer_pattern == "question":
            self.answer_pattern = r"<question>\s*(.*?)\s*</question>"
        else:
            self.answer_pattern = r"<answer>\s*(.*?)\s*</answer>"

        self.information_pattern = r"<information>\s*(.*?)\s*</information>"

        self.llm_judge = get_global_judge(
            base_url=os.getenv("QUARK_BASE_URL"), api_key="dummy_api_key", model=os.getenv("QUARK_MODEL")
        )

        self.use_rag_filter = use_rag_filter

        self.use_search_terms_filter = use_search_terms_filter

        self.noisy_rag_materials = noisy_rag_materials

        if lang == "R-Search":
            self.solver_system_prompt = """You are a helpful assistant that can solve the given question step by step. For each step, start by explaining your thought process. If additional information is needed, provide a specific query enclosed in <search> and </search>. The system will return the top search results within <observation> and </observation>. You can perform multiple searches as needed. When you know the final answer, use <original_evidence> and </original_evidence> to provide all potentially relevant original information from the observations. Ensure the information is complete and preserves the original wording without modification. If no searches were conducted or observations were made, omit the evidence section. Finally, provide the final answer within <answer> and </answer> tags."""
            self.solver_user_prompt = """{}"""
        else:
            self.solver_system_prompt = """You are a helpful and harmless assistant."""
            self.solver_user_prompt = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> xxx </answer>. Question: {}"""

    def extract_problems_from_trajectory_with_stats(
        self, input_text: str, output_text: str, metadata: Dict[str, Any] = None, batch_materials: List[str] = None
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Extract search-based problems from proposer generation trajectory with detailed statistics.

        Args:
            input_text: The input prompt used for proposer generation
            output_text: The proposer's generated output containing <answer></answer>
            metadata: Additional metadata about the generation
            batch_materials: List of materials from other trajectories in the batch

        Returns:
            Tuple of (extracted problems list, trajectory statistics dict)
        """
        extracted_problems = []
        metadata = metadata or {}

        traj_stats = {
            "answer_matches": 0,
            "format_error": 0,
            "valid_questions": 0,
            "successful_problems": 0,
        }

        if SELF_PLAY_DEBUG:
            logger.debug(f"Extracting problems from trajectory {metadata.get('trajectory_index', 'unknown')}")

        try:
            answer_matches = re.finditer(self.answer_pattern, output_text, re.DOTALL)

            match_list = list(answer_matches)
            traj_stats["answer_matches"] = len(match_list)

            if len(match_list) == 0:
                traj_stats["format_error"] += 1
                return extracted_problems, traj_stats

            if SELF_PLAY_DEBUG:
                logger.debug(f"Found {len(match_list)} <answer></answer> matches")

            for match_idx, match in enumerate(match_list[:1]):
                question_text = match.group(1).strip()

                question_text = re.sub(r"<\\?/?answer\s*>", "", question_text)
                question_text = re.sub(r"</\s*$", "", question_text)
                question_text = re.sub(r"</answer\s*$", "", question_text)

                if self._is_valid_search_question(question_text, output_text, metadata, batch_materials):
                    traj_stats["valid_questions"] += 1

                    formatted_problem = self._format_search_problem(question_text, input_text, output_text, metadata)
                    if formatted_problem:
                        extracted_problems.append(formatted_problem)
                        traj_stats["successful_problems"] += 1

        except Exception as e:
            logger.warning(f"Error extracting search problems with stats: {e}")

        return extracted_problems, traj_stats

    def _extract_information_content(self, output_text: str) -> str:
        """Extract all content from <information></information> tags in the trajectory."""
        information_matches = re.findall(self.information_pattern, output_text, re.DOTALL)
        if information_matches:
            return "\n\n".join(match.strip() for match in information_matches)
        return None

    def _extract_search_terms(self, output_text: str) -> List[str]:
        """Extract search terms from <search></search> tags in the output text."""
        search_pattern = r"<search>\s*(.*?)\s*</search>"
        search_matches = re.findall(search_pattern, output_text, re.DOTALL)

        search_terms = []
        for match in search_matches:
            try:
                terms = json.loads(match)
                if isinstance(terms, list):
                    search_terms.extend([term.strip() for term in terms if term.strip()])
                else:
                    search_terms.append(match.strip())
            except (json.JSONDecodeError, ValueError):
                search_terms.append(match.strip())

        return search_terms

    def _validate_with_external_llm(
        self, question_text: str, materials: str, ground_truth: Any, assigned_noisy_materials: List[str] = None
    ) -> bool:
        """Use external LLM to validate if the question can be answered correctly with the materials."""
        if not materials or not ground_truth:
            if SELF_PLAY_DEBUG:
                logger.debug(f"Skipping external validation - no judge or no materials/ground_truth")
            return False

        try:
            final_materials = materials

            if self.noisy_rag_materials > 0 and assigned_noisy_materials:
                all_materials = [materials] + assigned_noisy_materials
                random.shuffle(all_materials)
                final_materials = "\n\n".join(all_materials)

                if SELF_PLAY_DEBUG:
                    logger.debug(
                        f"Added {len(assigned_noisy_materials)} pre-assigned noisy RAG materials to validation, material: {final_materials[:100]}..."
                    )

            llm_answer = self.llm_judge.model_based_answer(final_materials, question_text)

            if not llm_answer:
                if SELF_PLAY_DEBUG:
                    logger.debug(f"External LLM returned empty answer for question: {question_text[:100]}...")
                return False

            answer_match = re.search(r"Answer[:：]?\s*(.*)", llm_answer, re.IGNORECASE)
            if answer_match:
                llm_answer = answer_match.group(1).strip()

            is_consistent = (
                em_check(
                    prediction=llm_answer,
                    golden_answers=ground_truth,
                )
                == 1
            )

            if not is_consistent:
                is_consistent = self.llm_judge.model_based_match(
                    question=question_text, golden_answer=ground_truth, model_answer=llm_answer
                )

            if SELF_PLAY_DEBUG:
                logger.debug(f"External validation result for question '{question_text[:50]}...': {is_consistent}")
                logger.debug(f"  Ground truth: {ground_truth}")
                logger.debug(f"  LLM answer: {llm_answer[:100]}...")
                logger.debug(f"LLM Judge is consistent: {is_consistent}")

            return is_consistent

        except Exception as e:
            logger.warning(f"External LLM validation failed: {e}")
            return True

    def _is_valid_search_question(
        self,
        question_text: str,
        output_text: str = "",
        metadata: Dict = None,
        assigned_noisy_materials: List[str] = None,
    ) -> bool:
        """Check if the extracted question is a valid search-based question."""

        if (
            not question_text
            or len(question_text.strip()) < 10
            or len(question_text.strip()) > 1000
            or len(question_text.split(" ")) < 10
        ):
            if SELF_PLAY_DEBUG:
                logger.debug(f"[Question Validation] question_text is invalid: {question_text}")
            return False

        reward_model = metadata.get("reward_model", {})
        ground_truth = None
        if isinstance(reward_model, dict):
            ground_truth_info = reward_model.get("ground_truth")
            if isinstance(ground_truth_info, dict):
                ground_truth = ground_truth_info.get("target")
            elif ground_truth_info is not None:
                ground_truth = ground_truth_info

        if ground_truth.lower() in question_text.lower():
            if SELF_PLAY_DEBUG:
                logger.debug(f"[Question Validation] ground_truth is in question_text, Invalid question")
            return False

        materials = self._extract_information_content(output_text)

        if not materials:
            if SELF_PLAY_DEBUG:
                logger.debug(f"[Question Validation] materials is None-Invalid question")
            return False

        search_terms = self._extract_search_terms(output_text)
        if len(search_terms) <= 1:
            if SELF_PLAY_DEBUG:
                logger.debug(f"[Question Validation] search_terms count <= 1: {len(search_terms)}")
            return False

        if len(search_terms) != len(set(search_terms)):
            if SELF_PLAY_DEBUG:
                logger.debug(f"[Question Validation] duplicate search_terms found: {search_terms}")
            return False

        if self.use_search_terms_filter:
            for search_term in search_terms:
                if ground_truth.lower() in search_term.lower():
                    if SELF_PLAY_DEBUG:
                        logger.debug(f"[Question Validation] ground_truth is in search_term, Invalid question")
                    return False

        if self.llm_judge and self.use_rag_filter:
            return self._validate_with_external_llm(question_text, materials, ground_truth, assigned_noisy_materials)

        return True

    def _format_search_problem(
        self, question_text: str, input_text: str, output_text: str, metadata: Dict
    ) -> Optional[Dict[str, Any]]:
        """Format an extracted search question into solver phase format."""
        try:
            if SELF_PLAY_DEBUG:
                logger.debug(f"Formatting search problem: {question_text[:100]}...")
                logger.debug(f"Metadata received: {metadata}")

            formatted_prompt = []
            if self.solver_system_prompt:
                formatted_prompt.append({"role": "system", "content": self.solver_system_prompt})

            formatted_prompt.append(
                {
                    "role": "user",
                    "content": self.solver_user_prompt.format(question_text),
                }
            )

            reward_model = metadata.get("reward_model", {"ground_truth": {"style": "rule"}})

            data_source = metadata.get("data_source", "self_generated")

            if SELF_PLAY_DEBUG:
                logger.debug(f"Using reward_model: {reward_model}")
                logger.debug(f"Using data_source: {data_source}")

            formatted_problem = {
                "data_source": data_source,
                "prompt": formatted_prompt,
                "ability": "fact-reasoning",
                "reward_model": reward_model,
                "extra_info": {
                    "question": question_text,
                    "need_tools_kwargs": True,
                    "split": "train",
                    "tools_kwargs": {
                        "search": {
                            "create_kwargs": {
                                "data_source": "self_generated",
                                "question": question_text,
                                "ground_truth": reward_model.get("ground_truth"),
                            }
                        }
                    },
                },
                "metadata": None,
                "extracted_question": question_text,
                "formatted_prompt": formatted_prompt,
                "problem_type": "search",
                "trajectory_index": metadata.get("trajectory_index", -1),
            }

            if SELF_PLAY_DEBUG:
                logger.debug(f"Successfully formatted problem: {formatted_problem['extracted_question']}")

            return formatted_problem

        except Exception as e:
            logger.warning(f"Error formatting search problem: {e}")
            if SELF_PLAY_DEBUG:
                logger.debug(f"Exception details: {e}", exc_info=True)
                logger.debug(f"Failed on question: {question_text}")
                logger.debug(f"Failed with metadata: {metadata}")
            return None


def _process_single_trajectory(
    trajectory_data: tuple[int, Dict[str, Any], ProblemExtractor, List[str]],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Process a single trajectory for problem extraction.

    Args:
        trajectory_data: Tuple of (trajectory_index, trajectory_dict, extractor, assigned_noisy_materials)

    Returns:
        Tuple of (extracted problems list, trajectory statistics dict)
    """
    i, trajectory, extractor, assigned_noisy_materials = trajectory_data
    problems = []

    traj_stats = {
        "answer_matches": 0,
        "valid_questions": 0,
        "successful_problems": 0,
        "format_error": 0,
    }

    try:
        input_text = trajectory.get("input", "")
        output_text = trajectory.get("output", "")
        metadata = trajectory.get("metadata", {}).copy()

        if SELF_PLAY_DEBUG:
            logger.debug(f"Processing trajectory {i}")

        problems, traj_stats = extractor.extract_problems_from_trajectory_with_stats(
            input_text, output_text, metadata, assigned_noisy_materials
        )

        if SELF_PLAY_DEBUG:
            logger.debug(f"Trajectory {i} yielded {len(problems)} problems")

    except Exception as e:
        logger.warning(f"Error processing trajectory {i}: {e}")
        if SELF_PLAY_DEBUG:
            logger.debug(f"Exception details for trajectory {i}: {e}", exc_info=True)

    return problems, traj_stats


def extract_problems_batch(
    trajectories: List[Dict[str, Any]], extractor: ProblemExtractor = None, max_workers: Optional[int] = None
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Extract search-based problems from a batch of proposer generation trajectories using multi-threading.

    Args:
        trajectories: List of trajectory dictionaries with 'input', 'output', and optional 'metadata'
        extractor: ProblemExtractor instance (creates default if None)
        max_workers: Maximum number of threads to use (defaults to min(32, number of trajectories))

    Returns:
        Tuple of (extracted problems list, extraction statistics dict)
    """
    if extractor is None:
        extractor = ProblemExtractor()

    if not trajectories:
        return [], {
            "trajectories_count": 0,
            "answer_matches_count": 0,
            "valid_questions_count": 0,
            "successful_problems_count": 0,
            "format_error_count": 0,
        }

    if max_workers is None:
        max_workers = min(32, len(trajectories))

    all_problems = []

    stats = {
        "trajectories_count": len(trajectories),
        "answer_matches_count": 0,
        "valid_questions_count": 0,
        "successful_problems_count": 0,
        "format_error_count": 0,
    }

    if SELF_PLAY_DEBUG:
        logger.debug(f"Starting batch extraction from {len(trajectories)} trajectories using {max_workers} threads")

    trajectory_data = []
    if extractor.noisy_rag_materials > 0:
        all_materials = []
        for trajectory in trajectories:
            output_text = trajectory.get("output", "")
            materials = extractor._extract_information_content(output_text)
            if materials:
                all_materials.append(materials)

        if SELF_PLAY_DEBUG:
            logger.debug(
                f"Extracted {len(all_materials)} materials for noisy RAG from {len(trajectories)} trajectories"
            )

        for i, trajectory in enumerate(trajectories):
            current_materials = extractor._extract_information_content(trajectory.get("output", ""))

            available_materials = [mat for mat in all_materials if mat != current_materials]
            assigned_noisy_materials = []

            if available_materials and current_materials:
                num_to_select = min(extractor.noisy_rag_materials, len(available_materials))
                if num_to_select > 0:
                    assigned_noisy_materials = random.sample(available_materials, num_to_select)

                    if SELF_PLAY_DEBUG:
                        logger.debug(f"Trajectory {i}: assigned {num_to_select} noisy materials")

            trajectory_data.append((i, trajectory, extractor, assigned_noisy_materials))
    else:
        trajectory_data = [(i, trajectory, extractor, []) for i, trajectory in enumerate(trajectories)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(_process_single_trajectory, data): data[0] for data in trajectory_data}

        for future in as_completed(future_to_index):
            trajectory_index = future_to_index[future]
            try:
                problems, traj_stats = future.result()

                all_problems.extend(problems)

                if traj_stats["answer_matches"] > 0:
                    stats["answer_matches_count"] += 1
                stats["valid_questions_count"] += traj_stats["valid_questions"]
                stats["successful_problems_count"] += traj_stats["successful_problems"]
                stats["format_error_count"] += traj_stats["format_error"]

            except Exception as e:
                logger.warning(f"Thread processing trajectory {trajectory_index} failed: {e}")
                if SELF_PLAY_DEBUG:
                    logger.debug(f"Thread exception details for trajectory {trajectory_index}: {e}", exc_info=True)

    if SELF_PLAY_DEBUG:
        logger.debug(f"Batch extraction complete: {len(all_problems)} total problems extracted")
        logger.debug(f"Extraction statistics: {stats}")

    logger.info(
        f"Extracted {len(all_problems)} search problems from {len(trajectories)} trajectories using {max_workers} threads"
    )
    return all_problems, stats
