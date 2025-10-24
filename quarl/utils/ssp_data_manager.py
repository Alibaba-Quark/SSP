"""
Self-Play Data Manager for managing dynamic data flow in self-play training.

This module implements a Ray remote data manager that handles:
1. Problem generation phase data
2. Problem solving phase data
3. Dynamic switching between data sources
4. Question extraction and processing from generation trajectories
"""

import json
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import ray


@dataclass
class SelfPlayPhase:

    PROBLEM_GENERATION = "problem_generation"
    PROBLEM_SOLVING = "problem_solving"


@ray.remote
class SSPDataManager:
    """
    Self-Play Data Manager for handling dynamic data flow in self-play training.

    This manager orchestrates the data flow between problem generation and solving phases,
    maintaining separate data pools and enabling dynamic dataloader switching.
    """

    def __init__(self):
        # Data storage for different phases
        self.problem_generation_data = []
        self.problem_solving_data = []
        self.generated_problems_pool = []
        self.current_step_problems = []

        # Metadata tracking
        self.current_phase = SelfPlayPhase.PROBLEM_GENERATION
        self.global_step = 0
        self.phase_step_counter = defaultdict(int)

        # Statistics tracking
        self.stats = {
            "total_problems_generated": 0,
            "total_trajectories_processed": 0,
            "current_step_trajectories": 0,
            "current_step_answer_matches": 0,
            "current_step_valid_questions": 0,
            "current_step_successful_problems": 0,
            "current_step_problems_count": 0,
            "total_no_answer_pattern": 0,
            "total_invalid_questions": 0,
            "total_format_failures": 0,
        }

        self.lock = threading.RLock()

    def set_problem_generation_data(self, data: List[Dict[str, Any]]):
        with self.lock:
            self.problem_generation_data = data
            print(f"Loaded {len(data)} problem generation samples")

    def get_current_phase_data(self) -> List[Dict[str, Any]]:
        with self.lock:
            if self.current_phase == SelfPlayPhase.PROBLEM_GENERATION:
                return self.problem_generation_data
            elif self.current_phase == SelfPlayPhase.PROBLEM_SOLVING:
                return self.problem_solving_data
            else:
                raise ValueError(f"Unknown phase: {self.current_phase}")

    def switch_to_phase(self, phase: str) -> bool:
        
        with self.lock:
            if phase == SelfPlayPhase.PROBLEM_GENERATION:
                self.current_phase = SelfPlayPhase.PROBLEM_GENERATION
                print(f"Switched to PROBLEM_GENERATION phase")
                return True

            elif phase == SelfPlayPhase.PROBLEM_SOLVING:
                # Only switch if we have current step problems to solve
                if len(self.current_step_problems) > 0:
                    self.current_phase = SelfPlayPhase.PROBLEM_SOLVING
                    self._prepare_solving_data()
                    print(
                        f"Switched to PROBLEM_SOLVING phase with {len(self.problem_solving_data)} current step problems"
                    )
                    return True
                else:
                    print("Cannot switch to PROBLEM_SOLVING: no current step problems available")
                    return False
            else:
                raise ValueError(f"Unknown phase: {phase}")

    def _prepare_solving_data(self):
        self.problem_solving_data = []
        for problem_data in self.current_step_problems:
            solving_entry = {
                "prompt": problem_data["formatted_prompt"],
                "problem": problem_data["extracted_question"],
                "data_source": "self_generated",
                "reward_model": problem_data.get("reward_model", {"style": "rule"}),
                "extra_info": {
                    "generation_step": problem_data["generation_step"],
                    "extraction_success": True,
                    "problem_type": "self_play_generated",
                },
            }
            self.problem_solving_data.append(solving_entry)

        print(f"Prepared {len(self.problem_solving_data)} problems from current step for solving")

    def reset_current_step_stats(self):
        with self.lock:
            self.stats["current_step_trajectories"] = 0
            self.stats["current_step_answer_matches"] = 0
            self.stats["current_step_valid_questions"] = 0
            self.stats["current_step_successful_problems"] = 0
            self.stats["current_step_problems_count"] = 0

    def record_extraction_stats(
        self,
        trajectories_count: int,
        answer_matches_count: int,
        valid_questions_count: int,
        successful_problems_count: int,
        format_error_count: int,
    ):
        
        with self.lock:
            # Update current step stats
            self.stats["current_step_trajectories"] = trajectories_count
            self.stats["current_step_answer_matches"] = answer_matches_count
            self.stats["current_step_valid_questions"] = valid_questions_count
            self.stats["current_step_successful_problems"] = successful_problems_count

            # Update cumulative stats
            self.stats["total_trajectories_processed"] += trajectories_count
            self.stats["total_no_answer_pattern"] += trajectories_count - answer_matches_count
            self.stats["total_invalid_questions"] += answer_matches_count - valid_questions_count
            self.stats["total_format_failures"] += valid_questions_count - successful_problems_count

            self.stats["total_format_failures"] += format_error_count

            if trajectories_count > 0:
                answer_match_rate = answer_matches_count / trajectories_count
                valid_rate = valid_questions_count / trajectories_count if trajectories_count > 0 else 0
                success_rate = successful_problems_count / trajectories_count if trajectories_count > 0 else 0

                print(f"Extraction stats for current step:")
                print(f"  Trajectories processed: {trajectories_count}")
                print(f"  Found <answer></answer>: {answer_matches_count} ({answer_match_rate:.2%})")
                print(f"  Valid questions: {valid_questions_count} ({valid_rate:.2%})")
                print(f"  Successful problems: {successful_problems_count} ({success_rate:.2%})")

    def add_generated_problems(self, problems: List[Dict[str, Any]], generation_step: int):
        
        with self.lock:
            self.current_step_problems = []

            for problem in problems:
                problem["generation_step"] = generation_step
                self.generated_problems_pool.append(problem)
                self.current_step_problems.append(problem)

            self.stats["total_problems_generated"] += len(problems)
            self.stats["current_step_problems_count"] = len(self.current_step_problems)

            print(
                f"Added {len(problems)} newly generated problems to current step. "
                f"Total pool size: {len(self.generated_problems_pool)}"
            )

    def add_generated_problems_with_success_rate(self, problems: List[Dict[str, Any]], generation_step: int):
        
        with self.lock:
            self.current_step_problems = []

            for problem in problems:
                if "success_rate" not in problem:
                    print(
                        f"Warning: Problem missing success_rate field: {problem.get('extracted_question', '')[:50]}..."
                    )
                    continue

                problem["generation_step"] = generation_step
                self.generated_problems_pool.append(problem)
                self.current_step_problems.append(problem)

            self.stats["total_problems_generated"] += len(self.current_step_problems)
            self.stats["current_step_problems_count"] = len(self.current_step_problems)

            print(
                f"Added {len(self.current_step_problems)} newly generated problems with success_rate to current step. "
                f"Total pool size: {len(self.generated_problems_pool)}"
            )

    def get_problems_by_success_rate(self, count: int, max_success_rate: float = 0.5) -> List[Dict[str, Any]]:
    
        with self.lock:
            if not self.generated_problems_pool:
                print(f"No problems found in the pool")
                return []

            import random

            filtered_problems = [
                problem
                for problem in self.generated_problems_pool
                if problem.get("success_rate", 1.0) <= max_success_rate
            ]

            print(
                f"Found {len(filtered_problems)} problems with success_rate <= {max_success_rate} out of {len(self.generated_problems_pool)} total"
            )

            if len(filtered_problems) >= count:
                return random.sample(filtered_problems, count)
            elif len(filtered_problems) > 0:
                remaining_count = count - len(filtered_problems)
                remaining_problems = [p for p in self.generated_problems_pool if p.get("success_rate", 1.0) > max_success_rate]
                additional_problems = random.sample(remaining_problems, min(remaining_count, len(remaining_problems)))
                return filtered_problems + additional_problems
            else:
                print(f"No problems found with success_rate <= {max_success_rate}, falling back to random selection")
                if count <= len(self.generated_problems_pool):
                    return random.sample(self.generated_problems_pool, count)
                else:
                    return random.sample(self.generated_problems_pool, len(self.generated_problems_pool))

    def get_current_phase(self) -> str:
        with self.lock:
            return self.current_phase

    def get_random_existing_problems(self, count: int) -> List[Dict[str, Any]]:
        
        with self.lock:
            if not self.generated_problems_pool:
                return []

            import random

            if count <= len(self.generated_problems_pool):
                return random.sample(self.generated_problems_pool, count)
            else:
                return [random.choice(self.generated_problems_pool) for _ in range(count)]

    def has_existing_problems(self) -> bool:
        with self.lock:
            return len(self.generated_problems_pool) > 0

    def update_global_step(self, step: int):
        with self.lock:
            self.global_step = step
            self.phase_step_counter[self.current_phase] += 1

    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            stats_with_rates = dict(self.stats)

            if self.stats["current_step_trajectories"] > 0:
                traj_count = self.stats["current_step_trajectories"]
                stats_with_rates["current_step_answer_match_rate"] = (
                    self.stats["current_step_answer_matches"] / traj_count
                )
                stats_with_rates["current_step_valid_question_rate"] = (
                    self.stats["current_step_valid_questions"] / traj_count
                )
                stats_with_rates["current_step_success_rate"] = (
                    self.stats["current_step_successful_problems"] / traj_count
                )
            else:
                stats_with_rates["current_step_answer_match_rate"] = 0.0
                stats_with_rates["current_step_valid_question_rate"] = 0.0
                stats_with_rates["current_step_success_rate"] = 0.0

            if self.stats["total_trajectories_processed"] > 0:
                total_traj = self.stats["total_trajectories_processed"]
                stats_with_rates["overall_answer_match_rate"] = 1.0 - (
                    self.stats["total_no_answer_pattern"] / total_traj
                )
                stats_with_rates["overall_valid_question_rate"] = 1.0 - (
                    (self.stats["total_no_answer_pattern"] + self.stats["total_invalid_questions"]) / total_traj
                )
                stats_with_rates["overall_success_rate"] = self.stats["total_problems_generated"] / total_traj
            else:
                stats_with_rates["overall_answer_match_rate"] = 0.0
                stats_with_rates["overall_valid_question_rate"] = 0.0
                stats_with_rates["overall_success_rate"] = 0.0

            stats_with_rates.update(
                {
                    "generation_data_size": len(self.problem_generation_data),
                }
            )

            return stats_with_rates

    def clear_solving_pool(self, keep_ratio: float = 0.5):
        """
        Clear part of the solving pool to prevent memory issues.

        Args:
            keep_ratio: Ratio of problems to keep (newest ones)
        """
        with self.lock:
            if len(self.generated_problems_pool) > 0:
                keep_count = int(len(self.generated_problems_pool) * keep_ratio)
                if keep_count > 0:
                    self.generated_problems_pool = self.generated_problems_pool[-keep_count:]
                else:
                    self.generated_problems_pool = []
                print(f"Cleared solving pool, kept {len(self.generated_problems_pool)} problems")