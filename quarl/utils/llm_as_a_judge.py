import logging
import random
import re
from typing import Any, Dict, List

from openai import OpenAI

logger = logging.getLogger(__name__)


class SyncLLMAsAJudge:
    """Synchronous LLM-as-a-Judge implementation"""

    def __init__(self, base_url: str, api_key: str, model: str):
        """
        Initialize synchronous LLM judge

        Args:
            base_url: Base URL for OpenAI API
            api_key: API key
            model: Model name to use
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=3)
        self.model = model

    def model_based_match(self, question: str, golden_answer: Any, model_answer: str) -> bool:
        """Model-based answer matching"""
        prompt = f"""
Please determine whether the model’s answer is consistent with the reference answer:

Question: {question}
Model Answer: {model_answer}
Reference Answer: {golden_answer}

Evaluation Criteria:
1. The model answer must accurately respond to the question and be consistent with the reference answer in meaning.
2. For numerical questions, the values must be equal or very close.
3. For textual questions, the core meaning must be correct.
4. Differences in wording or language are allowed as long as the core answer is the same.
5. If the model answer includes the correct answer and does not contain conflicting information, it is also considered correct.

Please respond only with "Correct" or "Wrong". Do not provide any additional explanation.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional judge who evaluates the correctness of answers based on given criteria.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=10,
            )

            judgment = response.choices[0].message.content.strip()
            logger.debug(f"LLM judgment result: {judgment}")

            return "correct" in judgment.lower()

        except Exception as e:
            logger.error(f"LLM judgment error: {e}")
            return False

    def batch_model_based_match(self, evaluations: list[dict]) -> list[bool]:
        """
        Batch evaluation of answers

        Args:
            evaluations: List of dictionaries containing question, golden_answer, model_answer

        Returns:
            list[bool]: Judgment result for each answer
        """
        results = []
        for eval_data in evaluations:
            try:
                result = self.model_based_match(
                    question=eval_data["question"],
                    golden_answer=eval_data["golden_answer"],
                    model_answer=eval_data["model_answer"],
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Batch evaluation error: {e}")
                results.append(False)
        return results

    def model_based_answer(self, materials, question) -> str:
        """
        Model-based answer generation
        """

        prompt = f"""
Answer the given question based on the provided materials. You should first conduct very concise reasoning within 50 words, and then directly provide your answer without detailed illustrations after saying 'Answer:'. Materials: {materials}\n Question: {question}\n
"""

        try:
            response_result = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=70,
            )

            answer = response_result.choices[0].message.content.strip()

            return answer

        except Exception as e:
            logger.error(f"LLM answer error: {e}")
            return ""

    def close(self):
        """Close client connection"""
        if hasattr(self, "client"):
            self.client.close()

    def __del__(self):
        """Auto close on destruction"""
        self.close()


# Singleton pattern management
_global_judge = None


def get_global_judge(base_url: str, api_key: str, model: str = "gpt-4") -> SyncLLMAsAJudge:
    """Get global judge instance"""
    global _global_judge
    if _global_judge is None:
        _global_judge = SyncLLMAsAJudge(base_url=base_url, api_key=api_key, model=model)
    return _global_judge


def create_sync_llm_judge(base_url: str, api_key: str, model: str = "gpt-4") -> SyncLLMAsAJudge:
    """Create synchronous LLM judge instance"""
    return SyncLLMAsAJudge(base_url=base_url, api_key=api_key, model=model)


# Usage example
def example_usage():
    """Usage example"""
    import os

    base_url = os.getenv("QUARK_BASE_URL")
    model = os.getenv("QUARK_MODEL")
    judge = create_sync_llm_judge(base_url=base_url, api_key="your-api-key", model=model)

    # Single evaluation
    result = judge.model_based_match(
        question="What is the capital of France?", golden_answer="Paris", model_answer="Paris is the capital city of France"
    )
    print(f"Evaluation result: {result}")

    # Batch evaluation
    evaluations = [
        {"question": "What is 1+1?", "golden_answer": "2", "model_answer": "The answer is 2"},
        {"question": "How many moons does Earth have?", "golden_answer": "1", "model_answer": "Earth has one natural satellite, the Moon"},
    ]

    batch_results = judge.batch_model_based_match(evaluations)
    print(f"Batch evaluation results: {batch_results}")

    # Explicit close (optional)
    judge.close()


if __name__ == "__main__":
    example_usage()
