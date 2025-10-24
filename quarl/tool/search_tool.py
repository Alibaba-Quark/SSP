import json
import logging
import os
from typing import Any, Callable, Optional, Tuple, TypeVar

from verl.tools.search_tool import SearchTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

USE_WIKI = [
    "searchR1_nq",
    "searchR1_triviaqa",
    "searchR1_popqa",
    "searchR1_hotpotqa",
    "searchR1_2wikimultihopqa",
    "searchR1_musique",
    "searchR1_bamboogle",
    "quark_selfplay_en",
    "self_generated",
]


class QuarkSearchTool(SearchTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        timeout = self.timeout
        query_list_from_params = parameters.get("query_list")
        datasource = parameters.get("data_source", None)
        # print("[DebugQuarkSearchToolSearchDataSource] data_source:", datasource)

        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return json.dumps({"result": error_msg}), 0.0, {}

        # Execute search using Ray execution pool
        try:
            if datasource in USE_WIKI:
                retrieval_service_url_exe = self.retrieval_service_url.get("wiki")
            else:
                retrieval_service_url_exe = self.retrieval_service_url.get("default")

            result_text, metadata = await self.execution_pool.execute.remote(
                self.execute_search, instance_id, query_list_from_params, retrieval_service_url_exe, self.topk, timeout
            )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Convert metadata to metrics
            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
            }

            return result_text, 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            return error_result, 0.0, {"error": str(e)}
