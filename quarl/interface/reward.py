from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class RewardFuncInfo:
    name: str  # Name of the reward function
    reward_fn: Callable[[List], List]  # Reward function object
    labels: List[str] = field(default_factory=list)  # List of labels the reward function applies to
    integration: str = "sum"  # Rule for combining multiple rewards: "sum" or "multiply"
    reward_fn_key: str = "data_source"
