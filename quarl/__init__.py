# Quark RL Team

# Import self-play components
from quarl.trainer.ppo.ssp_data_manager import SSPDataManager
from quarl.trainer.ppo.ssp_ray_trainer import SSPRayPPOTrainer
from quarl.utils.problem_extraction import ProblemExtractor

__all__ = ["SSPRayPPOTrainer", "SSPDataManager", "ProblemExtractor"]
