import logging
import os
import warnings
from functools import partial

import torch
import torch.distributed
import verl.utils.torch_functional as verl_F
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_name
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
)

from quarl.utils.patch.chat_template import switch_apply_chat_template

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class QuarkActorRolloutRefWorker(ActorRolloutRefWorker):
    def _switch_tokenizer_chat_template(self):
        from quarl.utils.chat_template import QuarkChatTemplate

        print(
            "[QuarkActorRolloutRefWorker]: QuarkChatTemplate._current_template =", QuarkChatTemplate._current_template
        )

        self.tokenizer = switch_apply_chat_template(
            self.tokenizer, apply_chat_template_fn=QuarkChatTemplate.apply_chat_template
        )

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert (
            self.world_size % infer_tp == 0
        ), f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(
            get_device_name(), mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )

        rollout_name = self.config.rollout.name

        if rollout_name in ["sglang", "sglang_async"]:
            if rollout_name == "sglang_async":
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. "
                    "Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            from quarl.worker.rollout.sglang_rollout import QuarkSGLangRollout

            rollout_cls = QuarkSGLangRollout
            logger.info("Using QuarkSGLangRollout with standard rollout")

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)

            self._switch_tokenizer_chat_template()

            rollout = rollout_cls(
                actor_module=local_path,
                config=self.config.rollout,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"

            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                rollout_config=self.config.rollout,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                multi_stage_wake_up=self.config.rollout.multi_stage_wake_up,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout, rollout_sharding_manager
