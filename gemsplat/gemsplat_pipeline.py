import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from gemsplat.data.gemsplat_datamanager import (
    GEMSPLATDataManager,
    GEMSPLATDataManagerConfig,
)
from gemsplat.gemsplat import SemanticGaussianSplattingModel, SemanticGaussianSplattingModelConfig
from gemsplat.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder


@dataclass
class GEMSPLATPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: GEMSPLATPipeline)
    """target class to instantiate"""
    datamanager: GEMSPLATDataManagerConfig = GEMSPLATDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = SemanticGaussianSplattingModelConfig()
    """specifies the model config"""
    network: BaseImageEncoderConfig = BaseImageEncoderConfig()
    """specifies the vision-language network config"""


class GEMSPLATPipeline(VanillaPipeline):
    def __init__(
        self,
        config: GEMSPLATPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        self.image_encoder: BaseImageEncoder = config.network.setup()

        self.datamanager: GEMSPLATDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            image_encoder=self.image_encoder,
        )
        # TODO make cleaner
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)
        self.datamanager.to(device)

        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            image_encoder=self.image_encoder,
            datamanager=self.datamanager,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(SemanticGaussianSplattingModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])
