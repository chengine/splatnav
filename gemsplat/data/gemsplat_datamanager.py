# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Datamanager.
"""

from __future__ import annotations

import random
import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
import numpy
import torch
# import torchvision
import yaml
from copy import deepcopy

from nerfstudio.cameras.cameras import Cameras, CameraType
from rich.progress import Console

CONSOLE = Console(width=120)
from gemsplat.encoders.image_encoder import BaseImageEncoder
from gemsplat.data.utils.maskclip_dataloader import MaskCLIPDataloader
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    ForwardRef,
    get_origin,
    get_args,
)

from PIL import Image
import pdb

@dataclass
class GemSplatDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: GemSplatDataManager)

class GemSplatDataManager(FullImageDatamanager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: GemSplatDataManagerConfig

    def __init__(
        self,
        config: GemSplatDataManagerConfig,
        device: Union[torch.device, str] = "cuda:0",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self.image_encoder: BaseImageEncoder = kwargs["image_encoder"]
        im_h = set(self.train_dataset.cameras.image_height.squeeze().tolist())
        im_w = set(self.train_dataset.cameras.image_width.squeeze().tolist())
        self.image_encoder.im_h = im_h
        self.image_encoder.im_w = im_w

        images = [self.train_dataset[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))]
        images = torch.cat(images)
    
        # path to GemSplat
        parent_path = Path(__file__).parent.parent.parent.parent.resolve()

        cache_dir = f"{parent_path}/outputs/{self.config.dataparser.data.name}/dataloader"
        clip_cache_path = Path(osp.join(cache_dir, f"clip_{self.image_encoder.name}.npy"))
        aff_cache_path = Path(osp.join(cache_dir, "affordance.npy"))
        
        # NOTE: cache config is sensitive to list vs. tuple, because it checks for dict equality
        self.clip_interpolator = MaskCLIPDataloader(
            image_list=images,
            device=self.device,
            cfg={
                "image_shape": list(images.shape[2:4]),
                "model_name": self.image_encoder.name,
            },
            cache_path=clip_cache_path,
            model=self.image_encoder,
        )
        torch.cuda.empty_cache()

        self.train_dataset.metadata["feature_dim"] = self.clip_interpolator.data.shape[-1]
        
        # resize CLIP embeddings
        # self.clip_interpolator.resize_data()

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        # TODO: Add functionality for RGBA images
        data = deepcopy(self.cached_train[image_idx])
        data["image"] = data["image"].to(self.device)[..., :3]
        
        # # CLIP embeddings
        data["clip"] = self.clip_interpolator(image_idx).to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
            
        camera.metadata["cam_idx"] = image_idx
        
        return camera, data
