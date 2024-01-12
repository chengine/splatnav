# TODO: FINISH THE DATALOADER
# TODO: ENSURE THE PATCH DIMENSIONALITY IS NOT AN ISSUE



import json
import os
from pathlib import Path

import numpy as np
import torch
from gemsplat.data.utils.feature_dataloader import FeatureDataloader
from gemsplat.encoders.maskclip_encoder import MaskCLIPNetwork
import torchvision.transforms.functional as TF
import torch
from tqdm import tqdm
from gemsplat.data.utils.utils import apply_pca_colormap_return_proj

import gc


class MaskCLIPDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: MaskCLIPNetwork,
        image_list: torch.Tensor = None,
        cache_path: str = None,
    ):
        self.model = model
        self.data_dict = {}
        self.image_list = image_list
        
        super().__init__(cfg, device, image_list, cache_path)

    def __call__(self, image_idx):
        # image_idx: index of the image in the training dataset
        output = self.data[image_idx].type(torch.float32).to(self.device)
        return output
    
    def load(self):
        # load the embeddings
        super().load()

        # Determine scaling factors for nearest neighbor interpolation
        feat_h, feat_w = self.data.shape[1:3]
        assert len(self.model.im_h) == 1, "All images must have the same height"
        assert len(self.model.im_w) == 1, "All images must have the same width"
        self.model.im_h, self.model.im_w = self.model.im_h.pop(), self.model.im_w.pop()
        self.model.scale_h = feat_h / self.model.im_h
        self.model.scale_w = feat_w / self.model.im_w

    def create(self, image_list):
            self.data = []
            self.data = self.model.encode_image(image_list).cpu().detach()
        
    def resize_data(self) -> None:
        # newsize
        newsize = [self.model.im_h // 4, self.model.im_w // 4]
        self.data_resized = torch.empty((self.data.shape[0], *newsize, self.data.shape[-1]),
                                        dtype=torch.half, device=self.device)
        
        # interpolation method
        if self.aux_data is not None:
            clip_interpolation_method = TF.InterpolationMode.BILINEAR
        else:
            # if not PCA
            clip_interpolation_method = TF.InterpolationMode.NEAREST
    
        for i in tqdm(
            range(0, self.data.shape[0], 
                1),
            desc="Resizing CLIP features",
        ):
            self.data_resized[i] = TF.resize(self.data[i].permute(2, 0, 1), newsize, 
                                        interpolation=clip_interpolation_method,
                                        antialias=None).permute(1, 2, 0)
            
        del self.data
        gc.collect()
        torch.cuda.empty_cache()
        
        self.data = self.data_resized
        self.data = self.data.cpu().detach()
        
        
               