import torch
import json
from pathlib import Path
import os

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.rays import Frustums, RaySamples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeRFWrapper():
    def __init__(self, config_fp, use_ns_coordinates=False) -> None:

        config_path = Path(config_fp + "/config.yml") # Path to config file 

        if os.path.isfile(config_fp + "/dataparser_transforms.json") and not use_ns_coordinates:
            with open(config_fp + "/dataparser_transforms.json", 'r') as f:
                meta = json.load(f)

            self.scale = meta["scale"]
            transform = meta["transform"]

            self.transform = torch.eye(4).to(device)
            self.transform[:3, :] = torch.tensor(transform)

        else: 
            self.scale = 1.
            self.transform = torch.eye(4).to(device)

        # Prepare model
        _, self.pipeline, _, _ = eval_setup(
            config_path, 
            test_mode="inference",
        )

    def get_density(self, xyz):

        pts = xyz.reshape(-1, 3)

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=pts,
                directions=torch.zeros_like(pts, device=device),
                starts=0,
                ends=0,
                pixel_area=None,
            )
        )

        density, _ = self.pipeline.model.field.get_density(ray_samples)
        return density

    def data_frame_to_ns_frame(self, points):
        transformed_points = (self.transform[:3, :3]@points.T).T + self.transform[:3, -1][None,:]
        transformed_points *= self.scale

        return transformed_points

    def ns_frame_to_data_frame(self, points):
        transformed_points = points / self.scale
        transformed_points = transformed_points - self.transform[:3, -1][None, :]
        transformed_points = (self.transform[:3, :3].T @ transformed_points.T).T

        return transformed_points
    
    def get_exp_name(self):
        return self.pipeline