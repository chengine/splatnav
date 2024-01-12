# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union, Literal
from nerfstudio.data.scene_box import OrientedBox

import torch
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.transforms.functional as TF
from tqdm import tqdm

from nerfstudio.cameras.cameras import Cameras
from gsplat._torch_impl import quat_to_rotmat
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig

from nerfstudio.cameras.cameras import Cameras
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.models.gaussian_splatting import GaussianSplattingModelConfig, GaussianSplattingModel
from nerfstudio.model_components.losses import (
    scale_gradients_by_distance_squared,
)

from gsplat.rasterize import RasterizeGaussians
from gsplat.project_gaussians import ProjectGaussians
from gsplat.sh import SphericalHarmonics, num_sh_bases
from pytorch_msssim import SSIM
from nerfstudio.model_components.scene_colliders import NearFarCollider

# need following import for background color override
from nerfstudio.model_components import renderers

from gemsplat.encoders.image_encoder import BaseImageEncoder
from gemsplat.data.gemsplat_datamanager import GEMSPLATDataManager

from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from gemsplat.viewer_utils import ViewerUtils
from nerfstudio.viewer.server.viewer_elements import (
    ViewerButton,
    ViewerNumber,
    ViewerText,
)


from torch import Tensor, TensorType, nn
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.mlp import MLPWithHashEncoding

try:
    import tinycudann as tcnn
except ImportError:
    pass

import time
from joblib import Parallel, delayed
from concurrent import futures

from enum import Enum

    
def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def projection_matrix(znear, zfar, fovx, fovy, device: Union[str, torch.device] = "cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )
    
            
@dataclass
class SemanticGaussianSplattingModelConfig(GaussianSplattingModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: SemanticGaussianSplattingModel)
    # enable_collider: bool = True
    # """Whether to create a scene collider to filter rays."""
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians"""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 4
    """maximum degree of spherical harmonics to use"""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="off")
    """camera optimizer config"""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    clip_img_loss_weight: float = 1e-1
    """weight for the CLIP-related term in the loss function."""
    clip_network_loss_weight: float = 1e-0
    """weight for the CLIP-related term in the loss function."""
    
    # MLP head
    hidden_dim: int = 64
    num_layers: int = 2


class SemanticGaussianSplattingModel(GaussianSplattingModel):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: SemanticGaussianSplattingModelConfig

    def __init__(self, *args, **kwargs):
        if "seed_points" in kwargs:
            self.seed_pts = kwargs["seed_points"]
        else:
            self.seed_pts = None
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]
        
        # datamanager
        self.datamanager: GEMSPLATDataManager = self.kwargs["datamanager"] 
        
        # CLIP embeddings input dimension
        self.clip_embeds_input_dim = self.kwargs["metadata"]["feature_dim"]
        
        # CLIP embeddings latent dimension
        self.clip_embeds_latent_dim = 3
        
        if self.seed_pts is not None and not self.config.random_init:
            self.means = torch.nn.Parameter(self.seed_pts[0])  # (Location, Color)
        else:
            self.means = torch.nn.Parameter((torch.rand((500000, 3)) - 0.5) * 10)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(self.means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        self.scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        self.quats = torch.nn.Parameter(random_quat_tensor(self.num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if self.seed_pts is not None and not self.config.random_init:
            fused_color = RGB2SH(self.seed_pts[1] / 255)
            shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().cuda()
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
            self.colors_all = torch.nn.Parameter(shs)
        else:
            colors = torch.nn.Parameter(torch.rand(self.num_points, 1, 3))
            shs_rest = torch.nn.Parameter(torch.zeros((self.num_points, dim_sh - 1, 3)))
            self.colors_all = torch.nn.Parameter(torch.cat([colors, shs_rest], dim=1))

        self.opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1)))
        
        # learned CLIP embeddings
        self.clip_embeds = torch.nn.Parameter(torch.rand(self.num_points, self.clip_embeds_latent_dim))
            
        # background CLIP rendering
        self.clip_background = torch.zeros(3, device=self.kwargs["device"])
        
        # Encoder-Decoder Network for the CLIP Embeddings
        # Encoder
        self.clip_encoder = tcnn.Network(
            n_input_dims=self.clip_embeds_input_dim,
            n_output_dims=self.clip_embeds_latent_dim,
            network_config={
                "otype": "CutlassMLP", # "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.config.hidden_dim,
                "n_hidden_layers": self.config.num_layers,
            },
        )
        
        # Decoder
        self.clip_decoder = tcnn.Network(
            n_input_dims=self.clip_embeds_latent_dim,
            n_output_dims=self.clip_embeds_input_dim,
            network_config={
                "otype": "CutlassMLP", # "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.config.hidden_dim,
                "n_hidden_layers": self.config.num_layers,
            },
        )
        
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        self.back_color = torch.zeros(3)

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        
        # initialize Viewer
        self.viewer_utils = ViewerUtils()
        
        self.setup_gui()\

    @property
    def colors(self):
        return self.colors_all[:, 0, :]

    @property
    def shs_rest(self):
        return self.colors_all[:, 1:, :]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        newp = dict["means"].shape[0]
        self.means = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.scales = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.quats = torch.nn.Parameter(torch.zeros(newp, 4, device=self.device))
        self.opacities = torch.nn.Parameter(torch.zeros(newp, 1, device=self.device))\
        
        # learned CLIP embeddings
        self.clip_embeds = torch.nn.Parameter(torch.zeros(newp, self.clip_embeds_latent_dim, device=self.device))\
        
        self.colors_all = torch.nn.Parameter(
            torch.zeros(newp, num_sh_bases(self.config.sh_degree), 3, device=self.device)
        )
        super().load_state_dict(dict, **kwargs)
            
    def setup_gui(self):
        self.viewer_utils.device = "cuda:0" #self.device #"cuda:0"
        # Note: the GUI elements are shown based on alphabetical variable names
        self.btn_refresh_pca = ViewerButton("Refresh PCA Projection", cb_hook=lambda _: self.viewer_utils.reset_pca_proj())

        # Only setup GUI for language features if we're using CLIP
        self.hint_text = ViewerText(name="Note:", disabled=True, default_value="Use , to separate labels")
        self.lang_1_pos_text = ViewerText(
            name="Language (Positives)",
            default_value="",
            cb_hook=lambda elem: self.viewer_utils.handle_language_queries(elem.value, is_positive=True),
        )
        self.lang_2_neg_text = ViewerText(
            name="Language (Negatives)",
            default_value="",
            cb_hook=lambda elem: self.viewer_utils.handle_language_queries(elem.value, is_positive=False),
        )
        self.softmax_temp = ViewerNumber(
            name="Softmax temperature",
            default_value=self.viewer_utils.softmax_temp,
            cb_hook=lambda elem: self.viewer_utils.update_softmax_temp(elem.value),
        )

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
        Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
        param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        
        repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
        param_state["exp_avg"] = torch.cat(
            [param_state["exp_avg"], torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims)],
            dim=0,
        )
        param_state["exp_avg_sq"] = torch.cat(
            [
                param_state["exp_avg_sq"],
                torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
            ],
            dim=0,
        )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def after_train(self, step: int):
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.grad.detach().norm(dim=-1)  # TODO fill in
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask], newradii / float(max(self.last_size[0], self.last_size[1]))
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, back_color: torch.Tensor):
        assert back_color.shape == (3,)
        self.back_color = back_color

    def refinement_after(self, optimizers: Optimizers, step):
        if self.step >= self.config.warmup_length:
            with torch.no_grad():
                # only split/cull if we've seen every image since opacity reset
                reset_interval = self.config.reset_alpha_every * self.config.refine_every
                if (
                    self.step < self.config.stop_split_at
                    and self.step % reset_interval > self.num_train_data + self.config.refine_every
                ):
                    # then we densify
                    assert (
                        self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                    )
                    avg_grad_norm = (
                        (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                    )
                    high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                    splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                    if self.step < self.config.stop_screen_size_at:
                        splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                    splits &= high_grads
                    nsamps = self.config.n_split_samples
                    (
                        split_means,
                        split_colors,
                        split_opacities,
                        split_scales,
                        split_quats,
                        split_clip_embeds,
                    ) = self.split_gaussians(splits, nsamps)

                    dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                    dups &= high_grads
                    
                    # dup_means, dup_colors, dup_opacities, dup_scales, dup_quats = self.dup_gaussians(dups)
                    dup_means, dup_colors, dup_opacities, dup_scales, dup_quats, dup_clip_embeds = self.dup_gaussians(dups)
                    
                    self.means = Parameter(torch.cat([self.means.detach(), split_means, dup_means], dim=0))
                    self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))

                    self.opacities = Parameter(
                        torch.cat([self.opacities.detach(), split_opacities, dup_opacities], dim=0)
                    )
                    self.scales = Parameter(torch.cat([self.scales.detach(), split_scales, dup_scales], dim=0))
                    self.quats = Parameter(torch.cat([self.quats.detach(), split_quats, dup_quats], dim=0))
                    
                    self.clip_embeds = Parameter(
                        torch.cat([self.clip_embeds.detach(), split_clip_embeds, dup_clip_embeds], dim=0)
                    )
                    
                    # append zeros to the max_2Dsize tensor
                    self.max_2Dsize = torch.cat(
                        [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                        dim=0,
                    )
                    split_idcs = torch.where(splits)[0]
                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        # if group == "clip_embeds":
                        #     continue
                        
                        self.dup_in_optim(optimizers.optimizers[group], split_idcs, param, n=nsamps)
                    dup_idcs = torch.where(dups)[0]

                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        # if group == "clip_embeds":
                        #     continue
                        
                        self.dup_in_optim(optimizers.optimizers[group], dup_idcs, param, 1)

                # Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
                if self.step % reset_interval > self.num_train_data + self.config.refine_every:
                    # then cull
                    deleted_mask = self.cull_gaussians()
                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        # if group == "clip_embeds":
                        #     continue
                        
                        self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)

                if self.step % reset_interval == self.config.refine_every:
                    reset_value = self.config.cull_alpha_thresh * 0.8
                    self.opacities.data = torch.full_like(
                        self.opacities.data, torch.logit(torch.tensor(reset_value)).item()
                    )
                    # reset the exp of optimizer
                    optim = optimizers.optimizers["opacity"]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                    param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            culls = culls | toobigs
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
        self.means = Parameter(self.means[~culls].detach())
        self.scales = Parameter(self.scales[~culls].detach())
        self.quats = Parameter(self.quats[~culls].detach())
        self.colors_all = Parameter(self.colors_all[~culls].detach())
        self.opacities = Parameter(self.opacities[~culls].detach())
        self.clip_embeds = Parameter(self.clip_embeds[~culls].detach())

        print(f"Culled {n_bef - self.num_points} gaussians")
        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        # step 6, sample new CLIP embeddings
        new_clip_embeds = self.clip_embeds[split_mask].repeat(samps, 1)
        return new_means, new_colors_all, new_opacities, new_scales, new_quats, new_clip_embeds

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        dup_means = self.means[dup_mask]
        dup_colors = self.colors_all[dup_mask]
        dup_opacities = self.opacities[dup_mask]
        dup_scales = self.scales[dup_mask]
        dup_quats = self.quats[dup_mask]
        dup_clip_embeds = self.clip_embeds[dup_mask]
        return dup_means, dup_colors, dup_opacities, dup_scales, dup_quats, dup_clip_embeds

    @property
    def num_points(self):
        return self.means.shape[0]

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "xyz": [self.means],
            "color": [self.colors_all],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
            "clip_embeds": [self.clip_embeds],
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        
        # insert parameters for the CLIP Encoder-Decoder
        gps["clip_encoder"] = list(self.clip_encoder.parameters())
        gps["clip_decoder"] = list(self.clip_decoder.parameters())              
        
        # add camera optimizer param groups
        self.camera_optimizer.get_param_groups(gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.config.num_downscales - self.step // self.config.resolution_schedule), 0)
        else:
            return 1
        
    @torch.no_grad()
    def get_semantic_outputs(self, outputs: Dict[str, torch.Tensor]):
        if not self.training:
            # Normalize CLIP features rendered by feature field
            clip_features = outputs["clip"]
            # clip_features /= clip_features.norm(dim=-1, keepdim=True)
            
            clip_features = self.clip_decoder(clip_features.view(-1, self.clip_embeds_latent_dim)).view(*outputs["clip"].shape[:-1],
                                                                                                        self.clip_embeds_input_dim).float()
            clip_features /= clip_features.norm(dim=-1, keepdim=True)
            
            # if self.datamanager.clip_interpolator.aux_data is None:
            #     clip_features /= clip_features.norm(dim=-1, keepdim=True) # if not PCA

            if self.viewer_utils.has_positives:
                if self.viewer_utils.has_negatives:
                    # Use paired softmax method as described in the paper with positive and negative texts
                    text_embs = torch.cat([self.viewer_utils.pos_embed, self.viewer_utils.neg_embed], dim=0)
                    raw_sims = clip_features @ text_embs.T

                    # Broadcast positive label similarities to all negative labels
                    pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
                    pos_sims = pos_sims.broadcast_to(neg_sims.shape)
                
                    # Updated Code
                    paired_sims = torch.cat((pos_sims.reshape((-1, 1)), neg_sims.reshape((-1, 1))), dim=-1)
                    
                    # compute the paired softmax
                    probs = paired_sims.softmax(dim=-1)[..., :1]
                    probs = probs.reshape((-1, neg_sims.shape[-1]))
                    
                    torch.nan_to_num_(probs, nan=0.0)
                    
                    sims, _ = probs.min(dim=-1, keepdim=True)
                    outputs["similarity"] = sims.reshape((*pos_sims.shape[:-1], 1))
                else:
                    # positive embeddings
                    text_embs = self.viewer_utils.pos_embed
                    sims = clip_features @ text_embs.T
                    # Show the mean similarity if there are multiple positives
                    if sims.shape[-1] > 1:
                        sims = sims.mean(dim=-1, keepdim=True)
                    outputs["similarity"] = sims

                # for outputs similar to the GUI
                similarity_clip = outputs[f"similarity"] - outputs[f"similarity"].min()
                similarity_clip /= (similarity_clip.max() + 1e-10)
                outputs["similarity_GUI"] = apply_colormap(similarity_clip,
                                                        ColormapOptions("turbo"))

    
            if self.viewer_utils.has_positives:
                # composited similarity
                p_i = torch.clip(outputs["similarity"] - 0.5, 0, 1)
                
                outputs["composited_similarity"] = apply_colormap(p_i / (p_i.max() + 1e-6), ColormapOptions("turbo"))
                mask = (outputs["similarity"] < 0.51).squeeze()
                outputs["composited_similarity"][mask, :] = outputs["rgb"][mask, :]
            
        return outputs


    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        
        if self.training:
            # currently relies on the branch vickie/camera-grads
            self.camera_optimizer.apply_to_camera(camera)
        if self.training:
            background = torch.rand(3, device=self.device)
        else:
            # logic for setting the background of the scene
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE
            else:
                background = self.back_color.to(self.device)
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return {"rgb": background.repeat(int(camera.height.item()), int(camera.width.item()), 1)}
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device="cuda", dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = camera.width.item(), camera.height.item()
        self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            (W + BLOCK_X - 1) // BLOCK_X,
            (H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            colors_crop = self.colors_all[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            clip_embeds_crop = self.clip_embeds[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            colors_crop = self.colors_all
            scales_crop = self.scales
            quats_crop = self.quats
            clip_embeds_crop = self.clip_embeds
        self.xys, depths, self.radii, conics, num_tiles_hit, _ = ProjectGaussians.apply(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            tile_bounds,
        )
        if (self.radii).sum() == 0:
            return {"rgb": background.repeat(int(camera.height.item()), int(camera.width.item()), 1)}

        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()
        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = SphericalHarmonics.apply(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)  # type: ignore
        else:
            rgbs = self.get_colors.squeeze()  # (N, 3)
            rgbs = torch.sigmoid(rgbs)
            
        # start = time.perf_counter()
        rgb = RasterizeGaussians.apply(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            rgbs,
            torch.sigmoid(opacities_crop),
            H,
            W,
            background,
        )
        
        # print(f"time rendering img: {time.perf_counter() - start}")
        
        depth_im = None
        if not self.training:
            depth_im = RasterizeGaussians.apply(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,
                depths[:, None].repeat(1, 3),
                torch.sigmoid(opacities_crop),
                H,
                W,
                torch.ones(3, device=self.device) * 10,
            )[..., 0:1]
        
        # CLIP Embeddings
        clip_embeds_im = RasterizeGaussians.apply(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            clip_embeds_crop,
            torch.sigmoid(opacities_crop),
            H,
            W,
            self.clip_background,
        )
            
        # rescale the camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)
        
        # outputs
        outputs = {"rgb": rgb,
                   "depth": depth_im,
                   "clip": clip_embeds_im,
                }  # type: ignore
        
        if not self.training:
            # Compute semantic inputs, e.g., composited semantic similarity.
            outputs = self.get_semantic_outputs(outputs=outputs)
            
        return outputs

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
        metrics_dict = {}
        gt_rgb = gt_img.to(self.device)  # RGB or RGBA image
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        metrics_dict["gaussian_count"] = self.num_points
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        # Dimensionality reduction on the CLIP embeddings
        clip_enc_inputs = batch["clip"].view(-1, self.clip_embeds_input_dim)
        
        # latent CLIP embeddings
        clip_latent = self.clip_encoder(clip_enc_inputs)
        
        # reconstructed CLIP embeddings
        clip_recon = self.clip_decoder(clip_latent).view(*batch["clip"].shape[:-1], self.clip_embeds_input_dim).float()
        clip_recon = torch.nn.functional.normalize(clip_recon, p=2.0)
        
        # # reshape latent CLIP embeddings
        clip_latent = clip_latent.view(*batch["clip"].shape[:-1], self.clip_embeds_latent_dim).float()
        
        d = self._get_downscale_factor()
        
        # start = time.perf_counter()
        if d > 1:
            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
            
        if clip_latent.shape[:-1] != outputs["clip"].shape[:-1]:
            # CLIP Embeddings
            clip_latent_img = TF.resize(clip_latent.permute(2, 0, 1), outputs["clip"].shape[:-1],
                                        interpolation=TF.InterpolationMode.BILINEAR,  # TF.InterpolationMode.NEAREST
                                        antialias=None).permute(1, 2, 0)
        
        # CLIP-related Loss
        # Encoder-Decoder Loss
        clip_network_loss = self.config.clip_network_loss_weight * torch.nn.functional.mse_loss(
            clip_recon, batch["clip"])
        
        # Rendering Loss 
        clip_img_loss = self.config.clip_img_loss_weight * torch.nn.functional.mse_loss(
            outputs["clip"], clip_latent_img)
        
        Ll1 = torch.abs(gt_img - outputs["rgb"]).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], outputs["rgb"].permute(2, 0, 1)[None, ...])
        if self.step % 10 == 0:
            # Before, we made split sh and colors onto different optimizer, with shs having a low learning rate
            # This is slow, instead we apply a regularization every few steps
            sh_reg = self.colors_all[:, 1:, :].norm(dim=1).mean()
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(self.config.max_gauss_ratio)
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            sh_reg = torch.tensor(0.0).to(self.device)
            scale_reg = torch.tensor(0.0).to(self.device)
        
        # print(f"time LOSS img: {time.perf_counter() - start}")
        
        # main loss
        main_loss = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss \
            + clip_network_loss + clip_img_loss
        
        return {
            "main_loss": main_loss,
            "sh_reg": sh_reg,
            "scale_reg": scale_reg,
        }

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
            predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
            predicted_rgb = outputs["rgb"]

        gt_rgb = gt_img.to(self.device)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict