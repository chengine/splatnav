from __future__ import annotations

import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union, Tuple

from typing_extensions import Annotated
import pickle
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib as mpl
from tqdm import tqdm
import open3d as o3d

from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.models.splatfacto import SplatfactoModel
# from gemsplat.gemsplat import GemSplatModel
# from nerfstudio.scripts.render import RenderInterpolated

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio
from nerfstudio.data.datasets.base_dataset import InputDataset

# # # # #
# # # # # Utils
# # # # #

# From INRIA
# Base auxillary coefficient
C0 = 0.28209479177387814

def SH2RGB(sh):
    return sh * C0 + 0.5

class GaussianSplat():
    def __init__(self, config_path: Path, res_factor=None,
        test_mode: Literal["test", "val", "inference"] = "inference",
        dataset_mode: Literal["train", "val", "test"] = 'test',
        device: Union[torch.device, str] = "cpu"
    ) -> None:
        # config path
        self.config_path = config_path

        # camera rescale resolution factor
        self.res_factor = res_factor

        # device
        self.device = device

        # initialize pipeline
        self.init_pipeline(test_mode)

        # load dataset
        self.load_dataset(dataset_mode)

        # load cameras
        self.get_cameras()

    def init_pipeline(self,
        test_mode: Literal["test", "val", "inference"]
    ):
        # Get config and pipeline
        self.config, self.pipeline, _, _ = eval_setup(
            self.config_path, 
            test_mode=test_mode,
        )

    def load_dataset(self,
        dataset_mode: Literal["train", "val", "test"]
    ):
        # return dataset
        if dataset_mode == "train":
            self.dataset = self.pipeline.datamanager.train_dataset
        elif dataset_mode in ["val", "test"]:
            self.dataset = self.pipeline.datamanager.eval_dataset
        else:
            ValueError('Incorrect value for datset_mode. Accepted values include: dataset_mode: Literal["train", "val", "test"].')

    def get_cameras(self):
        # Camera object contains camera intrinsics and extrinsics
        self.cameras = self.dataset.cameras

        if self.res_factor is not None:
            self.cameras.rescale_output_resolution(self.res_factor)

    def get_poses(self):
        return self.cameras.camera_to_worlds
    
    def get_images(self):
        # images
        images = [self.dataset.get_image_float32(image_idx)
                  for image_idx 
                  in range(len(self.dataset._dataparser_outputs.image_filenames))]
        
        return images
    
    def get_camera_intrinsics(self):
        K = self.cameras[0].get_intrinsics_matrices().squeeze()
        # width and height
        W = int(self.cameras[0].width.item())
        H = int(self.cameras[0].height.item())
        return H, W, K

    def render(self, pose, 
               compute_semantics: Optional[bool] = False,
               debug_mode=False):
        # Render from a single pose
        camera_to_world = pose[None,:3, ...]

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=self.cameras[0].fx,
            fy=self.cameras[0].fy,
            cx=self.cameras[0].cx,
            cy=self.cameras[0].cy,
            width=self.cameras[0].width,
            height=self.cameras[0].height,
            camera_type=CameraType.PERSPECTIVE,
        )
        
        cameras = cameras.to(self.device)
        
        # render outputs
        if isinstance(self.pipeline.model, NerfactoModel):
            aabb_box = None
            camera_ray_bundle = cameras.generate_rays(camera_indices=0, aabb_box=aabb_box)

            tnow = time.perf_counter()
            with torch.no_grad():
                outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

            if debug_mode:
                print('Rendering time: ', time.perf_counter() - tnow)

            # insert ray bundles
            outputs['ray_bundle'] = camera_ray_bundle
        elif isinstance(self.pipeline.model, SplatfactoModel): # or isinstance(self.pipeline.model, GemSplatModel):
            obb_box = None
            
            tnow = time.perf_counter()
            with torch.no_grad():
                try:
                    outputs = self.pipeline.model.get_outputs_for_camera(cameras, obb_box=obb_box,
                                                                        compute_semantics=compute_semantics)
                except:  
                    outputs = self.pipeline.model.get_outputs_for_camera(cameras, obb_box=obb_box)

            if debug_mode:
                print('Rendering time: ', time.perf_counter() - tnow)

        return outputs
    
    def generate_point_cloud(self,
                             use_bounding_box: bool = False,
                             bounding_box_min: Optional[Tuple[float, float, float]] = (-1, -1, -1),
                             bounding_box_max: Optional[Tuple[float, float, float]] = (1, 1, 1),
                             densify_scene: bool = False,
                             split_params: Dict = {
                                 'n_split_samples': 2
                             },
                             cull_scene: bool = False,
                             cull_params: Dict = {
                                 'cull_alpha_thresh': 0.1,
                                 'cull_scale_thresh': 0.5
                             }
                             )->None:
        if densify_scene:
            if cull_scene:
                # cache the previous values of all parameters
                means_prev = self.pipeline.model.means.clone()
                scales_prev = self.pipeline.model.scales.clone()
                quats_prev = self.pipeline.model.quats.clone() 
                features_dc_prev = self.pipeline.model.features_dc.clone()
                features_rest_prev = self.pipeline.model.features_rest.clone()
                opacities_prev = self.pipeline.model.opacities.clone()
                try:
                    clip_embeds_prev = self.pipeline.model.clip_embeds.clone()
                except:
                    clip_embeds_prev = None
                # cull Gaussians
                self.pipeline.model.cull_gaussians_refinement(cull_alpha_thresh=cull_params['cull_alpha_thresh'],
                                                              cull_scale_thresh=cull_params['cull_scale_thresh'],
                                                              )
                
            # split mask
            split_mask = torch.ones(len(self.pipeline.model.scales),
                                    dtype=torch.bool).to(self.device)

            try:
                # split Gaussians
                means, features_dc, features_rest, opacities, scales, quats, \
                clip_embeds \
                    = self.pipeline.model.split_gaussians(split_mask, split_params['n_split_samples'])
            except:
                # split Gaussians
                means, features_dc, features_rest, opacities, scales, quats, \
                 \
                    = self.pipeline.model.split_gaussians(split_mask, split_params['n_split_samples'])
            
            # 3D points
            pcd_points = means

            # colors computed from the term of order 0 in the Spherical Harmonic basis
            # coefficient of the order 0th-term in the Spherical Harmonics basis
            pcd_colors_coeff = features_dc
        else:
            # 3D points
            pcd_points = self.pipeline.model.means

            # colors computed from the term of order 0 in the Spherical Harmonic basis
            # coefficient of the order 0th-term in the Spherical Harmonics basis
            pcd_colors_coeff = self.pipeline.model.features_dc
            
            try:
                # other attributes of the Gaussian
                opacities, scales, quats, clip_embeds \
                    = self.pipeline.model.opacities, self.pipeline.model.scales, self.pipeline.model.quats, \
                        self.pipeline.model.clip_embeds
            except:
                # other attributes of the Gaussian
                opacities, scales, quats \
                    = self.pipeline.model.opacities, self.pipeline.model.scales, self.pipeline.model.quats

        # color computed from the Spherical Harmonics
        pcd_colors = SH2RGB(pcd_colors_coeff).squeeze()
        
        # mask points using a bounding box
        if use_bounding_box:
            mask = ((pcd_points[:, 0] > bounding_box_min[0]) & (pcd_points[:, 0] < bounding_box_max[0])
                    & (pcd_points[:, 1] > bounding_box_min[1]) & (pcd_points[:, 1] < bounding_box_max[1])
                    & (pcd_points[:, 2] > bounding_box_min[2]) & (pcd_points[:, 2] < bounding_box_max[2])
            )
            
            pcd_points = pcd_points[mask]
            pcd_colors = pcd_colors[mask]
            
            try:
                # other attributes of the Gaussian
                opacities, scales, quats, clip_embeds \
                    = opacities[mask], scales, quats[mask], clip_embeds[mask]
            except:
                # other attributes of the Gaussian
                opacities, scales, quats, clip_embeds \
                    = opacities[mask], scales, quats[mask], None
                    
        else:
            mask = None
            
        # apply transformation to the opacities and scales
        scales = torch.exp(scales)
        opacities = torch.sigmoid(opacities)

        # enviromment attributes
        env_attr = {'means': pcd_points,
                    'quats': quats,
                    'scales': scales,
                    'opacities': opacities,
                    'clip_embeds': clip_embeds}

        # create the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points.double().cpu().detach().numpy())
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors.double().cpu().detach().numpy())
        
        # reset the values of all parameters
        if cull_scene:
            self.pipeline.model.means = torch.nn.Parameter(means_prev)
            self.pipeline.model.scales = torch.nn.Parameter(scales_prev)
            self.pipeline.model.quats = torch.nn.Parameter(quats_prev)
            self.pipeline.model.features_dc = torch.nn.Parameter(features_dc_prev)
            self.pipeline.model.features_rest = torch.nn.Parameter(features_rest_prev)
            self.pipeline.model.opacities = torch.nn.Parameter(opacities_prev)
            try:
                self.pipeline.model.clip_embeds = torch.nn.Parameter(clip_embeds_prev)
            except:
                pass
                
        return pcd, mask, env_attr
    
    def get_semantic_point_cloud(self,
                                 positives: str = "",
                                 negatives: str = "object, things, stuff, texture",
                                 pcd_attr: Dict[str, torch.Tensor] = {}):
        # update the list of positives (objects)
        self.pipeline.model.viewer_utils.handle_language_queries(raw_text=positives,
                                                                 is_positive=True)
        
        # update the list of positives (objects)
        self.pipeline.model.viewer_utils.handle_language_queries(raw_text=negatives,
                                                                 is_positive=False)
        
        # semantic point cloud
        # CLIP features
        if 'clip_embeds' in pcd_attr.keys():
            pcd_clip = pcd_attr['clip_embeds']
        else:
            pcd_clip = self.pipeline.model.clip_embeds
        
        # get the semantic outputs
        pcd_clip = {'clip': pcd_clip, }
        semantic_pcd = self.pipeline.model.get_semantic_outputs(pcd_clip)
        
        return semantic_pcd
    
    # Generate RGB-D Image
    def generate_RGBD_point_cloud(self, pose,
                                  save_image: bool=False,
                                  filename: Optional[str]='/',
                                  compute_semantics: Optional[bool] = False,
                                  max_depth: Optional[float] = 1.0,
                                  return_pcd: Optional[bool] = True,
                                  positives: str = "",
                                  negatives: str = "object, things, stuff, texture"
                                 ):
        # update the semantic-query information
        if compute_semantics:
            # update the list of positives (objects)
            self.pipeline.model.viewer_utils.handle_language_queries(raw_text=positives,
                                                                     is_positive=True)
            
            # update the list of positives (objects)
            self.pipeline.model.viewer_utils.handle_language_queries(raw_text=negatives,
                                                                     is_positive=False)
            
        # pose to render from
        pose = pose.to(self.device)
        
        # render
        outputs = self.render(pose,
                              compute_semantics=compute_semantics)
        
        if save_image:
            # figure
            fig, axs = plt.subplots(2, 1, figsize=(15, 15))
            plt.tight_layout()

            # plot rendering
            axs[0].imshow(outputs['rgb'].cpu().numpy())
            axs[1].imshow(outputs['depth'].cpu().numpy())

            for ax in axs:
                ax.set_axis_off()
            plt.show()
            
            # create directory, if needed
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # save figure
            fig.savefig(filename)

        # create a point cloud from RGB-D image
        # depth channel
        cam_depth = outputs['depth'].squeeze()
        cam_rgb = outputs['rgb']
        
        # depth mask
        if max_depth is None:
            depth_mask = torch.ones_like(cam_depth, dtype=bool, device=cam_depth.device)
        else:
            depth_mask = cam_depth < max_depth 
        
        # start time
        t0 = time.perf_counter()
        
        # camera intrinsics
        H, W, K = self.get_camera_intrinsics()

        # unnormalized pixel coordinates
        u_coords = torch.arange(W, device=self.device)
        v_coords = torch.arange(H, device=self.device)

        # meshgrid
        U_grid, V_grid = torch.meshgrid(u_coords, v_coords, indexing='xy')

        # transformed points in camera frame
        # [u, v, 1] = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]] @ [x/z, y/z, 1]
        cam_pts_x = (U_grid - K[0, 2]) * cam_depth / K[0, 0]
        cam_pts_y = (V_grid - K[1, 2]) * cam_depth / K[1, 1]
        
        cam_pcd_points = torch.stack((cam_pts_x, cam_pts_y, cam_depth), axis=-1)

        if return_pcd:
            # point cloud
            cam_pcd = o3d.geometry.PointCloud()
            cam_pcd.points = o3d.utility.Vector3dVector(cam_pcd_points[depth_mask, ...].view(-1, 3).double().cpu().detach().numpy())
            cam_pcd.colors = o3d.utility.Vector3dVector(cam_rgb[depth_mask, ...].view(-1, 3).double().cpu().detach().numpy())
            
        else:
            cam_pcd = None
        
        return cam_rgb, cam_pcd_points, cam_pcd, depth_mask, outputs
    
def load_dataset(data_path: Path,
    dataset_mode: Literal["train", "val", "test", "all"] # 'all' uses the entire dataset.
):
    # init Nerfstudio dataset config
    nerfstudio_data_parser_config = NerfstudioDataParserConfig(data=data_path,
                                                               eval_mode="all" if dataset_mode == "all" else "fraction")

    # init data parser
    nerfstudio_data_parser = Nerfstudio(nerfstudio_data_parser_config)

    # data parser outputs
    data_parser_ouputs = nerfstudio_data_parser._generate_dataparser_outputs(split=dataset_mode
                                                                             if dataset_mode != "all" else "val")

    # load dataset
    dataset = InputDataset(data_parser_ouputs)
    
    return dataset

def load_model(config_path: Path):
    # rescale factor
    res_factor = None

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize NeRF
    gsplat = GaussianSplat(config_path=config_path,
                res_factor=res_factor,
                test_mode="test", # [options: "test", "inference", "val"]
                dataset_mode="val",
                device=device)
    
    return gsplat
