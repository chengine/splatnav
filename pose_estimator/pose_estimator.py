# %%
from __future__ import annotations

import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union

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

from ns_utils.nerfstudio_utils import GaussianSplat
from pose_estimator.utils import *

# # #
# # # Pose Estimator
# # #

from dataclasses import dataclass

@dataclass
class SplatLoc():
    def __init__(self, gsplat):
        # option to visualize point clouds
        self.visualize_RGBD_point_clouds = False

        # option to display computation time statistics
        self.display_compute_stats = True

        # option to print the error stats
        self.print_error_stats = True

        # option to visualize registration results
        self.enable_registration_visualization = False

        # option to visualize matches in PnP-RANSAC
        self.visualize_PnP_matches = False

        # global registration method (RANSAC, FGR)
        self.global_registration_method = Global_Registration.RANSAC

        # local pose refinement method (ICP, PnP_RANSAC)
        self.local_refinement_method = Local_Refinement.PnP_RANSAC

        # local pose refinement method (ICP, COLORED_ICP)
        self.local_registration_method = Local_Registration.COLORED_ICP
        
        # figures
        self.figures_dir = 'figures'

        # option to disable global registration after the first timestep
        self.disable_global_registration = True
        

        # Global registration is required for local registration.
        self.disable_global_registration = self.disable_global_registration \
            if self.local_refinement_method != Local_Refinement.ICP \
                else False

        # feature detector for PnP-RANSAC
        # options: POI_Detector.SIFT, POI_Detector.SURF, POI_Detector.ORB,
        #          POI_Detector.LIGHTGLUE
        # options = [detector for detector in POI_Detector]
        self.feature_detector = POI_Detector.LIGHTGLUE

        # LightGlue configuration parameters
        self.lightglue_configs = {'extractor': LIGHTGLUE_Extractor.SUPERPOINT,
                                  'max_num_keypoints': 2048
                                }

        # detector parameters
        self.detector_params = {}   

        # voxel size for downsampling the point cloud in the pose estimation module
        self.voxel_size = 0.05
        
        if self.feature_detector == POI_Detector.LIGHTGLUE:
            # setup LightGlue
            extractor, matcher = setup_lightglue(feature_extractor=self.lightglue_configs['extractor'],
                                                max_num_keypoints=self.lightglue_configs['max_num_keypoints'])
            
            # detector parameters
            self.detector_params['extractor'] = extractor
            self.detector_params['matcher'] = matcher
            
        # print options
        self.print_options = PrintOptions()
        
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # GSplat
        self.gsplat = gsplat

        # prior distribution for the UKF
        # mean of the prior distribution
        self.ukf_prior_mean = None

        # covariance of the prior distribution
        self.ukf_prior_cov = None

        # update the current estimate
        # mean of the distribution
        self.ukf_mean = self.ukf_prior_mean

        # covariance of the distribution
        self.ukf_cov = self.ukf_prior_cov

        # initialize the previous estimates
        self.ukf_mean_prev = self.ukf_mean

        # parameter Kappa for the UKF
        self.ukf_kappa = 2.0

        # parameter dt for the UKF's timestep
        self.ukf_dt = 1e-4

        # set the process and measurement noise covariances
        # self.ukf_process_noise_cov = 4.5e1 * torch.eye(6, device=self.gsplat.device)
        self.ukf_process_noise_cov = torch.diag(torch.tensor([2.5e1, 2.5e1, 2.5e1, 4.5e1, 4.5e1, 4.5e1])).to(self.device)
        self.ukf_meas_noise_cov = 3e-3 * torch.eye(6, device=self.gsplat.device)

    def set_prior_distribution(self, 
                               mu: torch.Tensor,
                               sigma: torch.Tensor
                               ):
        # mean of the prior distribution
        self.ukf_prior_mean = SE3_to_se3(mu)

        # covariance of the prior distribution
        self.ukf_prior_cov = sigma

        # update the current estimate
        # mean of the distribution
        self.ukf_mean = self.ukf_prior_mean

        # covariance of the distribution
        self.ukf_cov = self.ukf_prior_cov

        # initialize the previous estimates
        self.ukf_mean_prev = self.ukf_mean

    def set_kappa(self,
                  kappa: float
                  ):
        
        # set the parameter Kappa for the UKF
        self.ukf_kappa = kappa

    def set_dt(self,
               dt: float
               ):
        
        # set the parameter Kappa for the UKF
        self.ukf_dt = dt
        
    def estimate(self,
                 init_guess: torch.Tensor,
                 cam_rgb: torch.Tensor,
                 cam_K: torch.Tensor
                 ):

        if self.local_refinement_method == Local_Refinement.PnP_RANSAC:
            # #
            # # Perspective-n-Point (PnP)
            # #

            if self.print_error_stats or self.display_compute_stats:
                print(self.print_options.sep_0)
                print(self.print_options.sep_1)
                print(f"Local Refinement: PnP-RANSAC")
                print(self.print_options.sep_1)
                
            # start time
            t0 = time.perf_counter()
            
            # estimated pose
            local_est_pose = execute_PnP_RANSAC(self.gsplat, init_guess,
                                                camera_intrinsics_K=cam_K,
                                                rgb_input=cam_rgb,
                                                feature_detector=self.feature_detector,
                                                save_image=False,
                                                pnp_figure_filename=f'{self.figures_dir}/pnp_init_guess.png',
                                                print_stats=True,
                                                visualize_PnP_matches=False,
                                                pnp_matches_figure_filename=f'{self.figures_dir}/pnp_matches.png',
                                                detector_params=self.detector_params)
            # end time
            t1 = time.perf_counter()
            
            if self.display_compute_stats:
                print(f"Local Registration took {t1 - t0} seconds!")
            
            return torch.tensor(local_est_pose, device=self.device).float()

    def estimate_ukf(self,
                     init_guess: torch.Tensor,
                     cam_rgb: torch.Tensor,
                     cam_K: torch.Tensor
                     ):
        
        # start time
        t0_ukf = time.perf_counter()
        
        # # map from SE(3) to se(3), the lie algebra
        # y_mean_SE3 = SE3_to_se3(init_guess)

        # compute the sigmapoints of the prior
        prior_sigmapts = compute_sigmapoints(self.ukf_mean, self.ukf_cov, kappa=self.ukf_kappa)

        # estimated velocity
        est_vel = self.ukf_mean - self.ukf_mean_prev

        # propagate the dynamics
        prop_dynamics_sigmapts = dynamics_step(prior_sigmapts,
                                               vel=est_vel,
                                               dt=self.ukf_dt)

        # compute the prediction distribution
        pred_dist_mu, pred_dist_cov, mu_x_diff = inverse_sigma_transform_prediction(prop_dynamics_sigmapts,
                                                                                     kappa=self.ukf_kappa,
                                                                                     process_noise_cov=self.ukf_process_noise_cov)

        # update the initial guess for PnP-RANSAC
        # map from SE(3) to se(3), the lie algebra
        init_guess = se3_to_SE3(pred_dist_mu)
 
        if self.local_refinement_method == Local_Refinement.PnP_RANSAC:
            # #
            # # Perspective-n-Point (PnP)
            # #

            if self.print_error_stats or self.display_compute_stats:
                print(self.print_options.sep_0)
                print(self.print_options.sep_1)
                print(f"UKF Local Refinement: PnP-RANSAC")
                print(self.print_options.sep_1)
                
            # start time
            t0 = time.perf_counter()
            
            # estimated pose
            local_est_pose = execute_PnP_RANSAC(self.gsplat, init_guess,
                                                camera_intrinsics_K=cam_K,
                                                rgb_input=cam_rgb,
                                                feature_detector=self.feature_detector,
                                                save_image=True,
                                                pnp_figure_filename=f'{self.figures_dir}/pnp_init_guess.png',
                                                print_stats=True,
                                                visualize_PnP_matches=False,
                                                pnp_matches_figure_filename=f'{self.figures_dir}/pnp_matches.png',
                                                detector_params=self.detector_params)
            # end time
            t1 = time.perf_counter()
            
            if self.display_compute_stats:
                print(f"Local Registration took {t1 - t0} seconds!")

            
            # get the pseudo-measurement
            pseudo_meas = torch.tensor(local_est_pose, device=self.device).float()

        # compute the sigmapoints associated with the prediction distribution
        pred_sigmapts = compute_sigmapoints(pred_dist_mu, pred_dist_cov, kappa=self.ukf_kappa)

        # compute the posterior distribution
        meas_sigmapts = measurement_function(pred_sigmapts)

        # compute the posterior distribution prior to new measurements
        y_mean_sigmapts, yy_cov_sigmapts, xy_cov_sigmpts = inverse_sigma_transform_measurement(mu_x_diff=mu_x_diff,
                                                                                               pred_x_mean=pred_dist_mu,
                                                                                               meas_mu_samples=meas_sigmapts,
                                                                                               kappa=self.ukf_kappa,
                                                                                               meas_noise_cov=self.ukf_meas_noise_cov
                                                                                               )
        
        # compute the posterior incorporating new measurements
        posterior_mu, posterior_cov = correction_phase_post_measurement(y=pseudo_meas,
                                                                        pred_x_mean=pred_dist_mu,
                                                                        y_mean=y_mean_sigmapts,
                                                                        pred_xx_cov=pred_dist_cov,
                                                                        yy_cov=yy_cov_sigmapts,
                                                                        xy_cov=xy_cov_sigmpts
                                                                        )
        
        # update the previous estimates
        self.ukf_mean_prev = self.ukf_mean

        # current estimates
        self.ukf_mean = posterior_mu
        self.ukf_cov = posterior_cov

        if self.display_compute_stats:
            print(f"UKF took {time.perf_counter() - t0_ukf} seconds!")
            # print(f"Determinant of the covariance: {torch.linalg.det(self.ukf_cov)}")

        return se3_to_SE3(self.ukf_mean)
