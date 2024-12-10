import numpy as np
import os, sys
import time
import json
import imageio
import open3d as o3d
import copy
import cv2
import matplotlib.pyplot as plt
from enum import Enum, auto
import torch
import open3d as o3d

from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union
from open3d.pipelines.registration import Feature

from lightglue import LightGlue, SuperPoint, DISK, ALIKED, DoGHardNet
from lightglue.utils import rbd
from scipy.linalg import logm as sci_logm

from torch.autograd.gradcheck import get_numerical_jacobian

import copy

# add to the PATH
sys.path.append(f"{Path(__file__).parent.parent}")

from ns_utils.nerfstudio_utils import *
import pdb

# # # # #
# # # # # Utils
# # # # #

# scaling factor for FPFH
FPFH_MAX = 200.0


class DisablePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class PrintOptions:
    """
    Print Options for printing to the CONSOLE.
    """

    def __init__(self, width=100):
        # separators
        self.sep_0 = "*" * width
        self.sep_1 = "-" * width
        self.sep_space = "\x0c" * 3


class POI_Detector(Enum):
    SIFT = auto()
    ORB = auto()
    SURF = auto()
    LIGHTGLUE = auto()


class LIGHTGLUE_Extractor(Enum):
    SUPERPOINT = auto()
    DISK = auto()
    ALIKED = auto()
    DOGHARDNET = auto()


class Global_Registration(Enum):
    # RANSAC
    RANSAC = auto()
    # Fast Global Registration
    FGR = auto()


class Local_Registration(Enum):
    # Iterative Closest Point (ICP)
    ICP = auto()
    # Colored Iterative Closest Point
    COLORED_ICP = auto()


class Local_Refinement(Enum):
    # Iterative Closest Point (ICP)and its variant
    ICP = auto()
    # PnP-RANSAC
    PnP_RANSAC = auto()
    # iNeRF
    iNeRF = auto()


class Open3dCustomFeature(Feature):
    def __init__(self, feat):
        super().__init__(feat)

        @Feature.data.setter
        def data(self, value):
            self.data = value


def SE3error(T, That):
    Terr = np.linalg.inv(T) @ That
    rerr = abs(np.arccos(min(max(((Terr[0:3, 0:3]).trace() - 1) / 2, -1.0), 1.0)))
    terr = np.linalg.norm(Terr[0:3, 3])
    return (rerr * 180 / np.pi, terr)


def downsample_point_cloud(pcd, voxel_size=0.01, print_stats: bool = False):
    # downsample point cloud
    if print_stats:
        print(f":: Downsample with a voxel size {voxel_size:3f}")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    return pcd_down


def visualize_point_cloud(
    pcds: List[object] = [],
    enable_downsampled_visualization: bool = True,
    downsampled_voxel_size: float = 0.01,
):
    # visualize point cloud
    if enable_downsampled_visualization:
        pcds_down = [
            downsample_point_cloud(pcd, voxel_size=downsampled_voxel_size)
            for pcd in pcds
        ]

        o3d.visualization.draw_plotly(pcds_down)
    else:
        o3d.visualization.draw_plotly(pcds)


def visualize_registration_result(
    source,
    target,
    transformation,
    enable_downsampled_visualization: bool = True,
    downsampled_voxel_size: float = 0.01,
):
    source_disp = copy.deepcopy(source)
    source_disp.transform(transformation)

    visualize_point_cloud(
        [target, source_disp],
        enable_downsampled_visualization=enable_downsampled_visualization,
        downsampled_voxel_size=downsampled_voxel_size,
    )


def extract_geometric_feature(
    pcd, voxel_size, downsample_pcd: bool = True, print_stats: bool = False
):
    # downsample point cloud
    if downsample_pcd:
        if print_stats:
            print(f":: Downsample with a voxel size {voxel_size:3f}")
        pcd_down = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_down = copy.deepcopy(pcd)

    # estimate the normals
    radius_normal = voxel_size * 2
    if print_stats:
        print(f":: Estimate normal with search radius {radius_normal:3f}")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # FPFH feature
    radius_feature = voxel_size * 5
    if print_stats:
        print(f":: Compute FPFH feature with search radius {radius_feature:3f}")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    fpfh_data = np.asarray(pcd_fpfh.data)
    fpfh_data = fpfh_data / np.linalg.norm(fpfh_data)
    fpfh_rgb = np.vstack((fpfh_data, np.asarray(pcd_down.colors).T))
    pcd_fpfh = Open3dCustomFeature(pcd_fpfh)

    pcd_fpfh.data = fpfh_rgb

    return pcd_down, pcd_fpfh


def preprocess_point_clouds(
    source,
    target,
    voxel_size,
    downsample_pcd: bool = True,
    preprocess_target: Optional[bool] = True,
):
    # extract the geometric features
    source_down, source_fpfh = extract_geometric_feature(
        pcd=source, voxel_size=voxel_size, downsample_pcd=downsample_pcd
    )

    if preprocess_target:
        target_down, target_fpfh = extract_geometric_feature(
            pcd=target, voxel_size=voxel_size, downsample_pcd=downsample_pcd
        )
    else:
        target_down, target_fpfh = None, None

    return source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    voxel_size,
    print_stats: bool = False,
):
    # distance threshold for convergence criterion
    distance_threshold = voxel_size * 0.8  # 1.5

    if print_stats:
        print(f":: RANSAC Registration on downsampled point clouds.")
        print(
            f"   Using a distance threshold of {distance_threshold}, given a downsampling voxel size of {voxel_size}."
        )

    # transformation
    transformation = (
        o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold
                ),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
    )

    return transformation


def execute_fast_global_registration(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    voxel_size,
    print_stats: bool = False,
):
    # distance threshold for convergence criterion
    distance_threshold = voxel_size * 0.95

    if print_stats:
        print(f":: Fast Global Registration on downsampled point clouds.")
        print(
            f"   Using a distance threshold of {distance_threshold}, given a downsampling voxel size of {voxel_size}."
        )

    # transformation
    transformation = (
        o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold
            ),
        )
    )

    return transformation


# local refinement
def ICP_refinement_registration(
    source,
    target,
    source_fpfh,
    target_fpfh,
    voxel_size,
    init_transformation,
    distance_threshold: float = None,
    point_to_plane_registration: bool = True,
    print_stats: bool = False,
):
    # distance threshold
    if distance_threshold is None:
        distance_threshold = voxel_size * 0.4  #  * 1e1

    # Point-to-Plane ICP
    if print_stats:
        if point_to_plane_registration:
            method_descrip = "Point-to-Plane ICP registration"
        else:
            method_descrip = "Point-to-Point ICP registration"

        print(
            f"{method_descrip} is applied on the original point clouds, with a distance threshold of {distance_threshold:.3f}."
        )

    if point_to_plane_registration:
        icp_registration_option = (
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
    else:
        icp_registration_option = (
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

    # transformation
    transformation = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_transformation, icp_registration_option
    )

    return transformation


def Colored_ICP_refinement_registration(
    source,
    target,
    source_fpfh,
    target_fpfh,
    voxel_size,
    init_transformation,
    print_stats: bool = False,
):
    # radius
    radius = voxel_size

    # Colored ICP
    if print_stats:
        print(
            f"Colored ICP registration is applied on the original point clouds, with a radius of {radius:.3f}."
        )

    transformation = o3d.pipelines.registration.registration_colored_icp(
        source,
        target,
        radius,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
        ),
    )

    return transformation


def setup_lightglue(
    feature_extractor: LIGHTGLUE_Extractor, max_num_keypoints: Optional[int] = 2048
):
    if feature_extractor == LIGHTGLUE_Extractor.SUPERPOINT:
        # SuperPoint+LightGlue
        # load the extractor
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().cuda()

        # load the matcher
        matcher = LightGlue(features="superpoint").eval().cuda()
    elif feature_extractor == LIGHTGLUE_Extractor.DISK:
        # DISK+LightGlue
        # load the extractor
        extractor = DISK(max_num_keypoints=2048).eval().cuda()

        # load the matcher
        matcher = LightGlue(features="disk").eval().cuda()
    elif feature_extractor == LIGHTGLUE_Extractor.ALIKED:
        # ALIKED+LightGlue
        # load the extractor
        extractor = ALIKED(max_num_keypoints=2048).eval().cuda()

        # load the matcher
        matcher = LightGlue(features="aliked").eval().cuda()
    elif feature_extractor == LIGHTGLUE_Extractor.DOGHARDNET:
        # DoGHardNet+LightGlue
        # load the extractor
        extractor = DoGHardNet(max_num_keypoints=2048).eval().cuda()

        # load the matcher
        matcher = LightGlue(features="doghardnet").eval().cuda()
    else:
        raise NotImplementedError(
            "The specified extractor is not implemented in LightGlue!"
        )

    # speed optimization (Spikes in the computation time become more likely.)
    # matcher.compile(mode='reduce-overhead')

    # further speed optimization
    torch.set_float32_matmul_precision("medium")

    return extractor, matcher


def find_POI(
    img_rgb, npts, gray=True, viz=False, mask=None, detector=POI_Detector.SIFT
):  # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_rgb2 = np.copy(img_rgb)
    if gray is True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = img_rgb[..., 0]

    if detector == POI_Detector.SIFT:
        # detector = cv2.SIFT_create(nfeatures=npts, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=15, sigma=1.)
        detector = cv2.SIFT_create(nfeatures=npts)
    elif detector == POI_Detector.ORB:
        detector = cv2.ORB_create(npts)
    elif detector == POI_Detector.SURF:
        detector = cv2.xfeatures2d.SURF_create(npts)
    else:
        raise ValueError(f"Detector: {detector} does not exist!")

        # # FAST Detector
        # detector = cv2.FastFeatureDetector_create()

        # # SURF Detector
        # detector = cv2.xfeatures2d.SURF_create(400)
        # keypoints = detector.detect(img, None)
        # descriptors = detector.compute(img, keypoints)

    keypoints, descriptors = detector.detectAndCompute(img, mask)

    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    descriptors = np.array(descriptors)

    if gray is False:
        for i in range(2):
            keypoints_rgb, descriptors_rgb = detector.detectAndCompute(
                img_rgb[..., i + 1], mask
            )

            xy_rgb = [keypoint.pt for keypoint in keypoints_rgb]
            xy_rgb = np.array(xy_rgb).astype(int)
            descriptors_rgb = np.array(descriptors_rgb)

            xy = np.concatenate([xy, xy_rgb], axis=0)
            descriptors = np.concatenate([descriptors, descriptors_rgb], axis=0)
            keypoints = list(keypoints)
            keypoints.extend(list(keypoints_rgb))
            keypoints = tuple(keypoints)

    if viz is True:
        kp_img = cv2.drawKeypoints(img_rgb2, keypoints, img_rgb2, color=(0, 255, 0))

    # Perform lex sort and get sorted data
    if len(xy) > 0:
        sorted_idx = np.lexsort(xy.T)
        xy = xy[sorted_idx, :]
        descriptors = descriptors[sorted_idx, :]
        keypoints = [keypoints[i] for i in sorted_idx]
    else:
        print(f"xy: {xy}")
        print(f"keypoints:{keypoints}")
        raise RuntimeError("No keypoint was detected!")

    # Get unique row mask
    row_mask = np.append([True], np.any(np.diff(xy, axis=0), 1))

    # Get unique rows
    xy = xy[row_mask]
    descriptors = descriptors[row_mask]
    keypoints = [keypoints[i] for i in np.where(row_mask)[0]]

    return xy, descriptors, keypoints  # pixel coordinatess


def ratio_test(matches, thresh=0.95):
    # store all the good matches as per Lowe's ratio test.
    match_passed = []

    for match in matches:
        if len(match) == 2:
            m, n = match
        else:
            continue

        if m.distance < thresh * n.distance:
            match_passed.append(match)

    return match_passed


def sym_test(matches12, matches21):
    good = []
    for m1, n1 in matches12:
        for m2, n2 in matches21:
            if (m1.queryIdx == m2.trainIdx) and (m1.trainIdx == m2.queryIdx):
                good.append(m1)

    return good


def feature_matching(des1, des2, detector=POI_Detector.SIFT):
    # FLANN parameters
    if detector == POI_Detector.SIFT or detector == POI_Detector.SURF:
        FLAN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLAN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
    elif detector == POI_Detector.ORB:
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,  # 12
            key_size=12,  # 20
            multi_probe_level=0,
        )  # 2
        search_params = dict(checks=10)  # or pass empty dictionary
    else:
        raise RuntimeError(f"Detector: {detector} does not exist!")

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    Matches = flann.knnMatch(des1, des2, k=2)

    # matches21 = flann.knnMatch(des2, des1, k=2)

    matches12 = ratio_test(Matches, thresh=0.9)

    # matches21 = ratio_test(matches21, thresh=0.8)

    # matches = sym_test(matches12, matches21)

    matches = [m1 for (m1, n1) in matches12]

    return matches, Matches


def execute_PnP_RANSAC(
    nerf,
    init_guess,
    rgb_input,
    camera_intrinsics_K,
    feature_detector: POI_Detector,
    save_image: bool = False,
    pnp_figure_filename: Optional[str] = "/",
    print_stats: bool = False,
    visualize_PnP_matches: bool = False,
    pnp_matches_figure_filename: Optional[str] = "/",
    detector_params: Optional[Dict[str, Any]] = {},
):
    # generate RGB-D point cloud
    target_rgb, target_pcd_pts, _, depth_mask, *_ = nerf.generate_RGBD_point_cloud(
        init_guess,
        save_image=save_image,
        filename=pnp_figure_filename,
        compute_semantics=False,
        return_pcd=False,
    )

    # target point cloud
    target_pcd_cam = target_pcd_pts.view(-1, 3)
    pts_shape = target_pcd_cam.shape

    # convert from OPENGL Camera convention to OPENCV convention
    init_guess_gl = init_guess.detach().clone()
    init_guess_gl[:, 1] = -init_guess_gl[:, 1]
    init_guess_gl[:, 2] = -init_guess_gl[:, 2]

    # transform to the world space
    target_pcd = (
        init_guess_gl
        @ torch.cat(
            (target_pcd_cam, torch.ones(pts_shape[0], 1, device=nerf.device)), axis=-1
        ).T
    )
    target_pcd = target_pcd.T.view((*target_pcd_pts.shape[:2], target_pcd.shape[0]))[
        ..., :3
    ]

    # source image (Image obtained from the camera.)
    source_img = (rgb_input.cpu().numpy() * 255).astype(np.uint8)

    # target image (Image rendered at the initial guess.)
    target_img = (target_rgb.cpu().numpy() * 255).astype(np.uint8)

    # start time
    t0 = time.perf_counter()

    # feature matching
    if feature_detector == POI_Detector.LIGHTGLUE:
        # extract local features
        source_feats = detector_params["extractor"].extract(rgb_input.permute(2, 0, 1))
        target_feats = detector_params["extractor"].extract(target_rgb.permute(2, 0, 1))

        # match the features
        matches = detector_params["matcher"](
            {"image0": source_feats, "image1": target_feats}
        )

        # remove batch dimension
        source_feats, target_feats, matches = [
            rbd(x) for x in [source_feats, target_feats, matches]
        ]

        # macthes: indices with shape (K, 2)
        matches = matches["matches"]

        # matched points
        # coordinates in the source image with shape (K, 2)
        source_xy_matches = (
            source_feats["keypoints"][matches[..., 0]].long().cpu().numpy()
        )

        # coordinates in target image with shape (K, 2)
        target_xy_matches = target_feats["keypoints"][matches[..., 1]].long()
    else:
        source_xy, source_descriptors, source_kp_img = find_POI(
            source_img, 1000, viz=False, detector=feature_detector
        )
        target_xy, target_descriptors, target_kp_img = find_POI(
            target_img, 1000, viz=False, detector=feature_detector
        )
        matches, Matches = feature_matching(
            np.array(source_descriptors),
            np.array(target_descriptors),
            detector=feature_detector,
        )

        # number of matches
        num_matches = len(matches)

        if print_stats:
            print(f"Found {num_matches} matches!")

        # Start matching as points (x, y)
        source_xy_matches = []
        target_xy_matches = []

        for match in matches:
            target_xy_matches.append(target_xy[match.trainIdx])
            source_xy_matches.append(source_xy[match.queryIdx])

        # source and target matches
        source_xy_matches = np.stack(source_xy_matches, axis=0)
        target_xy_matches = np.stack(target_xy_matches, axis=0)

    # apply depth mask
    source_xy_matches[
        depth_mask[source_xy_matches[:, 1], source_xy_matches[:, 0]].cpu().numpy()
    ]
    target_xy_matches[
        depth_mask[target_xy_matches[:, 1], target_xy_matches[:, 0]].cpu().numpy()
    ]

    # Select matches from target point cloud
    target_3d_matches = target_pcd[target_xy_matches[:, 1], target_xy_matches[:, 0]]
    target_3d_matches = target_3d_matches.cpu().numpy()

    # Perform pnp ransac
    guess = init_guess.cpu().numpy()
    guess_w2c = np.linalg.inv(guess)
    t_guess = guess_w2c[:3, -1].reshape(-1, 1)
    r_guess = guess_w2c[:3, :3]
    R_vec_guess = cv2.Rodrigues(r_guess)[0]

    # PnP-RANSAC
    success, R_vec, t, inliers = cv2.solvePnPRansac(
        target_3d_matches,
        source_xy_matches.astype(np.float32),
        camera_intrinsics_K.cpu().numpy(),
        distCoeffs=None,
        rvec=R_vec_guess,
        tvec=t_guess,
        flags=cv2.SOLVEPNP_EPNP,  # SOLVEPNP_ITERATIVE
        confidence=0.99,
        reprojectionError=8.0,
        # useExtrinsicGuess=True
    )

    t1 = time.perf_counter()

    if print_stats:
        print(f"time PnP: {t1 - t0} secs.")

    # Retrieve the camera to world transform
    # openCV camera frame is x to right, y down, z forward
    # Also, r and t rotate world to camera
    est_pose = np.eye(4)
    r = cv2.Rodrigues(R_vec)[0]
    w2c_cv = np.hstack([r, t])

    est_pose[:3] = w2c_cv
    est_pose = np.linalg.inv(est_pose)
    est_pose[:, 1] = -est_pose[:, 1]
    est_pose[:, 2] = -est_pose[:, 2]

    if not success:
        print(f"PNP RANSAC FAILED!")
        raise RuntimeError(f"PNP RANSAC FAILED!")
    else:
        if print_stats:
            print(f"PNP RANSAC SUCCEEDED!")

    if visualize_PnP_matches:
        fig = plt.figure()
        flann_matches = cv2.drawMatchesKnn(
            source_img, source_kp_img, target_img, target_kp_img, Matches, None
        )
        plt.imshow(flann_matches)
        plt.show()

        # save figure
        fig.savefig(pnp_matches_figure_filename)

    return est_pose


def skew_matrix(vec):
    batch_dims = vec.shape[:-1]
    S = torch.zeros(*batch_dims, 3, 3).to(vec.device)
    S[..., 0, 1] = -vec[..., 2]
    S[..., 0, 2] = vec[..., 1]
    S[..., 1, 0] = vec[..., 2]
    S[..., 1, 2] = -vec[..., 0]
    S[..., 2, 0] = -vec[..., 1]
    S[..., 2, 1] = vec[..., 0]
    return S


def rot_matrix_to_vec(R):
    batch_dims = R.shape[:-2]

    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)

    def acos_safe(x, eps=1e-7):
        """https://github.com/pytorch/pytorch/issues/8069"""
        slope = np.arccos(1 - eps) / eps
        # TODO: stop doing this allocation once sparse gradients with NaNs (like in
        # th.where) are handled differently.
        buf = torch.empty_like(x)
        good = abs(x) <= 1 - eps
        bad = ~good
        sign = torch.sign(x[bad])
        buf[good] = torch.acos(x[good])
        buf[bad] = torch.acos(sign * (1 - eps)) - slope * sign * (abs(x[bad]) - 1 + eps)
        return buf

    # angle = torch.acos((trace - 1) / 2)[..., None]
    angle = acos_safe((trace - 1) / 2)[..., None]
    # print(trace, angle)

    vec = (
        1
        / (2 * torch.sin(angle + 1e-10))
        * torch.stack(
            [
                R[..., 2, 1] - R[..., 1, 2],
                R[..., 0, 2] - R[..., 2, 0],
                R[..., 1, 0] - R[..., 0, 1],
            ],
            dim=-1,
        )
    )

    # needed to overwrite nanes from dividing by zero
    vec[angle[..., 0] == 0] = torch.zeros(3, device=R.device)

    # eg TensorType["batch_size", "views", "max_objects", 3, 1]
    rot_vec = (angle * vec)[...]

    return rot_vec


def vec_to_rot_matrix(rot_vec):
    assert not torch.any(torch.isnan(rot_vec))

    angle = torch.norm(rot_vec, dim=-1, keepdim=True)

    axis = rot_vec / (1e-10 + angle)
    S = skew_matrix(axis)
    # print(S.shape)
    # print(angle.shape)
    angle = angle[..., None]
    rot_matrix = (
        torch.eye(3).to(rot_vec.device)
        + torch.sin(angle) * S
        + (1 - torch.cos(angle)) * S @ S
    )
    return rot_matrix


def SE3_to_se3(pose):
    # pose in se(3)
    pose_se3 = torch.zeros(6).to(pose.device)

    # translation components
    pose_se3[:3] = pose[:3, 3]

    # rotation component
    pose_se3[3:] = rot_matrix_to_vec(pose[:3, :3])

    return pose_se3


def se3_to_SE3(pose):
    # pose in SE(3)
    pose_SE3 = torch.eye(4).to(pose.device)

    # translation components
    pose_SE3[:3, 3] = pose[:3]

    # rotation component
    pose_SE3[:3, :3] = vec_to_rot_matrix(pose[3:])

    return pose_SE3


# function
def compute_sigmapoints(
    state_mu: torch.Tensor,
    state_cov: torch.Tensor,
    kappa: float = 2.0,
    device: torch.device = "cuda",
):
    # state dimension
    state_dim = state_mu.shape[-1]

    # Cholesky decomposition
    L = torch.linalg.cholesky(state_cov)

    # Sigmapoints
    sigmapts = torch.empty(2 * state_dim + 1, state_dim, device=device)

    # zeroth sigmapoints
    sigmapts[0, :] = state_mu

    # weight
    weight = np.sqrt(state_dim + kappa)

    sigmapts[1 : state_dim + 1, :] = state_mu + L * weight

    # associated sigmapoints
    sigmapts[state_dim + 1 :, :] = state_mu - L * weight

    return sigmapts


def dynamics_step(
    state: torch.Tensor,
    vel: torch.Tensor = torch.zeros(6, device="cuda"),
    dt: float = 1.0 / 50,
):
    # simple dynamics
    # state dimension
    state_dim = state.shape[-1]

    # number of position states
    pos_state_dim = state_dim // 4

    # number of state samples
    n_samples = state.shape[0] if len(state.shape) > 1 else 1

    # next state
    next_state = state.clone()

    # position states
    # state_SE_3 = se_3_to_SE_3(next_state[:, :pos_state_dim])
    # next_state_SE_3 = state_SE_3.clone()
    # next_state_SE_3[:3, -1] += dt * state[:, pos_state_dim + pos_state_dim  // 2]

    # # orientation
    # next_state_SE_3[:3, :3] @= torch.linalg.matrix_exp(dt * state[:, pos_state_dim  // 2: 2 * pos_state_dim])

    # next_state[:, :pos_state_dim // 2] = state[:, :pos_state_dim // 2] + dt * state[:, pos_state_dim + pos_state_dim  // 2]

    # next_state[:, :pos_state_dim // 2] = SE_3_to_se_3(se_3_to_SE_3(next_state[:, pos_state_dim // 2: pos_state_dim]))

    # next_state[:, :pos_state_dim] = state[:, :pos_state_dim] + dt * state[:, pos_state_dim: 2 * pos_state_dim]

    # velocity states

    # account for noise
    # next_state[:, :2 * pos_state_dim] += state[:, 2 * pos_state_dim:]

    return next_state + dt * vel


def inverse_sigma_transform_prediction(
    state_mu_samples: torch.Tensor,
    kappa: float = 2.0,
    process_noise_cov: torch.Tensor = None,
    device: torch.device = "cuda",
):
    # state dimension
    state_dim = state_mu_samples.shape[-1]

    if process_noise_cov is None:
        process_noise_cov = 1e-2 * torch.eye(state_mu_samples.shape[-1], device=device)

    # weights
    weights = torch.concat(
        (
            torch.tensor([kappa / (state_dim + kappa)]),
            torch.broadcast_to(
                torch.tensor([1 / (2 * (state_dim + kappa))]),
                (state_mu_samples.shape[0] - 1,),
            ),
        )
    ).to(device)

    # prediction mean
    pred_mean = weights @ state_mu_samples

    # prediction covariance
    mu_diff = torch.diag(torch.sqrt(weights)) @ (state_mu_samples - pred_mean[None])
    pred_cov = torch.bmm(mu_diff[..., None], mu_diff[..., None, :])

    pred_cov = torch.sum(pred_cov, dim=0) + process_noise_cov

    return pred_mean, pred_cov, mu_diff


def measurement_function(state: torch.Tensor):
    # return the position/orientation states
    # # state dimension
    state_dim = state.shape[-1]

    # # number of position states
    # pos_state_dim = state_dim // 4

    # return  state[:, :pos_state_dim] + state[:, 2 * pos_state_dim:3 * pos_state_dim]

    return state


def inverse_sigma_transform_measurement(
    mu_x_diff: torch.Tensor,
    pred_x_mean: torch.Tensor,
    meas_mu_samples: torch.Tensor,
    kappa: float = 2.0,
    meas_noise_cov: torch.Tensor = None,
    device: torch.device = "cuda",
):
    # measurement dimension
    meas_dim = meas_mu_samples.shape[-1]

    if meas_noise_cov is None:
        meas_noise_cov = 1e-2 * torch.eye(meas_mu_samples.shape[-1], device=device)

    # weights
    weights = torch.concat(
        (
            torch.tensor([kappa / (meas_dim + kappa)]),
            torch.broadcast_to(
                torch.tensor([1 / (2 * (meas_dim + kappa))]),
                (meas_mu_samples.shape[0] - 1,),
            ),
        )
    ).to(device)

    # measurement mean
    y_mean = weights @ meas_mu_samples

    # covariance (yy)
    mu_y_diff = torch.diag(torch.sqrt(weights)) @ (meas_mu_samples - y_mean[None])
    yy_cov = torch.bmm(mu_y_diff[..., None], mu_y_diff[..., None, :])

    yy_cov = torch.sum(yy_cov, dim=0) + meas_noise_cov

    # # We can optimize the next couple of lines.
    # # state dimension
    # state_dim = state_mu_samples.shape[-1]

    # # weights
    # weights = torch.concat((torch.tensor([kappa / (state_dim + kappa)]),
    #                           torch.broadcast_to(torch.tensor([1 / (2 * (state_dim + kappa))]),
    #                                              (state_mu_samples.shape[0] - 1,)))).to(device)

    # # covariance (xy)
    # mu_x_diff = torch.diag(torch.sqrt(weights)) @ (state_mu_samples - pred_x_mean[None])
    xy_cov = torch.bmm(mu_x_diff[..., None], mu_y_diff[..., None, :])

    xy_cov = torch.sum(xy_cov, dim=0)

    return y_mean, yy_cov, xy_cov


def measurement_innovation(
    y: torch.Tensor, y_mean: torch.Tensor, device: torch.device = "cuda"
):
    # innovation in se(3)
    return SE3_to_se3(y) - y_mean


def correction_phase_post_measurement(
    y: torch.Tensor,
    pred_x_mean: torch.Tensor,
    y_mean: torch.Tensor,
    pred_xx_cov: torch.Tensor,
    yy_cov: torch.Tensor,
    xy_cov: torch.Tensor,
    device: torch.device = "cuda",
):
    # Kalman gain
    K_gain = xy_cov @ torch.linalg.solve(
        yy_cov, torch.eye(yy_cov.shape[0], device=device)
    )

    # posterior covariance
    posterior_cov = pred_xx_cov - K_gain @ xy_cov.T

    # posterior mean
    posterior_mean = pred_x_mean + K_gain @ measurement_innovation(y, y_mean)

    return posterior_mean, posterior_cov
