# %%
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber, TimeSynchronizer
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
import sensor_msgs.msg as sensor_msgs
from scipy.spatial.transform import Rotation
import json

import struct
from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2, PointField, Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
import cv2

import numpy as np
import open3d as o3d
from pathlib import Path
import torch
import time

from ns_utils.nerfstudio_utils import GaussianSplat
from pose_estimator.utils import SE3error, PrintOptions
from pose_estimator.pose_estimator import SplatLoc
from filter_utils import KalmanFilter

goal_queries = ['beachball', 'phonebook', 'microwave', 'lightpole']
GOAL_ID = 2
TRIAL_ID = 10
MODE = 'closed-loop'
POSE_SOURCE = 'splat-loc'

# Our drone specific
camera_to_body = np.array([[ 0.00000000e+00, -2.22044605e-16,  1.00000000e+00,  6.65000000e-02],
 [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00,  1.05000000e-02],
 [-2.22044605e-16,  1.00000000e+00,  2.22044605e-16, -2.94000000e-02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])


class SplatLocNode(Node):
    def __init__(
        self,
        config_path: Path,
        use_compressed_image: bool = True,
        drone_config=None,
        enable_UKF: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(node_name="splatlocnode")
        # map name
        self.map_name = "camera_link"

        # device
        self.device = device

        # slam mode
        self.slam_method = "modal"

        # load the Gaussian Splat
        self.gsplat = GaussianSplat(
            config_path=config_path, dataset_mode="train", device=self.device
        )

        # applied to data
        self.T_data2gsplat = torch.eye(4).cuda()
        self.T_data2gsplat[:3, :] = (
            self.gsplat.pipeline.datamanager.train_dataparser_outputs.dataparser_transform
        )
        self.s_data2gsplat = (
            self.gsplat.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        )
        # applied to gsplat poses
        self.T_gsplat2data = torch.linalg.inv(self.T_data2gsplat)
        self.s_gsplat2data = 1 / self.s_data2gsplat

        # drone config parameters
        self.drone_config = drone_config

        # set the camera intrinsics
        self.update_camera_intrinsics()

        # distortion
        self.dparams = (
            torch.tensor(
                [
                    drone_config["k1"],
                    drone_config["k2"],
                    0.,     # k3
                    drone_config["p1"],
                    drone_config["p2"],
                ]
            )
            .float()
            .to(self.device)
        )

        # option to undistort
        self.undistort_images = True

        # blur
        self.blur_threshold = 1.0

        # self.T_handeye = torch.eye(4).cuda()

        if drone_config is not None:
            # camera-to-body-frame
            self.T_handeye = (
                torch.from_numpy(np.array(drone_config["pose_to_camera"]))
                .to(self.device)
                .float()
            )

        # inverse of the handeye
        # body-frame-to-camera
        self.T_inv_handeye = torch.linalg.inv(self.T_handeye)

        # Splat-Loc Node
        self.splat_loc = SplatLoc(self.gsplat)

        # setting to enable UKF
        self.enable_UKF = enable_UKF

        if self.enable_UKF:
            # set the parameters: kappa and dt for the UKF
            self.splat_loc.set_kappa(kappa=2.0)
            self.splat_loc.set_dt(dt=1e-3)

        # computation time
        self.computation_time = []

        # pose error
        self.pose_error = []

        # success rate
        self.cache_success_flag = []

        if self.enable_UKF:
            # computation time
            self.computation_time_ukf = []

            # pose error
            self.pose_error_ukf = []

            # success rate
            self.cache_success_flag_ukf = []

        # OpenCV brideTimeSynchronizer
        self.opencv_bridge = CvBridge()

        # QOS Profile
        qos_profile_incoming = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # subscriber to the ground-truth pose. #NOTE: CHANGE THIS!!!
        self.gt_pose_subscription = Subscriber(
            self, PoseStamped, "/republished_pose", qos_profile=qos_profile_incoming
        )

        # option to use compressed images
        self.use_compressed_image = use_compressed_image
        self.print_options = PrintOptions()

        # sensor message type (for images)
        sensor_img_type = CompressedImage if self.use_compressed_image else Image

        # subscriber to the RGB Image topic
        self.cam_rgb_subscription = Subscriber(
            self,
            sensor_img_type,
            "/republished_image",
            qos_profile=qos_profile_incoming,
        )

        # self.gt_pose_subscriber = self.create_subscription(
        #         PoseStamped,
        #         '/republished_pose',
        #         self.gt_pose_callback,
        #         10)
        
        # self.img_subscriber = self.create_subscription(
        #         sensor_img_type,
        #         '/republished_image',
        #         self.img_callback,
        #         10)

        self.ts = ApproximateTimeSynchronizer(
            [self.gt_pose_subscription, self.cam_rgb_subscription], 40, 0.1
        )
        self.ts.registerCallback(self.ts_callback)

        # topic_sync = "exact"

        # if topic_sync == "approx":
        #     self.ts = ApproximateTimeSynchronizer(self.subs, 40, topic_slop)
        # elif topic_sync == "exact":
        #     self.ts = TimeSynchronizer(self.subs, 40)
        # else:
        #     raise NameError("Unsupported topic sync method. Must be {approx, exact}.")

        # self.ts.registerCallback(self.ts_callback)

        # self.update_RGB_image,
        # self.pose_callback,

        # publisher for the current pose estimate
        self.est_pose_publisher = self.create_publisher(
            PoseArray, "/estimated_and_relative_pose", 10
        )

        # current RGB image
        self.cam_rgb = None
        # self.cam_K = None

        # initial guess of the pose
        self.init_guess = None

        # debug mode
        self.debug_mode = True

        # Kalman filter on the RELATIVE transform
        ndim = 6
        kf_x0 = np.zeros(ndim)
        kf_sig_x0 = 0.01*np.eye(ndim)
        kf_sig_R = 0.005*np.eye(ndim)
        kf_sig_Q = 0.005*np.eye(ndim)
        kf_A = np.eye(ndim)
        kf_C = np.eye(ndim)

        self.kalmanfilter = KalmanFilter(kf_x0, kf_sig_Q, kf_sig_R, kf_sig_x0, kf_A, kf_C)

        print('Splat-Loc Ready...')

    def ts_callback(self, msg_pose, msg_rgb):
        print('Received pose')

        # NOTE: ALWAYS INITIALIZE FROM THE VIO !!! #
        #if self.init_guess is None:
            #self.init_guess = torch.eye(4, device=self.device)
        self.update_init_guess(msg_pose)

        self.gt_pose = torch.eye(4, device=self.device)
        # translation
        self.gt_pose[:3, -1] = torch.tensor(
            [
                msg_pose.pose.position.x,
                msg_pose.pose.position.y,
                msg_pose.pose.position.z,
            ]
        )
        # orientation
        R_msg = Rotation.from_quat(
            np.array(
                [
                    msg_pose.pose.orientation.x,
                    msg_pose.pose.orientation.y,
                    msg_pose.pose.orientation.z,
                    msg_pose.pose.orientation.w,
                ]
            )
        ).as_matrix()
        self.gt_pose[:3, :3] = torch.from_numpy(R_msg).to(self.device)

        self.update_RGB_image(msg_rgb)

    # def gt_pose_callback(self, msg_pose):
    #     print('Received pose')
    #     if self.init_guess is None:
    #         self.update_init_guess(msg_pose)

    #     self.gt_pose = torch.eye(4, device=self.device)
    #     # translation
    #     self.gt_pose[:3, -1] = torch.tensor(
    #         [
    #             msg_pose.pose.position.x,
    #             msg_pose.pose.position.y,
    #             msg_pose.pose.position.z,
    #         ]
    #     )
    #     # orientation
    #     R_msg = Rotation.from_quat(
    #         np.array(
    #             [
    #                 msg_pose.pose.orientation.x,
    #                 msg_pose.pose.orientation.y,
    #                 msg_pose.pose.orientation.z,
    #                 msg_pose.pose.orientation.w,
    #             ]
    #         )
    #     ).as_matrix()
    #     self.gt_pose[:3, :3] = torch.from_numpy(R_msg).to(self.device)

    # def img_callback(self, msg_rgb):
    #     print('Received image')

    #     self.update_RGB_image(msg_rgb)

    def update_camera_intrinsics(self):
        # focal lengths and optical centers
        fx = self.drone_config["fl_x"]
        fy = self.drone_config["fl_y"]
        cx = self.drone_config["cx"]
        cy = self.drone_config["cy"]

        # cmaera intrinsics matrix
        self.cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.cam_K = torch.from_numpy(self.cam_K).to(self.device).float()

    def update_RGB_image(self, msg):
        # update the current image of the camera
        if self.use_compressed_image:
            # image in OpenCV format
            img_cv = self.opencv_bridge.compressed_imgmsg_to_cv2(msg)
            img_tensor = torch.from_numpy(img_cv).to(dtype=torch.float32) / 255.0

            # convert from BGR to RGB
            img_tensor = img_tensor[..., [2, 1, 0]]
        else:
            im_cv = self.opencv_bridge.imgmsg_to_cv2(msg, msg.encoding)
            if self.slam_method == "modal":
                im_cv = cv2.cvtColor(im_cv, cv2.COLOR_YUV2RGB_Y422)
            else:
                im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)

            # img_cv = self.opencv_bridge.imgmsg_to_cv2(msg, msg.encoding)
            # img_tensor = torch.from_numpy(im_cv).to(dtype=torch.float32) / 255.0

        if self.undistort_images:
            im_cv = cv2.undistort(
                im_cv, self.cam_K.cpu().numpy(), self.dparams.cpu().numpy(), None, None
            )

        # Filter motion blur
        if self.blur_threshold > 0:
            laplacian = cv2.Laplacian(im_cv, cv2.CV_64F).var()
            if laplacian < self.blur_threshold:
                return False, None

        img_tensor = torch.from_numpy(im_cv).to(dtype=torch.float32) / 255.0

        # cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("frame", im_cv)
        # import matplotlib.pyplot as plt
        # img = (img_tensor.cpu().numpy() * 255).astype(np.uint8)

        # plt.figure()
        # plt.imshow(img)
        # plt.show()

        self.cam_rgb = img_tensor

        # breakpoint()
        # estimate the pose
        self.estimate_pose()

        print("ESTIMATE COMPLETE")

    def pose_callback(self, msg):
        # set the initial guess
        if self.init_guess is None:
            self.update_init_guess(msg)

    def update_init_guess(self, msg):
        # This is MOCAP z-up frame
        if self.debug_mode:
            print("Updating the Initial Guess!")

        init_guess = torch.eye(4, device=self.device)

        # translation
        init_guess[:3, -1] = torch.tensor(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )

        # orientation
        # init_guess[:3, :3] = quaternion_to_rotation_matrix(torch.tensor([msg.pose.orientation.x,
        #                                                                       msg.pose.orientation.y,
        #                                                                       msg.pose.orientation.z,
        #                                                                       msg.pose.orientation.w])[None])
        R_msg = Rotation.from_quat(
            np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ]
            )
        ).as_matrix()
        init_guess[:3, :3] = torch.from_numpy(R_msg).to(self.device)
        init_guess = init_guess @ self.T_handeye

        # init_guess = init_guess[:, [1,2,0,3]]
        init_guess[:, [1, 2]] *= -1

        init_guess = self.T_data2gsplat @ init_guess
        init_guess[:3, 3] *= self.s_data2gsplat

        self.init_guess = init_guess
        # init_guess = self.gsplat.get_poses()[0]
        # self.init_guess = torch.eye(4).cuda()
        # self.init_guess[:3, :] = init_guess

        if self.enable_UKF:
            # Set the prior distribution of the UKF
            self.splat_loc.set_prior_distribution(
                mu=init_guess, sigma=torch.eye(6, device=self.device)
            )

    def estimate_pose(self):
        # estimate the pose
        if (
            (self.cam_rgb is not None)
            and (self.init_guess is not None)
            and (self.cam_K is not None)
        ):
            print(self.print_options.sep_0)
            print(self.print_options.sep_1)
            print(f"Running Pose Estimator...")
            print(self.print_options.sep_1)

            if self.enable_UKF:
                # estimate the pose (UKF)
                # try:
                # computation time
                start_time = time.perf_counter()

                est_pose_ukf = self.splat_loc.estimate_ukf(
                    init_guess=None,
                    cam_rgb=self.cam_rgb.to(self.device),
                    cam_K=self.cam_K,
                )

                # total computation time (for this iteration)
                self.computation_time_ukf.append(time.perf_counter() - start_time)

                # cache the success flag
                self.cache_success_flag_ukf.append(1)
            # except:
            #     print("Failed UKF!")
            #     # cache the success flag
            #     self.cache_success_flag_ukf.append(0)

            # estimate the pose (PnP-RANSAC only)
            try:
                # computation time
                start_time = time.perf_counter()

                est_pose = self.splat_loc.estimate(
                    init_guess=self.init_guess,
                    cam_rgb=self.cam_rgb.to(self.device),
                    cam_K=self.cam_K,
                )

                # total computation time (for this iteration)
                self.computation_time.append(time.perf_counter() - start_time)

                # cache the success flag
                self.cache_success_flag.append(1)
            except Exception as excp:
                print("Failed PnP!", "Exception:", excp)
                # cache the success flag
                self.cache_success_flag.append(0)
                return

            # update the initial guess
            #self.init_guess = torch.tensor(est_pose, device=self.device)

            ### NOTE: !!! WE ARE INITIALIZAING SPLAT_LOC WITH THE VIO HERE ###
            #self.init_guess = torch.tensor(self.gt_pose, device=self.device) #est_pose
            pub_pose = est_pose.clone()
            pub_pose[:3, 3] *= self.s_gsplat2data
            pub_pose = self.T_gsplat2data @ pub_pose

            # convert from opengl frame ros
            pub_pose[:, [1, 2]] *= -1
            pub_pose = pub_pose @ self.T_inv_handeye
            # pub_pose = pub_pose[:, [2,0,1,3]]

            if self.debug_mode:
                # pose error
                error = SE3error(self.gt_pose.cpu().numpy(), pub_pose.cpu().numpy())

                print(
                    f"{self.print_options.sep_space} PnP-RANSAC --- SE(3) Estimation Error -- Rotation: {error[0]}, Translation: {error[1]}"
                )

                # store the pose error
                self.pose_error.append(error)

            if self.enable_UKF:
                if self.debug_mode:
                    ukf_pose = est_pose_ukf.clone()
                    ukf_pose[:3, 3] *= self.s_gsplat2data
                    ukf_pose = self.T_gsplat2data @ ukf_pose

                    # convert from opengl frame ros
                    ukf_pose[:, [1, 2]] *= -1
                    ukf_pose = ukf_pose @ self.T_inv_handeye
                    # pub_pose = pub_pose[:, [2,0,1,3]]

                    # pose error
                    error = SE3error(self.gt_pose.cpu().numpy(), ukf_pose.cpu().numpy())

                    print(
                        f"{self.print_options.sep_space} UKF SE(3) --- Estimation Error -- Rotation: {error[0]}, Translation: {error[1]}"
                    )

                    # store the pose error
                    self.pose_error.append(error)

            # Publish the estimated pose
            # msg = PoseStamped()
            # # msg.header.frame_id = self.map_name

            # # translation
            # msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (
            #     pub_pose[0, -1].item(),
            #     pub_pose[1, -1].item(),
            #     pub_pose[2, -1].item(),
            # )

            # # orientation
            # (
            #     msg.pose.orientation.x,
            #     msg.pose.orientation.y,
            #     msg.pose.orientation.z,
            #     msg.pose.orientation.w,
            # ) = Rotation.from_matrix(pub_pose[:3, :3].cpu().numpy()).as_quat()

            # # header
            # msg.header.frame_id = self.map_name

            # self.est_pose_publisher.publish(msg)

            # Publish estimated pose and relative pose
            # estimated pose
            msg = Pose()
            # translation
            msg.position.x, msg.position.y, msg.position.z = (
                pub_pose[0, -1].item(),
                pub_pose[1, -1].item(),
                pub_pose[2, -1].item(),
            )

            # orientation
            (
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ) = Rotation.from_matrix(pub_pose[:3, :3].cpu().numpy()).as_quat()

            # relative
            relative_transform = self.gt_pose.cpu().numpy() @ np.linalg.inv(pub_pose.cpu().numpy())
            error = SE3error(self.gt_pose.cpu().numpy(), pub_pose.cpu().numpy())

            relative_transform_og = relative_transform

            # Kalman filter on relative pose
            rotvec = Rotation.from_matrix(relative_transform[:3, :3]).as_rotvec()
            trans = relative_transform[:3, -1]
            measurement = np.concatenate([trans, rotvec])
            self.kalmanfilter.update(measurement)

            state, covariance = self.kalmanfilter.get_estimate()

            relative_transform = np.eye(4)
            relative_transform[:3, -1] = state[:3]
            relative_transform[:3, :3] = Rotation.from_rotvec(state[3:]).as_matrix()

            # RAW MEASUREMENT
            relative_og = Pose()
            # translation
            relative_og.position.x, relative_og.position.y, relative_og.position.z = (
                relative_transform_og[0, -1].item(),
                relative_transform_og[1, -1].item(),
                relative_transform_og[2, -1].item(),
            )

            # orientation
            (
                relative_og.orientation.x,
                relative_og.orientation.y,
                relative_og.orientation.z,
                relative_og.orientation.w,
            ) = Rotation.from_matrix(relative_transform_og[:3, :3]).as_quat()

            # FILTERED MEASUREMENT
            relative = Pose()
            # translation
            relative.position.x, relative.position.y, relative.position.z = (
                relative_transform[0, -1].item(),
                relative_transform[1, -1].item(),
                relative_transform[2, -1].item(),
            )

            # orientation
            (
                relative.orientation.x,
                relative.orientation.y,
                relative.orientation.z,
                relative.orientation.w,
            ) = Rotation.from_matrix(relative_transform[:3, :3]).as_quat()

            out_msg = PoseArray()
            out_msg.header.frame_id = self.map_name
            out_msg.poses = [msg, relative, relative_og]
            self.est_pose_publisher.publish(out_msg)

            print(self.print_options.sep_0)
            print(self.print_options.sep_1)
            print(f"Finished Estimating the Pose...")
            print(self.print_options.sep_1)

    def save_results(self):
        # save the results

        save_path = f'traj/{POSE_SOURCE}/{MODE}/{goal_queries[GOAL_ID]}/{TRIAL_ID}'
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # computation time
        save_dir = os.path.join(save_path, "estimator")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(
            f"{save_dir}/computation_time.npy",
            self.computation_time,
        )

        # success rate
        np.save(
            f"{save_dir}/success_flag.npy",
            self.cache_success_flag,
        )

        # pose error
        np.save(f"{save_dir}/pose_error.npy", self.pose_error)

        if self.enable_UKF:
            # computation time
            save_dir = os.path.join(save_path, "estimator")
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            np.save(
                f"{save_dir}/computation_time_ukf.npy",
                self.computation_time,
            )

            # success rate
            np.save(
                f"{save_dir}/success_flag_ukf.npy",
                self.cache_success_flag,
            )

            # pose error
            np.save(f"{save_dir}/pose_error_ukf.npy", self.pose_error)

if __name__ == "__main__":
    # init
    rclpy.init()
    timestart = time.strftime("%Y%m%d-%H%M")

    # config path
    config_path = Path(
        # "outputs/configs/ros-depth-splatfacto/2024-10-30_140655/config.yml"
        "outputs/registered/gemsplat/2024-12-05_155215/config.yml"
    )

    # drone info
    drone_config_path = "splatnav12052024/registered/transforms.json"  # "configs/modal.json"

    if drone_config_path is not None:
        with open(drone_config_path, "r") as f:
            drone_config = json.load(f)

    # # Handeye Calibration
    # handeye_transforms_path = "configs/modal.json" # "transforms/handeye.json"

    # if handeye_transforms_path is not None:
    #     with open(handeye_transforms_path, "r") as f:
    #         handeye_dict = json.load(f)

    drone_config["pose_to_camera"] = camera_to_body

    # enable UKF
    enable_UKF = False  # True

    # Pose Estimator
    loc_node = SplatLocNode(
        config_path=config_path,
        use_compressed_image=False,
        drone_config=drone_config,
        enable_UKF=enable_UKF,
    )

    try:
        rclpy.spin(loc_node)
    except KeyboardInterrupt:
        pass
    finally:
        # save the results
        loc_node.save_results()

        # Clean-up
        loc_node.destroy_node()

        # shutdown
        rclpy.shutdown()
