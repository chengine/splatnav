#%%
import os
import torch
from pathlib import Path    
import time
import numpy as np
from splat.splat_utils import GSplatLoader
from splatplan.splatplan import SplatPlan
from splatplan.spline_utils import SplinePlanner

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
#from px4_msgs.msg import VehicleOdometry  # Adjusted to use the PX4-specific odometry message
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.duration import Duration
import time
import json

import threading
import sys
import select
import tty
import termios
import subprocess

from ros_utils import make_point_cloud
from semantic_utils import get_goal_locations

def record_bag(bag_name, topics):
    """Records a rosbag file."""
    command = ["ros2", "bag", "record", "-o", bag_name] + topics
    process = subprocess.Popen(command)

    return process

def get_yaw_from_rotation_matrix(R):
    """
    Extracts the yaw angle from a 3x3 rotation matrix.

    Args:
        R (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        float: Yaw angle in radians.
    """

    yaw = np.arctan2(R[1, 0], R[0, 0])
    return yaw

def yaw_to_rotation_matrix(yaw):
    """Converts a yaw angle to a 3x3 rotation matrix.

    Args:
        yaw: The yaw angle in radians.

    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """

    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GOAL_QUERIES = ['keyboard', 'beachball', 'phonebook', 'microwave']

GOAL_ID = 0
TRIAL_ID = 0
MODE = 'closed-loop'
POSE_SOURCE = 'external'
#POSE_SOURCE = 'splat-loc'

# we precomute this for faster start up time.
GOALS = np.array([
    [2.0180538,  1.106223,  -1.2050834],
    [2.2288399, -1.8753463, -1.2965584],
    [4.0521755,  0.9269549, -1.2108827],
    [4.3809066, -1.6589532, -1.2726507]
])

class ControlNode(Node):

    def __init__(self, mode='open-loop', pose_source='external'):
        super().__init__('control_node')
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_c = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_incoming = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        # Set the map name for visualization in RVIZ
        self.map_name = "camera_link"

        self.mode = mode
        self.original_mode = mode

        # Publish to the control topic
        self.control_publisher = self.create_publisher(
            Float32MultiArray,
            '/control',
            qos_profile_c
        )

        # Publishes the static voxel grid as a point cloud
        self.pcd_publisher = self.create_publisher(
            PointCloud2, "/gsplat_pcd", 10
        )
        self.timer = self.create_timer(1.0, self.pcd_callback)
        self.pcd_msg = None

        # This publishes the goal pose
        self.goal_publisher = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.goal_timer = self.create_timer(1.0, self.goal_callback)

        # When rebpulished pose is updated, we update the trajectory too! ONLY WORKS FOR SLOW REBPUB RATES
        self.pose_source = pose_source
        if pose_source == 'external':
            print('Getting poses externally!!!')
            self.position_subscriber = self.create_subscription(
                PoseStamped,
                '/republished_pose',
                self.trajectory_callback,
                10)
            
        ### Uses splat-loc ###
        elif pose_source == 'splat-loc':
            print('Getting poses from Splat-Loc!!!')
            self.position_subscriber = self.create_subscription(
                PoseArray,
                '/estimated_and_relative_pose',
                self.trajectory_callback,
                10)
        else:
            raise ValueError('Not a valid pose source')
        
        # Publish vio pose source
        self.vio_publisher = self.create_publisher(PoseStamped, "/vio_pose", 10)
        # The state of SplatPlan. This is used to trigger replanning. 
        self.current_pose_publisher = self.create_publisher(PoseStamped, "/current_pose", 10)

        ### Initialize variables  ###
        self.velocity_output = [0.0, 0.0, 0.0]
        self.position_output = [0.0, 0.0, -0.75]
        self.current_position = [0.0, 0.0, -0.75]
        self.acceleration_output = [0.0, 0.0, 0.0]

        #TODO: SET TO 0!!! #
        self.des_yaw_rate = 0.0
        self.yaw = 0.
        self.yaw0 = 0.
        self.outgoing_waypoint = [0.0, 0., -0.75, 0., 0., 0., 0., 0., 0., 0., 0., 0., self.yaw]

        self.timer = self.create_timer(1.0 / 10.0, self.publish_control)

        self.start_mission = False

        # Start the keyboard listener thread
        self.keyboard_thread = threading.Thread(target=self.key_listener)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

        ### SPLATPLAN INITIALIZATION ###
        ############# Specify scene specific data here  #############
        # Points to the config file for the GSplat
        path_to_gsplat = Path('outputs/registered/gemsplat/2024-12-05_155215/config.yml')

        self.radius = 0.25       # radius of robot
        self.amax = 1.
        self.vmax = 1.

        self.distance_between_points = 0.2

        lower_bound = torch.tensor([0., -2.5, -1.5], device=device)
        upper_bound = torch.tensor([5., 2., 0.25], device=device)
        resolution = 80

        #################
        # Robot configuration
        robot_config = {
            'radius': self.radius,
            'vmax': self.vmax,
            'amax': self.amax,
        }

        # Environment configuration (specifically voxel)
        voxel_config = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'resolution': resolution,
        }

        tnow = time.time()
        self.gsplat = GSplatLoader(path_to_gsplat, device)
        print('Time to load GSplat:', time.time() - tnow)

        print(f'There are {len(self.gsplat.means)} Gaussians in the Splat.')


        ### CALCULATE GOAL LOcATIONS ###
        self.goal_queries = ['keyboard', 'beachball', 'phonebook', 'microwave']

        # goal = get_goal_locations(self.gsplat.splat,
        #                    [self.goal_queries[GOAL_ID]],
        #                    bounding_box_min=lower_bound,
        #                    bounding_box_max=upper_bound
        # )[0]
        # print('Goal', goal)

        goal = GOALS[GOAL_ID]

        self.goal = goal.tolist()
        
        spline_planner = SplinePlanner(spline_deg=6, device=device)
        self.planner = SplatPlan(self.gsplat, robot_config, voxel_config, spline_planner, device)

        # Publishes the trajectory as a Pose Array
        self.trajectory_publisher = self.create_publisher(PoseArray, "/trajectory", 10)

        self.traj = None
        self.do_replan = True

        self.outputs = []

        
        self.transform = np.eye(4)
        self.current_rotation = np.array([0., 0., 0., 1.])
        self.current_rotation_matrix = np.eye(3)
        self.current_yaw = 0
        self.current_pose = np.eye(4)

        print("SplatPlan Initialized...")
        
    def trajectory_callback(self, msg):
        if msg is not None:

            if self.pose_source == 'splat-loc':
            # It's a pose array (pose in splat-loc frame, relative transform filtered, relative transform raw)
                pose, relative, _ = msg.poses

                self.current_position = [pose.position.x, pose.position.y, pose.position.z]
                self.current_rotation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                self.current_rotation_matrix = Rotation.from_quat(self.current_rotation).as_matrix()
                self.current_yaw = get_yaw_from_rotation_matrix(self.current_rotation_matrix)

                self.current_pose = np.eye(4)
                self.current_pose[:3, -1] = np.array(self.current_position)
                self.current_pose[:3, :3] = self.current_rotation_matrix

                self.transform = np.eye(4)
                self.transform[:3, -1] = np.array([relative.position.x, relative.position.y, relative.position.z])
                self.transform[:3, :3] = Rotation.from_quat(np.array([relative.orientation.x, relative.orientation.y, relative.orientation.z, relative.orientation.w])).as_matrix()

                out_msg = PoseStamped()
                out_msg.header.frame_id = self.map_name
                out_msg.pose.position = pose.position
                out_msg.pose.orientation = pose.orientation

                vio_pose = self.transform @ self.current_pose
                vio_msg = PoseStamped()
                vio_msg.header.frame_id = self.map_name
                vio_msg.pose.position.x, vio_msg.pose.position.y, vio_msg.pose.position.z = vio_pose[0, -1], vio_pose[1, -1], vio_pose[2, -1]
                vio_msg.pose.orientation.x, vio_msg.pose.orientation.y, vio_msg.pose.orientation.z, vio_msg.pose.orientation.w = Rotation.from_matrix(vio_pose[:3, :3]).as_quat().tolist()

                self.current_pose_publisher.publish(out_msg)
                self.vio_publisher.publish(vio_msg)

            else:
            # It's a PoseStamped
                self.current_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                self.current_rotation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
                self.current_rotation_matrix = Rotation.from_quat(self.current_rotation).as_matrix()
                self.current_yaw = get_yaw_from_rotation_matrix(self.current_rotation_matrix)

                self.current_pose = np.eye(4)
                self.current_pose[:3, -1] = np.array(self.current_position)
                self.current_pose[:3, :3] = self.current_rotation_matrix

                msg.header.frame_id = self.map_name
                # There's no difference
                self.current_pose_publisher.publish(msg)
                self.vio_publisher.publish(msg)

        if not self.do_replan:
            self.do_replan = True

            return

        # The open-loop case only executes this code block once.
        if self.traj is None or self.mode == 'closed-loop':
            print('Updating plan')
            try:
                if self.traj is None:
                    traj, output = self.plan_path(self.outgoing_waypoint[:3], self.goal)
                    # NOTE: !!! self.traj is the np array of traj( a list)
                    traj = np.array(traj)

                else:
                    traj, output = self.plan_path(self.current_position, self.goal)
                    # NOTE: !!! self.traj is the np array of traj( a list)
                    traj = np.array(traj)

                yaw = self.outgoing_waypoint[-1]
                yaws = [yaw]
                for idx, pt in enumerate(traj):
                    if idx > 0:

                        diff = traj[idx] - traj[idx - 1]

                        prev_yaw = yaw
                        yaw = np.arctan2(diff[1], diff[0])

                        closest_k = np.round(-(yaw - prev_yaw) / (2*np.pi))

                        yaw = yaw + 2*np.pi*closest_k     

                        yaws.append(yaw)

                yaws = np.stack(yaws)

                self.traj = np.concatenate([traj, yaws.reshape(-1, 1)], axis=-1)

                output['traj'] = self.traj.tolist()

                self.outputs.append(output)

                # IF we are close to goal
                if np.linalg.norm(np.array(self.current_position) - np.array(self.goal)) < 0.25:
                    print('Reached Goal!')

                    # Stop replanning by switching modes
                    self.mode = 'open-loop'

            except Exception as e:
                print(e)

        poses = []
        for idx, pt in enumerate(self.traj):
            msg = Pose()
            # msg.header.frame_id = self.map_name
            msg.position.x, msg.position.y, msg.position.z = pt[0], pt[1], pt[2]
            yaw = pt[-1]
            quat = Rotation.from_euler("z", yaw).as_quat()
            (
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ) = (quat[0], quat[1], quat[2], quat[3])

            poses.append(msg)

        print('Traj Start:', self.traj[0, :3])
        # print('Traj End:', self.traj[-1])

        msg = PoseArray()
        msg.header.frame_id = self.map_name
        msg.poses = poses

        self.trajectory_publisher.publish(msg)

        self.do_replan = False

        # waypoint [pos, vel, accel, jerk]
                    # Find the closest position to the current position
        distance = np.linalg.norm( self.traj[:, :3] - np.array(self.current_position)[None] , axis=-1 ).squeeze()
                    
        # Find all distances within a ball to the drone
        within_ball = distance <= self.distance_between_points

        # If no points exist within the ball, choose the closest point
        if not np.any(within_ball):
            min_index = np.argmin(distance)
            # self.outgoing_waypoint = ( self.traj[min_index] + self.outgoing_waypoint ) / 2
            self.traj = self.traj[min_index:]
        
        #else
        else:
            # find the point that makes the most progress
            indices = np.arange(len(distance))[within_ball]
            max_index = np.max(indices)
            # self.outgoing_waypoint = ( self.traj[max_index] + self.outgoing_waypoint ) / 2
            self.traj = self.traj[max_index:]                   

        return

    def goal_callback(self):
        msg = PoseStamped()
        msg.header.frame_id = self.map_name

        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = self.goal[0], self.goal[1], self.goal[2]
        yaw = 0.
        quat = Rotation.from_euler("z", yaw).as_quat()
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = (quat[0], quat[1], quat[2], quat[3])

        self.goal_publisher.publish(msg)

    def pcd_callback(self):
        if self.pcd_msg is None:
            points = self.gsplat.means.cpu().numpy()
            colors = (255 * torch.clip(self.gsplat.colors, 0., 1.).cpu().numpy()).astype(np.uint32)

            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgba", offset=12, datatype=PointField.UINT32, count=1),
            ]

            self.pcd_msg = make_point_cloud(points, colors, self.map_name, fields)

        self.pcd_publisher.publish(self.pcd_msg)

    def key_listener(self):
        print("Press the space bar to start the mission.")
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.start_mission:
                dr, dw, de = select.select([sys.stdin], [], [], 0)
                if dr:
                    c = sys.stdin.read(1)
                    if c == "1":
                        self.start_mission = True
                        print("Starting trajectory...")
                        break
        except Exception as e:
            print(e)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def publish_control(self):
        # current_time = self.get_clock().now().to_msg()
        # current_time_f = current_time.sec + current_time.nanosec * 1e-9
        # print(current_time_f)
        control_msg = Float32MultiArray()

        # pop from the first element of the trajectory 
        if (self.start_mission) and self.traj is not None:

            if self.traj.shape[0] > 0:

                # If open loop, pop the trajectory
                #if self.mode == 'open-loop' or not self.do_replan:

                yaw_difference = self.traj[0, -1] - self.outgoing_waypoint[-1]
                self.outgoing_waypoint = ( 0.4*np.array(self.traj[0]) + 0.6*np.array(self.outgoing_waypoint) ).tolist()

                allowed_yaw_diff = np.clip(yaw_difference, -np.pi/30, np.pi/30)
                self.outgoing_waypoint[-1] = self.outgoing_waypoint[-1] + allowed_yaw_diff

                self.traj = self.traj[2:]

                # else:
                #     # waypoint [pos, vel, accel, jerk]
                #     # Find the closest position to the current position
                #     distance = np.linalg.norm( self.traj[:, :3] - np.array(self.current_position)[None] , axis=-1 ).squeeze()
                    
                #     # Find all distances within a ball to the drone
                #     within_ball = distance <= self.distance_between_points

                #     # If no points exist within the ball, choose the closest point
                #     if not np.any(within_ball):
                #         min_index = np.argmin(distance)
                #         self.outgoing_waypoint = ( self.traj[min_index] + self.outgoing_waypoint ) / 2
                    
                #     #else
                #     else:
                #         # find the point that makes the most progress
                #         indices = np.arange(len(distance))[within_ball]
                #         max_index = np.max(indices)
                #         self.outgoing_waypoint = ( self.traj[max_index] + self.outgoing_waypoint ) / 2                        
                    
                self.position_output = [self.outgoing_waypoint[0], self.outgoing_waypoint[1], self.outgoing_waypoint[2]]
                self.velocity_output = [self.outgoing_waypoint[3], self.outgoing_waypoint[4], self.outgoing_waypoint[5]]
                acceleration_output = [self.outgoing_waypoint[6], self.outgoing_waypoint[7], self.outgoing_waypoint[8]]        # We set this to 0 for now
                self.jerk = [self.outgoing_waypoint[9], self.outgoing_waypoint[10], self.outgoing_waypoint[11]]
                self.yaw_output = self.outgoing_waypoint[-1]

            else:
                print("Trajectory complete.")
        else:
            self.yaw_output = self.yaw


        # PUBLISH OUTGOING CONTROL
        outgoing_transform = np.eye(4)
        outgoing_transform[:3, -1] = self.position_output[:3]
        outgoing_transform[:3, :3] = yaw_to_rotation_matrix(self.yaw_output)

        outgoing_transform = self.transform @ outgoing_transform
        outgoing_velocity = self.transform[:3, :3] @ self.velocity_output[:3]
        outgoing_acceleration = self.transform[:3, :3] @ self.acceleration_output[:3]

        control_msg.data = [
            outgoing_acceleration[0], outgoing_acceleration[1], outgoing_acceleration[2],
            outgoing_velocity[0], outgoing_velocity[1], outgoing_velocity[2],
            outgoing_transform[0, -1], outgoing_transform[1, -1], outgoing_transform[2, -1], get_yaw_from_rotation_matrix(outgoing_transform[:3, :3])
        ]

        self.control_publisher.publish(control_msg)
        self.publish_control_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9  
        # print("control message: ", control_msg.data)

    def plan_path(self, start, goal):
        start = torch.tensor(start).to(device).to(torch.float32)
        goal = torch.tensor(goal).to(device).to(torch.float32)
        output = self.planner.generate_path(start, goal)

        return output['traj'], output
    
    def save_data(self):
        save_path = f'traj/{POSE_SOURCE}/{self.original_mode}/{self.goal_queries[GOAL_ID]}/{TRIAL_ID}'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.data_savepath = os.path.join(save_path, 'data.json')
    
        out_data = {
            'total_data': self.outputs,
            'radius': self.radius,
            'amax': self.amax,
            'vmax': self.vmax
        }

        with open(self.data_savepath, 'w') as f:
            json.dump(out_data, f, indent=4)

        print('Saved data!')

def main(args=None):
    bag_name = f"traj/{POSE_SOURCE}/{MODE}/{GOAL_QUERIES[GOAL_ID]}/{TRIAL_ID}"
    topics = ["/republished_pose", "/republished_image", "/republished_pointcloud", "/estimated_and_relative_pose", "/control", "/vrpn_mocap/modal/pose"]
    bag_process = record_bag(bag_name, topics)

    rclpy.init(args=args)
    control_node = ControlNode(mode=MODE, pose_source=POSE_SOURCE)

    try:
        rclpy.spin(control_node)
    except KeyboardInterrupt:
        pass
    finally:
        bag_process.terminate()
        control_node.save_data()
        control_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
