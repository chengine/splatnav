#%%
import os
import threading
import sys
import select
import tty
import termios
import subprocess
from pathlib import Path    
import time
import json

# Computation imports
import torch
import numpy as np

# ros imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation
from rclpy.duration import Duration

# Splat-Nav imports
from splat.splat_utils import GSplatLoader
from splatplan.splatplan import SplatPlan
from splatplan.opt_utils import SplineOptimizer

from ros_utils import make_point_cloud, record_bag, get_yaw_from_rotation_matrix, yaw_to_rotation_matrix, yaw_from_heading_vector, state_to_waypoint
from semantic_utils import get_goal_locations

# Start of code #

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

        qos_profile_c = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
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

        ### Initialize state variables  ###
        self.position_output = np.array([0.0, 0.0, -0.75])
        self.velocity_output = np.array([0.0, 0.0, 0.0])
        self.acceleration_output = np.array([0.0, 0.0, 0.0])
        self.jerk_output = np.array([0.0, 0.0, 0.0])
        self.yaw_output = 0.

        self.outgoing_waypoint = state_to_output(self.position_output, self.velocity_output, self.acceleration_output, self.jerk_output, self.yaw_output)

        # INITIALIZE PUBLISH CONTROL!!!
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
        self.vmax = 1.      # max velocity of robot. THIS IS ALSO USED FOR ROS CONTROL!

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

        self.spline_optimizer = SplineOptimizer(num_control_points=6, num_deriv=3, dim=3, device=device)
        self.planner = SplatPlan(self.gsplat, robot_config, voxel_config, self.spline_optimizer, device)

        # Publishes the trajectory as a Pose Array
        self.trajectory_publisher = self.create_publisher(PoseArray, "/trajectory", 10)

        self.traj = None
        self.time_of_last_replan = None
        self.time_of_last_state_estimate = None

        self.outputs = []
        
        self.transform = np.eye(4)
        self.current_rotation = np.array([0., 0., 0., 1.])
        self.current_rotation_matrix = np.eye(3)
        self.current_yaw = self.yaw_output
        self.current_pose = np.eye(4)

        print("SplatPlan Initialized...")
        
    def trajectory_callback(self, msg):
        if msg is not None:
            current_time = self.get_clock().now().to_msg()
            current_time_f = current_time.sec + current_time.nanosec * 1e-9     # in seconds!

            self.time_of_last_state_estimate = current_time_f

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

        # The open-loop case only executes this code block once.
        if self.traj is None or self.mode == 'closed-loop':
            print('Updating plan')
            try:
                # TODO: TIME HOW LONG THIS TAKES
                if self.traj is None:
                    x0 = self.outgoing_waypoint
                else:
                    x0 = self.current_position
                
                current_time = self.get_clock().now().to_msg()
                current_time_f = current_time.sec + current_time.nanosec * 1e-9
                self.time_of_last_replan = current_time_f
                # NOTE: !!! self.traj is the np array of traj( a list)
                traj, output = self.plan_path(x0, self.goal)        # traj shape: (opt: num_t) x (opt: num splines)  x num_deriv x dim
                traj = np.array(traj).transpose(1, 0, 2, 3).reshape(-1, self.spline_optimizer.num_deriv + 1, self.spline_optimizer.dim)

                # We're not actually going to use the above trajectory. We will be timing and querying the spline at times.
                # traj is just to visualize.
                velocity = traj[:, 1, :]
                yaw = np.arctan2(velocity[:, 1], velocity[:, 0])        # num_pts x 1

                self.traj = np.concatenate([traj, yaws.reshape(-1, 1, 1)], axis=-1)

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
        # Query positions of traj
        for idx, pt in enumerate(self.traj[:, 0, :]):
            msg = Pose()
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

        # print('Traj Start:', self.traj[0, :3])
        # print('Traj End:', self.traj[-1])

        msg = PoseArray()
        msg.header.frame_id = self.map_name
        msg.poses = poses

        self.trajectory_publisher.publish(msg)               

        return

    def publish_control(self):
        # Get current time so we know how much time as elapsed since last replan
        current_time = self.get_clock().now().to_msg()
        current_time_f = current_time.sec + current_time.nanosec * 1e-9     # in seconds!
        
        control_msg = Float32MultiArray()

        # pop from the first element of the trajectory 
        if (self.start_mission) and self.traj is not None:

            t_query = current_time_f - self.time_of_last_replan
            t_0 = self.time_of_last_state_estimate - self.time_of_last_replan
            trans = self.current_pose[:3, -1]
            x0 = torch.zeros(self.spline_optimizer.num_deriv + 1, self.spline_optimizer.dim, device=device)

            # TODO: Unsure if we want to send the positions... since there is time delay on when we get the state estimate
            x0[0] = torch.tensor(trans, device=device)
            x0[1] = torch.tensor(self.velocity_output, device=device)
            x0[2] = torch.tensor(self.acceleration_output, device=device)
            x0[3] = torch.tensor(self.jerk_output, device=device)

            # Queries the waypoint
            output = self.planner.query_waypoint(t_query, t_0, x0).cpu().numpy()

            yaw = yaw_from_heading_vector(output[1, 1], output[1, 0])
            self.outgoing_waypoint = state_to_waypoint(output[0], output[1], output[2], output[3], yaw)

            self.position_output = [self.outgoing_waypoint[0], self.outgoing_waypoint[1], self.outgoing_waypoint[2]]
            self.velocity_output = [self.outgoing_waypoint[3], self.outgoing_waypoint[4], self.outgoing_waypoint[5]]
            self.acceleration_output = [self.outgoing_waypoint[6], self.outgoing_waypoint[7], self.outgoing_waypoint[8]]        # We set this to 0 for now
            self.jerk_output = [self.outgoing_waypoint[9], self.outgoing_waypoint[10], self.outgoing_waypoint[11]]
            self.yaw_output = self.outgoing_waypoint[-1]

            else:
                print("Trajectory complete.")

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

        # NOTE: Can change the shape of this to accomodate varying number of derivatives
        start = torch.tensor(start, dtype=torch.float32, device=device)
        x0 = start[:-1].reshape(self.spline_optimizer.num_deriv + 1, self.spline_optimizer.dim)

        goal = torch.tensor(goal, dtype=torch.float32, device=device)
        xf = goal[:-1].reshape(self.spline_optimizer.num_deriv + 1, self.spline_optimizer.dim)

        output = self.planner.generate_path(x0, xf)

        return output['traj'], output

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
