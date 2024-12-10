#%%
import os
import torch
from pathlib import Path    
import time
import numpy as np
from tqdm import tqdm
import json

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import cv2
from rosbags.image import message_to_cvimage
from scipy.spatial.transform import Rotation as R

import polytope
from splat.gsplat_utils import GSplatLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load file
current_path = Path.cwd()  # Get the current working directory as a Path object
parent_path = current_path.parent  # Get the parent directory

path_to_gsplat = Path('outputs/configs/ros-depth-splatfacto/2024-10-30_140655/config.yml')
path_to_gsplat = parent_path.joinpath(path_to_gsplat)

trial_number = ['0', '1', '2', '3']
exp_types = ['open-loop']

radius = 0.2       # radius of robot
amax = 1.
vmax = 1.

tnow = time.time()
gsplat = GSplatLoader(path_to_gsplat, device)
print('Time to load GSplat:', time.time() - tnow)


for j in trial_number:
    for exp_type in exp_types:
        json_path = f'traj/data_{exp_type}_{j}.json'

        if exp_type == 'open-loop':
            rosbag_path = f'traj/traj_ol_{j}'
        elif exp_type == 'closed-loop':
            rosbag_path = f'traj/traj_cl_{j}'
        else:
            raise ValueError('Invalid experiment type')

        with open( os.path.join(str(parent_path), json_path), 'r') as f:
            meta = json.load(f)

        # Load in the data
        total_data = meta['total_data']

        total_data_processed = []
        for i, data in enumerate(total_data):
            print(f"Processing trajectory {i}/{len(total_data)}")

            traj = torch.tensor(data['traj'], device=device)[:, :3]

            # Compute the distance to the GSplat
            safety_margin = []
            for pt in traj:
                h, grad_h, hess_h, info = gsplat.query_distance(pt, radius=radius, distance_type='ball-to-ellipsoid')

                # NOTE: IMPORTANT!!! h is the squared signed distance minus the radius squared, so we need to undo this, because we want signed distance - radius
                # record min value of h
                squared_signed_distance = torch.min(h) + radius**2
                sign_dist = torch.sign(squared_signed_distance)
                mag_dist = torch.abs(squared_signed_distance)
                signed_distance = sign_dist * torch.sqrt(mag_dist) - radius
                safety_margin.append(signed_distance.item())

            # Compute the total path length
            path_length = torch.sum(torch.norm(traj[1:] - traj[:-1], dim=1)).item()

            # Quality of polytopes
            polytopes = data['polytopes'] #[torch.cat([polytope[0], polytope[1].unsqueeze(-1)], dim=-1).tolist() for polytope in polytopes]

            polytope_vols = []
            polytope_radii = []

            polytope_margin = []
            for poly in polytopes:

                poly = np.array(poly)
                A = poly[:, :-1]
                b = poly[:, -1]

                p = polytope.Polytope(A, b)
                polytope_vols.append(p.volume)
                polytope_radii.append(np.linalg.norm(p.chebR))

                vertices = torch.tensor(polytope.extreme(p), device=device, dtype=torch.float32)

                for vertex in vertices:
                    h, grad_h, hess_h, info = gsplat.query_distance(vertex, radius=radius, distance_type='ball-to-ellipsoid')

                    squared_signed_distance = torch.min(h) + radius**2
                    sign_dist = torch.sign(squared_signed_distance)
                    mag_dist = torch.abs(squared_signed_distance)
                    signed_distance = sign_dist * torch.sqrt(mag_dist) - radius

                    # record min value of h
                    polytope_margin.append(signed_distance.item())

            data['safety_margin'] = safety_margin
            data['path_length'] = path_length
            data['polytope_vols'] = polytope_vols
            data['polytope_radii'] = polytope_radii
            data['polytope_margin'] = polytope_margin

            total_data_processed.append(data)

        meta['total_data'] = total_data_processed

        # bag could be empty
        try:
            # We want to add another item to the dictionary for ros data
            bag_name = os.path.join(str(parent_path), rosbag_path)

            # Create a typestore and get the string class.
            typestore = get_typestore(Stores.LATEST)

            # Create reader instance and open for reading.
            with Reader(bag_name) as reader:
                # Topic and msgtype information is available on .connections list.
                for connection in reader.connections:
                    print(connection.topic, connection.msgtype)

                # Iterate over messages.
                poses = []
                poses_timestamps = []

                for connection, timestamp, rawdata in reader.messages():
                    if connection.topic == '/republished_pose':
                        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                        positions = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                        quaternion = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

                        rot_object = R.from_quat(quaternion)
                        rot_mat = rot_object.as_matrix()

                        transform = np.eye(4)
                        transform[:3, :3] = rot_mat
                        transform[:3, 3] = positions

                        poses.append(transform)
                        poses_timestamps.append(timestamp)

            poses = np.stack(poses, axis=0)
            traj = torch.tensor(poses, device=device, dtype=torch.float32)   # N x 4 x 4
            points = traj[..., :3, 3]

            # Compute the distance to the GSplat
            safety_margin_ros = []
            for pt in points:
                h, grad_h, hess_h, info = gsplat.query_distance(pt, radius=radius, distance_type='ball-to-ellipsoid')

                # NOTE: IMPORTANT!!! h is the squared signed distance minus the radius squared, so we need to undo this, because we want signed distance - radius
                # record min value of h
                squared_signed_distance = torch.min(h) + radius**2
                sign_dist = torch.sign(squared_signed_distance)
                mag_dist = torch.abs(squared_signed_distance)
                signed_distance = sign_dist * torch.sqrt(mag_dist) - radius
                safety_margin_ros.append(signed_distance.item())

        except:
            safety_margin_ros = None
            poses_timestamps = None
            traj = None

        ros_data = {
            'poses': traj.tolist() if traj is not None else None,
            'poses_timestamps': poses_timestamps,
            'safety_margin': safety_margin_ros
        }

        meta['ros_data'] = ros_data

        write_path = f'traj/{exp_type}_{j}_processed.json'

        # Save the data
        with open( os.path.join(str(parent_path), write_path), 'w') as f:
            json.dump(meta, f, indent=4)

#%%