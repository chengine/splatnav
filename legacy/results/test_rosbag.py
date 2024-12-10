#%%
import os
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import cv2
from rosbags.image import message_to_cvimage
from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path
import imageio
import json

exp_type = 'open-loop'
trial = 3

current_path = Path.cwd()  # Get the current working directory as a Path object
parent_path = current_path.parent  # Get the parent directory

if exp_type == 'open-loop':
    rosbag_path = f'traj/traj_ol_{trial}'
elif exp_type == 'closed-loop':
    rosbag_path = f'traj/traj_cl_{trial}'
else:
    raise ValueError('Invalid experiment type')

bag_name = os.path.join(str(parent_path), rosbag_path)

# Create a typestore and get the string class.
typestore = get_typestore(Stores.LATEST)

# Create reader instance and open for reading.
with Reader(bag_name) as reader:
    # Topic and msgtype information is available on .connections list.
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # Iterate over messages.
    imgs = []
    img_timestamps = []

    poses = []
    poses_timestamps = []

    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/republished_image':
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            img = message_to_cvimage(msg, 'rgb8')

            imgs.append(img)
            img_timestamps.append(timestamp)

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

    # create directory
    total_data = []
    os.makedirs(f'{exp_type}_{trial}', exist_ok=True)

    if len(imgs) == 0:
        for i, pose in enumerate(poses):
            data = {
                'transforms_matrix': pose.tolist(),
                'timestamp': poses_timestamps[i],
                'file_path': None
            }

            total_data.append(data)
    else:
        for i, (img, pose) in enumerate(zip(imgs, poses)):
            imageio.imwrite(f'{exp_type}_{trial}/r_{i}.png', img)

            data = {
                'transforms_matrix': pose.tolist(),
                'timestamp': poses_timestamps[i],
                'file_path': f'{exp_type}_{trial}/r_{i}.png'
            }

            total_data.append(data)

    with open(f'{exp_type}_{trial}/transforms.json', 'w') as f:
        json.dump(total_data, f, indent=4)

# %%
