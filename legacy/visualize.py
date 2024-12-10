import os
import time
from pathlib import Path
from splat.splat_utils import GSplatLoader
import torch
import numpy as np
import trimesh
import json
import viser
import viser.transforms as tf
import matplotlib as mpl
import scipy
from polytopes.polytopes_utils import find_interior
from scipy.spatial.transform import Rotation as R

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some useful helper functions
def create_polytope_trimesh(polytopes, colors=None):
    for i, (A, b) in enumerate(polytopes):
        # Transfer all tensors to numpy
        pt = find_interior(A, b)

        halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
        hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)
        qhull_pts = hs.intersections

        output = trimesh.convex.convex_hull(qhull_pts)

        if colors is not None:
            output.visual.face_colors = colors[i]
            output.visual.vertex_colors = colors[i]

        if i == 0:
            mesh = output
        else:
            mesh += output
    
    return mesh

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

### ------------------- ###

### PARAMETERS###
# NOTE: THIS REQUIRES CHANGING TO THE SCENE YOU WANT TO VISUALIZE
scene_name = 'ros'
path_to_gsplat = Path('outputs/registered/gemsplat/2024-11-20_155935/config.yml')

modes = ['open-loop']
goal_queries = ['beachball', 'crate', 'microwave', 'ladder']
GOAL_IDS = [0, 1, 2, 3]
TRIAL_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

bounds = None
rotation = tf.SO3.from_x_radians(np.pi).wxyz      # identity rotation

### ------------------- ###
gsplat = GSplatLoader(path_to_gsplat, device)

server = viser.ViserServer()

### ------------------- ###
# Only visualize the gsplat within some bounding box set by bounds
if bounds is not None:
    mask = torch.all((gsplat.means - bounds[:, 0] >= 0) & (bounds[:, 1] - gsplat.means >= 0), dim=-1)
else:
    mask = torch.ones(gsplat.means.shape[0], dtype=torch.bool, device=device)

means = gsplat.means[mask]
covs = gsplat.covs[mask]
colors = gsplat.colors[mask]
opacities = gsplat.opacities[mask]

# Add splat to the scene
server.scene.add_gaussian_splats(
    name="/splats",
    centers= means.cpu().numpy(),
    covariances= covs.cpu().numpy(),
    rgbs= colors.cpu().numpy(),
    opacities= opacities.cpu().numpy(),
    wxyz=rotation,
)

### ------------------- ###

os.makedirs('assets', exist_ok=True)

# Will save the ellipsoids to a file
if not os.path.exists(f"assets/{scene_name}.obj"):
    gsplat.save_mesh(f"assets/{scene_name}.obj", bounds=bounds, res=4)

# Load the ellipsoidal gsplat
mesh = trimesh.load_mesh(str(Path(__file__).parent / f"assets/{scene_name}.obj"))
assert isinstance(mesh, trimesh.Trimesh)
vertices = mesh.vertices
faces = mesh.faces
print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

# Load the ellipsoidal representation
server.scene.add_mesh_simple(
    name="/ellipsoids",
    vertices=vertices,
    faces=faces,
    color=np.array([0.5, 0.5, 0.5]),
    wxyz=rotation,
    opacity=0.5
)

### ------------------- ###

try:
    voxels = trimesh.load_mesh(str(Path(__file__).parent / f"assets/{scene_name}_voxel.obj"))

    # Load the voxel representation
    server.scene.add_mesh_simple(
    name="/voxel",
    vertices=voxels.vertices,
    faces=voxels.faces,
    wireframe=True,
    opacity=0.2,
    wxyz=rotation
    )
except:
    print("No voxel mesh found")

### ------------------- ###
for mode in modes:
    for GOAL_ID in GOAL_IDS:
        for TRIAL_ID in TRIAL_IDS:

            data_path = f'traj/{mode}/{goal_queries[GOAL_ID]}/{TRIAL_ID}'

            # Create a typestore and get the string class.
            typestore = get_typestore(Stores.LATEST)

            bag_name = data_path
            # Create reader instance and open for reading.
            with Reader(bag_name) as reader:
                # Topic and msgtype information is available on .connections list.
                for connection in reader.connections:
                    print(connection.topic, connection.msgtype)

                # Iterate over messages.
                mocap_poses = []
                mocap_timestamps = []

                qvio_poses = []
                qvio_timestamps = []

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

                        transform_c2w = transform

                        qvio_poses.append(transform_c2w)
                        qvio_timestamps.append(timestamp)

                    if connection.topic == '/vrpn_mocap/modalai/pose':
                        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                        positions = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                        quaternion = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

                        rot_object = R.from_quat(quaternion)
                        rot_mat = rot_object.as_matrix()

                        transform = np.eye(4)
                        transform[:3, :3] = rot_mat
                        transform[:3, 3] = positions

                        transform_c2w = transform

                        mocap_poses.append(transform_c2w)
                        mocap_timestamps.append(timestamp)

            mocap_poses = np.array(mocap_poses)
            qvio_poses = np.array(qvio_poses)

            # Visualize the trajectory and series of line segments
            traj = qvio_poses[..., :3, -1]

            points = np.stack([traj[:-1], traj[1:]], axis=1)
            progress = np.linspace(0, 1, len(points))

            # Safety margin color
            cmap = mpl.cm.get_cmap('jet')
            colors = np.array([cmap(prog) for prog in progress])[..., :3]
            colors = colors.reshape(-1, 1, 3)

            # Add trajectory to scene
            server.scene.add_line_segments(
                name=f'traj/{mode}/{goal_queries[GOAL_ID]}/{TRIAL_ID}',
                points=points,
                colors=colors,
                line_width=10,
                wxyz=rotation,
            )

            # # Visualize the polytopes as well
            # polytopes = data['polytopes']
            # polytopes = [(np.array(polytope)[..., :3], np.array(polytope)[..., 3]) for polytope in polytopes]

            # colors = np.array([cmap(i) for i in np.linspace(0, 1, len(polytopes))])[..., :3]
            # colors = colors.reshape(-1, 3)
            # colors = np.concatenate([colors, 0.1*np.ones((len(polytopes), 1))], axis=-1)
            # colors = (255*colors).astype(np.uint8)

            # # Create polytope corridor mesh object
            # corridor_mesh = create_polytope_trimesh(polytopes, colors=colors)

            # # Add the corridor to the scene
            # server.scene.add_mesh_trimesh(
            # name=f"/corridor_{i}",
            # mesh=corridor_mesh,
            # wxyz=rotation,
            # visible=False
            # )

while True:
    time.sleep(10.0)