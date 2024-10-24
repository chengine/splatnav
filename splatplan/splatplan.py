import torch
import numpy as np
import open3d as o3d
import scipy
from scipy import sparse
import clarabel
import time

from polytopes.polytopes_utils import h_rep_minimal, find_interior, compute_segment_in_polytope
from initialization.grid_utils import GSplatVoxel
from polytopes.collision_set import GSplatCollisionSet
from polytopes.decomposition import compute_polytope
from ellipsoids.intersection_utils import compute_intersection_linear_motion

class SplatPlan():
    def __init__(self, gsplat, robot_config, env_config, spline_planner, device):
        # gsplat: GSplat object

        self.gsplat = gsplat
        self.device = device

        # Robot configuration
        self.radius = robot_config['radius']
        self.vmax = robot_config['vmax']
        self.amax = robot_config['amax']
        self.collision_set = GSplatCollisionSet(self.gsplat, self.vmax, self.amax, self.radius, self.device)

        # Environment configuration (specifically voxel)
        self.lower_bound = env_config['lower_bound']
        self.upper_bound = env_config['upper_bound']
        self.resolution = env_config['resolution']

        tnow = time.time()
        torch.cuda.synchronize()
        self.gsplat_voxel = GSplatVoxel(self.gsplat, lower_bound=self.lower_bound, upper_bound=self.upper_bound, resolution=self.resolution, radius=self.radius, device=device)
        torch.cuda.synchronize()
        print('Time to create GSplatVoxel:', time.time() - tnow)

        # Spline planner
        self.spline_planner = spline_planner

        # Save the mesh
        # gsplat_voxel.create_mesh(save_path=save_path)
        # gsplat.save_mesh(scene_name + '_gsplat.obj')

        # Record times
        self.times_cbf = []
        self.times_qp = []
        self.times_prune = []

    def generate_path(self, x0, xf):
        # Part 1: Computes the path seed using A*
        tnow = time.time()
        path = self.gsplat_voxel.create_path(x0, xf)
        torch.cuda.synchronize()
        time_astar = time.time() - tnow

        times_collision_set = 0
        times_polytope = 0

        polytopes = []      # List of polytopes (A, b)
        segments = torch.tensor(np.stack([path[:-1], path[1:]], axis=1), device=self.device)

        for it, segment in enumerate(segments):
            # If this is the first line segment, we always create a polytope. Or subsequently, we only instantiate a polytope if the line segment
            if it == 0:

                # Part 2: Computes the collision set

                tnow = time.time()
                output = self.collision_set.compute_set_one_step(segment)
                torch.cuda.synchronize()
                times_collision_set += time.time() - tnow

                # Part 3: Computes the polytope
                tnow = time.time()
                polytope = self.get_polytope_from_outputs(output)
                torch.cuda.synchronize()
                times_polytope += time.time() - tnow

            else:
                # Test if the line segment is within the polytope
                # If the segment is within the polytope, we proceed to next segment
                if compute_segment_in_polytope(polytope[0], polytope[1], segment):

                    continue

                else:
                    # If the segment is not within the polytope, we create a new polytope
                    # Part 2: Computes the collision set

                    tnow = time.time()
                    output = self.collision_set.compute_set_one_step(segment)
                    torch.cuda.synchronize()
                    times_collision_set += time.time() - tnow 

                    # Part 3: Computes the polytope
                    tnow = time.time()
                    polytope = self.get_polytope_from_outputs(output)
                    torch.cuda.synchronize()
                    times_polytope += time.time() - tnow

            polytopes.append(polytope)
            #print(f"Instantiated polytope at segment {it}")

        # Step 4: Perform Bezier spline optimization
        tnow = time.time()
        traj, feasible = self.spline_planner.optimize_b_spline(polytopes, x0, xf)
        if not feasible:
            traj = torch.stack([x0, xf], dim=0)
        torch.cuda.synchronize()
        times_opt = time.time() - tnow
  
        # Save outgoing information
        traj_data = {
            'path': path.tolist(),
            'polytopes': [torch.cat([polytope[0], polytope[1].unsqueeze(-1)], dim=-1).tolist() for polytope in polytopes],
            'num_polytopes': len(polytopes),
            'traj': traj.tolist(),
            'times_astar': time_astar,
            'times_collision_set': times_collision_set,
            'times_polytope': times_polytope,
            'times_opt': times_opt,
            'feasible': feasible
        }

        return traj_data
    
    def get_polytope_from_outputs(self, data):
        # For every single line segment, we always create a polytope at the first line segment, 
        # and then we subsequently check if future line segments are within the polytope before creating new ones.
        gs_ids = data['gaussian_ids']

        A_bb = data['A_bb']
        b_bb = data['b_bb_shrunk']
        segment = data['path']
        delta_x = segment[1] - segment[0]

        midpoint = data['midpoint']

        if len(gs_ids) == 0:
            return (A_bb, b_bb)
        elif len(gs_ids) == 1:
            rots = data['rots'].expand(2, -1, -1)
            scales = data['scales'].expand(2, -1)
            means = data['means'].expand(2, -1)
        else:
            rots = data['rots']
            scales = data['scales']
            means = data['means']

        # Perform the intersection test
        intersection_output = compute_intersection_linear_motion(segment[0], delta_x, rots, scales, means, 
                                R_B=None, S_B=self.radius, collision_type='sphere', 
                                mode='bisection', N=10)

        # Compute the polytope
        A, b, pts = compute_polytope(intersection_output['deltas'], intersection_output['Q_opt'], intersection_output['K_opt'], intersection_output['mu_A'])

        # The full polytope is a concatenation of the intersection polytope and the bounding box polytope
        A = torch.cat([A, A_bb], dim=0)
        b = torch.cat([b, b_bb], dim=0)

        norm_A = torch.linalg.norm(A, dim=-1, keepdims=True)
        A = A / norm_A
        b = b / norm_A.squeeze()

        # By manageability, the midpoint should always be clearly within the polytope
        # NOTE: Let's hope there are no errors here.
        A, b, qhull_pts = h_rep_minimal(A.cpu().numpy(), b.cpu().numpy(), midpoint.cpu().numpy())

        return (torch.tensor(A, device=self.device), torch.tensor(b, device=self.device))

    def save_polytope(self, polytopes, save_path):
        # Initialize mesh object
        mesh = o3d.geometry.TriangleMesh()

        for (A, b) in polytopes:
            # Transfer all tensors to numpy
            A = A.cpu().numpy()
            b = b.cpu().numpy()

            pt = find_interior(A, b)

            halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
            hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)
            qhull_pts = hs.intersections

            pcd_object = o3d.geometry.PointCloud()
            pcd_object.points = o3d.utility.Vector3dVector(qhull_pts)
            bb_mesh, qhull_indices = pcd_object.compute_convex_hull()
            mesh += bb_mesh
        
        success = o3d.io.write_triangle_mesh(save_path, mesh, print_progress=True)

        return success