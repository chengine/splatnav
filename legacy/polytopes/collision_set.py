import numpy as np
import scipy
import torch
import open3d as o3d

from ellipsoids.mesh_utils import create_gs_mesh
from ellipsoids.covariance_utils import quaternion_to_rotation_matrix

# This function calculates the bounding box for all line segments, but this is not strictly necessary
# Since we will probably prune polytopes and so sequential one-at-a-time may be fine as well.
def compute_bounding_box(path, rs):
    # path: N+1 x 3

    # Find the bounding box hyperplanes of the path

    # Without loss of generality, let us assume the local coordinate system is at the midpoint of the path and local x is along the path
    # We artibrarily choose y and z accordingly.
    midpoints = 0.5 * (path[1:] + path[:-1])        # N x 3
    lengths = torch.linalg.norm(path[1:] - path[:-1], dim=1)        # N

    lengths_x = lengths/2 + rs
    lengths_y = rs*torch.ones_like(lengths_x)
    lengths_z = rs*torch.ones_like(lengths_x)

    local_x = (path[1:] - path[:-1]) / torch.linalg.norm(path[1:] - path[:-1], dim=-1, keepdim=True)      # This is the pointing direction of the path (N x 3)

    # TODO: May have to treat the case where the path is just two points.!!!
    # Do Gram-Schmidt to find the other two directions
    random_vec = torch.randn(local_x.shape[-1], device=local_x.device)  # take a random vector
    local_y = random_vec[None, :] -  torch.sum(random_vec[None, :]*local_x, dim=-1, keepdim=True) * local_x       # make it orthogonal to x
    local_y = local_y / torch.linalg.norm(local_y, dim=-1, keepdim=True)            # normalize it
    local_z = torch.cross(local_x, local_y)    # This is the direction perpendicular to the path and the y-axis

    rotation_matrix = torch.stack([local_x, local_y, local_z], dim=-1)    # This is the local x,y,z to world frame rotation (N x 3 x 3)

    # These vectors form the normal of the hyperplanes. We simply need to find their intercepts. 
    # We are basically trying to find a_i.T * (x_0 + l_i * a_i) = b_i = a_i.T * x_0 + l_i (since a_i.T * a_i = 1)
    xyz_lengths = torch.stack([lengths_x, lengths_y, lengths_z], dim=-1)
    intercepts_pos = torch.bmm(midpoints[..., None, :], rotation_matrix).squeeze() + xyz_lengths    # N x 3
    intercepts_neg = -(intercepts_pos) + 2*xyz_lengths    # N x 3

    # Represent as x.T A <= b
    A = torch.cat([rotation_matrix, -rotation_matrix], dim=-1).transpose(1, 2)    # N x 6 x 3
    b = torch.cat([intercepts_pos, intercepts_neg], dim=-1)    # N x 6

    return A, b

def ellipsoid_halfspace_intersection(means, rots, scales, A, b):
    # This comparison tensor will be N (num of Gaussians) x 6 (num of hyperplanes)
    a_times_mu = means @ A.T
    numerator = b[None, :] - a_times_mu        # N x 6

    a_times_R = rots.transpose(1, 2) @ A.T # N x 3 x 6 
    denominator = torch.linalg.norm( a_times_R * scales[..., None] , dim=1)    # N x 6

    distance = -numerator / denominator

    # A gaussian must satisfy the metric for all 6 hyperplanes
    keep_gaussian = (distance <= 1.).all(dim=-1)      # mask of length N

    return keep_gaussian

# def bounding_box_and_points(self, path, point_cloud):
#     # path: N+1 x 3
#     # point_cloud: M x 3

#     # Find the bounding box hyperplanes of the path

#     # Without loss of generality, let us assume the local coordinate system is at the midpoint of the path and local x is along the path
#     # We artibrarily choose y and z accordingly.
#     # NOTE: Might have to keepdims to the norms
#     midpoints = 0.5 * (path[1:] + path[:-1])        # N x 3
#     lengths = torch.linalg.norm(path[1:] - path[:-1], dim=1)        # N
#     lengths_x = lengths + self.rs
#     local_x = (path[1:] - path[:-1]) / torch.linalg.norm(path[1:] - path[:-1], dim=-1, keepdim=True)      # This is the pointing direction of the path (N x 3)
#     local_y = torch.cross(self.up, local_x)   # This is the direction perpendicular to the path and the z-axis
#     local_z = torch.cross(local_x, local_y)    # This is the direction perpendicular to the path and the y-axis

#     rotation_matrix = torch.stack([local_x, local_y, local_z], dim=-1)    # This is the local x,y,z to world frame rotation (N x 3 x 3)

#     # These vectors form the normal of the hyperplanes. We simply need to find their intercepts. 
#     # We are basically trying to find a_i.T * (x_0 + l_i * a_i) = b_i = a_i.T * x_0 + l_i (since a_i.T * a_i = 1)
#     xyz_lengths = torch.stack([lengths_x, lengths, lengths], dim=-1)
#     intercepts_pos = torch.bmm(midpoints[..., None, :], rotation_matrix).squeeze() + torch.stack([lengths_x, lengths, lengths], dim=-1)    # N x 3
#     intercepts_neg = -(intercepts_pos) + 2*xyz_lengths    # N x 3

#     # Represent as x.T A <= b
#     A = torch.cat([rotation_matrix, -rotation_matrix], dim=-1)    # N x 3 x 6
#     b = torch.cat([intercepts_pos, intercepts_neg], dim=-1)    # N x 6

#     # Now return points that are only within the bounding box
#     xA = torch.einsum('mk, nkl -> nml', point_cloud, A)    # N x M x 6

#     # In order for a pt to be within the bounding box, it must satisfy xA <= b, i.e. xA-b <= 0
#     mask = (xA - b[:, None, :] <= 0.)    # N x M x 6
#     keep_mask = mask.all(dim=-1)    # N x M      # For every path segment N, keep_pts is a boolean of every point in M if it is in the bounding box or not

#     # Returning just the points is not parallelizable because there can be different number of points for each bounding box, so we need to loop, selecting the points
#     # where the mask is True.
#     keep_points = []
#     for keep_per_n in keep_mask:
#         keep_points.append(point_cloud[keep_per_n])

#     # saves bounding box constraints for polytope definition
#     self.box_As = torch.transpose(A[0], 0, 1)
#     self.box_bs = b[0]

#     # Return both the bounding box half-space representation and the relevant points
#     output = {'A': A, 'b': b, 'midpoints': midpoints, 'keep_points': keep_points}
#     return output

class GSplatCollisionSet():
    def __init__(self, gsplat, vmax, amax, radius, device):
        self.gsplat = gsplat
        self.vmax = vmax
        self.amax = amax
        self.radius = radius        # robot radius
        self.device = device

        self.rs = vmax**2 / (2*amax) + self.radius    # Safety radius

        self.means = self.gsplat.means
        self.rots = quaternion_to_rotation_matrix(self.gsplat.rots)
        self.scales = self.gsplat.scales
        self.gaussian_ids = torch.arange(self.means.shape[0], device=self.device)
     
    # NOTE: THIS COMPUTES THE COLLISION SET FOR ALL LINE SEGMENTS IN THE PATH!!!
    def compute_set(self, path, save_path=None):

        # Compute the bounding box
        A, b = compute_bounding_box(path, self.rs)

        # Then compute the Gaussians that lie within the bounding box
        # NOTE: We want to consider if we only create one bounding box and collision set for ONE line segment. Computationally
        # it probably makes no difference.

        # TODO: We could definitely batch this. However, there's not a big need for this since we are only considering one line segment at a time.
        gaussian_collision_set = []
        
        collision_set_ids = []
        for i, (A0, b0) in enumerate(zip(A, b)):

            # # This comparison tensor will be N (num of Gaussians) x 6 (num of hyperplanes)
            # a_times_mu = self.means @ A0.T
            # numerator = b0[None, :] - a_times_mu        # N x 6

            # a_times_R = self.rots.transpose(1, 2) @ A0.T # N x 3 x 6 
            # denominator = torch.linalg.norm( a_times_R * self.scales[..., None] , dim=1)    # N x 6

            # distance = -numerator / denominator

            # # A gaussian must satisfy the metric for all 6 hyperplanes
            # keep_gaussian = (distance <= 1.).all(dim=-1)      # mask of length N

            keep_gaussian = ellipsoid_halfspace_intersection(self.means, self.rots, self.scales, A0, b0)

            # Save the indices of the gaussians that are within the bounding box
            data = {
                'gaussian_ids': self.gaussian_ids[keep_gaussian],
                'A_bb': A0,
                'b_bb': b0,
                'b_bb_shrunk': b0 - self.radius,
                'path': path[i:i+2],
                'midpoint': 0.5 * (path[i] + path[i+1]),
                'means': self.means[keep_gaussian],
                'rots': self.rots[keep_gaussian],
                'scales': self.scales[keep_gaussian],
                'id': i
            }

            gaussian_collision_set.append(data)

            collision_set_ids.append(self.gaussian_ids[keep_gaussian])

        # We want to save the bounding box as a viewable mesh using Open3d
        if save_path is not None:
            self.save_bounding_box(path, A, b, save_path + '_bounding_box.obj')
            self.save_collision_set(collision_set_ids, save_path + '_collision_set.obj')

        # Return a dictionary of the Gaussian collision set and the shrunk bounding box constraints representing only the free space for the robot centroid.
        return gaussian_collision_set
    
    # NOTE: THIS COMPUTES THE COLLISION SET FOR ONE LINE SEGMENT IN THE PATH!!!
    def compute_set_one_step(self, segment):

        # Compute the bounding box
        A, b = compute_bounding_box(segment, self.rs)
        A = A.squeeze()
        b = b.squeeze()

        # tnow = time.time()
        # torch.cuda.synchronize()
        
        # This comparison tensor will be N (num of Gaussians) x 6 (num of hyperplanes)
        a_times_mu = self.means @ A.T
        numerator = b[None, :] - a_times_mu        # N x 6

        a_times_R = self.rots.transpose(1, 2) @ A.T # N x 3 x 6 
        denominator = torch.linalg.norm( a_times_R * self.scales[..., None] , dim=1)    # N x 6

        distance = -numerator / denominator

        # A gaussian must satisfy the metric for all 6 hyperplanes
        keep_gaussian = (distance <= 1.).all(dim=-1)      # mask of length N

        # torch.cuda.synchronize()
        # print('Time to compute gsplat check:', time.time() - tnow)

        # Save the indices of the gaussians that are within the bounding box
        output = {
            'gaussian_ids': self.gaussian_ids[keep_gaussian],
            'A_bb': A,
            'b_bb': b,
            'b_bb_shrunk': b - self.radius,
            'path': segment,
            'midpoint': 0.5*(segment[0] + segment[1]),
            'means': self.means[keep_gaussian],
            'rots': self.rots[keep_gaussian],
            'scales': self.scales[keep_gaussian],
        }

        # Return a dictionary of the Gaussian collision set and the shrunk bounding box constraints representing only the free space for the robot centroid.
        return output
    
    def save_collision_set(self, gaussian_ids, save_path):

        # This saves the whole collision set as one mesh
        unique_ids = torch.unique(torch.cat(gaussian_ids, dim=0))

        means = self.gsplat.means[unique_ids]
        rots = quaternion_to_rotation_matrix(self.gsplat.rots[unique_ids])
        scales = self.gsplat.scales[unique_ids]
        colors = self.gsplat.colors[unique_ids]
    
        scene = create_gs_mesh(means.cpu().numpy(), rots.cpu().numpy(), scales.cpu().numpy(), colors.cpu().numpy(), res=4, transform=None, scale=None)
        success = o3d.io.write_triangle_mesh(save_path, scene, print_progress=True)
        return success

    def save_bounding_box(self, path, A, b, save_path):
        # Initialize mesh object
        mesh = o3d.geometry.TriangleMesh()

        midpoints = 0.5 * (path[1:] + path[:-1])
        for A0, b0, mid_pt in zip(A, b, midpoints):
            # Transfer all tensors to numpy
            A0 = A0.cpu().numpy()
            b0 = b0.cpu().numpy()
            mid_pt = mid_pt.cpu().numpy()

            halfspaces = np.concatenate([A0, -b0[..., None]], axis=-1)
            hs = scipy.spatial.HalfspaceIntersection(halfspaces, mid_pt, incremental=False, qhull_options=None)
            qhull_pts = hs.intersections

            pcd_object = o3d.geometry.PointCloud()
            pcd_object.points = o3d.utility.Vector3dVector(qhull_pts)
            bb_mesh, qhull_indices = pcd_object.compute_convex_hull()
            mesh += bb_mesh
        
        success = o3d.io.write_triangle_mesh(save_path, mesh, print_progress=True)

        return success