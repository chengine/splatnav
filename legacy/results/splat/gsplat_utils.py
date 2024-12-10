import json
import torch
from pathlib import Path
import open3d as o3d
import time

from ellipsoids.mesh_utils import create_gs_mesh
from ellipsoids.covariance_utils import quaternion_to_rotation_matrix
from ellipsoids.covariance_utils import compute_cov
from splat.distances import distance_point_ellipsoid, batch_point_distance, batch_squared_point_distance, batch_mahalanobis_distance
from ns_utils.nerfstudio_utils import GaussianSplat, SH2RGB

class GSplatLoader():
    def __init__(self, gsplat_location, device):
        self.device = device

        if isinstance(gsplat_location, str):
            self.load_gsplat_from_json(gsplat_location)
        elif isinstance(gsplat_location, Path):
            self.load_gsplat_from_nerfstudio(gsplat_location)
        else:
            raise ValueError('GSplat file must be either a .json or .yml file.')
        
    def load_gsplat_from_nerfstudio(self, gsplat_location):

        self.splat = GaussianSplat(gsplat_location,
                    test_mode= "inference",
                    dataset_mode = 'train',
                    device = self.device)

        self.means = self.splat.pipeline.model.means.detach().clone()
        self.rots = self.splat.pipeline.model.quats.detach().clone()
        self.scales = self.splat.pipeline.model.scales.detach().clone()
        self.scales = torch.exp(self.scales)

        self.covs_inv = compute_cov(self.rots, 1 / self.scales)
        self.covs = compute_cov(self.rots, self.scales)

        self.colors = SH2RGB(self.splat.pipeline.model.features_dc.detach().clone())

        print(f'There are {self.means.shape[0]} Gaussians in the GSplat model')

        return

    def load_gsplat_from_json(self, gsplat_location):

        with open(gsplat_location, 'r') as f:
            data = json.load(f)
        
        keys = ['means', 'rotations', 'colors', 'opacities', 'scalings']
        tensors = {}

        # Measure time for loading tensors
        start_time = time.time()
        for key in keys:
            tensors[key] = torch.tensor(data[key]).to(dtype=torch.float32, device=self.device)
        print(f"Loading tensors took {time.time() - start_time:.4f} seconds")
        
        # Measure time for setting attributes
        start_time = time.time()
        self.means = tensors['means']
        self.rots = tensors['rotations']
        self.colors = tensors['colors']
        self.opacities = tensors['opacities']
        self.scales = tensors['scalings']

        print(f"Setting attributes took {time.time() - start_time:.4f} seconds")

        # Print tensor sizes
        print(f"Opacities tensor size: {self.opacities.size()}")
        print(f"Scales tensor size: {self.scales.size()}")

        # Measure time for normalization
        self.opacities = torch.sigmoid(self.opacities)
        self.scales = torch.exp(self.scales)

        # Measure time for computing Sigma inverse
        self.covs_inv = compute_cov(self.rots, 1. / self.scales)
        self.covs = compute_cov(self.rots, self.scales)

        return 

    def save_mesh(self, filepath):
        scene = create_gs_mesh(self.means.cpu().numpy(), quaternion_to_rotation_matrix(self.rots).cpu().numpy(), self.scales.cpu().numpy(), self.colors.cpu().numpy(), res=4, transform=None, scale=None)
        success = o3d.io.write_triangle_mesh(filepath, scene, print_progress=True)

        return success

    #NOTE: Need to provide the robot radius OR the robot R and S matrices
    def query_distance(self, x, distance_type = None, radius=0., R_robot=None, S_robot=None, epsilon=0.):
        # Queries varieties of distance from x to the GSplat.

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if distance_type == 'ball-to-ball':
            ball_radius = torch.max(self.scales, dim=-1)[0]
            dist, grad, hess = batch_point_distance(x[..., :3].squeeze(), self.means)

            h = dist - (ball_radius + radius + epsilon)
            grad_h = grad
            hess_h = hess

            info = None

        elif distance_type == 'ball-to-ball-squared': 
            ball_radius = torch.max(self.scales, dim=-1)[0]
            squared_dist, grad, hess = batch_squared_point_distance(x[..., :3].squeeze(), self.means)

            h = squared_dist - (ball_radius + radius + epsilon)**2
            grad_h = grad
            hess_h = hess

            info = None

        elif distance_type == 'ball-to-pt-squared': 
            squared_dist, grad, hess = batch_squared_point_distance(x[..., :3].squeeze(), self.means)

            h = squared_dist - (radius + epsilon)**2
            grad_h = grad
            hess_h = hess

            info = None

        elif distance_type == 'mahalanobis':
            maha_dist, grad, hess = batch_mahalanobis_distance(x[..., :3].squeeze(), self.means, self.covs_inv)

            h = maha_dist - 1.
            grad_h = grad
            hess_h = hess

            info = None

        elif (distance_type is None) or (distance_type == 'ball-to-ellipsoid'):
            # Queries the min Euclidian distance from point to ellipsoid
            # Rotate point into the ellipsoid frame

            # Convert rotations from quaternions to rotation matrices
            rots = quaternion_to_rotation_matrix(self.rots)

            # Sort the scales in descending order as required by the solver
            sorted_output = torch.sort(self.scales, dim=-1, descending=True)
            sorted_scales, sorted_inds = sorted_output[0], sorted_output[1]
      
            # NOTE:!!! IMPORTANT!!! When we sort, we need to change the rotation matrices accordingly
            rots = torch.gather(rots, 2, sorted_inds[..., None, :].expand_as(rots))

            # Translate robot w.r.t ellipsoid mean, then rotate point into ellipsoid aligned frame
            x_local_frame = torch.bmm( torch.transpose(rots, 1, 2) , (x[..., :3] - self.means).unsqueeze(-1) ).squeeze() + 1e-8

            # The solver requires the point to be in the first octant. Calculate the sign of the point and flip the point.
            flip = torch.sign(x_local_frame)
            x_local_frame = torch.abs(x_local_frame)

            # solve for the distance in the local frame
            dist, _, hess, yhat = distance_point_ellipsoid(sorted_scales + 1e-8, x_local_frame)

            # flip, rotate, and translate the closest point back to the global frame
            y = torch.bmm(rots, (flip * yhat).unsqueeze(-1)).squeeze(-1) + self.means

            # Calculate cbf 
            phi = torch.sign( torch.sum( (1./ sorted_scales)**2 * (x_local_frame**2) , dim=-1) - 1.)

            h = phi * dist - (radius + epsilon)**2

            # Compute gradient in world frame. 
            grad_h = 2 * phi[..., None] * (x[..., :3] - y)

            # Mutliple Hessian by phi
            hess_h = phi[..., None, None] * hess

            info = {'y': y, 'phi': phi}

        else:
            raise ValueError('Distance type not recognized. Please provide a valid distance type.')

        return h, grad_h, hess_h, info
    
# The purpose of this loader is to run toy examples and for figures.
class DummyGSplatLoader(GSplatLoader):
    def __init__(self, device):
        self.device = device

    def initialize_attributes(self, means, rots, scales, colors=None):
        self.means = means.to(self.device)
        self.rots = rots.to(self.device)
        self.scales = scales.to(self.device)

        self.cov_inv = compute_cov(self.rots, 1 / self.scales)
        self.covs = compute_cov(self.rots, self.scales)

        if colors is not None:
            self.colors = colors.to(self.device)
        else:
            self.colors = 0.5*torch.ones(means.shape[0], 3).to(self.device)

        return