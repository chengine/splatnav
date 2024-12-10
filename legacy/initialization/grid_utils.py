import torch
import open3d as o3d
import unfoldNd

from initialization.astar_utils import astar3D

class GSplatVoxel():
    def __init__(self, gsplat, lower_bound, upper_bound, resolution, radius, device):
        self.gsplat = gsplat
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.radius = radius            # Robot radius
        self.device = device

        self.resolution = resolution    # Can be a vector (discretization per dimension) or a scalar
        if isinstance(self.resolution, int):
            self.resolution = torch.tensor([self.resolution, self.resolution, self.resolution], device=self.device)
        
        # Define max and minimum indices
        self.min_index = torch.zeros(3, dtype=int, device=self.device)
        self.max_index = torch.tensor(self.resolution, dtype=int, device=self.device) - 1
        self.cell_sizes = (upper_bound - lower_bound) / self.resolution

        self.grid_centers = None
        self.non_navigable_grid = None

        with torch.no_grad():
            self.generate_kernel()
            self.create_navigable_grid()

    # We employ a subdividing strategy to populate the voxel grid in order to avoid
    # having to check every point/index in the grid with all bounding boxes in the scene

    # NOTE: Might be useful to visualize this navigable grid to see if it is correct and for paper.
    def create_navigable_grid(self, unfold=True, chunk=100000):

        # Create a grid
        self.non_navigable_grid = torch.zeros( (self.resolution[0], self.resolution[1], self.resolution[2]), dtype=bool, device=self.device)

        # ...along with its corresponding grid centers
        X, Y, Z = torch.meshgrid(
            torch.linspace(self.lower_bound[0] + self.cell_sizes[0]/2, self.upper_bound[0] - self.cell_sizes[0]/2, self.resolution[0], device=self.device),
            torch.linspace(self.lower_bound[1] + self.cell_sizes[1]/2, self.upper_bound[1] - self.cell_sizes[1]/2, self.resolution[1], device=self.device),
            torch.linspace(self.lower_bound[2] + self.cell_sizes[2]/2, self.upper_bound[2] - self.cell_sizes[2]/2, self.resolution[2], device=self.device)
        )
        self.grid_centers = torch.stack([X, Y, Z], dim=-1)

        if unfold:
            # Compute the bounding box properties, accounting for robot radius inflation ### TODO: WIP, NEED TO ACCOUNT FOR ROBOT RADIUS IN UNFOLD
            bb_mins = self.gsplat.means - torch.sqrt(torch.diagonal(self.gsplat.covs, dim1=1, dim2=2))
            bb_maxs = self.gsplat.means + torch.sqrt(torch.diagonal(self.gsplat.covs, dim1=1, dim2=2))

            # A majority of the Gaussians are extremely small, smaller than the discretization size of the grid. 
            # TODO:???

        else:
            bb_mins = self.gsplat.means - torch.sqrt(torch.diagonal(self.gsplat.covs, dim1=1, dim2=2)) - self.radius
            bb_maxs = self.gsplat.means + torch.sqrt(torch.diagonal(self.gsplat.covs, dim1=1, dim2=2)) + self.radius

        # Mask out ellipsoids that have bounding boxes outside of grid bounds
        # reference: https://math.stackexchange.com/questions/2651710/simplest-way-to-determine-if-two-3d-boxes-intersect
        # If any of the conditions are true, the intervals overlap
        condition1 = (self.lower_bound[None, :] - bb_mins <= 0.) & (self.upper_bound[None, :] - bb_mins >= 0.)
        condition2 = (self.lower_bound[None, :] - bb_maxs <= 0.) & (self.upper_bound[None, :] - bb_maxs >= 0.)
        condition3 = (self.lower_bound[None, :] - bb_mins >= 0.) & (self.lower_bound[None, :] - bb_maxs <= 0.)
        condition4 = (self.upper_bound[None, :] - bb_mins >= 0.) & (self.upper_bound[None, :] - bb_maxs <= 0.)

        overlap = condition1 | condition2 | condition3 | condition4

        # there must be overlap in all 3 dimensions in order for the boxes to overlap
        overlap = torch.all(overlap, dim=-1)

        bb_mins = bb_mins[overlap]
        bb_maxs = bb_maxs[overlap]

        # The vertices are min, max, min + x, min + y, min + z, min + xy, min + xz, min + yz
        axes_mask = [torch.tensor([0, 0, 0], device=self.device),
                torch.tensor([1, 0, 0], device=self.device),
                torch.tensor([0, 1, 0], device=self.device),
                torch.tensor([0, 0, 1], device=self.device),
                torch.tensor([1, 1, 0], device=self.device),
                torch.tensor([1, 0, 1], device=self.device),
                torch.tensor([0, 1, 1], device=self.device),
                torch.tensor([1, 1, 1], device=self.device)]

        ### NEWER CODE ### Is too slow...
        # bb_index_size = (bb_maxs - bb_mins) / self.cell_sizes

        # # Clamp the bounding box indices to the grid resolution
        # bb_index_size = torch.clamp(bb_index_size, min=torch.zeros(3, device=self.device)[None, :], max=self.resolution[None, :].float())

        # max_sizes = torch.round(torch.max(bb_index_size, dim=0).values).to(torch.int64)        # This forms our meshgrid

        # print(max_sizes)
        # X_bb, Y_bb, Z_bb = torch.meshgrid(
        #     torch.linspace(0., 1., max_sizes[0].item()+1, device=self.device),
        #     torch.linspace(0., 1., max_sizes[1].item()+1, device=self.device),
        #     torch.linspace(0., 1., max_sizes[2].item()+1, device=self.device)
        # )       # This forms the grid between bb_min and bb_max

        # bb_indices_all = torch.stack([X_bb, Y_bb, Z_bb], dim=-1).reshape(-1, 3)

        # # For every bounding box, we need to produce this meshgrid
        # bb_index_size_list = torch.split(bb_index_size, 1000)
        # bb_mins_list = torch.split(bb_mins, 1000)

        # counter = 0
        # for bb_index_size_, bb_mins_ in zip(bb_index_size_list, bb_mins_list):
        #     bb_indices = bb_index_size_[:, None, :] * bb_indices_all[None, :, :] + bb_mins_[:, None, :]
        #     print(bb_indices.shape, bb_indices.max(), bb_indices.min())
        #     bb_indices = torch.round( bb_indices.reshape(-1, 3) ).to(torch.int64)

        #     # Kill everything that is outside of the grid bounds
        #     in_grid = ( torch.all( (self.max_index - bb_indices) >= 0. , dim=-1) ) & ( torch.all( bb_indices >= 0. , dim=-1) )
        #     bb_indices = bb_indices[in_grid]

        #     # Might want to do a unique to prune out duplicates
        #     bb_indices = torch.unique_consecutive(bb_indices, dim=0)
        #     #print(bb_indices.shape, bb_indices.max(), bb_indices.min())
        #     self.non_navigable_grid[bb_indices[:,0], bb_indices[:,1], bb_indices[:,2]] = True

        #     print('Iteration:', counter)
        #     counter += 1

        ### OLDER CODE ###

        # Subdivide the bounding box until it fits into the grid resolution
        # lowers = []
        # uppers = []
        counter = 0
        while len(bb_mins) > 0:

            bb_min_list = torch.split(bb_mins, chunk)
            bb_max_list = torch.split(bb_maxs, chunk)

            bb_mins = []
            bb_maxs = []
            for bb_min, bb_max in zip(bb_min_list, bb_max_list):
                # Compute the size of the bounding box
                bb_size = bb_max - bb_min

                # Check if the bounding box fits into the grid resolution
                mask = torch.all(bb_size - self.cell_sizes[None, :] <= 0., dim=-1)      # If true, the bounding box fits into the grid resolution and we pop

                # TODO:??? Do we need to check if what we append is only one element?
                # lowers.append(bb_min[mask])
                # uppers.append(bb_max[mask])
                bb_min_keep = bb_min[mask]
                bb_size_keep = bb_size[mask]
                if len(bb_min_keep) > 0:
                    for axis_mask in axes_mask:
                        vertices = bb_min_keep + axis_mask[None, :] * bb_size_keep
                        # Bin the vertices into the navigable grid
                        shifted_vertices = vertices - (self.grid_centers[0, 0, 0])          # N x 3

                        vertex_index = torch.round( shifted_vertices / self.cell_sizes[None, :] ).to(dtype=int)

                        # Check if the vertex or subdivision is within the grid bounds. If not, ignore.
                        in_grid = ( torch.all( (self.max_index - vertex_index) >= 0. , dim=-1) ) & ( torch.all( vertex_index >= 0. , dim=-1) ) 
                        vertex_index = vertex_index[in_grid]
                        self.non_navigable_grid[vertex_index[:,0], vertex_index[:,1], vertex_index[:,2]] = True

                # If the bounding box does not fit within the subdivisions, divide the max dimension by 2

                # First, record the remaining bounding boxes
                bb_min = bb_min[~mask]
                bb_max = bb_max[~mask]
                bb_size = bb_size[~mask]

                # Calculate the ratio in order to know which dimension to divide by
                bb_ratio = bb_size / self.cell_sizes[None, :]
                max_dim = torch.argmax(bb_ratio, dim=-1)
                indices_to_change = torch.stack([torch.arange(max_dim.shape[0], device=self.device), max_dim], dim=-1)  # N x 2

                # Create a left and right partition (effectively doubling the size of the bounding box variables)
                bb_min_1 = bb_min.clone()
                bb_min_2 = bb_min.clone()
                bb_min_2[indices_to_change[:, 0], indices_to_change[:, 1]] += 0.5 * bb_size[indices_to_change[:, 0], indices_to_change[:, 1]]

                bb_max_1 = bb_max.clone()
                bb_max_1[indices_to_change[:, 0], indices_to_change[:, 1]] -= 0.5 * bb_size[indices_to_change[:, 0], indices_to_change[:, 1]]
                bb_max_2 = bb_max.clone()

                bb_min = torch.cat([bb_min_1, bb_min_2], dim=0)
                bb_max = torch.cat([bb_max_1, bb_max_2], dim=0)

                bb_mins.append(bb_min)
                bb_maxs.append(bb_max)

            bb_mins = torch.cat(bb_mins)
            bb_maxs = torch.cat(bb_maxs)

            print('Iteration:', counter)
            counter += 1

        # If unfold, we need to now do a maxpool3d with the grid
        if unfold:
            kernel_size = tuple(self.robot_mask.shape)
            padding = tuple( (torch.tensor(self.robot_mask.shape) - 1) // 2)
            lib_module = unfoldNd.UnfoldNd(
                kernel_size, dilation=1, padding=padding, stride=1
            )
            unfolded = lib_module(self.non_navigable_grid.to(dtype=torch.float32)[None, None]).squeeze()     # kernel_size x N x N x N
            unfolded = unfolded.to(dtype=bool)

            # Take the maxpool3d of the binary grid
            mask = self.robot_mask.reshape(-1)
            unfolded = unfolded[mask]       # mask_size x N x N x N
            non_navigable = torch.any(unfolded, dim=0).reshape(self.resolution[0], self.resolution[1], self.resolution[2])  
            self.non_navigable_grid = non_navigable

        return
    
    def generate_kernel(self):
        # Functions find the voxelized overapproximation of the Minkowski sum 

        rad_cell = torch.ceil(self.radius / self.cell_sizes - 0.5)
        lower_bound = -rad_cell
        upper_bound = rad_cell

        resolution = (2*upper_bound + 1).to(dtype=torch.uint8)

        # Forms the vertices of a unit cube
        axes_mask = torch.stack([torch.tensor([0, 0, 0], device=self.device),
                torch.tensor([1, 0, 0], device=self.device),
                torch.tensor([0, 1, 0], device=self.device),
                torch.tensor([0, 0, 1], device=self.device),
                torch.tensor([1, 1, 0], device=self.device),
                torch.tensor([1, 0, 1], device=self.device),
                torch.tensor([0, 1, 1], device=self.device),
                torch.tensor([1, 1, 1], device=self.device)], dim=0)    # 8 x 3

        # Form the lower grid vertex
        X, Y, Z = torch.meshgrid(
            torch.linspace(lower_bound[0], upper_bound[0], resolution[0], device=self.device),
            torch.linspace(lower_bound[1], upper_bound[1], resolution[1], device=self.device),
            torch.linspace(lower_bound[2], upper_bound[2], resolution[2], device=self.device)
        )

        # bottom vertex
        grid_vertices = torch.stack([X, Y, Z], dim=-1)
        grid_vertices = grid_vertices.reshape(-1, 3)

        vertices = (grid_vertices[:, None, :] + axes_mask[None, ...]) * self.cell_sizes[None, None, :]
        vertex_length = torch.linalg.norm(vertices, dim=-1)     # N x 8

        vertex_mask = torch.any(vertex_length <= self.radius, dim=-1)      # N

        # reshape back into grid
        vertex_mask = vertex_mask.reshape(resolution[0], resolution[1], resolution[2])

        self.robot_mask = vertex_mask

        return 

    def create_mesh(self, save_path=None):
        # Create a mesh from the navigable grid
        non_navigable_grid_centers = self.grid_centers[self.non_navigable_grid]
        non_navigable_grid_centers_flatten = non_navigable_grid_centers.view(-1, 3).cpu().numpy()

        scene = o3d.geometry.TriangleMesh()
        for cell_center in non_navigable_grid_centers_flatten:
            box = o3d.geometry.TriangleMesh.create_box(width=self.cell_sizes[0].cpu().numpy(), 
                                                        height=self.cell_sizes[1].cpu().numpy(), 
                                                        depth=self.cell_sizes[2].cpu().numpy())
            box = box.translate(cell_center, relative=False)
            scene += box

        if save_path is not None:
            o3d.io.write_triangle_mesh(save_path, scene, print_progress=True)

        return scene

    def create_path(self, x0, xf):
        source = self.get_indices(x0)   # Find nearest grid point and find its index
        target = self.get_indices(xf)

        source_occupied = self.non_navigable_grid[source[0], source[1], source[2]]
        target_occupied = self.non_navigable_grid[target[0], target[1], target[2]]

        # If either target or source is occupied, we do a nearest neighbor search to find the closest navigable point
        if target_occupied:
            print('Target is in occupied voxel. Projecting end point to closest unoccupied.')

            xf = self.find_closest_navigable(xf)
            target = self.get_indices(xf)

        if source_occupied:
            print('Source is in occupied voxel. Projecting starting point to closest unoccupied.')

            x0 = self.find_closest_navigable(x0)
            source = self.get_indices(x0)
        
        # Plans A*. Only accepts numpy objects. Returns numpy array N x 3.
        path3d, indices = astar3D(self.non_navigable_grid.cpu().numpy(), source.cpu().numpy(), target.cpu().numpy(), self.grid_centers.cpu().numpy())

        try:
            assert len(path3d) > 0
            #path3d = np.concatenate([x0.reshape(1, 3).cpu().numpy(), path3d, xf.reshape(1, 3).cpu().numpy()], axis=0)
        except:
            print('Could not find a feasible initialize path. Please change the initial/final positions to not be in collision.')
            path3d = None

        return path3d

    def get_indices(self, point):
        transformed_pt = point - self.grid_centers[0, 0, 0]

        indices = torch.round(transformed_pt / self.cell_sizes).to(dtype=int)

        # If querying points outside of the bounds, project to the nearest side
        for i, ind in enumerate(indices):
            if ind < 0.:
                indices[i] = 0

                print('Point is outside of minimum bounds. Projecting to nearest side. This may cause unintended behavior.')

            elif ind > self.non_navigable_grid.shape[i]-1:
                indices[i] = self.non_navigable_grid.shape[i]-1

                print('Point is outside of maximum bounds. Projecting to nearest side. This may cause unintended behavior.')

        return indices

    def find_closest_navigable(self, point):
        navigable_centers = self.grid_centers[~self.non_navigable_grid].reshape(-1, 3)
        dist = torch.norm(navigable_centers - point[None, :], dim=-1)
        min_point_idx = torch.argmin(dist)

        closest_navigable = navigable_centers[min_point_idx]

        return closest_navigable