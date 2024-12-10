import dijkstra3d
import numpy as np

# Perform coarse path planning
def astar3D(field, source, target, feasible):

    #Performs A* 
    inds = dijkstra3d.binary_dijkstra(field, source, target, connectivity=6, background_color=1)
    inds = inds.astype(np.int32)

    # print(f'Returning path of length {len(inds)}')
    traj = feasible[inds[:, 0], inds[:, 1], inds[:, 2]]

    return traj, inds

def chunk_path(path, num_sec):
    # Path: N x 3
    # num_sec: number of sections to split path into

    # Returns path M x 3

    N_path = len(path)-1

    path_no_end = path[:-1]
    path_no_start = path[1:]

    new_path = []
    cat_path = None

    # 3 cases
    if N_path == num_sec:
        # No need to do anything. 
        # for (start, end) in zip(path_no_end, path_no_start):
        #     sub_path = np.concatenate([start, end], axis=0)

        #     new_path.append(sub_path.reshape(-1, 3))

        cat_path = path
        print('Request number of sections is equal to length of shortest path. Returning same path.')
        
    elif N_path < num_sec:
        # Need to interpolate points to match num_sections
        indices = np.arange(num_sec)
        split_indices = np.array_split(indices, N_path)

        sub_paths = []
        for (start, end, sample_ind) in zip(path_no_end, path_no_start, split_indices):
            t = np.linspace(0., 1., len(sample_ind), endpoint=False)
            sub_path = np.reshape(start, (1, 3)) + t[:, None] * np.reshape(end - start, (1, 3))

            # Convert sub_path to list of paths
            # sub_path_ = [np.concatenate([start_, end_], axis=0) for (start_, end_) in zip(sub_path[:-1], sub_path[1:])]

            # new_path.extend(sub_path_)
            sub_paths.append(sub_path)

        cat_path = np.concatenate(sub_paths, axis=0)
        cat_path = np.concatenate([cat_path, end[None]], axis=0)
        print('Request number of sections is greater than length of shortest path. Interpolating path.')

    elif N_path > num_sec:
        # Undesirable condition, as the input path is already the shortest it can be. Raise an error asking
        # to bump up the number of sections. Return the path.

        # for (start, end) in zip(path_no_end, path_no_start):
        #     sub_path = np.concatenate([start, end], axis=0)

        #     new_path.append(sub_path.reshape(-1, 3))

        cat_path = path
        print('Requested number of sections is less than the fewest amount of sections. Returning the path with fewest sections.')
    
    else:
        raise ValueError('Chunk Path function has error.')

    return cat_path

def straight_splits(path, traj):
    assert path.shape[0] == traj.shape[0]

    sorted_traj = [traj[0]]

    running_path = [path[0]]
    running_traj = [traj[0]]
    run_direction = path[1] - running_path[0]
    running_path.append(path[1])
    running_traj.append(traj[1])

    for it, (index, pt) in enumerate(zip(path[2:], traj[2:])):
        # Check if the current point is in the same direction
        current_dir = index - running_path[-1]
        metric = np.sum((current_dir - run_direction)**2)
        if metric > 0:
            # Direction has changed
            sorted_traj.append(np.array(running_traj)[-1])

            running_path = [running_path[-1], index]
            running_traj = [running_traj[-1], pt]
            run_direction = current_dir
        else:
            # Direction has not changed
            running_path.append(index)
            running_traj.append(pt)

            if it == len(path[2:]) - 1:
                sorted_traj.append(pt)

    straight_traj = np.stack(sorted_traj, axis=0)

    return straight_traj


### SOME PARTS OF THIS CLASS IS CANABALIZED FROM THE GSPLAT GRID CLASS ###
class Astar():
    # Uses A* as path initialization
    def __init__(self, grid_occupied, grid_points) -> None:
        self.grid_occupied = grid_occupied      # Binary Field (Nx x Ny x Nz)
        self.grid_points = grid_points          # Points corresponding to the field (Nx x Ny x Nz)

        self.cell_sizes = self.grid_points[1, 1, 1] - self.grid_points[0, 0, 0]

    def create_path(self, x0, xf, num_sec=10):
        source = self.get_indices(x0)   # Find nearest grid point and find its index
        target = self.get_indices(xf)

        source_occupied = self.grid_occupied[source[0], source[1], source[2]]
        target_occupied = self.grid_occupied[target[0], target[1], target[2]]

        if target_occupied:
            raise ValueError('Target is in occupied voxel. Please choose another end point.')

        if source_occupied:
            raise ValueError('Source is in occupied voxel. Please choose another starting point.')
        
        path3d, indices = astar3D(self.grid_occupied, source, target, self.grid_points)

        try:
            assert len(path3d) > 0
        except:
            raise AssertionError('Could not find a feasible initialize path. Please change the initial/final positions to not be in collision.')

        # straight_path = straight_splits(indices, path3d)

        # path = self.fewest_straight_lines(path3d)       # Reduces the length of the A* path

        # path = chunk_path(path, num_sec)

        # # Remove first and last points, replace with original values
        # path = path[1:-1]
        # path = np.concatenate([x0.reshape(1, -1), path, xf.reshape(1, -1)])

        # straight_path = straight_path[1:-1]
        # straight_path = np.concatenate([x0.reshape(1, -1), straight_path, xf.reshape(1, -1)])

        # return path, straight_path, path3d

        return path3d

    def get_indices(self, point):
        min_bound = self.grid_points[0, 0, 0]

        transformed_pt = point - min_bound

        indices = np.round(transformed_pt / self.cell_sizes).astype(np.uint32)

        return_indices = indices.copy()
        # If querying points outside of the bounds, project to the nearest side
        for i, ind in enumerate(indices):
            if ind < 0.:
                return_indices[i] = 0

                print('Point is outside of minimum bounds. Projecting to nearest side. This may cause unintended behavior.')

            elif ind > self.grid_occupied.shape[i]:
                return_indices[i] = self.grid_occupied.shape[i]

                print('Point is outside of maximum bounds. Projecting to nearest side. This may cause unintended behavior.')

        return_indices = return_indices.astype(np.uint32)

        return return_indices

    # Process path so you have fewest number of straight line paths as possible
    def fewest_straight_lines(self, path, num_test_pts=100):
        root_index = 0
        candidate_index = 1

        root = path[root_index]
        candidate = path[candidate_index]
        direction = (candidate - root) / np.linalg.norm(candidate-root, keepdims=True)

        new_path = [root]
        while True:
            candidate_index += 1
            candidate = path[candidate_index]

            new_direction = (candidate - root) / np.linalg.norm(candidate-root, keepdims=True)

            if np.linalg.norm(new_direction - direction) < 1e-2:
                if candidate_index < len(path) - 1:
                    # Going in a straight line, proceed to next iteration
                    continue
                else:
                    # Corresponds to straight line path to the goal
                    new_path.append(path[-1])
                    break

            direction = new_direction

            # Sample from the root to new_candidate 
            t = np.linspace(0., 1., num_test_pts, endpoint=False)
            test_points = np.reshape(root, (1, 3)) + t[..., None] * np.reshape(candidate - root, (1, 3))

            test_pts_is_collide = []
            for test_pt in test_points:
                index = self.get_indices(test_pt)
                test_pts_is_collide.append(self.grid_occupied[index[..., 0], index[..., 1], index[..., 2]])

            if np.any(np.array(test_pts_is_collide)):
                # Some point is in collision, then we make the previous candidate the root
                root = path[candidate_index - 1]
                candidate_index -= 1

                new_path.append(root)

                # Edge case: If second to last pt is clear but last pt is not.
                if candidate_index == len(path) - 2:
                    new_path.append(path[-1])
                    break

            # Termination condition if last segment is clear
            if candidate_index == len(path) - 1:
                new_path.append(path[-1])
                break
