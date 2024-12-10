import numpy as np
import scipy
import clarabel
from scipy import sparse
from scipy.optimize import linprog
import torch

def compute_path_in_polytope(A, b, path):
    # Path is a list of points. Sequential points indicate line segments.
    # We check each point in path to see if it satisfies Ax <= b. The first point that does not satisfy
    # the check is the point where the path exits the polytope.
    # A: (n_constraints, n_dim)
    # b: (n_constraints,)

    # Outputs the last index of path that satisfies the polytope

    criterion = torch.all( (A @ path.T - b[:, None]) <= 0., dim=0 )
    idx = torch.arange(len(path))[~criterion]

    if len(idx) == 0:
        # All points in path satisfy polytope
        return None
    else:
        first_exit = idx[0]
        return first_exit - 1

def compute_segment_in_polytope(A, b, segment):
    # Segment is a list of two points. We check if the line segment between the two points
    # satisfies Ax <= b. If it does, we return True. If not, we return False.
    # A: (n_constraints, n_dim)
    # b: (n_constraints,)

    # Outputs whether the segment satisfies the polytope
    criterion = torch.all( (A @ segment.T - b[:, None]) <= 0.)

    return criterion

def h_rep_minimal(A, b, pt):
    halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
    hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)

    qhull_pts = hs.intersections
    # NOTE: It's possible that hs.dual_vertices errors out due to it not being to handle large number of facets. In that case, use the following code:
    try:
        minimal_Ab = halfspaces[hs.dual_vertices]
    except:
        convex_hull = scipy.spatial.ConvexHull(qhull_pts, incremental=False, qhull_options=None)
        minimal_Ab = convex_hull.equations
    

    minimal_A = minimal_Ab[:, :-1]
    minimal_b = -minimal_Ab[:, -1]

    return minimal_A, minimal_b, qhull_pts

def find_interior(A, b):
    # by way of Chebyshev center
    norm_vector = np.reshape(np.linalg.norm(A, axis=1),(A.shape[0], 1))
    c = np.zeros(A.shape[1]+1)
    c[-1] = -1
    A = np.hstack((A, norm_vector))

    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))

    return res.x[:-1]

def polytopes_to_matrix(As, bs):
    A_sparse = scipy.linalg.block_diag(*As)
    b_sparse = np.concatenate(bs)

    return A_sparse, b_sparse

def check_and_project(A, b, point):
    # Check if Ax <= b

    criteria = A @ point - b
    is_valid = np.all(criteria < 0)

    if is_valid:
        return point, True
    else:
        # project point to nearest facet

        # Setup workspace
        n_constraints = A.shape[0]
        P = sparse.eye(3, format='csc')
        A = sparse.csc_matrix(A)    
        q = -2*point

        settings = clarabel.DefaultSettings()
        settings.verbose = False

        solver = clarabel.DefaultSolver(P, q, A, b, [clarabel.NonnegativeConeT(n_constraints)], settings)
        sol = solver.solve()

        # Check solver status
        if str(sol.status) != 'Solved':
            print(f"Solver status: {sol.status}")
            print(f"Number of iterations: {sol.iterations}")
            print('Clarabel did not solve the problem!')
            solver_success = False
            solution = None
        else:
            solver_success = True
            solution = sol.x

        print('Closest point:', solution, 'Success?', solver_success)

        return solution, solver_success