import numpy as np
import scipy
from scipy.optimize import linprog

def h_rep_minimal(A, b, pt):
    halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
    hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)

    # NOTE: It's possible that hs.dual_vertices errors out due to it not being to handle large number of facets. In that case, use the following code:
    try:
        minimal_Ab = halfspaces[hs.dual_vertices]
    except:
        qhull_pts = hs.intersections
        convex_hull = scipy.spatial.ConvexHull(qhull_pts, incremental=False, qhull_options=None)
        minimal_Ab = convex_hull.equations

    minimal_A = minimal_Ab[:, :-1]
    minimal_b = -minimal_Ab[:, -1]

    return minimal_A, minimal_b

def find_interior(A, b):
    # by way of Chebyshev center
    norm_vector = np.reshape(np.linalg.norm(A, axis=1),(A.shape[0], 1))
    c = np.zeros(A.shape[1]+1)
    c[-1] = -1
    A = np.hstack((A, norm_vector))

    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))

    return res.x[:-1]