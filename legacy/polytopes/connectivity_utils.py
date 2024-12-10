import numpy as np
import polytope
from cvxopt import matrix, solvers
from polytopes.polytopes_utils import polytopes_to_matrix

# This file is mostly deprecated since we can guarantee that the polytope corridor is connected.
# It is here only for completeness and sanity checks.

def test_connected_union_polys(As, bs):
    # polys: list of polytopes

    # Test if the union of polytopes is a connected region

    As_1 = As[:-1]
    As_2 = As[1:]

    bs_1 = bs[:-1]
    bs_2 = bs[1:]

    adjacents = []
    for A1, A2, b1, b2 in zip(As_1, As_2, bs_1, bs_2):
        adjacent = polytope.is_adjacent(polytope.Polytope(A1, b1), polytope.Polytope(A2, b2))  # returns boolean
        adjacents.append(adjacent)

    adjacents = np.array(adjacents)
    # print(adjacents)
    return np.all(adjacents)

def test_connected_union_polys_approximate(As, bs, points):
    As_1 = As[:-1]
    As_2 = As[1:]

    bs_1 = bs[:-1]
    bs_2 = bs[1:]

    pts_1 = points[:-1]
    pts_2 = points[1:]

    adjacents = []
    for A1, A2, b1, b2, p1, p2 in zip(As_1, As_2, bs_1, bs_2, pts_1, pts_2):
        # Use only points that are interior
        keep1 = np.all((A1 @ p1.T - b1[..., None]) <= 1e-2, axis=0)
        keep2 = np.all((A2 @ p2.T - b2[..., None]) <= 1e-2, axis=0)

        keep_pts1 = p1[keep1]
        keep_pts2 = p2[keep2]

        # Test if valid points in 1 satisfy polytope 2
        poly1_in_2 = np.any(np.all(A2 @ keep_pts1.T - b2[..., None] <= 1e-2, axis=0))
        poly2_in_1 = np.any(np.all(A1 @ keep_pts2.T - b1[..., None] <= 1e-2, axis=0))

        # print(poly2_in_1)
        adjacents.append(poly1_in_2 or poly2_in_1)

    adjacents = np.array(adjacents)
    # print(adjacents)

    return np.all(adjacents)

def test_connected_union_polys_LP(As, bs):
    N = len(As)-1
    dim = As[0].shape[-1]

    As1 = As[:-1]
    As2 = As[1:]
    bs1 = bs[:-1]
    bs2 = bs[1:]

    A_intersection = [np.concatenate([A1, A2], axis=0) for (A1, A2) in zip(As1, As2)]
    b_intersection = [np.concatenate([b1, b2], axis=0) for (b1, b2) in zip(bs1, bs2)]
    A, b = polytopes_to_matrix(A_intersection, b_intersection)

    c = matrix(np.zeros(A.shape[-1]))
    A = matrix(A)
    b = matrix(b)
    sol=solvers.lp(c,A,b, solver='glpk')

    # test_pts = cvx.Variable((N, dim))

    # obj = cvx.Minimize(0.)

    # constraints = [A @ cvx.reshape(test_pts, N*dim, order='C') <= b]

    # prob = cvx.Problem(obj, constraints)
    # prob.solve(solver='GLPK')
    # prob.solve()

    # print(sol['x'])
    if sol['x'] is not None:
        return True
    else:
        return False