import numpy as np 
import torch
import clarabel
from scipy import sparse

def get_qp_matrices(bernstein_coefficients, polytopes, x0, xf):
    # NOTE: IMPORTANT! We assume x0 and xf are feasible to begin with! To maintain feasibility, we need
    # to check if x0 and xf are in their corresponding polytopes, and if not, project them into the polytope before
    # calling this function.

    device = bernstein_coefficients.device

    # num_t x num_splines x num_control_points x num_deriv : bernstein coefficients
    num_eval_points, num_splines, num_control_points, num_deriv = bernstein_coefficients.shape
    dim = x0.shape[-1]

    list_As = []
    list_bs = []
    for A, b in polytopes:
        list_As.extend([A]*num_control_points)
        list_bs.extend([b]*num_control_points)
    
    # Linear inequality constraints Ax <= b for the control points
    A = torch.block_diag(*list_As)
    b = torch.cat(list_bs, dim=0)

    # Linear equality constraints Cx = d for the control points
    first_coefficient = bernstein_coefficients[0].permute(0, 2, 1)       # num_splines x num_deriv x num_control_points
    last_coefficient = bernstein_coefficients[-1].permute(0, 2, 1)       # num_splines x num_deriv x num_control_points

    first_coefficient_ = first_coefficient.reshape(num_splines*num_deriv, num_control_points)
    last_coefficient_ = last_coefficient.reshape(num_splines*num_deriv, num_control_points)

    kron_eye = torch.eye(dim, device=device)

    kron1 = torch.kron(first_coefficient_.contiguous(), kron_eye)
    kron2 = torch.kron(last_coefficient_.contiguous(), kron_eye)

    split_by_spline1 = torch.split(kron1, num_deriv*dim, dim=0)     # num_splines x ...
    split_by_spline2 = torch.split(kron2, num_deriv*dim, dim=0)

    diag1 = torch.block_diag(*split_by_spline1)
    diag2 = torch.block_diag(*split_by_spline2)

    # Continuity
    continuity_C = diag2[:-num_deriv*dim] - diag1[num_deriv*dim:]
    continuity_d = torch.zeros(continuity_C.shape[0], device=device)

    # Continuity at the beginning
    B0, _ = x0.shape
    num_deriv_at_x0 = B0
    if B0 == num_deriv:
        continuity_x0_A = diag1[:num_deriv*dim]
    else:
        continuity_x0_A = diag1[:num_deriv_at_x0*dim]
    continuity_x0_d = x0.reshape(-1)

    # Continuity at the end
    Bf, _ = xf.shape
    num_deriv_at_xf = Bf
    if Bf == num_deriv:
        continuity_xf_A = diag2[-num_deriv*dim:]
    else:
        continuity_xf_A = diag2[-num_deriv*dim : -num_deriv*dim + num_deriv_at_xf*dim]
    continuity_xf_d = xf.reshape(-1)

    # Concatenate all equality constraints
    C = torch.cat([continuity_C, continuity_xf_A, continuity_x0_A], dim=0)
    d = torch.cat([continuity_d, continuity_xf_d, continuity_x0_d], dim=0)

    # Cost matrix: rubber-band loss. sum_i ||c_i - c_{i+1}||^2 is equivalent to sum_i c_i^2 - c_i c_{i+1} - c_i c_{i-1}, which can be expressed
    # as matrix of 2s on diagonal (except first and last element on diag), and -1s on the off-diagonal corresponding to the 
    # adjacent control points.
    Q = torch.eye(num_splines*num_control_points*dim, device=device)
    Q_arange = torch.arange(Q.shape[0], device=device)[:-dim]
    Q[Q_arange, Q_arange + dim] = -1.
    Q = Q + Q.T
    Q[0, 0] = 1.
    Q[-1, -1] = 1.

    return A, b, C, d, Q

def get_continuity_at_point(bernstein_coefficients, x0, spline_index):
    # NOTE: IMPORTANT! We assume x0 and xf are feasible to begin with! To maintain feasibility, we need
    # to check if x0 and xf are in their corresponding polytopes, and if not, project them into the polytope before
    # calling this function.

    device = bernstein_coefficients.device

    # num_t x num_splines x num_control_points x num_deriv : bernstein coefficients
    num_splines, num_control_points, num_deriv = bernstein_coefficients.shape
    dim = x0.shape[-1]

    first_coefficient = bernstein_coefficients[spline_index].permute(1, 0)       # num_deriv x num_control_points

    kron_eye = torch.eye(dim, device=device)

    kron = torch.kron(first_coefficient.contiguous(), kron_eye)

    if x0.shape[0] == num_deriv:
        C = torch.zeros(kron.shape[0], num_splines*num_control_points*dim, device=device)
        C[:, spline_index*num_control_points*dim : (spline_index + 1)*num_control_points*dim] = kron
    else:
        C = torch.zeros(x0.shape[0]*dim, num_splines*num_control_points*dim, device=device)
        C[:, spline_index*num_control_points*dim : (spline_index + 1)*num_control_points*dim] = kron[:x0.shape[0]*dim]

    d = x0.reshape(-1)
    return C, d

# Computes a single spline within the polytope to some future point
def get_single_spline_matrices(bernstein_coefficients, polytope, x0, xf):
    # NOTE: Again, we assume that x0 and xf are already feasible to begin with!

    device = bernstein_coefficients.device
    num_eval_points, num_control_points, num_deriv = bernstein_coefficients.shape
    dim = x0.shape[-1]

    # Polytope constraint on control points
    As = [polytope[0]]*num_control_points
    bs = [polytope[1]]*num_control_points
    A = torch.block_diag(*As)
    b = torch.cat(bs, dim=0)

    first_coefficient = bernstein_coefficients[0].permute(1, 0)       # num_deriv x num_control_points
    last_coefficient = bernstein_coefficients[-1].permute(1, 0)       # num_deriv x num_control_points

    kron_eye = torch.eye(dim, device=device)

    kron1 = torch.kron(first_coefficient.contiguous(), kron_eye)
    kron2 = torch.kron(last_coefficient.contiguous(), kron_eye)

    # Continuity at the beginning
    num_deriv_at_x0 = len(x0)

    if num_deriv_at_x0 == num_deriv:
        continuity_x0_A = kron1
    else:
        continuity_x0_A = kron1[:num_deriv_at_x0*dim]
    continuity_x0_d = x0.reshape(-1)

    # Continuity at the end
    num_deriv_at_xf = len(xf)
    if num_deriv_at_xf == num_deriv:
        continuity_xf_A = kron2
    else:
        continuity_xf_A = kron2[:num_deriv_at_xf*dim]
    continuity_xf_d = xf.reshape(-1)

    # Concatenate all equality constraints
    C = torch.cat([continuity_x0_A, continuity_xf_A], dim=0)
    d = torch.cat([continuity_x0_d, continuity_xf_d], dim=0)

    # Cost matrix: rubber-band loss. sum_i ||c_i - c_{i+1}||^2 is equivalent to sum_i c_i^2 - c_i c_{i+1} - c_i c_{i-1}, which can be expressed
    # as matrix of 2s on diagonal (except first and last element on diag), and -1s on the off-diagonal corresponding to the
    # adjacent control points.
    Q = torch.eye(num_control_points*dim, device=device)
    Q_arange = torch.arange(Q.shape[0], device=device)[:-dim]
    Q[Q_arange, Q_arange + dim] = -1.
    Q = Q + Q.T
    Q[0, 0] = 1.
    Q[-1, -1] = 1.

    return A, b, C, d, Q

# This operation we do in numpy since it's simple.
def get_polytope_projection_matrix(A, b, pt):
    # A: num_constraints x dim
    # b: num_constraints
    dim = A.shape[-1]
    Q = np.eye(dim)
    w = -pt

    return A, b, Q, w

def project_point_into_polytope(A, b, pt):
    # A: num_constraints x dim
    # b: num_constraints
    # pt: dim
    # Returns the projection of pt into the polytope defined by A and b
    A, b, Q, w = get_polytope_projection_matrix(A, b, pt)
    cones = [clarabel.NonnegativeConeT(A.shape[0])]

    ###### CLARABEL #######

    Q = sparse.csc_matrix(Q)
    A = sparse.csc_matrix(A)

    settings = clarabel.DefaultSettings()
    settings.verbose = False

    solver = clarabel.DefaultSolver(Q, w, A, b, cones, settings)

    sol = solver.solve()

    # Check solver status
    if str(sol.status) != 'Solved':
        print(f"Solver status: {sol.status}")
        print('Clarabel did not solve polytope projection!')
        solver_success = False
        projected_pt = None

    else:
        solver_success = True
        projected_pt = np.array(sol.x)

    return projected_pt, solver_success