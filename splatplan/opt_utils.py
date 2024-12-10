import numpy as np 
import torch
import clarabel
from scipy import sparse

from spline_utils import BezierCurve
from matrices import get_qp_matrices, project_point_into_polytope, get_single_spline_matrices, get_continuity_at_point

# --------------------------------------------------------------------------------#
######################################################################################################
# Spline optimizer for poly Bezier curve
class SplineOptimizer():
    def __init__(self, num_control_points=6, num_deriv=3, dim=3, device='cpu') -> None:
        self.num_control_points = num_control_points
        self.dim = dim              # dimension of state
        self.num_deriv = num_deriv  # number of derivatives to enforce continuity, plus position
        self.device = device

        ### Instantiate Bezier Curve
        self.bspline = BezierCurve(num_control_points, num_deriv, dim)
        self.control_points = None

    def optimize_bspline(self, polytopes, x0, xf, time_scales=None):
        # If x0 and xf are positions and higher order derivatives (less than the set number of derivatives to enforce continuity) are provided (full or partially)
        # the optimizer will automatically enforce continuity. Continuity is enforced in order, specifically if x0 is 3 x dim, then position, velocity,
        # and acceleration continuity will be enforced. If x0 is 2 x dim, then position and velocity continuity will be enforced.
        # NOTE: If you pass in only positions, make sure that x0 is size 1 x dim and xf is size 1 x dim.

        if x0.ndim == 1:
            x0 = x0[None, :]
        if xf.ndim == 1:
            xf = xf[None, :]

        # Check if the position x0 and xf are in the polytopes
        position_x0 = x0[0]
        position_xf = xf[0]

        A_x0, b_x0 = polytopes[0]
        A_xf, b_xf = polytopes[-1]

        # Project if they are not in the polytope
        if not torch.all( (A_x0 @ position_x0 - b_x0) <= 0.):
            print('Initial position not in the first polytope!')
            projected_x0 = project_point_into_polytope(A_x0, b_x0, position_x0)
            x0[0] = projected_x0

        if not torch.all( (A_xf @ position_xf - b_xf) <= 0.):
            print('Final position not in the last polytope!')
            projected_xf = project_point_into_polytope(A_xf, b_xf, position_xf)
            xf[0] = projected_xf

        # Optimize coefficients
        control_points, solver_success = self.optimize_bspline_coefficients(polytopes, x0, xf, time_scales)

        if solver_success:
            # Set control points
            self.bspline.set_control_points( control_points, time_scales )
            self.polytopes = polytopes
            self.control_points = control_points
            self.num_splines = self.bspline.num_splines
            self.time_scales = time_scales
            self.xf = xf
            self.x0 = x0

        return control_points, solver_success

    def optimize_bspline_coefficients(self, polytopes, x0, xf, time_scales=None):
        num_splines = len(polytopes)         # number of splines

        if time_scales is not None:
            assert len(time_scales) == num_splines, "Time scales must be provided for each section of the Bezier curve"
            # We need to set the time scales for each section of the Bezier cure
            self.bspline.set_time_scale(time_scales)

        # To enforce continuity, we want to evaluate the bspline bernstein coefficients at the start and end of each spline, where
        # start is t=0, and end is t=1.
        progress_eval_points = torch.tensor([0., 1.], device=self.device)[None, :].expand(num_splines, 2)       # num_splines x 2
        bernstein_coefficients = self.bspline.evaluate_coefficients(progress_eval_points)  # num_t x num_splines x num_control_points x num_deriv

        #Set up matrices
        A, b, C, d, Q = get_qp_matrices(bernstein_coefficients, polytopes, x0, xf)
        n_var = C.shape[-1]
        w = np.zeros(n_var)

        A = A.cpu().numpy()
        b = b.cpu().numpy()
        C = C.cpu().numpy()
        d = d.cpu().numpy()
        Q = Q.cpu().numpy()
        cones = [clarabel.ZeroConeT(C.shape[0]), clarabel.NonnegativeConeT(A.shape[0])]

        ###### CLARABEL #######

        P = sparse.csc_matrix(Q)
        A = sparse.csc_matrix(np.concatenate([C, A], axis=0))
        b = np.concatenate([d, b], axis=0)

        settings = clarabel.DefaultSettings()
        settings.verbose = False

        solver = clarabel.DefaultSolver(P, w, A, b, cones, settings)

        sol = solver.solve()

        # Check solver status
        if str(sol.status) != 'Solved':
            print(f"Solver status: {sol.status}")
            print('Clarabel did not solve the problem!')
            solver_success = False
            control_points = None

        else:
            solver_success = True
            control_points = torch.tensor(sol.x, device=self.device)
            control_points = control_points.reshape(num_splines, self.num_control_points, self.dim)

        # Cost
        cost = torch.sum( (control_points.reshape(-1, self.dim)[:-1] - control_points.reshape(-1, self.dim)[1:] )**2 ).item()
        print('Cost: ', cost)
        return control_points, solver_success

    # Evaluates the bspline at some scalar wall clock time t
    def evaluate_bspline_at_t(self, t):
        if self.bspline.control_points is not None:
            output = self.bspline.evaluate_at_t(t)      # num_deriv + 1 x dim
        else:
            output = None

        return output
    
    # Discretizes  the poly Bezier curve based on t in [0, 1], i.e. linspace(0., 1., 10)
    def evaluate_bspline(self, t):
        output = self.bspline.evaluate(t)
        return output

    # At some wall clock time t, given the state, locally construct the Bezier curve linking to the next spline
    def solve_local_waypoint(self, query_t, t, x0):
        if x0.ndim == 1:
            x0 = x0[None, :]

        # Using t, find which spline / polytope we are in.
        if self.time_scales is None:
            # If no time scale is provided, we assume that the time is normalized between 0 and 1
            spline_ind = np.clip(np.floor(t).astype(np.int32), 0, self.num_splines - 1)

            spline_begin = torch.arange(self.num_splines, device=self.device)

            progress_along_spline = t - spline_begin[spline_ind]

        else:
            spline_end = torch.cumsum(self.time_scales, dim=0)

            t = np.clip(t, 0., spline_end[-1].item())       # Clip t to be within the total time of the poly spline

            spline_ind = torch.searchsorted(spline_end, t, right=True)

            spline_begin = torch.cat([torch.zeros(1, device=self.device), spline_end[:-1]], dim=0)

            spline_ind = torch.clip(spline_ind, 0, self.num_splines - 1)

            progress_along_spline = (t - spline_begin[spline_ind]) / self.time_scales[spline_ind]

        polytope = self.polytopes[spline_ind]

        # First project the state x0 into the first polytope
        position_x0 = x0[0]

        A_x0, b_x0 = polytope

        # Project if they are not in the polytope
        if not torch.all( (A_x0 @ position_x0 - b_x0) <= 0.):
            print('Initial position not in the first polytope!')
            projected_x0, success = project_point_into_polytope(A_x0.cpu().numpy(), b_x0.cpu().numpy(), position_x0.cpu().numpy())
            x0[0] = torch.tensor(projected_x0, device=self.device, dtype=torch.float32)

        # Solves for the spline from the current polytope all the way to the end
        progress_eval_points = torch.tensor([0., 1.], device=self.device)[None, :].expand(self.num_splines, 2)       # num_splines x 2
        bernstein_coefficients = self.bspline.evaluate_coefficients(progress_eval_points)  # num_t x num_splines x num_control_points x num_deriv

        #Set up matrices
        A, b, C, d, Q = get_qp_matrices(bernstein_coefficients, self.polytopes, self.x0, self.xf)

        # TODO: There's a faster way to do this without having to querying these eval points for every spline when we only care about one
        progress_eval_points = torch.tensor([progress_along_spline, 1.], device=self.device)[None, :]
        bernstein_coefficients = self.bspline.evaluate_coefficients(progress_eval_points)[0]  # num_splines x num_control_points x num_deriv
        C_, d_ = get_continuity_at_point(bernstein_coefficients, x0, spline_ind)

        # C[-C_.shape[0]:, :] = C_
        # d[-d_.shape[0]:] = d_

        # C = torch.cat([C, C_], dim=0)
        # d = torch.cat([d, d_], dim=0)

        # Add continuity of current point in the cost
        Q_ = C_.T @ C_
        Q = Q + Q_
        w = -C_.T @ d_

        n_var = C.shape[-1]
        #w = np.zeros(n_var)

        A = A.cpu().numpy()
        b = b.cpu().numpy()
        C = C.cpu().numpy()
        d = d.cpu().numpy()
        Q = Q.cpu().numpy()
        w = w.cpu().numpy()
        cones = [clarabel.ZeroConeT(C.shape[0]), clarabel.NonnegativeConeT(A.shape[0])]

        ###### CLARABEL #######

        P = sparse.csc_matrix(Q)
        A = sparse.csc_matrix(np.concatenate([C, A], axis=0))
        b = np.concatenate([d, b], axis=0)

        settings = clarabel.DefaultSettings()
        settings.verbose = False

        solver = clarabel.DefaultSolver(P, w, A, b, cones, settings)

        sol = solver.solve()

        # Check solver status
        if str(sol.status) != 'Solved':
            print(f"Solver status: {sol.status}")
            print('Clarabel did not solve the problem!')
            solver_success = False
            output = None
        
            meta = None

        else:
            solver_success = True
            control_points = torch.tensor(sol.x, device=self.device)
            control_points_all = self.control_points.clone()

            control_points = control_points.reshape(-1, self.num_control_points, self.dim)
            control_points_all[spline_ind:] = control_points[spline_ind:]

            self.bspline.set_control_points(control_points_all, self.time_scales)
            output = self.bspline.evaluate_at_t(query_t)

            # NOTE: IMPORTANT! REMEMBER TO SET THE CONTROL POINTS BACK!
            self.bspline.set_control_points(self.control_points, self.time_scales)

            meta = {
                'control_points': control_points_all,
                'control_points_local': control_points[0],
                'spline_ind': spline_ind,
                'cost': torch.sum( (control_points.reshape(-1, self.dim)[:-1] - control_points.reshape(-1, self.dim)[1:] )**2 ).item()
            }

        return output, solver_success, meta