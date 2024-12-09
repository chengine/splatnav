import numpy as np 
import torch
import clarabel
from scipy import sparse

from spline_utils import BezierCurve
from matrices import get_qp_matrices
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

    def optimize_bspline(self, polytopes, x0, xf, time_scales=None):
        # If x0 and xf are positions and higher order derivatives (less than the set number of derivatives to enforce continuity) are provided (full or partially)
        # the optimizer will automatically enforce continuity. Continuity is enforced in order, specifically if x0 is 3 x dim, then position, velocity,
        # and acceleration continuity will be enforced. If x0 is 2 x dim, then position and velocity continuity will be enforced.
        # NOTE: If you pass in only positions, make sure that x0 is size 1 x dim and xf is size 1 x dim.

        # Optimize coefficients
        control_points, solver_success = self.optimize_bspline_coefficients(polytopes, x0, xf, time_scales)

        if solver_success:
            # Set control points
            self.bspline.set_control_points( control_points, time_scales )

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
        
        # TODO
        else:
            solver_success = True
            control_points = torch.tensor(sol.x, device=self.device)
            control_points = control_points.reshape(num_splines, self.num_control_points, self.dim)

        return control_points, solver_success

    # Evaluates the bspline at some scalar wall clock time t
    def evaluate_bspline(self, t):
        if self.bspline.control_points is not None:
            output = self.bspline.evaluate_at_t(t)      # num_deriv + 1 x dim
        else:
            output = None

        return output