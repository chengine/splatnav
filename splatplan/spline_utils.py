import torch
import numpy as np
import sympy as sym
import scipy
import time

class BezierCurve():
    def __init__(self, num_control_points, num_deriv, dim):
        self.n = num_control_points - 1     # degree of the Bezier curve

        assert self.n >= 1, 'Number of control points must be at least 2'
        self.num_deriv = num_deriv
        self.dim = dim

        self.compute_bernstein_coefficients()
        self.time_scale = None

    def compute_bernstein_coefficients(self):
        # Computes the Bernstein coefficients symbolically
        t = sym.Symbol('t')

        tnow = time.time()
        mat_fx = []

        for k in range(self.n + 1):
            rows = []

            for deriv in range( self.num_deriv + 1):
                # 0-th order
                if deriv == 0:
                    deriv_fx = ( ( 1 - t )**( self.n - k ) ) * ( t**k )

                # higher order
                else:
                    deriv_fx = sym.simplify(sym.diff( deriv_fx , t, 1))

                # NOTE: This is a workaround in order to lambidfy matrices that contain constants.
                if deriv_fx.is_constant():
                    deriv_fx = deriv_fx + 1e-12*t

                rows.append(deriv_fx)

            mat_fx.append(rows)

        function_matrix = sym.Matrix(mat_fx)      # num_control_points x num_deriv
        print('Time to compute Bernstein coefficients:', time.time() - tnow)
 
        self.bernstein_fx = sym.lambdify(t, function_matrix)

        combo_scaling = scipy.special.comb(self.n, np.arange(self.n + 1))

        self.coefficient_fx = lambda x: torch.tensor( (self.bernstein_fx(x.cpu().numpy()).transpose() * combo_scaling).transpose() , dtype=torch.float32).to(x.device)

    def set_control_points(self, control_points, time_scale=None):
        # control_points: tensor of shape (num_control_points, dim) or (num_splines, num_control_points, dim)
        if control_points.ndim == 2:
            assert control_points.shape == (self.n + 1, self.dim)
            self.num_splines = 1

        elif control_points.ndim == 3:
            assert control_points.shape[1:] == (self.n + 1, self.dim)   
            self.num_splines = control_points.shape[0]  

        else:
            raise ValueError('Invalid shape for control_points')

        self.control_points = control_points

        if time_scale is not None:
            assert torch.numel(time_scale) == self.num_splines, 'Time scale must be a scalar or a tensor of shape (num_splines)'
        
        self.time_scale = time_scale            # NOTE: We don't check if time_scale is of valid shape!

    # NOTE: We do not check if the time scale is of valid shape, or if the number of splines match the number of control points if you set it afterward!
    def set_time_scale(self, time_scale):
        self.num_splines = time_scale.numel()
        self.time_scale = time_scale

    def evaluate_coefficients(self, t):
        # Evaluates the Bernstein coefficients at time t, where t is 1D tensor or 2D tensor
        if t.ndim == 1:
            coefficients = self.coefficient_fx(t).permute(2, 0, 1)       # num_t x num_control_points x num_deriv

            if self.num_splines > 1:
                coefficients = coefficients.unsqueeze(0).expand(self.num_splines, -1, -1, -1).permute(1, 2, 0, 3)       # num_t x num_control_points x num_splines x num_deriv

        elif t.ndim == 2:
            B, N = t.shape
            coefficients = self.coefficient_fx(t.flatten())
            coefficients = coefficients.reshape(self.n + 1, self.num_deriv + 1, B, N).permute(3, 0, 2, 1)       # num_t x num_control_points x num_splines x num_deriv

        elif t.ndim > 2:
            raise ValueError('Invalid shape for t')

        else:   # scalar case
            coefficients = self.coefficient_fx(t)

        if (self.time_scale is not None):
            time_scale = self.time_scale.unsqueeze(-1)
            time_scaling = (1. / time_scale) ** torch.arange(self.num_deriv + 1, device=t.device).unsqueeze(0) # (opt: num_splines) x num_deriv
            time_scaling = time_scaling.squeeze()

            coefficients = coefficients * time_scaling 

        if coefficients.ndim == 4:
            coefficients = coefficients.permute(0, 2, 1, 3)

        return coefficients

    # NOTE: Make sure to set the control points before calling this function!
    def evaluate(self, t):
        # Evaluates the Bezier curve at time t, subject to batching (i.e. multiple splines) and time scaling
        # t: scalar or 1D tensor or 2D tensor. NOTE THAT THESE t ARE NORMALIZED (i.e. PROGRESS PER SPLINE)
        # output: array of shape (num_t, num_deriv, dim) or (num_splines, num_t, num_deriv, dim)

        if t.ndim == 2:
            assert t.shape[0] == self.num_splines, 'Number of splines must match the number of control points'

        # Compute the Bernstein coefficients
        coefficients = self.evaluate_coefficients(t)       # (opt: num_t) x (opt: num splines) x num_control_points x num_deriv

        # Compute the Bezier curve subject to time scaling
        if self.num_splines == 1:
            output = torch.einsum('...ij, il -> ...jl', coefficients, self.control_points)
        else:
            output = torch.einsum('...bij, bil -> ...bjl', coefficients, self.control_points)
        return output

    # NOTE: Only works for scalar t
    def evaluate_at_t(self, t):
        assert t >= 0., 'Time must be non-negative'

        # Finds the correct Bezier curve to evaluate t at. t is a scalar or 1D tensor representing where among the total time of the poly spline.
        if self.time_scale is None:
            # If no time scale is provided, we assume that the time is normalized between 0 and 1
            spline_ind = np.clip(np.floor(t).astype(np.int32), 0, self.num_splines - 1)

            spline_begin = torch.arange(self.num_splines, device=self.control_points.device)

            normalized_t = t - spline_begin[spline_ind]

        else:
            spline_end = torch.cumsum(self.time_scale, dim=0)

            t = np.clip(t, 0., spline_end[-1].item())       # Clip t to be within the total time of the poly spline

            spline_ind = torch.searchsorted(spline_end, t, right=True)

            spline_begin = torch.cat([torch.zeros(1, device=self.time_scale.device), spline_end[:-1]], dim=0)

            spline_ind = torch.clip(spline_ind, 0, self.num_splines - 1)

            normalized_t = (t - spline_begin[spline_ind]) / self.time_scale[spline_ind]

        coefficients = self.coefficient_fx(normalized_t)       # num_control_points x num_deriv

        if (self.time_scale is not None):
            time_scale = self.time_scale[spline_ind].unsqueeze(-1)
            time_scaling = (1. / time_scale) ** torch.arange(self.num_deriv + 1, device=self.control_points.device).unsqueeze(0) # num_deriv
            time_scaling = time_scaling.squeeze()

            coefficients = coefficients * time_scaling 

        # print(coefficients)
        # print(self.control_points[spline_ind])
        # print(normalized_t)

        output = torch.einsum('ij, il -> jl', coefficients, self.control_points[spline_ind])

        return output
