import numpy as np
import torch
import time

### ________________________________________GENERAL INTERSECTION TEST FOR SHERE/ELLIPSOID-TO-ELLIPSOID____________________________________________________ ###

###NOTE: THIS FUNCTION COMPUTES INTERSECTION OVER A LINE SEGMENT ###
def compute_intersection_linear_motion(x0, delta_x, R_A, S_A, mu_A, R_B=None, S_B=None, collision_type='sphere', mode='bisection', N=10):
    # N parameter is overloaded. If mode is bisection, we will run the bisection iteration N times. If mode is uniform, we will sample N points uniformly.
    # x0 is ndim. So is delta_x.
    # TODO is to add support for batchable x0 and delta_x

    if mode == 'bisection':
        if collision_type == 'sphere':
            # S_B needs to be the robot radius
            assert isinstance(S_B, float) # 'S_B must be a scalar'

            eval_fn = lambda evals: compute_sphere_ellipsoid_Q(R_A, S_A, S_B, evals)
            lambdas, Phi, kappa = compute_K_parameters_sphere(R_A, S_A, S_B)

        elif collision_type == 'ellipsoid':
            eval_fn = lambda evals: compute_ellipsoid_ellipsoid_Q(R_A, S_A, R_B, S_B, evals)
            lambdas, Phi, kappa = compute_K_parameters_ellipsoid(R_A, S_A, R_B, S_B)

        else:
            raise ValueError('Collision type not supported')
        
        # Run the bisection algorithm
        s_l = torch.zeros(R_A.shape[0], device=R_A.device)      # lower bound
        s_u = torch.ones(R_A.shape[0], device=R_A.device)       # upper bound

        s_bounds = torch.stack([s_l, s_u], dim=-1)     # N x 2

        for iter in range(N):
            eval_pts = 0.5*( torch.sum( s_bounds, dim=-1) )     # midpoint

            # compute the Q matrix
            Q = eval_fn(eval_pts)       # N x dim x dim

            delta_x_Q = (Q @ delta_x.unsqueeze(-1)).squeeze()      # N x dim
            Q_diff = torch.bmm(Q, (mu_A - x0.unsqueeze(0)).unsqueeze(-1)).squeeze()      # N x dim

            numerator = torch.sum(delta_x_Q * Q_diff, dim=-1)   # N
            denominator = torch.sum(delta_x_Q**2, dim=-1)    # N

            t = torch.clamp(numerator / denominator, 0., 1.)        # N

            # Compute the derivative of K(s)
            x_opt = x0.unsqueeze(0) + t.unsqueeze(-1) * delta_x.unsqueeze(0)
            quadratic = x_opt - mu_A

            v = torch.sum(quadratic[..., None] * Phi, dim=-2)      # N x dim 

            s_grad_diag_numerator = kappa - 2*(eval_pts*kappa)[..., None] - (eval_pts**2)[..., None] * (lambdas - kappa)    # N x dim
            s_grad_diag_denominator = (kappa + eval_pts[..., None]*(lambdas - kappa))**2    # N x dim
            s_grad_diag = s_grad_diag_numerator / s_grad_diag_denominator           # N x dim
            s_grad = torch.sum( s_grad_diag * (v**2) , dim=-1 )   # N

            # Update the bounds
            mask = (s_grad >= 0.)
            s_bounds[mask, 0] = eval_pts[mask]
            s_bounds[~mask, 1] = eval_pts[~mask]

        # Optimal point along line segment
        deltas = x_opt - mu_A
        Q_opt = torch.bmm(Q.transpose(-2, -1), Q).squeeze()      # N x dim x dim
        K_opt = torch.bmm(Q, (deltas).unsqueeze(-1)).squeeze()      # N x dim
        K_opt = torch.sum(K_opt**2, dim=-1)      # N            delta_j Q delta_j

        # Is the line intersecting?
        is_not_intersect = (K_opt >= 1.)

        output = {
            'seedpoint': x_opt,
            'deltas': deltas,
            'Q_opt': Q_opt,
            'K_opt': K_opt,
            'mu_A': mu_A,
            'is_not_intersect': is_not_intersect
        }
        
    # TODO: We need to redo the uniform sampling for the line segment. Raise not implemented for now...
    elif mode == 'uniform':
        raise NotImplementedError('Uniform sampling not implemented yet')

        # eval_pts = torch.linspace(0., 1., N+2, device=R_A.device)[1:-1].reshape(1, -1, 1)

        # if collision_type == 'sphere':
        #     # S_B needs to be the robot radius
        #     try:
        #         assert len(S_B.shape) == 1, 'S_B must be a scalar'
        #     except:
        #         raise ValueError('S_B must be a scalar')

        #     is_intersect, K_values = gs_sphere_intersection_test(R_A, S_A, S_B, mu_A, mu_B, eval_pts)

        # elif collision_type == 'ellipsoid':
        #     is_intersect, K_values = ellipsoid_intersection_test(R_A, S_A, mu_A, R_B, S_B, mu_B, eval_pts)

        # else:
        #     raise ValueError('Collision type not supported')
        
    else:
        raise ValueError('Mode not supported')
    
    # Return a dictionary of variables
    return output

# Computes the intersection criterion for a set of means for the robot ellipsoid
# TODO: This function is not yet implemented.
def compute_intersection_point(R_A, S_A, mu_A, R_B=None, S_B=None, mu_B=None, collision_type='sphere', mode='bisection', N=10):

    raise NotImplementedError('Not implemented yet')

    if mode == 'bisection':
        if collision_type == 'sphere':
            # S_B needs to be the robot radius
            assert len(S_B.shape) == 1, 'S_B must be a scalar'

            eval_fn = lambda evals: compute_sphere_ellipsoid_Q(R_A, S_A, S_B, evals)

        elif collision_type == 'ellipsoid':
            eval_fn = lambda evals: compute_ellipsoid_ellipsoid_Q(R_A, S_A, R_B, S_B, evals)

        else:
            raise ValueError('Collision type not supported')
        
        # Run the bisection algorithm
        s_l = torch.zeros(R_A.shape[0], device=R_A.device)      # lower bound
        s_u = torch.ones(R_A.shape[0], device=R_A.device)       # upper bound

        s_bounds = torch.stack([s_l, s_u], dim=-1)     # N x 2

        for iter in range(N):
            eval_pts = 0.5*( torch.sum( s_bounds, dim=-1) )     # midpoint

            # compute the Q matrix
            Q = eval_fn(eval_pts)       # N x dim x dim

            delta_x_Q = (Q @ delta_x.unsqueeze(-1)).squeeze()      # N x dim
            Q_diff = torch.bmm(Q, (mu_A - x0.unsqueeze(0)).unsqueeze(-1)).squeeze()      # N x dim

            numerator = torch.sum(delta_x_Q * Q_diff, dim=-1)   # N
            denominator = torch.sum(delta_x_Q**2, dim=-1)    # N

            t = torch.clamp(numerator / denominator, 0., 1.)        # N

            # Compute the derivative of K(s)
            quadratic = x0.unsqueeze(0) + t.unsqueeze(-1) * delta_x.unsqueeze(0) - mu_A.unsqueeze(0)

            v = torch.sum(quadratic[..., None] * Phi, dim=-2)      # N x dim 

            s_grad_diag_numerator = kappa - 2*(eval_pts*kappa)[..., None] - (eval_pts**2)[..., None] * (lambdas - kappa)    # N x dim
            s_grad_diag_denominator = (kappa + eval_pts[..., None]*(lambdas - kappa))**2    # N x dim
            s_grad_diag = s_grad_diag_numerator / s_grad_diag_denominator           # N x dim
            s_grad = torch.sum( s_grad_diag * (v**2) , dim=-1 )   # N

            # Update the bounds
            mask = (s_grad >= 0.)
            s_bounds[mask, 0] = eval_pts[mask]
            s_bounds[~mask, 1] = eval_pts[~mask]

        
    # TODO: We need to redo the uniform sampling for the line segment. Raise not implemented for now...
    elif mode == 'uniform':
        raise NotImplementedError('Uniform sampling not implemented yet')

        eval_pts = torch.linspace(0., 1., N+2, device=R_A.device)[1:-1].reshape(1, -1, 1)

        if collision_type == 'sphere':
            # S_B needs to be the robot radius
            try:
                assert len(S_B.shape) == 1, 'S_B must be a scalar'
            except:
                raise ValueError('S_B must be a scalar')

            is_intersect, K_values = gs_sphere_intersection_test(R_A, S_A, S_B, mu_A, mu_B, eval_pts)

        elif collision_type == 'ellipsoid':
            is_intersect, K_values = ellipsoid_intersection_test(R_A, S_A, mu_A, R_B, S_B, mu_B, eval_pts)

        else:
            raise ValueError('Collision type not supported')
        
    else:
        raise ValueError('Mode not supported')


# TODO: ALLOW FOR BOTH UNIFORM SAMPLING OR BISECTION SEARCH !!!
### ________________________________________INTERSECTION TEST FOR ELLIPSOID-TO-ELLIPSOID____________________________________________________ ###
# In fact, we can just transform the world frame to a robot body spherical frame and then use the sphere to ellipsoid tests. 
# This is implicitly what the generalized eigenvalue problem is doing.

def generalized_eigen(A, B):
    # IMPORTANT!!! Assuming B is not batched (statedim x statedim), A is batched (batchdim x statedim x statedim)
    batch_dim = A.shape[0]
    state_dim = B.shape[0]

    # see NR section 11.0.5
    # L is a lower triangular matrix from the Cholesky decomposition of B
    L,_ = torch.linalg.cholesky_ex(B)

    L = L.reshape(-1, state_dim, state_dim).expand(batch_dim, -1, -1)

    # solve Y * L^T = A by a workaround
    # if https://github.com/tensorflow/tensorflow/issues/55371 is solved then this can be simplified
    Y = torch.transpose(torch.linalg.solve_triangular(L, torch.transpose(A, 1, 2), upper=False), 1, 2)

    # solve L * C = Y
    C = torch.linalg.solve_triangular(L, Y, upper=False)
    # solve the equivalent eigenvalue problem

    e, v_ = torch.linalg.eigh(C)

    # solve L^T * x = v, where x is the eigenvectors of the original problem
    v = torch.linalg.solve_triangular(torch.transpose(L, 1, 2), v_, upper=True)
    # # normalize the eigenvectors
    return e, v

def ellipsoid_intersection_test(R_A, S_A, mu_A, R_B, S_B, mu_B, eval_pts):
    # Compute the covariance
    # Cov_A_half = torch.bmm(R_A, S_A)        # N x dim x dim
    Cov_A_half = R_A * S_A[..., None, :]
    Cov_A = torch.bmm(Cov_A_half, Cov_A_half.transpose(-2, -1))

    Cov_B_half = R_B * S_B[None, :]
    Cov_B = Cov_B_half @ Cov_B_half.transpose(-2, -1)

    lambdas, _, v_squared = ellipsoid_intersection_test_helper(Cov_A, Cov_B, mu_A, mu_B)  # (batchdim x statedim), (batchdim x statedim x statedim), (batchdim x statedim)
    kappa = 1.
    
    K_values = K_function(lambdas, v_squared, kappa, eval_pts)      # batchdim x Nsamples

    intersection = torch.any(K_values > 1., dim=-1)

    return intersection, K_values

def ellipsoid_intersection_test_helper(Sigma_A, Sigma_B, mu_A, mu_B):
    lambdas, Phi = generalized_eigen(Sigma_A, Sigma_B) # eigh(Sigma_A, b=Sigma_B)
    v_squared = (torch.bmm(Phi.transpose(1, 2), (mu_A - mu_B)[..., None])).squeeze() ** 2
    return lambdas, Phi, v_squared

def compute_K_parameters_ellipsoid(R_A, S_A, R_B, S_B):
    # Compute the covariance
    # Cov_A_half = torch.bmm(R_A, S_A)        # N x dim x dim
    Cov_A_half = R_A * S_A[..., None, :]
    Cov_A = torch.bmm(Cov_A_half, Cov_A_half.transpose(-2, -1))

    Cov_B_half = R_B * S_B[None, :]
    Cov_B = Cov_B_half @ Cov_B_half.transpose(-2, -1)

    lambdas, Phi = generalized_eigen(Cov_A, Cov_B) # eigh(Sigma_A, b=Sigma_B)
    kappa = 1.

    return lambdas, Phi, kappa

### ________________________________________INTERSECTION TEST FOR SPHERE-TO-ELLIPSOID____________________________________________________ ###
def gs_sphere_intersection_test(R, S, radius, mu_A, mu_B, eval_pts):
    lambdas, v_squared = gs_sphere_intersection_test_helper(R, S, mu_A, mu_B)  # (batchdim x statedim), (batchdim x statedim x statedim), (batchdim x statedim)
    kappa = radius**2

    K_values = K_function(lambdas, v_squared, kappa, eval_pts)      # batchdim x Nsamples

    intersection = torch.any(K_values > 1., dim=-1)
 
    return intersection, K_values

def gs_sphere_intersection_test_helper(R, S, mu_A, mu_B):
    lambdas, v_squared = S**2, (torch.bmm(R.transpose(1, 2), (mu_A - mu_B)[..., None])).squeeze() ** 2
    return lambdas, v_squared

def compute_K_parameters_sphere(R_A, S_A, radius):
    lambdas = S_A**2
    Phi = R_A
    kappa = radius**2

    return lambdas, Phi, kappa

def K_function(lambdas, v_squared, kappa, eval_pts):
    batchdim = lambdas.shape[0]
    return torch.sum(v_squared.reshape(batchdim, 1, -1)*((eval_pts*(1.-eval_pts))/(kappa + eval_pts*(lambdas.reshape(batchdim, 1, -1) - kappa))), dim=2)

### ________________________________________COMPUTES Q____________________________________________________ ###
# NOTE: THIS ONLY RETURNS THE MATRIX SQUARE ROOT OF Q!!! 
def compute_Q(lambdas, Phi, kappa, eval_pts):
    # Compute square root of Q (this is for efficiency reasons).
    numerator = eval_pts * (1. - eval_pts)      # N
    denominator = kappa + eval_pts[:, None] * ( lambdas - kappa )    # N x dim
    diag = torch.sqrt(numerator[:, None] / denominator)                 # N x dim

    Q = diag[..., None] * Phi.transpose(-2, -1)   # N x dim x dim

    return Q

def compute_ellipsoid_ellipsoid_parameters(Sigma_A, Sigma_B):
    lambdas, Phi = generalized_eigen(Sigma_A, Sigma_B)
    return lambdas, Phi

# NOTE: THIS ONLY RETURNS THE MATRIX SQUARE ROOT OF Q!!! 
def compute_ellipsoid_ellipsoid_Q(R_A, S_A, R_B, S_B, eval_pts):

    # Compute the covariance
    Cov_A_half = R_A * S_A[..., None, :]
    Cov_A = torch.bmm(Cov_A_half, Cov_A_half.transpose(-2, -1))

    Cov_B_half = R_B * S_B[None, :]
    Cov_B = Cov_B_half @ Cov_B_half.transpose(-2, -1)

    # Compute generalized eigenvalue problem
    lambdas, Phi = compute_ellipsoid_ellipsoid_parameters(Cov_A, Cov_B)     # N x dim, N x dim x dim

    kappa = 1.

    Q = compute_Q(lambdas, Phi, kappa, eval_pts)

    return Q

# NOTE: THIS ONLY RETURNS THE MATRIX SQUARE ROOT OF Q!!! 
def compute_sphere_ellipsoid_Q(R_A, S_A, radius, eval_pts):
    lambdas, Phi = S_A**2, R_A     # N x dim, N x dim x dim
    kappa = radius**2
    Q = compute_Q(lambdas, Phi, kappa, eval_pts)

    return Q

### ________________________________________INTERSECTION TEST FOR SPHERE-TO-ELLIPSOID (NUMPY VARIANTS)____________________________________________________ ###
# This section is just for timing and comparison purposes.
def gs_sphere_intersection_test_np(R, D, kappa, mu_A, mu_B, tau):
    tnow = time.time()
    lambdas, v_squared = gs_sphere_intersection_test_helper_np(R, D, mu_A, mu_B)  # (batchdim x statedim), (batchdim x statedim x statedim), (batchdim x statedim)
    print('helper:' , time.time() - tnow)
    tnow = time.time()
    KK = gs_K_function_np(lambdas, v_squared, kappa, tau)      # batchdim x Nsamples
    print('function eval:' , time.time() - tnow)

    tnow = time.time()
    test_result = ~np.all(np.any(KK > 1., axis=-1))
    print('boolean:' , time.time() - tnow)

    return test_result

def gs_sphere_intersection_test_helper_np(R, D, mu_A, mu_B):
    lambdas, v_squared = D, (np.matmul(np.transpose(R, (0, 2, 1)), (mu_A - mu_B)[..., None])).squeeze() ** 2
    return lambdas, v_squared

def gs_K_function_np(lambdas, v_squared, kappa, tau):
    batchdim = lambdas.shape[0]
    ss = np.linspace(0., 1., 100)[1:-1].reshape(1, -1, 1)
    return (1./tau**2)*np.sum(v_squared.reshape(batchdim, 1, -1)*((ss*(1.-ss))/(kappa + ss*(lambdas.reshape(batchdim, 1, -1) - kappa))), axis=2)


