import torch

def batch_mahalanobis_distance(x, means, covs_inv):
    # Computes the Mahalanobis distance of a batch of points x to a batch of Gaussians with means and covariances.
    # x: n
    # means: B x n
    # covs: B x n x n

    x = x.unsqueeze(0)      # 1 x n
    # we only want x y z 
    x = x[..., :3]

    diff = x - means
    mahalanobis = torch.bmm(diff[..., None, :], torch.bmm(covs_inv, diff.unsqueeze(-1))).squeeze()
    grad = 2 * torch.bmm(covs_inv, diff[..., None]).squeeze() 
    hessian = 2 * covs_inv

    return mahalanobis, grad, hessian

def batch_point_distance(x, means):
    # Computes the Euclidean distance of a batch of points x to a batch of other points.
    # x: n
    # means: B x n
    eps = 1e-6
    x = x.unsqueeze(0)  # 1 x n

    # we only want x y z 
    x = x[..., :3]

    diff = x - means  # B x 3

    # Compute the Euclidean distance
    euclidean_distance = torch.norm(diff, dim=1)

    # Compute the gradient
    grad = diff / (euclidean_distance.unsqueeze(1) + eps)  # B x 3

    # Compute the Hessian in a batched manner
    B, n = diff.shape
    I = torch.eye(n, device=x.device).unsqueeze(0).expand(B, -1, -1)  # B x 3 x 3
    d = diff.unsqueeze(2)  # B x 3 x 1
    dT = diff.unsqueeze(1)  # B x 1 x 3
    euclidean_distance_squared = euclidean_distance ** 2 + eps
    euclidean_distance_cubed = euclidean_distance + eps

    hessian = (I - (d @ dT) / euclidean_distance_squared.unsqueeze(1).unsqueeze(2)) / euclidean_distance_cubed.unsqueeze(1).unsqueeze(2)

    return euclidean_distance, grad, hessian

def batch_squared_point_distance(x, means):
    # Computes the Squared Euclidean distance of a batch of points x to a batch of other points
    # x: n
    # means: B x n
    x = x.unsqueeze(0)  # 1 x n

    # we only want x y z 
    x = x[..., :3]
    diff = x - means  # B x 3

    # Compute the Euclidean distance
    euclidean_distance_squared = torch.sum(diff**2, dim=1)

    # Compute the gradient
    grad = 2*diff

    # Compute the Hessian in a batched manner
    B, n = diff.shape
    I = torch.eye(n, device=x.device).unsqueeze(0).expand(B, -1, -1)  # B x 3 x 3
    hessian = 2 * I

    return euclidean_distance_squared, grad, hessian

# This is a root-finding function that performs bisection search.
def real_get_root(r, z, g, max_iterations=25):
    n = r*z

    s = torch.zeros((len(n), 2), device=r.device)
    s[:, 0] = z[..., -1] - 1.
    g_pos = (g >= 0)
    s[g_pos, 1] = torch.linalg.norm(n[g_pos], dim=-1) - 1.
    
    for i in range(max_iterations):
        s_i = torch.mean(s, dim=-1, keepdims=True)
        ratio = n / (s_i + r)
        g = torch.sum(ratio**2, dim=-1) - 1
        
        g_pos = (g >= 0)

        s_i_ = s_i.squeeze()
        s[g_pos, 0] = s_i_[g_pos]
        s[~g_pos, 1] = s_i_[~g_pos]

    return s_i

# Calculates the min. distance from a point to an ellipsoid
def distance_point_ellipsoid(s, x):
    # NOTE: e0, e1, e2: semi-axes of the ellipsoid (e0 > e1 > e2)
    # Change of variables
    z = x / s
    g = torch.sum(z**2, dim=-1) - 1.
    r = (s[..., :] / s[..., -1][..., None])**2

    # Calculate dual variable
    lam = real_get_root(r, z, g)
    
    # Calculate optimal closest point
    y = r * x / (lam + r)
    
    # This is the unscaled labmda
    lam = ( lam.squeeze() * (s[..., -1])**2 ).unsqueeze(-1)

    # Calculate distance
    squared_distance = torch.sum( (y - x)**2, dim=-1)

    # Gradient in local frame
    grad = 2*(x - y)

    # Calculate intermediate variables for hessian
    dq_dlam = -2 * torch.sum(  (s[..., :]**2 * x**2) / (lam + s[..., :]**2)**3 , dim=-1, keepdim=True)
    dy_dlam = - (  (s[..., :]**2 * x) / (lam + s[..., :]**2)**2  )
    dq_dx = -2 * dy_dlam

    # Collect into Hessian
    hess = 2 * ( torch.diag_embed( 1. - (y / x))  + (1. / dq_dlam[..., None]) * torch.einsum('bi, bj -> bij', dy_dlam, dq_dx)   )

    return squared_distance, grad, hess, y
