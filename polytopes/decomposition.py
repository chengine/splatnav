import torch

def compute_polytope(deltas, Q, K, mu):
        
    delta_Q = torch.bmm(Q, deltas[..., None]).squeeze()
    rhs = torch.sqrt(K) + torch.sum(delta_Q * mu, dim=-1)

    A = -delta_Q.squeeze()
    b = -rhs

    # Boundary point (compute just for reference)
    boundary_points = mu + deltas / torch.sqrt(K)[..., None]

    return A, b, boundary_points