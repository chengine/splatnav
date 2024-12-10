import torch

def compute_polytope(deltas, Q, K, mu):
        
    delta_Q = torch.bmm(Q, deltas[..., None]).squeeze()
    rhs = torch.sqrt(K) + torch.sum(delta_Q * mu, dim=-1)

    A = -delta_Q.squeeze()
    b = -rhs

    # Boundary point (compute just for reference)
    boundary_points = mu + deltas / torch.sqrt(K)[..., None]

    return A, b, boundary_points

# def compute_supporting_hyperplanes(R, D, kappa, mu_A, test_pt, tau):

#     batch = R.shape[0]
#     dim = R.shape[-1]

#     evals = gs_sphere_intersection_eval(R, D, kappa, mu_A, test_pt, tau)

#     K_j = evals[0]
#     inds = evals[1]

#     ss = torch.linspace(0., 1., 100, device=R.device)[1:-1]
#     s_max = ss[inds]

#     lambdas = D

#     S_j_flat = (s_max*(1-s_max))[..., None] / (kappa + s_max[..., None] * (lambdas - kappa))

#     S_j = torch.diag_embed(S_j_flat)
#     A_j = torch.bmm(R, torch.bmm(S_j, R.transpose(1, 2)))

#     delta_j = test_pt - mu_A

#     A = -torch.bmm(delta_j.reshape(batch, 1, -1), A_j).squeeze()
#     b = -torch.sqrt(K_j) + torch.sum(A*mu_A, dim=-1)

#     proj_points = mu_A + delta_j / torch.sqrt(K_j)[..., None]

#     # TODO: Leaves the polytopes as tensors
#     return A.cpu().numpy().reshape(-1, dim), b.cpu().numpy().reshape(-1, 1), proj_points.cpu().numpy()

# #  TODO: Pull this into a class
# def compute_polytope(R, D, kappa, mu_A, test_pt, tau, A_bound, b_bound):
#     # Find safe polytope in A <= b form
#     A, b, _ = compute_supporting_hyperplanes(R, D, kappa, mu_A, test_pt, tau)

#     # A, b = A.cpu().numpy(), b.cpu().numpy()

#     dim = mu_A.shape[-1]

#     # Add in the bounding poly constraints
#     A = np.concatenate([A.reshape(-1, dim), A_bound.reshape(-1, dim)], axis=0)
#     b = np.concatenate([b.reshape(-1, 1), b_bound.reshape(-1, 1)], axis=0)

#     # TODO: Leave polytopes as tensors
#     return A, b