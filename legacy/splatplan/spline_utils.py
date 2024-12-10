import numpy as np 
import cvxpy as cvx
import torch
import sympy as sym
import scipy
import clarabel
from scipy import sparse
# --------------------------------------------------------------------------------#

def b_spline_terms(t, deg):
    # terms are (K choose k)(1-t)**(K-k) * t**k
    terms = []
    for i in range(deg + 1):
        scaling = scipy.special.comb(deg, i)
        term = scaling * (1-t)**(deg - i) *t**i
        terms.append(term)

    return np.array(terms).astype(np.float32)

def b_spline_term_derivs(pts, deg, d):
    # terms are (K choose k)(1-t)**(K-k) * t**k
    terms = []
    for i in range(deg + 1):
        scaling = scipy.special.comb(deg, i)
        t = sym.Symbol('t')
        term = []
        for pt in pts:
            term.append(scaling * sym.diff((1-t)**(deg - i) *t**i, t, d).subs(t, pt))
        terms.append(np.array(term))

    return np.array(terms).astype(np.float32)

def create_time_pts(deg=8, N_sec=10, tf=1., device='cpu'):
    #Find coefficients for T splines, each connecting one waypoint to the next
    
    # THESE COEFFICIENTS YOU CAN STORE, SO YOU ONLY NEED TO COMPUTE THEM ONCE!
    time_pts = np.linspace(0., tf, N_sec)

    T = b_spline_terms(time_pts, deg)   #(deg + 1) x 2
    dT = b_spline_term_derivs(time_pts, deg, 1)
    ddT = b_spline_term_derivs(time_pts, deg, 2)
    dddT = b_spline_term_derivs(time_pts, deg, 3)
    ddddT = b_spline_term_derivs(time_pts, deg, 4)

    data = {
    'time_pts': torch.tensor(T, device=device),
    'd_time_pts': torch.tensor(dT, device=device),
    'dd_time_pts': torch.tensor(ddT, device=device),
    'ddd_time_pts': torch.tensor(dddT, device=device),
    'dddd_time_pts': torch.tensor(ddddT, device=device)
    }

    return data

### 
def get_qp_matrices(T, dT, ddT, dddT, ddddT, polytopes, x0, xf, device):
    
    N_sec = len(polytopes)
    deg = T[0].shape[0]
    w = deg*N_sec*3
    k = deg*3
    k3 = deg

    index = torch.arange(deg-1, device=device)

    # Create cost
    Q_ = torch.eye(deg, device=device)
    off_diag = torch.stack([ index, index + 1 ], dim=-1)
    Q_[off_diag[:, 0], off_diag[:, 1]] = -1.
    Q_ = Q_ + Q_.T
    Q_[0, 0] = 1.
    Q_[-1, -1] = 1.
    Q__ = N_sec*3*[Q_]
    Q = torch.block_diag(*Q__)

    # Create inequality matrices
    A = []
    b = []

    # Create equality matrices
    C = torch.zeros((3* 4* N_sec, w), device=device)
    d = torch.zeros(C.shape[0], device=device)

    # Create cost matrix P, consisting only of jerk
    for i in range(N_sec):
        A_ = polytopes[i][0]
        b_ = polytopes[i][1]

        # Ax <= b
        A_x = deg*[A_[:, 0].reshape(-1, 1)]
        A_y = deg*[A_[:, 1].reshape(-1, 1)]
        A_z = deg*[A_[:, 2].reshape(-1, 1)]

        A_xs = torch.block_diag(*A_x)
        A_ys = torch.block_diag(*A_y)
        A_zs = torch.block_diag(*A_z)

        A_blck = torch.cat([A_xs, A_ys, A_zs], dim=-1)
        A.append(A_blck)
        b.extend(deg*[b_])

        # Cx = d
        if i < N_sec-1:
            pos1_cof = T[i][:, -1].reshape(1, -1)
            pos2_cof = -T[i+1][:, 0].reshape(1, -1)

            p1 = torch.block_diag(pos1_cof, pos1_cof, pos1_cof)
            p2 = torch.block_diag(pos2_cof, pos2_cof, pos2_cof)

            vel1_cof = dT[i][:, -1].reshape(1, -1)
            vel2_cof = -dT[i+1][:, 0].reshape(1, -1)

            v1 = torch.block_diag(vel1_cof, vel1_cof, vel1_cof)
            v2 = torch.block_diag(vel2_cof, vel2_cof, vel2_cof)

            acc1_cof = ddT[i][:, -1].reshape(1, -1)
            acc2_cof = -ddT[i+1][:, 0].reshape(1, -1)

            a1 = torch.block_diag(acc1_cof, acc1_cof, acc1_cof)
            a2 = torch.block_diag(acc2_cof, acc2_cof, acc2_cof)

            jer1_cof = dddT[i][:, -1].reshape(1, -1)
            jer2_cof = -dddT[i+1][:, 0].reshape(1, -1)

            j1 = torch.block_diag(jer1_cof, jer1_cof, jer1_cof)
            j2 = torch.block_diag(jer2_cof, jer2_cof, jer2_cof)

            C_t1 = torch.cat([p1, v1, a1, j1], dim=0)
            C_t2 = torch.cat([p2, v2, a2, j2], dim=0)
            C_t = torch.cat([C_t1, C_t2], dim=-1)

            n, m = C_t.shape
            n_e = m//2
            C[n*i: n*(i+1), n_e*i:n_e*(i+2)] = C_t
    
    # Create inequality matrices
    A = torch.block_diag(*A)
    b = torch.cat(b, dim=0)
    b = b.reshape((-1,))

    # Append initial and final position constraints
    p0_cof = T[0][:, 0].reshape(1, -1)
    pf_cof = T[-1][:, -1].reshape(1, -1)

    p0 = torch.block_diag(p0_cof, p0_cof, p0_cof)
    pf = torch.block_diag(pf_cof, pf_cof, pf_cof)

    C_ = torch.zeros((3*2, w), device=device)
    C_[:3, 0:n_e] = p0
    C_[3:, -n_e:] = pf

    d_ = torch.cat([x0, xf], dim=0)
    #d_ = torch.cat([x0, torch.zeros(3, device=device)], dim=0)

    # Concatenate G and h matrices
    C = torch.cat([C, C_], dim=0)

    d = torch.cat([d, d_], dim=0)
    d = d.reshape((-1,))

    return A, b, C, d, Q

######################################################################################################
class SplinePlanner():
    def __init__(self, spline_deg=6, N_sec=10, device='cpu', use_cvxpy=False) -> None:
        self.spline_deg = spline_deg
        self.N_sec = N_sec
        self.device = device
        self.use_cvxpy = use_cvxpy

        ### Create the time points matrix/coefficients for the Bezier curve
        self.time_pts = create_time_pts(deg=spline_deg, N_sec=N_sec, device=device)

    # def optimize_one_step(self, A, b, x0, xf):
    #     self.calculate_b_spline_coeff_one_step(A, b, x0, xf)
    #     return self.eval_b_spline()

    # def calculate_b_spline_coeff_one_step(self, A, b, x0, xf):
    #     N_sections = len(A)         #Number of segments

    #     T = self.time_pts['time_pts']
    #     dT = self.time_pts['d_time_pts']
    #     ddT = self.time_pts['dd_time_pts']
    #     dddT = self.time_pts['ddd_time_pts']
    #     ddddT = self.time_pts['dddd_time_pts']

    #     # Copy time points N times
    #     T_list = [T]*N_sections
    #     dT_list = [dT]*N_sections
    #     ddT_list = [ddT]*N_sections
    #     dddT_list = [dddT]*N_sections
    #     ddddT_list = [ddddT]*N_sections

    #     #Set up CVX problem
    #     A_prob, b_prob, C_prob, d_prob, Q_prob = get_qp_matrices(T_list, dT_list, ddT_list, dddT_list, ddddT_list, A, b, x0, xf)
        
    #     # eliminate endpoint constraint
    #     C_prob = C_prob[:-3]
    #     d_prob = d_prob[:-3]
        
    #     n_var = C_prob.shape[-1]

    #     x = cvx.Variable(n_var)

    #     final_point = cvx.reshape(x, (N_sections*3, -1), order='C')[-3:, -1]

    #     obj = cvx.Minimize(cvx.quad_form(x, Q_prob) + cvx.quad_form(final_point - xf, np.eye(3)))

    #     constraints = [A_prob @ x <= b_prob, C_prob @ x == d_prob]

    #     prob = cvx.Problem(obj, constraints)

    #     prob.solve(solver='CLARABEL')
        
    #     coeffs = []
    #     cof_splits = np.split(x.value, N_sections)
    #     for cof_split in cof_splits:
    #         xyz = np.split(cof_split, 3)
    #         cof = np.stack(xyz, axis=0)
    #         coeffs.append(cof)

    #     self.coeffs = np.array(coeffs)
    #     return self.coeffs, prob.value

    def optimize_b_spline(self, polytopes, x0, xf):
        _, solver_success = self.calculate_b_spline_coeff(polytopes, x0, xf)
        return self.eval_b_spline(), solver_success

    def calculate_b_spline_coeff(self, polytopes, x0, xf):
        N_sections = len(polytopes)         #Number of segments

        T = self.time_pts['time_pts']
        dT = self.time_pts['d_time_pts']
        ddT = self.time_pts['dd_time_pts']
        dddT = self.time_pts['ddd_time_pts']
        ddddT = self.time_pts['dddd_time_pts']

        # Copy time points N times
        T_list = [T]*N_sections
        dT_list = [dT]*N_sections
        ddT_list = [ddT]*N_sections
        dddT_list = [dddT]*N_sections
        ddddT_list = [ddddT]*N_sections

        #Set up matrices
        A_prob, b_prob, C_prob, d_prob, Q_prob = get_qp_matrices(T_list, dT_list, ddT_list, dddT_list, ddddT_list, polytopes, x0, xf, self.device)
        n_var = C_prob.shape[-1]

        A_prob = A_prob.cpu().numpy()
        b_prob = b_prob.cpu().numpy()
        C_prob = C_prob.cpu().numpy()
        d_prob = d_prob.cpu().numpy()
        Q_prob = Q_prob.cpu().numpy()

        ###### CLARABEL #######

        if self.use_cvxpy:
            x = cvx.Variable(n_var)

            loss = cvx.quad_form(x, Q_prob)
            objective = cvx.Minimize(loss)
            constraints = [A_prob @ x <= b_prob, C_prob @ x == d_prob]

            prob = cvx.Problem(objective, constraints)
            prob.solve(solver='ECOS')

            # Check solver status
            if prob.status in ["infeasible", "unbounded"]:
                print(f"Solver status: {prob.status}")
                #print(f"Number of iterations: {sol.iterations}")
                print('CVXPY did not solve the problem!')
                solver_success = False
                solution = None
                self.coeffs = None

            else:
                solver_success = True
                solution = np.array(x.value)

                coeffs = []
                cof_splits = np.split(solution, N_sections)
                for cof_split in cof_splits:
                    xyz = np.split(cof_split, 3)
                    cof = np.stack(xyz, axis=0)
                    coeffs.append(cof)
                self.coeffs = np.array(coeffs)

        else:
            P = sparse.csc_matrix(Q_prob)
            A = sparse.csc_matrix(np.concatenate([C_prob, A_prob], axis=0))

            q = np.zeros(n_var)
            b = np.concatenate([d_prob, b_prob], axis=0)

            cones = [clarabel.ZeroConeT(C_prob.shape[0]), clarabel.NonnegativeConeT(A_prob.shape[0])]

            settings = clarabel.DefaultSettings()
            settings.verbose = False

            solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)

            sol = solver.solve()

            # Check solver status
            if str(sol.status) != 'Solved':
                print(f"Solver status: {sol.status}")
                #print(f"Number of iterations: {sol.iterations}")
                print('Clarabel did not solve the problem!')
                solver_success = False
                solution = None
                self.coeffs = None

            else:
                solver_success = True
                solution = np.array(sol.x)

                coeffs = []
                cof_splits = np.split(solution, N_sections)
                for cof_split in cof_splits:
                    xyz = np.split(cof_split, 3)
                    cof = np.stack(xyz, axis=0)
                    coeffs.append(cof)
                self.coeffs = np.array(coeffs)

        return self.coeffs, solver_success

    def eval_b_spline(self):
        T = self.time_pts['time_pts'].cpu().numpy()
        dT = self.time_pts['d_time_pts'].cpu().numpy()
        ddT = self.time_pts['dd_time_pts'].cpu().numpy()
        dddT = self.time_pts['ddd_time_pts'].cpu().numpy()
        ddddT = self.time_pts['dddd_time_pts'].cpu().numpy()

        if self.coeffs is not None:
            full_traj = []
            for i, coeff in enumerate(self.coeffs):
                if i < len(self.coeffs) - 1:
                    pos = (coeff @ T[:, :-1]).T
                    vel = (coeff @ dT[:, :-1]).T
                    acc = (coeff @ ddT[:, :-1]).T
                    jerk = (coeff @ dddT[:, :-1]).T
                else:
                    pos = (coeff @ T).T
                    vel = (coeff @ dT).T
                    acc = (coeff @ ddT).T
                    jerk = (coeff @ dddT).T

                sub_traj = np.concatenate([pos, vel, acc, jerk], axis=-1)
                full_traj.append(sub_traj)

            return np.concatenate(full_traj, axis=0)
        
        else:
            return None