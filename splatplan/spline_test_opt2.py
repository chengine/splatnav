#%%
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from opt_utils import SplineOptimizer

#%%

centers_x = torch.linspace(-0.5, 0.5, 5)
centers_y = centers_x**2
centers = torch.stack([centers_x, centers_y], dim=1).cuda()

polytopes = []
for i in range(len(centers)):
    A = torch.cat([torch.eye(2, device='cuda'), -torch.eye(2, device='cuda')], dim=0)
    b = torch.cat([centers[i] + 0.2 + 0.1*(torch.rand(2).cuda() - 0.5) , -centers[i] + 0.2 + 0.1*(torch.rand(2).cuda() - 0.5)], dim=0).cuda()
    polytopes.append((A, b))

x0 = torch.tensor([[-0.65, 0.4], [0.5, 0.]], device='cuda')
xf = torch.tensor([[0.37, 0.4], [0.0, 0.], [0., 0.], [0., 0.]], device='cuda')

# Spline optimizer
spline_optimizer = SplineOptimizer(6, 3, 2, 'cuda')

time_scales = torch.linspace(2., 1., len(polytopes)).cuda()

tnow = time.time()
torch.cuda.synchronize()
control_points, success = spline_optimizer.optimize_bspline(polytopes, x0, xf, time_scales)
torch.cuda.synchronize()
print('Elapsed', time.time() - tnow)
# %%

fig, ax = plt.subplots()
fps = 3

scat_cntrl = ax.scatter(control_points[0, :, 0].cpu().numpy(), control_points[0, :, 1].cpu().numpy(), s=100)

# Plots poly spline
for control_points_ in control_points:
    ax.scatter(control_points_[:, 0].cpu().numpy(), control_points_[:, 1].cpu().numpy())

# Draw rectangle around the centers
patch = []
for A, b in polytopes:
    top = b[:2]
    bottom = -b[2:]
    diff = top - bottom
    p = ax.add_patch(plt.Rectangle(bottom.cpu().numpy(), diff[0].item(), diff[1].item(), fill=None))
    patch.append(p)

out = spline_optimizer.evaluate_bspline_at_t(0.).cpu().numpy()

current_position = x0[0].cpu().numpy()
current = ax.scatter(current_position[0], current_position[1], color='red', marker='x', s=100)

real_traj, = ax.plot([current_position[0]], [current_position[1]], color='red', linestyle='dashed')

scat = ax.scatter(out[0, 0], out[0, 1], color='black')
quiv = ax.quiver(out[0, 0], out[0, 1], out[1, 0], out[1, 1], color='black', angles='xy', scale_units='xy', scale=1)
quiv_acc = ax.quiver(out[0, 0], out[0, 1], out[2, 0], out[2, 1], color='red', alpha=0.5, linestyle='dashed', angles='xy', scale_units='xy', scale=1)

title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

ts = torch.linspace(0., time_scales.sum(), int(time_scales.sum() * 30))
data = [out]
real = [current_position]

def update(frame):

    # Simulating noisy measurements
    x0 = data[-1]
    #x0[0] = x0[0] + np.sin(frame) * 0.05
    x0[0] = x0[0] + np.random.randn(2)* 0.05

    current_position = real[-1] + x0[1]*ts[1].item()

    real.append(current_position)

    tnow = time.time()
    torch.cuda.synchronize()
    out, success, meta = spline_optimizer.solve_local_waypoint(ts[frame+1].item(), ts[frame].item(), torch.tensor(x0, device='cuda'))
    torch.cuda.synchronize()
    print('Elapsed', time.time() - tnow)

    # set color of rectangle patch to green if the spline is inside the polytope
    for i, p in enumerate(patch):
        if i == meta['spline_ind']:
            p.set_edgecolor('green')
        else:
            p.set_edgecolor('red')

    if success:
        print('Cost: ', meta['cost'])
        print('Success')
        
        out = out.cpu().numpy()
        data.append(out)

        scat.set_offsets(out[0])
        scat_cntrl.set_offsets(meta['control_points'].cpu().numpy().reshape(-1, 2))

        quiv.set_offsets(out[0])
        quiv.set_UVC(out[1, 0], out[1, 1])

        quiv_acc.set_offsets(out[0])
        quiv_acc.set_UVC(out[2, 0], out[2, 1])

        title.set_text(f"Time: {ts[frame].item():.2f}")

        current.set_offsets(current_position)

        real_traj.set_data([r[0] for r in real], [r[1] for r in real])

    return (scat, quiv, quiv_acc, title, current, scat_cntrl, real_traj)

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(ts)-1, interval=1./fps)
ani.save(filename="pillow_example.gif", writer="pillow")

# %%