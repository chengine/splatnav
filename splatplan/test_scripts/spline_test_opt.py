#%%
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from spline_utils import BezierCurve
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

x0 = torch.tensor([[-0.65, 0.4], [0.5, 0.], [0., 0.], [0., 0.]], device='cuda')
xf = torch.tensor([[0.37, 0.4], [-0.25, 0.], [0., 0.], [0., 0.]], device='cuda')

# Spline optimizer
spline_optimizer = SplineOptimizer(10, 3, 2, 'cuda')

time_scales = torch.linspace(2., 1., len(polytopes)).cuda()

tnow = time.time()
torch.cuda.synchronize()
control_points, success = spline_optimizer.optimize_bspline(polytopes, x0, xf, time_scales)
torch.cuda.synchronize()
print('Elapsed', time.time() - tnow)
# %%

fig, ax = plt.subplots()
fps = 3

# Plots poly spline
for control_points_ in control_points:
    ax.scatter(control_points_[:, 0].cpu().numpy(), control_points_[:, 1].cpu().numpy())

# Draw rectangle around the centers
for A, b in polytopes:
    top = b[:2]
    bottom = -b[2:]
    diff = top - bottom
    ax.add_patch(plt.Rectangle(bottom.cpu().numpy(), diff[0].item(), diff[1].item(), fill=None))

out = spline_optimizer.evaluate_bspline(0.).cpu().numpy()

scat = ax.scatter(out[0, 0], out[0, 1], color='black')
quiv = ax.quiver(out[0, 0], out[0, 1], out[1, 0], out[1, 1], color='black', angles='xy', scale_units='xy', scale=1)
quiv_acc = ax.quiver(out[0, 0], out[0, 1], out[2, 0], out[2, 1], color='red', alpha=0.5, linestyle='dashed', angles='xy', scale_units='xy', scale=1)

title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

ts = torch.linspace(0., time_scales.sum(), 100)
data = []
def update(frame):
    out = spline_optimizer.evaluate_bspline(ts[frame].item()).cpu().numpy()
    data.append(out)

    scat.set_offsets(out[0])

    quiv.set_offsets(out[0])
    quiv.set_UVC(out[1, 0], out[1, 1])

    quiv_acc.set_offsets(out[0])
    quiv_acc.set_UVC(out[2, 0], out[2, 1])

    title.set_text(f"Time: {ts[frame].item():.2f}")

    return (scat, quiv, quiv_acc, title)

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(ts), interval=1./fps)
ani.save(filename="pillow_example.gif", writer="pillow")

# %%