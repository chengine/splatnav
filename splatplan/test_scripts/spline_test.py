#%%
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from spline_utils import BezierCurve

num_control_points = 30

# Straight line
control_points = torch.linspace(0, 1, num_control_points).reshape(-1, 1) * torch.ones(2).reshape(1, -1)
# Add some noise
control_points += torch.randn(num_control_points, 2) * 0.1

control_points = control_points.reshape(5, 6, -1)

control_points[[0, 1, 2, 3], -1] = control_points[[1, 2, 3, 4], 0]

bcurve = BezierCurve(control_points.shape[1], 3, 2)

tnow = time.time()
torch.cuda.synchronize()
bcurve.set_control_points(control_points.cuda(), time_scale=torch.linspace(1., 2., control_points.shape[0]).cuda())
output = bcurve.evaluate(torch.linspace(0, 1, 100).cuda())
torch.cuda.synchronize()
print('Elapsed', time.time() - tnow)

# %%

fig, ax = plt.subplots()
fps = 2

# Plots poly spline
for batch in range(output.shape[1]):
    points = output[:, batch, 0, :].cpu().numpy()

    ax.plot(points[:, 0], points[:, 1])

    ax.scatter(control_points[batch, :, 0].cpu().numpy(), control_points[batch, :, 1].cpu().numpy())

    # # Plot velocities
    # velocities = output[:, batch, 1, :].cpu().numpy()

    # for i in range(velocities.shape[0]):
    #     if i % 5 == 0:
    #         #plt_points = np.stack([points[i] - velocities[i]/ 50, points[i] + velocities[i]/ 50], axis=0)
    #         plt_points = np.stack([points[i], points[i] + velocities[i]/ 50], axis=0)
    #         ax.plot(plt_points[:, 0], plt_points[:, 1], color='b')

# Dynamically update the plot
# for i, t in enumerate(torch.linspace(0., 10., 100)):
#     out = bcurve.evaluate_at_t(t.item()).cpu().numpy()

#     if i == 0:
#         scat = ax.scatter(out[0, 0], out[0, 1], color='black')
#         quiv = ax.quiver(out[0, 0], out[0, 1], out[1, 0], out[1, 1], color='black')
#         quiv_acc = ax.quiver(out[0, 0], out[0, 1], out[2, 0], out[2, 1], color='red', alpha=0.5, linestyle='dashed')
#     else:
#         scat.set_offsets(out[0])

#         quiv.set_offsets(out[0])
#         quiv.set_UVC(5*out[1, 0], 5*out[1, 1])

#         quiv_acc.set_offsets(out[0])
#         quiv_acc.set_UVC(5*out[2, 0], 5*out[2, 1])

#     # Redraw the figure
#     plt.draw()
#     plt.pause(0.2)

out = bcurve.evaluate_at_t(0.).cpu().numpy()

scat = ax.scatter(out[0, 0], out[0, 1], color='black')
quiv = ax.quiver(out[0, 0], out[0, 1], out[1, 0], out[1, 1], color='black', angles='xy', scale_units='xy', scale=1)
quiv_acc = ax.quiver(out[0, 0], out[0, 1], out[2, 0], out[2, 1], color='red', alpha=0.5, linestyle='dashed', angles='xy', scale_units='xy', scale=1)

ts = torch.linspace(0., 10., 100)
def update(frame):
    out = bcurve.evaluate_at_t(ts[frame].item()).cpu().numpy()

    scat.set_offsets(out[0])

    quiv.set_offsets(out[0])
    quiv.set_UVC(out[1, 0], out[1, 1])

    quiv_acc.set_offsets(out[0])
    quiv_acc.set_UVC(out[2, 0], out[2, 1])

    return (scat, quiv, quiv_acc)

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(ts), interval=1./fps)
ani.save(filename="pillow_example.gif", writer="pillow")

# %%