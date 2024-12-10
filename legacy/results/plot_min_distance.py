#%%
import os
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

methods = ['open-loop', 'closed-loop']

fig, ax = plt.subplots(1, figsize=(20, 10), dpi=500)

font = {
        'family': 'Arial',
        'weight' : 'bold',
        'size'   : 25}

matplotlib.rc('font', **font)

for j, method in enumerate(methods):
    for k in range(4):

        read_path = f'traj/{method}_{k}_processed.json'
        save_fp = os.path.join(str(Path(os.getcwd()).parent.absolute()), read_path)

        with open(save_fp, 'r') as f:
            meta = json.load(f)

        ros_data = meta['ros_data']

        if method == 'open-loop':
            linestyle = '--'
            alpha=0.6

        else:
            linestyle = '-'
            alpha=1.0

        if k == 0:
            col = '#EA4335'
            cutoff = 30
        elif k == 1:
            col = '#FBBc05'
            cutoff = 16
        elif k == 2:
            col = '#4285F4'
            cutoff = 0
        elif k == 3:
            col = '#34A853'
            cutoff = 60
        else:
            raise ValueError('Invalid trial number')

        try:
            safety_margin = np.array(ros_data['safety_margin'])

            traj = np.array(ros_data['poses'])[..., :3,-1]

            # time_steps = np.array(ros_data['poses_timestamps'])
            # time_steps = time_steps - time_steps[0]
            # time_steps = time_steps / 1e9
            time_steps = np.arange(len(safety_margin)) / len(safety_margin)

            if method == 'closed-loop':
                margin = safety_margin[:cutoff]
                time_steps = time_steps[:cutoff]
                linewidth = 6
            else:
                mask = traj[..., -1] < -0.4
                margin = safety_margin[mask]
                time_steps = time_steps[mask]
                linewidth = 4

            time_steps = time_steps / time_steps[-1]
            print(f'{method} Trial {k}')
            ax.plot(time_steps, margin, linewidth=linewidth, color=col, linestyle=linestyle, alpha=alpha)
        except Exception as e:
            print(f'{method} Trial {k} failed')
            print(e)
            pass

# SAFETY MARGIN
ax.set_title('Distance to GSplat', fontsize=40, fontweight='bold')
# ax.get_xaxis().set_visible(False)
ax.set_xlabel('Progress', fontsize=40, fontweight='bold')
# ax.set_ylabel('Distance to GSplat', fontsize=40, fontweight='bold')
ax.axhline(y = 0., color = 'k', linestyle = '--', linewidth=3, alpha=0.7, zorder=0) 
ax.grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.25, zorder=0)
# ax.set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax.spines[location].set_linewidth(4)

plt.savefig(f'ros_traj.pdf', dpi=1000)

#%%