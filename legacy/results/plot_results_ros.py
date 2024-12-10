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

fig, ax = plt.subplots(2, 2, figsize=(10, 7), dpi=500)

linewidth=3
font = {
        'family': 'Arial',
        'weight' : 'bold',
        'size'   : 13}

matplotlib.rc('font', **font)

for j, method in enumerate(methods):

    success = []
    safety = []
    polytope_safety = []
    times = []
    polytope_vols = []
    polytope_radii = []
    num_facets = []
    ros_safety_margin = []
    for k in range(4):

        read_path = f'traj/{method}_{k}_processed.json'
        save_fp = os.path.join(str(Path(os.getcwd()).parent.absolute()), read_path)

        with open(save_fp, 'r') as f:
            meta = json.load(f)

        datas = meta['total_data']
        ros_data = meta['ros_data']

        if method == 'open-loop':
            col = '#4285F4'
        else:
            col = '#34A853'

        if k == 0:
            cutoff = 30
        elif k == 1:
            cutoff = 16
        elif k == 2:
            cutoff = 0
        elif k == 3:
            cutoff = 60

        # Per trajectory
        for i, data in enumerate(datas):

            # If the trajectory wasn't feasible to solve, we don't store any data on it besides the success.
            if not data['feasible']:
                success.append(False)
                continue

            else:
                success.append(True)

            num_polytopes = data['num_polytopes']

            # record the times
            traj_time = np.array([data['times_astar'], data['times_collision_set'],
                                data['times_polytope'], data['times_opt']])
            times.append(traj_time)
            
            # record the min safety margin
            safety.append(np.array(data['safety_margin']).min())
            polytope_safety.append(np.array(data['polytope_margin']).min())

            # record the polytope stats (min/max/mean/std)
            polytope_vols_entry = np.array(data['polytope_vols'])
            polytope_radii_entry = np.array(data['polytope_radii'])

            polytope_vols.append([polytope_vols_entry.min(), polytope_vols_entry.max(), polytope_vols_entry.mean(), polytope_vols_entry.std()])
            polytope_radii.append([polytope_radii_entry.min(), polytope_radii_entry.max(), polytope_radii_entry.mean(), polytope_radii_entry.std()])

            polytope_length = np.array([len(np.array(poly)) for poly in data['polytopes']]).mean()

            num_facets.append(polytope_length)

        if ros_data['safety_margin'] is not None:
            if method == 'closed-loop':
                safety_margin = np.array(ros_data['safety_margin'])[:cutoff]
                ros_safety_margin.append(safety_margin)
            else:
                traj = np.array(ros_data['poses'])[..., :3,-1]
                mask = traj[..., -1] < -0.4
                safety_margin = np.array(ros_data['safety_margin'])[mask]
                ros_safety_margin.append(safety_margin)

    success = np.array(success)
    safety = np.array(safety)
    polytope_safety = np.array(polytope_safety)
    times = np.array(times)
    polytope_vols = np.array(polytope_vols)
    polytope_radii = np.array(polytope_radii)
    ros_safety_margin = np.concatenate(ros_safety_margin)

    print(f'{method} Trial {k}', np.array(num_facets).mean(), 'Succeeded:', success.sum()/len(success))

    try:
        # Computation Time

        # TODO: This plots the individual times of each component of the algorithm
        # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, qp_solve_time.mean(), bottom = 0, width=0.15, color= adjust_lightness(col, 0.5), linewidth=3, ec='k', label='qp')
        # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, cbf_solve_time.mean(), bottom=qp_solve_time.mean(), width=0.15, color=adjust_lightness(col, 1.0), linewidth=3, ec='k', label='cbf')
        # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, prune_time.mean(), bottom=cbf_solve_time.mean() + qp_solve_time.mean(), width=0.15, color = adjust_lightness(col, 1.), linewidth=3, hatch='x', ec='k', label='prune')
        # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, prune_time.mean() + cbf_solve_time.mean() + qp_solve_time.mean(), width=0.15, color = adjust_lightness(col, 1.), linewidth=3,  ec='k', label='prune')

        # TODO: This plots just the total time
        # ax[0, 0].bar(j/len(methods), times.sum(axis=1).mean(), width=0.25, color=col, capsize=10, edgecolor='black', linewidth=linewidth, 
        #             linestyle='-', joinstyle='round', rasterized=True)
        ax[0, 0].scatter(j/len(methods), times.sum(axis=-1).mean(), s=200, color='k', alpha=1, marker='x')
        violinplot = ax[0, 0].violinplot(times.sum(axis=-1), positions=[j/len(methods)], widths=0.25, showmeans=False, showextrema=False, showmedians=False)

        for pc in violinplot['bodies']:
            # pc.set_facecolor(col)
            # pc.set_edgecolor('black')
            # pc.set_alpha(1)
            pc.set_color(col)
            pc.set_alpha(0.8)

        # Safety Margin
        # For trajectory points
        # errors = np.abs(safety.mean().reshape(-1, 1) - np.array([safety.min(), safety.max()]).reshape(-1, 1))

        # ax[0, 1].errorbar(k + 0.75*j/len(methods) + 0.25/2, safety.mean().reshape(-1, 1), yerr=errors, color=adjust_lightness(col, 0.5), markeredgewidth=5, capsize=15, elinewidth=5, alpha=0.5)
        # ax[0, 1].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(safety)), safety, s=250, color=col, alpha=0.04)
        # ax[0, 1].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.13 + l/10, safety.mean(), s=200, color=col, alpha=1, marker='>')
        ax[0, 1].scatter(j/len(methods), safety.mean(), s=200, color='k', alpha=1, marker='x')
        violinplot = ax[0, 1].violinplot(safety, positions=[j/len(methods)], widths=0.25, showmeans=False, showextrema=False, showmedians=False)

        for pc in violinplot['bodies']:
            # pc.set_facecolor(col)
            # pc.set_edgecolor('black')
            # pc.set_alpha(1)
            pc.set_color(col)
            pc.set_alpha(0.8)

        ax[0, 1].axhline(y = ros_safety_margin.min(), color = col, linestyle = '--', linewidth=3, alpha=0.7, zorder=0) 

        # For polytope vertices
        # errors = np.abs(polytope_safety.mean().reshape(-1, 1) - np.array([polytope_safety.min(), polytope_safety.max()]).reshape(-1, 1))

        # ax[1, 0].errorbar(k + 0.75*j/len(methods) + 0.25/2, polytope_safety.mean().reshape(-1, 1), yerr=errors, color=adjust_lightness(col, 0.5), markeredgewidth=5, capsize=15, elinewidth=5, alpha=0.5)
        # ax[1, 0].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(polytope_safety)), polytope_safety, s=250, color=col, alpha=0.04)
        # ax[1, 0].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.13, polytope_safety.mean(), s=200, color=col, alpha=1, marker='>')

        ax[1, 0].scatter(j/len(methods), polytope_safety.mean(), s=200, color='k', alpha=1, marker='x')
        violinplot = ax[1, 0].violinplot(polytope_safety, positions=[j/len(methods)], widths=0.25, showmeans=False, showextrema=False, showmedians=False)

        for pc in violinplot['bodies']:
            # pc.set_facecolor(col)
            # pc.set_edgecolor('black')
            # pc.set_alpha(1)
            pc.set_color(col)
            pc.set_alpha(0.8)

        # # Polytope Volume
        # errors = np.abs(polytope_vols[:, 2].mean().reshape(-1, 1) - np.array([polytope_vols[:, 0].min(), polytope_vols[:, 1].max()]).reshape(-1, 1))

        # ax[1, 0].errorbar(k + 0.75*j/len(methods) + 0.25/2, polytope_vols[:, 2].mean().reshape(-1, 1), yerr=errors, color=adjust_lightness(col, 0.5), markeredgewidth=5, capsize=15, elinewidth=5, alpha=0.5)
        # ax[1, 0].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(polytope_vols[:, 2])), polytope_vols[:, 2], s=250, color=col, alpha=0.04)
        # ax[1, 0].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.13, polytope_vols[:, 2].mean(), s=200, color=col, alpha=1, marker='>')

        ax[1, 1].scatter(j/len(methods), polytope_vols[:, 2].mean(), s=200, color='k', alpha=1, marker='x')
        violinplot = ax[1, 1].violinplot(polytope_vols[:, 2], positions=[j/len(methods)], widths=0.25, showmeans=False, showextrema=False, showmedians=False)

        for pc in violinplot['bodies']:
            # pc.set_facecolor(col)
            # pc.set_edgecolor('black')
            # pc.set_alpha(1)
            pc.set_color(col)
            pc.set_alpha(0.8)

        # # Polytope Radii
        # errors = np.abs(polytope_radii[:, 2].mean().reshape(-1, 1) - np.array([polytope_radii[:, 0].min(), polytope_radii[:, 1].max()]).reshape(-1, 1))

        # ax[1, 1].errorbar(k + 0.75*j/len(methods) + 0.25/2, polytope_radii[:, 2].mean().reshape(-1, 1), yerr=errors, color=adjust_lightness(col, 0.5), markeredgewidth=5, capsize=15, elinewidth=5, alpha=0.5)
        # ax[1, 1].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(polytope_radii[:, 2])), polytope_radii[:, 2], s=250, color=col, alpha=0.04)
        # ax[1, 1].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.13, polytope_radii[:, 2].mean(), s=200, color=col, alpha=1, marker='>')

        # Eccentricity
        # errors = np.abs(eccentricity[:, 2].mean().reshape(-1, 1) - np.array([eccentricity[:, 0].min(), eccentricity[:, 1].max()]).reshape(-1, 1))
        
        # ax[1, 1].errorbar(k + 0.75*j/len(methods) + 0.25/2, eccentricity[:, 2].mean().reshape(-1, 1), yerr=errors, color=adjust_lightness(col, 0.5), markeredgewidth=5, capsize=15, elinewidth=5, alpha=0.5)
        # ax[1, 1].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(eccentricity[:, 2])), eccentricity[:, 2], s=250, color=col, alpha=0.04)
        # ax[1, 1].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.13, eccentricity[:, 2].mean(), s=200, color=col, alpha=1, marker='>')

        # ax[1, 1].scatter(k +  + 0.75*j/len(methods) + 0.25/2 + l/10, eccentricity[:, 2].mean(), s=200, color='k', alpha=1, marker='x')
        # violinplot = ax[1, 1].violinplot(eccentricity[:, 2], positions=[k + 0.75*j/len(methods) + 0.25/2 + l/10], widths=0.1, showmeans=False, showextrema=False, showmedians=False)

        # for pc in violinplot['bodies']:
        #     # pc.set_facecolor(col)
        #     # pc.set_edgecolor('black')
        #     # pc.set_alpha(1)
        #     pc.set_color(col)
        #     pc.set_alpha(0.8)

    except Exception as e:
        print(f'{method} Trial {k} failed')
        print(e)
        pass

# COMPUTATION TIME
ax[0, 0].set_title('Computation Time (s)' , fontsize=20, fontweight='bold')
ax[0, 0].get_xaxis().set_visible(False)
ax[0, 0].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.25, zorder=0)
ax[0, 0].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[0, 0].spines[location].set_linewidth(4)
# ax[0,0].set_yscale('log')

# SAFETY MARGIN
ax[0, 1].set_title('Min. Distance (Traj)', fontsize=20, fontweight='bold')
ax[0, 1].get_xaxis().set_visible(False)
ax[0, 1].axhline(y = 0., color = 'k', linestyle = '--', linewidth=3, alpha=0.7, zorder=0) 
ax[0, 1].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.25, zorder=0)
ax[0, 1].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[0, 1].spines[location].set_linewidth(4)
#ax[0, 1].set_ylim(-0.0015, 0.00015)

# # POLYTOPE VOLUME
# ax[1, 0].set_title(r'Polytope Volume $\uparrow$', fontsize=25, fontweight='bold')
# ax[1, 0].get_xaxis().set_visible(False)
# ax[1, 0].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
# ax[1, 0].set_axisbelow(True)
# for location in ['left', 'right', 'top', 'bottom']:
#     ax[1, 0].spines[location].set_linewidth(4)

# # POLYTOPE RADII
# ax[1, 1].set_title(r'Polytope Radius $\uparrow$', fontsize=25, fontweight='bold')
# ax[1, 1].get_xaxis().set_visible(False)
# ax[1, 1].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
# ax[1, 1].set_axisbelow(True)
# for location in ['left', 'right', 'top', 'bottom']:
#     ax[1, 1].spines[location].set_linewidth(4)

# POLYTOPE VOLUME
ax[1, 0].set_title('Min. Distance (Vertices)', fontsize=20, fontweight='bold')
ax[1, 0].get_xaxis().set_visible(False)
ax[1, 0].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.25, zorder=0)
ax[1, 0].axhline(y = 0., color = 'k', linestyle = '--', linewidth=3, alpha=0.7, zorder=0) 
ax[1, 0].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[1, 0].spines[location].set_linewidth(4)

# POLYTOPE Volume
ax[1, 1].set_title(r'Polytope Volume', fontsize=20, fontweight='bold')
ax[1, 1].get_xaxis().set_visible(False)
ax[1, 1].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.25, zorder=0)
ax[1, 1].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[1, 1].spines[location].set_linewidth(4)

plt.savefig(f'ros_stats.pdf', dpi=1000)

#%%