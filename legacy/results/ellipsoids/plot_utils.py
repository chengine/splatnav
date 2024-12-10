import torch
from matplotlib.patches import Ellipse
import numpy as np

# Plotting function for ellipsoids
def plot_ellipse(mu, Sigma, n_std_tau, ax, **kwargs):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/plot_ellipse.ipynb

    ee, V = torch.linalg.eigh(Sigma)
    e_big = ee[1]
    e_small = ee[0]
    v_big = V[:, 1]
    theta = torch.arctan2(v_big[1] , v_big[0]) * 180. / np.pi

    long_length = n_std_tau * 2. * torch.sqrt(e_big)
    short_length = n_std_tau * 2. * torch.sqrt(e_small)

    if not ('facecolor' in kwargs):
        kwargs['facecolor'] = 'grey'
    if not ('edgecolor' in kwargs):
        kwargs['edgecolor'] = 'k'

    ellipse = Ellipse(mu, width=long_length, height=short_length, angle=theta, **kwargs)
    ax.add_artist(ellipse)
