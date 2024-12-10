import numpy as np
import torch
from matplotlib.patches import Ellipse, Polygon
from polytope import extreme, cheby_ball, bounding_box

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


def get_patch(poly1, color="green"):
    """Takes a Polytope and returns a Matplotlib Patch Polytope 
    that can be added to a plot
    
    Example::

    > # Plot Polytope objects poly1 and poly2 in the same plot
    > import matplotlib.pyplot as plt
    > fig = plt.figure()
    > ax = fig.add_subplot(111)
    > p1 = get_patch(poly1, color="blue")
    > p2 = get_patch(poly2, color="yellow")
    > ax.add_patch(p1)
    > ax.add_patch(p2)
    > ax.set_xlim(xl, xu) # Optional: set axis max/min
    > ax.set_ylim(yl, yu) 
    > plt.show()
    """
    V = extreme(poly1)
    rc,xc = cheby_ball(poly1)
    x = V[:,1] - xc[1]
    y = V[:,0] - xc[0]
    mult = np.sqrt(x**2 + y**2)
    x = x/mult
    angle = np.arccos(x)
    corr = np.ones(y.size) - 2*(y < 0)
    angle = angle*corr
    ind = np.argsort(angle) 

    patch = Polygon(V[ind,:], True, color=color, alpha=0.1)
    return patch

def plot_polytope(poly1, ax, color='green'):
    """Plots a 2D polytope or a region using matplotlib.
    
    Input:
    - `poly1`: Polytope or Region
    """
    if len(poly1) == 0:

        poly = get_patch(poly1, color)
        l,u = bounding_box(poly1)
        ax.add_patch(poly)        

    else:
        l,u = bounding_box(poly1, color)

        for poly2 in poly1.list_poly:
            poly = get_patch(poly2, color=np.random.rand(3))
            ax.add_patch(poly)
    return ax

def plot_halfplanes(A, b, lower, upper, ax):
    # In A x <= b form in 2D

    t = np.linspace(lower, upper, endpoint=True)

    y = (-A[:, 0][:, None]*t[None,...] + b[:, None]) / A[:, 1][:, None]

    for y_ in y:
        ax.plot(t, y_, linestyle='dotted')

    return ax