import numpy as np
import polytope

def sample_sphere(N_samples):
    # Samples the unit sphere uniformly
    # reference: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    i = np.arange(N_samples)
    y = 1. - (i / (N_samples - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - y * y)  # radius at y

    theta = phi * i  # golden angle increment

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.stack([x, y, z], axis=-1)

def sphere_to_poly(N_samples, radius, center):

    sphere_v_rep = sample_sphere(N_samples)

    # multiply by radius
    sphere_v_rep *= radius

    # then shift by the center
    sphere_v_rep = sphere_v_rep + center[None,...]

    poly = polytope.qhull(sphere_v_rep)

    return poly, sphere_v_rep

def circle_to_poly(N_samples, radius, center):

    t = np.linspace(0., 2*np.pi, N_samples)
    circle_v_rep = np.stack([np.cos(t), np.sin(t)], axis=-1)
    circle_v_rep = radius * circle_v_rep + center[None,...]

    poly = polytope.qhull(circle_v_rep)

    return poly