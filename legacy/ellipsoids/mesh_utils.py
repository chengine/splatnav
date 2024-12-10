import numpy as np
import open3d as o3d

def create_gs_mesh(means, rotations, scalings, colors, res=4, transform=None, scale=None):
    scene = o3d.geometry.TriangleMesh()

    # Nerfstudio performs the transform first, then does the scaling
    if scale is not None:
        means = means * scale
        scalings = scalings * scale

    if transform is not None:
        rot = transform[:3, :3]
        t = transform[:3, -1]

        means = np.matmul(rot, means[..., None]).squeeze() + t

        rotations = np.matmul(rot, rotations)

    for i, (mean, R, S, col) in enumerate(zip(means, rotations, scalings, colors)):
        one_gs_mesh = o3d.geometry.TriangleMesh.create_sphere(resolution=res)
        points = np.asarray(one_gs_mesh.vertices)
        new_points = points * S[None]
        one_gs_mesh.vertices = o3d.utility.Vector3dVector(new_points)
        one_gs_mesh = one_gs_mesh.paint_uniform_color(col)
        one_gs_mesh = one_gs_mesh.rotate(R)
        one_gs_mesh = one_gs_mesh.translate(mean)
        scene += one_gs_mesh

    return scene