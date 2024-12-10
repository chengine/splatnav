import torch

def quaternion_to_angle_axis(quaternion, eps=1e-6):
    """Convert quaternion vector to angle axis of rotation

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (Tensor): batch with quaternions

    Return:
        Tensor: batch with angle axis of rotation

    Shape:
        - Input: :math:`(N, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 4)  # Nx4
        >>> output = tgm.quaternion_to_angle_axis(input)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    input_shape = quaternion.shape
    if len(input_shape) == 1:
        quaternion = torch.unsqueeze(quaternion, dim=0)

    assert quaternion.size(1) == 4, 'Input must be a vector of length 4'
    normalizer = 1 / torch.norm(quaternion, dim=1)
    q1 = quaternion[:, 1] * normalizer
    q2 = quaternion[:, 2] * normalizer
    q3 = quaternion[:, 3] * normalizer

    sin_squared = q1 * q1 + q2 * q2 + q3 * q3
    mask = (sin_squared > eps).to(sin_squared.device)
    mask_pos = (mask).type_as(sin_squared)
    mask_neg = (mask == False).type_as(sin_squared)  # noqa
    batch_size = quaternion.size(0)
    angle_axis = torch.zeros(
        batch_size, 3, dtype=quaternion.dtype).to(
        quaternion.device)

    sin_theta = torch.sqrt(sin_squared)
    cos_theta = quaternion[:, 0] * normalizer
    mask_theta = (cos_theta < eps).view(1, -1)
    mask_theta_neg = (mask_theta).type_as(cos_theta)
    mask_theta_pos = (mask_theta == False).type_as(cos_theta)  # noqa

    theta = torch.atan2(-sin_theta, -cos_theta) * mask_theta_neg \
        + torch.atan2(sin_theta, cos_theta) * mask_theta_pos

    two_theta = 2 * theta
    k_pos = two_theta / sin_theta
    k_neg = 2.0
    k = k_neg * mask_neg + k_pos * mask_pos

    angle_axis[:, 0] = q1 * k
    angle_axis[:, 1] = q2 * k
    angle_axis[:, 2] = q3 * k

    if len(input_shape) == 1:
        angle_axis = angle_axis.squeeze(0)

    return angle_axis

def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4

def quaternion_to_rotation_matrix(quat):
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdims=True)
    return angle_axis_to_rotation_matrix(quaternion_to_angle_axis(quat))[..., :3, :3]

def scaling_to_mat(scaling, exp=False):
    # NOTE: SCALING FROM GS REQUIRES EXPONENTIATING
    if exp:
        return torch.diag_embed(torch.exp(scaling), 0)
    else:
        return torch.diag_embed(scaling, 0)

def compute_cov(quat, scaling, exp=False):
    R = quaternion_to_rotation_matrix(quat)[..., :3, :3]
    S = scaling_to_mat(scaling, exp=exp)

    M = torch.bmm(R, S)
    return torch.bmm(M, M.transpose(-2, -1))
