import struct
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2

point_struct = struct.Struct("<fffBBBB")

opengl_to_opencv = np.array([ 
    [1., 0., 0., 0.],
    [0., -1., 0., 0.],
    [0., 0., -1., 0.],
    [0., 0., 0., 1.]
])

def state_to_waypoint(position, velocity, acceleration, jerk, yaw):
    output = np.zeros(13)
    output[0:3] = position
    output[3:6] = velocity
    output[6:9] = acceleration
    output[9:12] = jerk
    output[-1] = yaw
    return output

def make_point_cloud(points, colors, frame_id, fields, stamp=1e-3):
    buffer = bytearray(point_struct.size * len(points))
    for i, (point, color) in enumerate(zip(points, colors)):
        # r, g, b, a = color(point, t)
        point_struct.pack_into(
            buffer,
            i * point_struct.size,
            point[0],
            point[1],
            point[2],
            color[2],
            color[1],
            color[0],
            255,
        )

    return PointCloud2(
        header=Header(frame_id=frame_id),
        height=1,
        width=len(points),
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=point_struct.size,
        row_step=len(buffer),
        data=buffer,
    )


def record_bag(bag_name, topics):
    """Records a rosbag file."""
    command = ["ros2", "bag", "record", "-o", bag_name] + topics
    process = subprocess.Popen(command)

    return process

def get_yaw_from_rotation_matrix(R):
    """
    Extracts the yaw angle from a 3x3 rotation matrix.

    Args:
        R (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        float: Yaw angle in radians.
    """

    yaw = np.arctan2(R[1, 0], R[0, 0])
    return yaw

def yaw_to_rotation_matrix(yaw):
    """Converts a yaw angle to a 3x3 rotation matrix.

    Args:
        yaw: The yaw angle in radians.

    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """

    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

def yaw_from_heading_vector(vector):
    """Computes the yaw angle from a 2D vector.

    Args:
        vector: A 2D vector.

    Returns:
        The yaw angle in radians.
    """

    return np.arctan2(vector[1], vector[0])

