import struct
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2

point_struct = struct.Struct("<fffBBBB")

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