import numpy as np


def camera_top1_to_world(path_xy, camera_position_xyz, camera_rotation_3x3):
    path = np.asarray(path_xy, float); position = np.asarray(camera_position_xyz, float); rotation = np.asarray(camera_rotation_3x3, float)
    local = np.column_stack([path[:, :2], np.zeros(len(path))])
    return (position.reshape(1,3) + (rotation @ local.T).T)[:, :2]
