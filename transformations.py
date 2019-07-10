import numpy as np
'''
    Functions to manipulate coordinate data with translational and rotational transformations, and cart/pol conversions

'''


def vectorNorm(vec, axis=None):
    ''' Returns vector norm. Singleton in dimension along which the norm was made, so that numpy math can proceed'''
    vecShape = len(vec.shape)
    # do along last axis if none is given
    if not axis:
        axis = vecShape - 1

    if axis >= vecShape:
        raise IndexError("Attempted to sum along axis {}, but the vector has shape {}".format(axis, vecShape))
    return np.expand_dims(np.sqrt((vec ** 2).sum(axis=axis)), axis)


def angleFromVectors(v1, v2):
    axis = len(v1.shape) - 1
    nv1 = v1 / vectorNorm(v1)
    nv2 = v2 / vectorNorm(v2)
    return np.arccos(np.clip((nv1 * nv2).sum(axis=axis), -1.0, 1.0))


def rotate(xyz, rotation_angles):
    ''' Rotational transform of coordinate system about origin. Angles are in radians
    '''

    r_x = np.array([[1,                       0,                              0],
                    [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
                    [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]])

    r_y = np.array([[np.cos(rotation_angles[1]) , 0, np.sin(rotation_angles[1])],
                    [0                       ,    1,                        0  ],
                    [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]])

    r_z = np.array([[np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
                    [np.sin(rotation_angles[2]),  np.cos(rotation_angles[2]), 0],
                    [0,                       0,                              1]])

    rotmat = r_x.dot(r_y).dot(r_z)       # net rotation matrix
    return np.dot(xyz, rotmat)   # do rotation


def pol2cart(theta, rho, z):
    ''' Converts polar to cartesian coordinates, outputs as nparts * xyz array
    '''
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    cart_coords = np.stack((x, y, z), axis=2)
    return cart_coords


def center_on_rings(traj, index_dict):
    ring_indices = index_dict['ring1'] + index_dict['ring2']
    traj.xyz[:, :, :2]  -= traj.xyz[:, ring_indices, :2].mean(axis=1)[:, np.newaxis, :]

    ring_vec = traj.xyz[:, index_dict['ring2'], :2].mean(axis=1) - traj.xyz[:, index_dict['ring1'], :2].mean(axis=1)
    ring_vec /= np.sqrt((ring_vec ** 2).sum(axis=1))[:, np.newaxis]
    rot_angles = angleFromVectors(ring_vec, np.array((1, 0)))

    for i in range(traj.xyz.shape[0]):
        traj.xyz[i, :, :] = rotate(traj.xyz[i, :, :], [0, 0, rot_angles[i]])
