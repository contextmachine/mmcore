from multipledispatch import dispatch

from mmcore.base.ecs.components import apply, component, todict

import numpy as np


@component()
class Controls:
    ...


CONTROL_POINTS_ATTRIBUTE_NAME = "points"


def controls_to_dict(controls):
    return todict(controls)


def find_points_in_controls(controls):

    if CONTROL_POINTS_ATTRIBUTE_NAME in controls:
        return controls[CONTROL_POINTS_ATTRIBUTE_NAME]
    elif 'path' in controls:
        return find_points_in_controls(controls['path'])
    else:
        return


def set_points_in_controls(controls, points: dict):
    controls['path'] = {CONTROL_POINTS_ATTRIBUTE_NAME: points}


def decode_control_points(control_points: dict):
    return [[pt['x'], pt['y'], pt['z']] for pt in control_points.values()]


def encode_control_points(control_points: 'np.ndarray| list| tuple'):

    return {f'pt{i}': dict(x=pt[0], y=pt[1], z=pt[2]) for i, pt in enumerate(control_points)}
