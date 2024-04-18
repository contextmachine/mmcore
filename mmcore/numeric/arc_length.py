import numpy as np
import math



def curvature_based_step(tolerance, curvature_radius):
    return 2 * np.sqrt(2 * curvature_radius * tolerance - tolerance ** 2)


def arc_height(chord_length, curvature_radius):
    """
    ___
    :param chord_length:
    :param curvature_radius:
    :return:
    """
    return curvature_radius - math.sqrt((curvature_radius - chord_length/2)*(curvature_radius +  chord_length/2))

