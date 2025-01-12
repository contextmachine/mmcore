from mmcore.numeric.vectors import scalar_cross


def bounding_planes(a,b,c,d):
    return scalar_cross(c-a,d-b)