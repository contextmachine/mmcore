from collections import namedtuple



ClosestPointSolution1D = namedtuple('ClosestPointSolution1D', ['point', 'distance', 'bounded', 't'])
ClosestPointSolution2D = namedtuple('ClosestPointSolution2D', ['point', 'distance', 'bounded', 'u', 'v'])
ClosestPointSolution3D = namedtuple('ClosestPointSolution3D', ['point', 'distance', 'bounded', 'u', 'v', 'w'])

