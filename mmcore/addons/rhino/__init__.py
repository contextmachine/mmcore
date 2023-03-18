# Native imports
import rpyc, sys
from mmcore.addons import ModuleResolver

with ModuleResolver() as rsl:
    import rhino3dm


import rhino3dm


print(rhino3dm.Point3d(1, 2, 3))


from mmcore.addons.rhino.native import random
from mmcore.addons.rhino.native.utils import *
