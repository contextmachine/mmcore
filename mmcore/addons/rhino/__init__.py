# Native imports
import rpyc, sys
try:
    import rhino3dm
except ImportError as err:
    cn = rpyc.connect("84.201.152.88", 18812)
    rhino3dm = cn.root.getmodule("rhino3dm")
    sys.modules["rhino3dm"] = rhino3dm
    import rhino3dm

from mmcore.addons.rhino.native import random
from mmcore.addons.rhino.native.utils import *
