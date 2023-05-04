# Usage example:
# from mmcore.base.registry.fcpickle import FSDB
# from mmcore.base.basic import Object3D
# c= Object3D(name="A")
# FSDB['obj']= obj
# ...
# shell:
# python -m pickle .pkl/obj
#[mmcore] : Object3D(priority=1.0,
#                    children_count=0,
#                    name=A,
#                    part=NE) at cf3d55d7-677e-4f96-9e31-b628c3962520
#
import os
import pickleshare
PICKLE_FS_ROOT=f"{os.getcwd()}/.pkl"

FSDB = pickleshare.PickleShareDB(PICKLE_FS_ROOT)


