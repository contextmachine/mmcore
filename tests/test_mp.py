import numpy as np
import time

from mmcore.geom.parametric import Pipe, NurbsCurve, Circle


def test():
    """


    @return:
    >>> path = NurbsCurve([[0, 0, 0],
    ...               [-47, -315, 0],
    ...               [-785, -844, 0],
    ...               [-704, -1286, 0],
    ...               [-969, -2316, 0]])

    >>> profile = Circle(r=10.5)
    >>> pipe = Pipe(path, profile)
    >>> pipe.mpeval(uvs=(2000, 200)) # 400,000 points
    [mp] time 8.37929892539978 s
    >>> pipe.veval(uvs=(2000, 200)) # 400,000 points
    [single] time 40.84727501869202 s

    """

    nb2 = NurbsCurve([[0, 0, 0],
                      [-47, -315, 0],
                      [-785, -844, 0],
                      [-704, -1286, 0],
                      [-969, -2316, 0]])

    r = Circle(r=10.5)
    oo = Pipe(nb2, r)
    s=time.time()
    oo.mpeval(uvs=(2000, 200))
    print(f"[mp] time {time.time() - s} s")

    s = time.time()
    oo.veval(uvs=(2000, 200))
    print(f"[single] time {time.time() - s} s")


test()
