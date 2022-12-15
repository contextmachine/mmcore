#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"

import compas.geometry as cg
import numpy as np

from ..parametric import Arc

js = {}


class ArcForArc(Arc):
    evalute_param = np.pi / 2
    origin = [0, 0, 0]
    normal = [0, 0, 1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.evalute_point = self.evaluate(self.evalute_param)

    def to_compas_vector(self):
        self.cc = cg.NurbsCurve.from_circle(cg.Circle(cg.Plane([self.x0, self.y0, 0.0], [0, 0, 1]), self.r))
        point, ts = self.cc.closest_point(self.evalute_point.to_compas(), return_parameter=True)
        vec = self.cc.tangent_at(ts)

        vec_pl = cg.Vector.from_start_end(self.origin, self.normal)
        toframe = cg.Frame.from_plane(cg.Plane(self.origin, vec_pl.unitized()))
        tr = cg.Transformation.from_frame(toframe)
        self.point = point.transformed(tr)
        self.ts = vec.transformed(tr)
        return self.ts.unitized(), self.point, cg.Circle(cg.Plane([self.x0, self.y0, 0.0], [0, 0, 1]), self.r)


test = ArcForArc(origin=[5.542538, 35.977149, -12.14216], normal=[68.076713, 40.180588, -14.656795], r=15,
                 evalute_param=np.radians(10))
print(test.to_compas_vector())

js['tang'] = [test.to_compas_vector()[0].x, test.to_compas_vector()[0].y, test.to_compas_vector()[0].z]
js['point'] = [test.to_compas_vector()[1].x, test.to_compas_vector()[1].y, test.to_compas_vector()[1].z]
js['circ'] = test.to_compas_vector()[2].to_jsonstring()
