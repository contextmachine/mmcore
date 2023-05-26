import numpy as np
from mmcore.base import A,AGroup,ALine,ColorRGB
from mmcore.geom.parametric import ParametricObject
from mmcore.base.sharedstate import serve


class ParametricSpring(AGroup, ParametricObject):

    def __init__(self, r=2, h=3, tess=100, tess2=20, **kwargs):
        super().__init__()
        self.__dict__ |= kwargs
        self.r, self.h, self.tess, self.tess2 = r, h, tess, tess2

    def evaluate(self, t):

        res = np.array([self.r * np.cos(t * 2 * np.pi), self.r * np.sin(t * 2 * np.pi), self.h, 1.0], dtype=float)
        return (res.T @ self.matrix_to_square_form()).tolist()[:3]

    def __call__(self, color=ColorRGB(250, 250, 250), **kwargs):

        self.__dict__ |= kwargs
        _points = []

        _points.append(list(map(self.evaluate, np.linspace(0, 1, 100))))

        if len(self.children) > 0:
            for chld in self.children:
                chld.dispose()
        ln = ALine(uuid=self.uuid + "-line", geometry=_points,
                   material=ALine.material_type(color=color.decimal))
        self.add(ln)
        return A.__call__(self, **kwargs)


if __name__ == "__main__":
    p = ParametricSpring(r=2, h=3, uuid="circle")
    serve.start_as_main()
