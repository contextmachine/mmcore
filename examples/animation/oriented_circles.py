# This example illustrates one of those idiotic tasks
# that you might encounter if you were taught grasshopper in the 2010s.
import time

import numpy as np
import typing
from collections import namedtuple

from mmcore.base import A, AGroup, ALine
from mmcore.base.models.gql import LineBasicMaterial
from mmcore.geom.parametric.sketch import Circle3D
from mmcore.geom.vectors import unit

GridProps = namedtuple("GridProps", ["xsize", "ysize", "xstep", "ystep"])


class Circle3DGrid(AGroup):
    props_class = GridProps
    props: props_class = props_class(5, 5, 16.5, 16.5)
    params: list[list[float]]
    target_point = (8.0, 8.0, 32.0)
    eval_point_count = 16

    def __call__(self, target_point=None, *args, **kwargs):
        # if len(self._children) > 0:
        #    for child in self.children:
        #        child.dispose()

        self.set_state(*args, **kwargs)
        if target_point is not None:
            self.target_point = target_point

        for circle in self.build_grid():
            self.add(circle)
        return A.__call__(self)

    @property
    def properties(self):
        return self.props._asdict()

    @property
    def gui(self):
        return self.props

    def set_state(self, *args, props=None, **kwargs):
        if props is not None:
            self.props |= props
        super().set_state(*args, **kwargs)

    def build_grid(self):
        # build_grid is a function that override the Circle parametrisation with grid rules

        for u in map(lambda x: x * self.props.xstep, range(self.props.xsize)):
            for v in map(lambda x: x * self.props.ystep, range(self.props.ysize)):
                circle = Circle3D(r=self.props.xstep / 3, origin=(u, v, 0),
                                  normal=unit(np.array(self.target_point) - np.array([u, v, 0])))
                *points, = map(circle.evaluate, np.linspace(0, 1, self.eval_point_count))

                yield ALine(uuid=self.uuid + f"-circle-{u}-{v}", geometry=np.array(points, dtype=float).tolist(),
                            material=LineBasicMaterial(color=1361990))

    def grid_bounds(self):
        return self.props.xstep * self.props.xsize, self.props.ystep * self.props.ysize


if __name__ == "__main__":
    IPYTHON = True
    from mmcore.base.sharedstate import serve
    grid = Circle3DGrid(uuid="parametric_grid")
    orbit = Circle3D(r=grid.grid_bounds()[0], origin=np.array(grid.grid_bounds() + (180.0,)) / 2)


    def animate():
        for j in range(5):
            for i in np.linspace(0, 1, 60):
                grid(target_point=orbit.evaluate(i))
                # #print(grid.target_point)
                time.sleep(0.001)
    if IPYTHON:
        serve.start()
        animate()
    else:
        serve.start_as_main(on_start=animate)
