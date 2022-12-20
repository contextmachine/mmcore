#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
import copy
from abc import ABC, ABCMeta, abstractmethod
from functools import wraps

import compas.geometry as cg
import numpy as np
import rhino3dm
from compas_occ.geometry.curves.nurbs import OCCNurbsCurve

from ..baseitems import Base, DictableItem, Item
from ..geom import Point


def to_cmp_point(func):
    @wraps(func)
    def wrp(*a, **kw):
        return cg.Point(*func(*a, **kw))

    return wrp


class SimpleCircle(DictableItem):
    x0 = 0.0
    y0 = 0.0
    r = 1.0

    def evaluate(self, t):
        x = self.x0 + self.r * np.cos(t)
        y = self.y0 + self.r * np.sin(t)
        return x, y


class PrmGenerator(Item, metaclass=ABCMeta):
    """
        Это пожалуй одно из лучших решений для реализации коллекции любого типа (Generator)
        Нюанс в том что не все объекты (тем более содержащиеся в контейнерах должны быть сгенерированы)
        Достаточно важное место в фреймворке отведено продолжительности жизн и объекта.
        Таким образом имеет смысл считать генераторы коллекций равносильными __init__ в Item.
        Контейнерные коллекции в свою очередь ближе к call
    """
    stop = 2 * np.pi
    start = 0.0

    def __init__(self, *args, **kwargs):
        self._step = None
        super().__init__(*args, **kwargs)
        self.si = self.start

    @abstractmethod
    def evaluate(self, t):
        ...

    def __call__(self, step=None, **kwargs):

        super().__call__(**kwargs)
        self.step = step
        self.si = self.start

    def __iter__(self):
        return self

    def __next__(self):

        if self.si <= self.stop + 0.001:
            # print(self.si, self.start, self.stop, self.step)
            t = copy.deepcopy(self.si)
            self.si += self.step
            return self.evaluate(t)
        else:
            raise StopIteration

    def __getitem__(self, item: slice):

        if isinstance(item, (float, int)):

            return self.evaluate(item)

        elif isinstance(item, slice):
            slf = copy.deepcopy(self)
            slf.__call__(start=item.start, stop=item.stop, step=item.step)

            return slf

    @abstractmethod
    def __repr__(self):
        ...

    @property
    def step(self):

        return self._step

    @step.setter
    def step(self, val):

        self._step = val


class ParametricType(PrmGenerator, ABC):

    @property
    def parameterisation(self):
        l = []
        for base in self.__class__.__bases__:
            if issubclass(base, ParametricType):
                l.append(base)
        return l

    def __format__(self, *format_spec):
        nm, literal = format_spec
        l = f"(parameterisation: Base "
        for base in self.parameterisation:
            l += f"-> {base.__name__}"

        return l + f" -> ({literal}))"


class Linear(ParametricType, ABC):
    a = 1.0
    b = 0.0

    def __format__(self, *format_spec):
        a, b = format_spec

        return f"{self.__class__.__name__}(a={a}, b={b})" + super().__format__("Linear",
                                                                               "Ax + By + Cz + D= 0")

    def __repr__(self):
        return self.__format__(self.a, self.b)

    def evaluate(self, t):
        pass


class Circular(ParametricType):
    stop = 2 * np.pi
    start = 0.0
    x0 = 0.0
    y0 = 0.0

    r = 1.0

    def __format__(self, *format_spec):
        r, x, y = format_spec

        return f"{self.__class__.__name__}(r={r}, x0={x}, y0={y})" + super().__format__("Circular",
                                                                                        "Ax^2 + By^2 + Cx + Dy + F = 0")

    def __repr__(self):
        return self.__format__(self.r, self.x0, self.y0)

    def evaluate(self, t):
        pass


class Quadratic(ParametricType):
    stop = 1 * np.pi
    start = -1 * np.pi
    x0 = 0.0
    y0 = 0.0

    a = 1
    b = 1

    def __format__(self, *format_spec):
        a, b, x, y = format_spec
        return f"{self.__class__.__name__}(a={a}, b={b}, x0={x}, y0={y}))" + super(
            Quadratic, self).__format__("Quadratic", "Ax^2 + By^2 + Cz^2 + Dx + Fy + Iz + J = 0")

    def __repr__(self):
        return self.__format__(self.a, self.b, self.x0, self.y0)

    def evaluate(self, t):
        pass


class ClassicLinear(Linear):
    x0 = 0.0
    y0 = 0.0
    a = 1.0
    b = 0.0

    def evaluate(self, t):
        x = self.x0 + self.a * t
        y = self.y0 + self.b * t
        return x, y


class Circle(Circular):
    r = 1.0

    def __init__(self, *args, **kwargs):
        self._plane = None
        super().__init__(*args, **kwargs)

    def evaluate(self, t):
        x = self.x0 + self.r * np.cos(t)
        y = self.y0 + self.r * np.sin(t)
        return x, y

    def tan_at(self, t):
        x, y = np.asarray(self.origin) - np.array(self.evaluate(t))
        return -y, x

    @property
    def plane(self):
        return self._plane

    @plane.setter
    def plane(self, value):
        self._plane = value

    @property
    def origin(self):
        return self.x0, self.y0


class Cone3d(Circle, Circular):
    z0 = 0.0
    vertex = np.array([0.0, 0.0, 0.0])

    @property
    def plane(self):
        return self._plane

    @property
    def target_pt(self):
        try:

            return self._target_pt
        except:
            return np.array([0, 0, 0])

    @target_pt.setter
    def target_pt(self, v):
        setattr(self, "_target_pt", v)

    def plane_from_normal(self, target):

        v = target - self.origin
        vunit = v / np.linalg.norm(v)
        v2 = np.cross(vunit, [0.0, 0.0, 1.0])
        v3 = np.cross(vunit, v2)

        return cg.Frame(self.origin, v2, v3)

    @property
    def old_plane(self):
        return cg.Frame(self.origin, np.asarray([1, 0, 0]), np.asarray([0, 1, 0]))

    @plane.setter
    def plane(self, value):
        self._plane = value

    def _plane_xf(self):

        new_plane = self.plane_from_normal(self.target_pt)
        old_plane = cg.Frame(self.origin, np.asarray([1, 0, 0]), np.asarray([0, 1, 0]))
        T = cg.Transformation.from_frame_to_frame(old_plane, new_plane)
        self.plane = new_plane

        return T

    def evaluate(self, t):
        x, y = super().evaluate(t)

        pt = cg.Point(x, y, self.z0)

        if self.target_pt is not None:
            pt.transform(self._plane_xf())
            return list(pt)
        else:

            return list(pt)

    def tan_at_world(self, t):
        return self.plane.local_to_local_coordinates(
            cg.Frame(self.origin, np.asarray([1, 0, 0]), np.asarray([0, 1, 0])), self.plane, self.tan_at(t))

    def tan_at(self, t):
        # x, y, z = np.array(self.evaluate(t)) - np.asarray(self.origin)
        # T=self._plane_xf()
        xyz = [cg.Point(*self.evaluate(t)), cg.Point(*self.origin)]
        al, bl = cg.world_to_local_coordinates(self.plane, xyz)
        # print(al, bl)
        rt = np.asarray(bl) - np.asarray(al)
        lx, ly, lz = rt / np.linalg.norm(rt)
        # print(lx, ly, lz)

        return -ly, lx, 0.0

    def tan_vec(self, t, ray_k=1):
        pt, vec = cg.world_to_local_coordinates(self.plane, [self.evaluate(t)])[0], self.tan_at(t)
        ptj = np.array(pt) + (np.asarray(vec) * ray_k)
        return np.asarray(cg.local_to_world_coordinates(self.plane, [pt, ptj]))

    @property
    def origin(self):

        return self.x0, self.y0, self.z0


class Ellipse(Quadratic):
    a = 1
    b = 1

    def evaluate(self, t):
        x = self.x0 + self.a * np.cos(t)
        y = self.y0 + self.b * np.sin(t)
        return x, y


class Hyperbola(Quadratic):
    """
    The equation is y2 / a2 − x2 / b2 = 1,
    where the asymptotes of the hyperbola are x = [b / a] * y and x = [−b / a] * y.
    """
    a = 1
    b = 1

    def evaluate(self, t):
        x = self.x0 + self.a * np.cosh(t)
        y = self.y0 + self.b * np.sinh(t)
        return x, y


class HyperEllipse(Quadratic):
    a = 1
    b = 1
    c = 0.6

    def evaluate(self, t):
        x = self.x0 + (self.a * np.cosh(t) if np.abs(t) < self.c else self.a * np.sin(t))
        y = self.y0 + (self.b * np.sinh(t) if np.abs(t) < self.c else self.b * np.cos(t))
        return x, y


class ParametricEquation(Base):
    ...


class Cone(ParametricEquation):
    def __call__(self, *args, **kwargs):
        ...


class GeomCircle(Circle):
    def evaluate(self, t) -> Point:
        return Point(**dict(zip(("x", "y"), super(GeomCircle, self).evaluate(t))))

    @property
    def origin(self):
        return Point(x=self.x0, y=self.y0)

    def __repr__(self):
        return super(GeomCircle, self).__repr__()


class Arc1(Circle, DictableItem):
    r = 1.0
    x0 = 0.0
    y0 = 0.0
    start_angle = 0.0
    end_angle = np.pi / 2

    def __call__(self, *args, **kwargs):
        super(Arc1, self).__call__(*args, **kwargs)

        self.start = self.start_angle
        self.end = self.end_angle

    def __getitem__(self, item: float):
        return super(Arc1, self).__getitem__(slice(self.start, self.stop, item))

    def __next__(self):
        super(Arc1, self).__next__()


class Arc(SimpleCircle):
    r = 1.0
    x0 = 0.0
    y0 = 0.0
    start_angle = 0.0
    end_angle = np.pi / 2

    def __call__(self, *args, **kwargs):
        super(Arc, self).__call__(*args, **kwargs)

        self.start = self.evaluate(self.start_angle)
        self.end = self.evaluate(self.end_angle)

    def to_compas(self):
        self.cc = OCCNurbsCurve.from_circle(cg.Circle(cg.Plane([self.x0, self.y0, 0.0], [0, 0, 1]), self.r))
        _, self.ts = self.cc.closest_point(self.start.to_compas(), return_parameter=True)
        _, self.te = self.cc.closest_point(self.end.to_compas(), return_parameter=True)
        return self.cc.segmented(self.ts, self.te)

    def evaluate(self, t):
        x, y = super().evaluate(t)
        return Point(x=x, y=y)

    def to_rhino(self):
        rh_arc = rhino3dm.Arc(rhino3dm.Point3d(0.0, 0.0, 0.0), radius=self.r,
                              angleRadians=self.end_angle - self.start_angle)
        return rh_arc

    def to_rhino_json(self):
        return self.to_rhino().ToNurbsCurve().Encode()

    def to_compas_json(self):
        return self.to_compas().to_jsonstring()
