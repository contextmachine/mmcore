from CGAL.CGAL_Kernel import *
from mmcore.base import Delegate
from CGAL import CGAL_Kernel as cgk


@Delegate(delegate=Point_3)
class CgalPoint3:

    def __new__(cls, x=None, y=None, z=0.0, ref=None):

        if isinstance(x, Point_3):
            return cls.from_ref(x)
        elif ref is not None:
            return cls.from_ref(ref)
        else:
            inst = super().__new__(cls)
            inst._ref = Point_3(x, y, z)
        return inst

    @classmethod
    def from_ref(cls, ref):
        inst = object.__new__(cls)
        inst._ref = ref
        return inst

    def __iter__(self):
        return iter(self.xyz)

    @property
    def xyz(self):
        return [self._ref.x(), self._ref.y(), self._ref.z()]

    @xyz.setter
    def xyz(self, v):
        self._ref.set_coordinates(*v)

    def __repr__(self):
        x, y, z = self.xyz

        return f"{self.__class__.__name__}({x}, {y}, {z})"

    def get_ref(self):
        return self._ref


@Delegate(delegate=Triangle_3)
class CgalTriangle3:
    def __new__(cls, a, b, c, ref=None):
        inst = object.__new__(cls)
        inst._ref = Triangle_3(Point_3(*a), Point_3(*b), Point_3(*c))
        return inst

    @property
    def pts(self):
        return list(map(CgalPoint3.from_ref, [self._ref.vertex(0), self._ref.vertex(1),
                self._ref.vertex(2)]))

    @pts.setter
    def pts(self, v):
        a, b, c = v
        self._ref.vertex(0).set_coordinates(*a)
        self._ref.vertex(1).set_coordinates(*b)
        self._ref.vertex(2).set_coordinates(*c)

    def __repr__(self):
        x, y, z = self.pts

        return f"{self.__class__.__name__}({x}, {y}, {z})"
