from typing import Optional, Union

from numpy._typing import NDArray

from mmcore.geom.curves.curve import Curve

Spline=Union[BSpline,NURBSpline]
class BSpline(Curve):

    knots: NDArray[float]
    degree:int

    def __init__(self,
                 control_points: NDArray[float],
                 degree: int=None,
                 knots: Optional[NDArray[float]]=None) -> None: ...
    @property
    def control_points(self) -> NDArray[float]:
        """Control Points array in shape (N, M)
        where
        N -- count of points, M= count of coordinate components
        """
        ...
    def generate_knots(self)->NDArray[float]: ...

    def basis_function(self, t:float, i:int, k:int)->float:...

    def set(self,
            control_points:Optional[ NDArray[float]]=None,
            degree:Optional[int]=None,
            knots:Optional[ NDArray[float]]=None)->None:...
    def split(self, t: float)->tuple[BSpline, BSpline]: ...

class NURBSpline(BSpline):
    """
    Non-Uniform Rational BSpline (NURBS)
    Example:
        >>> import numpy as np
        >>> spl = NURBSpline(np.array([(-26030.187675027133, 5601.3871095975337, 31638.841094491760),
        ...                   (14918.717302595671, -25257.061306278192, 14455.443462719517),
        ...                   (19188.604482326708, 17583.891501540096, 6065.9078795798523),
        ...                   (-18663.729281923122, 5703.1869371495322, 0.0),
        ...                   (20028.126297559378, -20024.715164607202, 2591.0893519960955),
        ...                   (4735.5467668945130, 25720.651181520021, -6587.2644037490491),
        ...                   (-20484.795362315021, -11668.741154421798, -14201.431195298581),
        ...                   (18434.653814767291, -4810.2095985021788, -14052.951382291201),
        ...                   (612.94310080525793, 24446.695569574043, -24080.735343204549),
        ...                   (-7503.6320665111089, 2896.2190847052334, -31178.971042788111)]
        ...                  ))

    """
    knots: NDArray[float]
    degree: int
    weights: NDArray[float]

    def __init__(self,
                 control_points:NDArray[float],
                 weights:Optional[ NDArray[float]]=None,
                 degree:int=None,
                 knots:Optional[ NDArray[float]]=None)->None:...

    @property
    def control_points(self) -> NDArray[float]:
        """Control Points array in shape (N, M)
        where
        N -- count of points, M= count of coordinate components
        """
        ...
    def set_weights(self, weights:Optional[ NDArray[float]]=None)->None:...

    def split(self, t: float) -> tuple[NURBSpline, NURBSpline]: ...

def nurbs_split(self:Spline,t:float) -> tuple[Spline, Spline]: ...
