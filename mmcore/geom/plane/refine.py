from typing import Dict, Tuple, Union
import numpy as np

from mmcore.geom.vec import *


class PlaneRefine:
    """
    The `PlaneRefine` class is used to perform vector operations on 3D coordinates.

    Attributes:
        axis (str): The axis along which operations are performed.
        _axis_indices (dict): A dictionary mapping each axis ('x', 'y', 'z') to a number (0, 1, 2)
        next (PlaneRefine): A recursive instance of PlaneRefine for the subsequent axis if provided.

    Methods:
        __init__(*axis: Tuple[str], axis_indices: Dict[str, int] = None)
            Constructor for PlaneRefine class. Raises a ValueError for more than 3 axes.

        __call__(xyz: np.ndarray, inplace: bool = False)
           Perform unitization of the array and the computation of the cross product depending on the axis.
           Can modify the input array in-place and applies recursion if a next PlaneRefine instance exists.

    Example:

        >>> import numpy as np
        >>> from mmcore.geom.plane.refine import PlaneRefine

        Define two 3x3 numpy arrays with random float values
        >>> pln1, pln2 = np.random.random((3,3)), np.random.random((3,3))

        Instantiate PlaneRefine class to perform operations along 'x' and 'y' axes
        >>> refine_xy = PlaneRefine('x', 'y')

        Apply the PlaneRefining operation to pln1
        The operations are performed along 'x' and 'y' axes in that order
        >>> pln1_refined = refine_xy(pln1)

        Print the refined array
        >>> print(pln1_refined)

        Instantiate PlaneRefine class to perform operations along 'z' and 'y' axes
        >>> refine_zy = PlaneRefine('z', 'y')

        Apply the PlaneRefining operation to pln2
        The operations are performed along 'z' and 'y' axes in that order
        >>> pln2_refined = refine_zy(pln2)


         Print the refined array
         >>> print(pln2_refined)

        After running the above code, `pln1_refined` and `pln2_refined` will contain the arrays `pln1` and `pln2`
        respectively, but after undergoing transformations along the specified axes. Note that the original arrays
        `pln1`
        and `pln2` haven't been modified, because we did not pass `True` for `inplace` argument. If you want to modify
        the original arrays, you can do as follows:

        Apply operations on pln1 in-place
        >>> refine_xy(pln1, inplace=True)

        Now pln1 has been modified by the refine_xy operations
        >>> print(pln1)


    """
    axis: str

    next: Union['PlaneRefine', None] = None

    cross_table = dict(x=lambda xyz: unit(cross(xyz[1], xyz[2])), y=lambda xyz: unit(cross(xyz[2], xyz[0])),
                       z=lambda xyz: unit(cross(xyz[0], xyz[1]))
                       )

    def __init__(self, *axis, axis_indices: Dict[str, int] = None) -> None:
        """
        The constructor for PlaneRefine class.

        Parameters:
            axis (tuple): The axes for which operations will be performed in the sequence provided.
            axis_indices (dict): A mapping of axes to their respective indices.

        Raises:
            ValueError: If more than three axes are provided.
            ValueError: If a provided axis is not an existing one.
        """

        if axis_indices is None:
            self._axis_indices = dict(x=0, y=1, z=2)
        else:
            self._axis_indices = axis_indices
        self.next = None

        if len(axis) > 3:
            raise ValueError(f'Too many axes {len(axis)} for 3D plane refinement!')
        if DEBUG_MODE:
            for ax in axis:
                if ax not in self._axis_indices:
                    raise ValueError(f"Axis {ax} is not doesn't exist. Please use one of the following: "
                                     f"{list(self._axis_indices.keys())}"
                            )

        self.axis = axis[0]
        if len(axis) > 1:

            self.next = PlaneRefine(*axis[1:], axis_indices=self._axis_indices)

    def _unit_check(self, arr: np.ndarray):

        return unit(arr)

    def _chain_call(self, xyz):

        xyz[self._axis_indices[self.axis]] = self.cross_table[self.axis](xyz)
        if self.next is None:
            return xyz
        else:
            return self.next._chain_call(xyz)

    @vectorize(excluded=[0, 'inplace'], signature='(i,i)->(i,i)')
    def __call__(self, xyz: np.ndarray, inplace: bool = False) -> np.ndarray:
        """
        Perform operations on the given 3D coordinates 'xyz'.
        This includes unitization of the array and computation of the cross product depending on 'axis'.
        Applies these operations recursively if a 'next' instance of PlaneRefine exists and modifies the input array
        in-place if 'inplace' is True.

        Parameters:
            xyz (np.ndarray): The 3D coordinates to perform operations on.
            inplace (bool, optional): If True, modifies 'xyz' in-place. Defaults to False.

        Returns:
            np.ndarray: The modified 'xyz' array after performing the specified operations.

        Raises:
            TypeError: If 'xyz' is not a numpy ndarray when 'inplace' is True.
            ValueError: If 'xyz' is not a 3 by 3 numpy ndarray.
        """

        if DEBUG_MODE:
            if not isinstance(xyz, np.ndarray) and inplace:
                raise TypeError(f"xyz should be a numpy ndarray to use inplace option")

            if isinstance(xyz, np.ndarray):
                if (xyz.shape[0], xyz.shape[-1]) != (3, 3):
                    raise ValueError(f"xyz should be a numpy ndarray, with shape {(3, 3)} exist: {xyz}")

        if inplace:

            xyz[:] = self._unit_check(xyz)
        else:
            xyz = self._unit_check(xyz)

        return self._chain_call(xyz)
