from __future__ import annotations

from ._typing import Type, TypeVar, Union, SupportsInt, SupportsFloat
import numpy as np
from mmcore.api._base import Base

from typing_extensions import Buffer

Numeric = Union[SupportsInt, SupportsFloat, Buffer]

from mmcore.geom.vec import dist, angle, unit, norm


class BaseVector(Base):
    __dim__: int

    def __init__(self, arr=None):
        super().__init__()
        if arr is None:
            self._array = np.zeros(self.__class__.__dim__, float)
        else:
            self._array = np.array(arr, float)

    def __add__(self, other: SupportVector):
        if isinstance(other, self.__class__):
            return self.__class__(self._array + other._array)
        else:
            return self.__class__(self._array + other)

    def __iadd__(self, other: SupportVector):
        if isinstance(other, self.__class__):
            self._array += other._array
        else:
            self._array += other

    def __mul__(self, other: SupportVector):
        if isinstance(other, self.__class__):
            return self.__class__(self._array * other._array)
        else:
            return self.__class__(self._array * other)

    def __imul__(self, other: SupportVector):
        if isinstance(other, self.__class__):
            self._array *= other._array
        else:
            self._array *= other

    def __sub__(self, other: SupportVector):
        if isinstance(other, self.__class__):
            return self.__class__(self._array - other._array)
        else:
            return self.__class__(self._array - other)

    def __isub__(self, other: SupportVector):
        if isinstance(other, self.__class__):
            self._array -= other._array
        else:
            self._array -= other

    def __eq__(self, other: V):
        return np.allclose(self._array, other._array)

    def __array__(self, dtype=float):
        return np.array(self._array, dtype=dtype)

    def scale(self, scale: float) -> bool:
        """
        Scales the vector by specifying a scaling factor.
        scale : The scale factor to multiple the vector by (i.e. 1.5).
        Returns true if successful.
        """
        self._array *= scale
        return True

    @property
    def length(self) -> float:
        """
        Get the length of this vector.
        """
        return norm(self._array)

    def angle_to(self, vector: V) -> float:
        """
        Determines the angle between this vector and the specified vector.
        vector : The vector to measure the angle to.
        The angle in radians between this vector and the specified vector.
        """
        return angle(self._array, vector._array)

    def is_equal(self, vector: Type[Self]) -> bool:
        """
        Compare this vector with another to check for equality.
        vector : The vector to compare with for equality.
        Returns true if the vectors are equal.
        """
        return np.allclose(self._array, vector._array)

    def normalize(self) -> bool:
        """
        Makes this vector of unit length.
        This vector should not be zero length.
        Returns true if successful.
        """
        self._array[:] = unit(self._array)
        return True

    def copy(self) -> V:
        """
        Creates and returns an independent copy of this Vector2D object.
        Returns a new Vector2D object that is a copy of this Vector2D object.
        """
        return self.__class__(np.copy(self._array))

    @classmethod
    def cast(cls, arg) -> Type[Self]:
        return arg if isinstance(arg, cls) else cls(arg)

    def as_array(self) -> np.ndarray:
        return np.asarray(self._array)

    def to_data(self):
        return self._array

    def dot(self, vector: Type[Self]) -> float:
        """
        Returns the dot product between this vector and the specified vector.
        vector : The vector to take the dot product to.
        Returns the dot product value.
        """
        return np.dot(self._array, vector._array)

    def is_parallel(self, vector: Type[Self]) -> bool:
        """
        Determines if the input vector is parallel with this vector.
        vector : The vector to test parallelism to.
        Returns true if the vectors are parallel.
        """
        return np.isclose(abs(self.dot(vector)), 1.0)

    def is_perpendicular(self, vector: Type[Self]) -> bool:
        """
        Determines if the input vector is perpendicular to this vector.
        vector : The vector to test perpendicularity to.
        Returns true if the vectors are perpendicular.
        """
        return np.isclose(self.dot(vector), 0.0)

    def set_with_array(self, coordinates: list[float]) -> bool:
        """
        Reset this vector with the coordinate values in an array [x, y, z].
        coordinates : The array of coordinate values.
        Returns true if successful.
        """
        self._array[:] = coordinates
        return True

    def subtract(self, vector: Type[Self]) -> bool:
        """
        Subtract a vector from this vector.
        vector : The vector to subtract.
        Returns true if successful.
        """
        self._array -= vector._array
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}()"


V = TypeVar("V", bound=BaseVector)
SupportVector = Union[V, Numeric]


class BaseMatrix(Base):
    _array: np.ndarray

    def invert(self) -> bool:
        """
        Inverts this matrix.
        Returns true if successful.
        """
        self._array[:] = np.linalg.inv(self._array)
        return True

    def copy(self) -> Self:
        """
        Creates an independent copy of this matrix.
        Returns the new matrix copy.
        """
        m = self.__class__()
        m._array[:] = self._array
        return m

    def get_cell(self, row: int, column: int) -> float:
        """
        Gets the value of the specified cell in the 4x4 matrix.
        row : The index of the row. The first row has in index of 0
        column : The index of the column. The first column has an index of 0
        The cell value at [row][column].
        """

        return float(self._array[row, column])

    def set_cell(self, row: int, column: int, value: float) -> bool:
        """
        Sets the specified cell in the 4x4 matrix to the specified value.
        row : The index of the row. The first row has in index of 0
        column : The index of the column. The first column has an index of 0
        value : The new cell value.
        Returns true if successful.
        """
        self._array[row, column] = value
        return True

    def as_array(self) -> np.ndarray:
        """
        Returns the contents of the matrix as a 16 element array.
        Returns the array of cell values.
        """
        return np.asarray(self._array)

    def set_with_array(self, cells: list[float]) -> bool:
        """
        Sets the contents of the array using a 16 element array.
        cells : The array of cell values.
        Returns true if successful.
        """
        self._array[:] = np.array(cells).reshape((4, 4))
        return True

    def is_equal_to(self, matrix: Self) -> bool:
        """
        Compares this matrix with another matrix and returns True if they're identical.
        matrix : The matrix to compare this matrix to.
        Returns true if the matrices are equal.
        """

        return np.allclose(self._array, matrix._array)

    @property
    def determinant(self) -> float:
        """
        Returns the determinant of the matrix.
        """
        return np.linalg.det(self._array)

    def set_to_identity(self) -> bool:
        """
        Resets this matrix to an identify matrix.
        Returns true if successful.
        """
        self._array[:] = np.eye(self._array.shape[0], dtype=float)
        return True

    def transform(self, matrix: "Self") -> bool:
        """
        Transforms this matrix using the input matrix.
        matrix : The transformation matrix.
        Returns true if successful.
        """
        self._array = self._array.dot(matrix.as_array())
        return True

    def __matmul__(self, other):
        if isinstance(other, BaseVector):
            return other.__class__.cast(
                self._array.dot(np.append(other.as_array(), 1.0))
            )
        else:
            m = self.copy()
            m.transform(other)
            return m
