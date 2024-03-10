from __future__ import annotations
import adsk.core
from argparse import ArgumentTypeError

import numpy as np

from mmcore.api._base import Base
from mmcore.api._base_vectors import BaseVector
from mmcore.geom.vec import angle, cross, unit, dist


class Vector2D(BaseVector):
    """
    2D vector. This object is a wrapper for 2D vector data and is used to
    pass vector data in and out of the API.
    They are created statically using the create method of the Vector2D class.
    """
    __dim__: int = 2

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, y={self.y}) at {hex(id(self))}'
    @classmethod
    def create(cls, x: float, y: float) -> Vector2D:
        """
        Creates a 2D vector object.
        x : The x coordinate of the vector.
        y : The y coordinate of the vector.
        Returns the new Vector2D object or null if the creation failed.
        """
        return Vector2D(np.array([x, y]))

    def dot(self, vector: Vector2D) -> float:
        """
        Calculates the Dot Product of this vector and an input vector.
        vector : The vector to use in the dot product calculation.
        Returns the dot product of the two vectors.
        """
        return np.dot(self._array, vector._array)

    @classmethod
    def cast(cls, arg) -> Vector2D:
        if isinstance(arg, (tuple, np.ndarray, list)):
            if len(arg) == 2:
                return cls(arg)
            else:
                return cls.cast(arg[:2])

        elif isinstance(arg, (Point3D, Vector3D)):
            return cls.cast(arg.as_array()[:2])
        elif isinstance(arg, (Point2D, Vector2D)):
            return cls.cast(arg._array)
        else:
            raise ValueError(f'{arg}')


    def is_parallel(self, vector: Vector2D) -> bool:
        """
        Compare this vector with another to check for parallelism.
        vector : The vector to compare with for parallelism.
        Returns true if the vectors are parallel.
        """
        return np.isclose(np.dot(self._array, vector._array), 1)


    def is_perpendicular(self, vector: Vector2D) -> bool:
        """
        Compare this vector with another to check for perpendicularity.
        vector : The vector to compare with for perpendicularity.
        Returns true if the vectors are perpendicular.
        """
        return np.isclose(np.dot(self._array, vector._array), 0)

    def normalize(self) -> bool:
        """
        Normalizes the vector.
        Normalization makes the vector length equal to one.
        The vector should not be zero length.
        Returns true if successful.
        """
        self._array = unit(self._array)

    def set_with_array(self, coordinates: list[float]) -> bool:
        """
        Sets the definition of the vector by specifying an array containing the x and y coordinates.
        coordinates : An array that specifies the values for the x and y coordinates of the vector.
        Returns true if successful
        """
        self._array[:] = coordinates


    def subtract(self, vector: Vector2D) -> bool:
        """
        Subtract a vector from this vector.
        vector : The vector to subtract from this vector.
        Returns true if successful.
        """
        return bool()

    def transform(self, matrix: Matrix2D) -> bool:
        """
        Transforms the vector by specifying a 2D transformation matrix.
        matrix : The Matrix2D object that defines the transformation.
        Returns true if successful.
        """
        return bool()

    def as_point(self) -> Point2D:
        """
        Return a point with the same x and y values as this vector.
        Returns the new point.
        """
        return Point2D()

    @property
    def x(self) -> float:
        """
        The x value.
        """
        return self._array[0]

    @x.setter
    def x(self, value: float):
        """
        The x value.
        """
        self._array[0] = value

    @property
    def y(self) -> float:
        """
        The y value.
        """
        return self._array[1]

    @y.setter
    def y(self, value: float):
        """
        The y value.
        """
        self._array[1] = value

    def to_vector3d(self) -> Vector3D:
        return Vector3D(np.append(self._array, 0))

class Vector3D(BaseVector):
    """
    3D vector. This object is a wrapper over 3D vector data and is used as way to pass vector data
    in and out of the API and as a convenience when operating on vector data.
    They are created statically using the create method of the Vector3D class.
    """
    __dim__: int = 3

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z}) at {hex(id(self))}'
    @classmethod
    def cast(cls, arg) -> Vector3D:
        if isinstance(arg, (tuple, np.ndarray, list)):
            if len(arg) == 3:
                return cls(arg)
            else:
                return cls.cast(np.append(arg, np.zeros((3,), float))[:3])

        elif isinstance(arg, (Point3D, Vector3D)):
            return cls.cast(arg.as_array())
        elif isinstance(arg, (Point2D, Vector2D)):
            return cls.cast((*arg._array, 0.0))
        else:
            raise ValueError(f'{arg}')

    @classmethod
    def create(cls, x: float, y: float, z: float) -> Vector3D:
        """
        Creates a 3D vector object. This object is created statically using the Vector3D.create method.
        x : The optional x value.
        y : The optional y value.
        z : The optional z value.
        Returns the new vector.
        """
        return Vector3D((x, y, z))

    def add(self, vector: Vector3D) -> bool:
        """
        Adds a vector to this vector.
        vector : The vector to add to this vector.
        Returns true if successful.
        """
        self._array += vector._array
        return True

    def as_point(self) -> Point3D:
        """
        Returns a new point with the same coordinate values as this vector.
        Return the new point.
        """
        return Point3D(self._array)

    def cross(self, vector: Vector3D) -> Vector3D:
        """
        Returns the cross product between this vector and the specified vector.
        vector : The vector to take the cross product to.
        Returns the vector cross product.
        """
        return Vector3D(cross(self._array, vector._array))

    def to_vector2d(self) -> Vector2D:
        return Vector2D(self._array[:-1])


    def transform(self, matrix: Matrix3D) -> bool:
        """
        Transform this vector by the specified matrix.
        matrix : The transformation matrix.
        Returns true if successful.
        """
        self._array[:] = matrix.m.dot(self._array)
        return True

    def __array__(self, dtype=float):
        return np.array(self._array, dtype=dtype)

    @property
    def length(self) -> float:
        """
        Get the length of this vector.
        """
        return dist(self._array)

    @property
    def x(self) -> float:
        """
        The x value.
        """
        return self._array[0]

    @x.setter
    def x(self, value: float):
        """
        The x value.
        """
        self._array[0] = value

    @property
    def y(self) -> float:
        """
        The y value.
        """
        return self._array[1]

    @y.setter
    def y(self, value: float):
        """
        The y value.
        """
        self._array[1] = value

    @property
    def z(self) -> float:
        """
        The z value.
        """
        return self._array[2]

    @z.setter
    def z(self, value: float):
        """
        The z value.
        """
        self._array[2] = value


class Point2D(Vector2D): ...


class Point3D(Vector3D): ...


class Matrix2D(Base):
    """
    2D 3x3 matrix. This object is a wrapper over 2D matrix data and is used as way to pass matrix data
    in and out of the API and as a convenience when operating on matrix data.
    They are created statically using the create method of the Matrix2D class.
    """

    def __init__(self):
        pass

    @staticmethod
    def cast(arg) -> Matrix2D:
        return Matrix2D()

    @staticmethod
    def create() -> Matrix2D:
        """
        Creates a 2D matrix (3x3) object. It is initialized as an identity matrix.
        Returns the new matrix.
        """
        return Matrix2D()

    def set_to_identity(self) -> bool:
        """
        Resets this matrix to be an identity matrix.
        Returns true if successful.
        """
        return bool()

    def invert(self) -> bool:
        """
        Invert this matrix.
        Returns true if successful.
        """
        return bool()

    def copy(self) -> Matrix2D:
        """
        Creates an independent copy of this matrix.
        Returns the new matrix copy.
        """
        return Matrix2D()

    def transform_by(self, matrix: Matrix2D) -> bool:
        """
        Transforms this matrix using the input matrix.
        matrix : The transformation matrix.
        Returns true if successful.
        """
        return bool()

    def get_cell(self, row: int, column: int) -> float:
        """
        Gets the value of the specified cell in the 3x3 matrix.
        row : The index of the row. The first row has in index of 0
        column : The index of the column. The first column has an index of 0
        Returns the value at [row][column].
        """
        return float()

    def set_cell(self, row: int, column: int, value: float) -> bool:
        """
        Sets the specified cell in the 3x3 matrix to the specified value.
        row : The index of the row. The first row has in index of 0
        column : The index of the column. The first column has an index of 0
        value : The new value of the cell.
        Returns true if successful.
        """
        return bool()

    def get_as_coordinate_system(self) -> tuple[Point2D, Vector2D, Vector2D]:
        """
        Gets the matrix data as the components that define a coordinate system.
        origin : The output origin point of the coordinate system.
        xAxis : The output x axis direction of the coordinate system.
        yAxis : The output y axis direction of the coordinate system.
        """
        return (Point2D(), Vector2D(), Vector2D())

    def as_array(self) -> list[float]:
        """
        Returns the contents of the matrix as a 9 element array.
        Returns the array of matrix values.
        """
        return [float()]

    def set_with_coordinate_system(self, origin: Point2D, xAxis: Vector2D, yAxis: Vector2D) -> bool:
        """
        Reset this matrix to align with a specific coordinate system.
        origin : The origin point of the coordinate system.
        xAxis : The x axis direction of the coordinate system.
        yAxis : The y axis direction of the coordinate system.
        Returns true if successful.
        """
        return bool()

    def set_with_array(self, cells: list[float]) -> bool:
        """
        Sets the contents of the array using a 9 element array.
        cells : The array of cell values.
        Returns true if successful.
        """
        return bool()

    def is_equal_to(self, matrix: Matrix2D) -> bool:
        """
        Compares this matrix with another matrix and returns True if they're identical.
        matrix : The matrix to compare to.
        Returns true if the matrix is equal to this matrix.
        """
        return bool()

    def set_to_align_coordinate_systems(self, fromOrigin: Point2D, fromXAxis: Vector2D, fromYAxis: Vector2D,
                                        toOrigin: Point2D, toXAxis: Vector2D, toYAxis: Vector2D) -> bool:
        """
        Sets this matrix to be the matrix that maps from the 'from' coordinate system to the 'to' coordinate system.
        fromOrigin : The origin point of the from coordinate system.
        fromXAxis : The x axis direction of the from coordinate system.
        fromYAxis : The y axis direction of the from coordinate system.
        toOrigin : The origin point of the to coordinate system.
        toXAxis : The x axis direction of the to coordinate system.
        toYAxis : The y axis direction of the to coordinate system.
        Returns true if successful.
        """
        return bool()

    def set_to_rotate_to(self, _from: Vector2D, to: Vector2D) -> bool:
        """
        Sets to the matrix of rotation that would align the 'from' vector with the 'to' vector.
        from : The from vector.
        to : The to vector.
        Returns true if successful.
        """
        return bool()

    def set_to_rotation(self, angle: float, origin: Point2D) -> bool:
        """
        Sets this matrix to the matrix of rotation by the specified angle, through the specified origin.
        angle : The rotation angle in radians.
        origin : The origin point of the rotation.
        Returns true if successful.
        """
        return bool()

    @property
    def determinant(self) -> float:
        """
        Returns the determinant of the matrix.
        Returns the determinant value of this matrix.
        """
        return float()


class Matrix3D(Base):
    """
    3D 4x4 matrix. This object is a wrapper over 3D matrix data and is used as way to pass matrix data
    in and out of the API and as a convenience when operating on matrix data.
    They are created statically using the create method of the Matrix3D class.
    """

    def __init__(self):
        self.m = np.eye(3, float)

    @classmethod
    def cast(cls, arg) -> Matrix3D:
        return Matrix3D()

    @classmethod
    def create(cls) -> Matrix3D:
        """
        Creates a 3d matrix object. It is initialized as an identity matrix and
        is created statically using the Matrix3D.create method.
        Returns the new matrix.
        """
        return Matrix3D()

    def set_to_identity(self) -> bool:
        """
        Resets this matrix to an identify matrix.
        Returns true if successful.
        """
        return bool()

    def invert(self) -> bool:
        """
        Inverts this matrix.
        Returns true if successful.
        """
        return bool()

    def copy(self) -> Matrix3D:
        """
        Creates an independent copy of this matrix.
        Returns the new matrix copy.
        """
        return Matrix3D()

    def transform(self, matrix: Matrix3D) -> bool:
        """
        Transforms this matrix using the input matrix.
        matrix : The transformation matrix.
        Returns true if successful.
        """
        return bool()

    def get_as_coordinate_system(self) -> tuple[Point3D, Vector3D, Vector3D, Vector3D]:
        """
        Gets the matrix data as the components that define a coordinate system.
        origin : The output origin point of the coordinate system.
        xAxis : The output x axis direction of the coordinate system.
        yAxis : The output y axis direction of the coordinate system.
        zAxis : The output z axis direction of the coordinate system.
        """
        return (Point3D(), Vector3D(), Vector3D(), Vector3D())

    def set_with_coordinate_system(self, origin: Point3D, xAxis: Vector3D, yAxis: Vector3D, zAxis: Vector3D) -> bool:
        """
        Sets the matrix based on the components of a coordinate system.
        origin : The origin point of the coordinate system.
        xAxis : The x axis direction of the coordinate system.
        yAxis : The y axis direction of the coordinate system.
        zAxis : The z axis direction of the coordinate system.
        Returns true if successful.
        """
        return bool()

    def get_cell(self, row: int, column: int) -> float:
        """
        Gets the value of the specified cell in the 4x4 matrix.
        row : The index of the row. The first row has in index of 0
        column : The index of the column. The first column has an index of 0
        The cell value at [row][column].
        """
        return float()

    def set_cell(self, row: int, column: int, value: float) -> bool:
        """
        Sets the specified cell in the 4x4 matrix to the specified value.
        row : The index of the row. The first row has in index of 0
        column : The index of the column. The first column has an index of 0
        value : The new cell value.
        Returns true if successful.
        """
        return bool()

    def as_array(self) -> list[float]:
        """
        Returns the contents of the matrix as a 16 element array.
        Returns the array of cell values.
        """
        return [float()]

    def set_with_array(self, cells: list[float]) -> bool:
        """
        Sets the contents of the array using a 16 element array.
        cells : The array of cell values.
        Returns true if successful.
        """
        return bool()

    def is_equal_to(self, matrix: Matrix3D) -> bool:
        """
        Compares this matrix with another matrix and returns True if they're identical.
        matrix : The matrix to compare this matrix to.
        Returns true if the matrices are equal.
        """
        return bool()

    def set_to_align_coordinate_systems(self, fromOrigin: Point3D, fromXAxis: Vector3D, fromYAxis: Vector3D,
                                        fromZAxis: Vector3D, toOrigin: Point3D, toXAxis: Vector3D, toYAxis: Vector3D,
                                        toZAxis: Vector3D) -> bool:
        """
        Sets this matrix to be the matrix that maps from the 'from' coordinate system to the 'to' coordinate system.
        fromOrigin : The origin point of the from coordinate system.
        fromXAxis : The x axis direction of the from coordinate system.
        fromYAxis : The y axis direction of the from coordinate system.
        fromZAxis : The z axis direction of the from coordinate system.
        toOrigin : The origin point of the to coordinate system.
        toXAxis : The x axis direction of the to coordinate system.
        toYAxis : The y axis direction of the to coordinate system.
        toZAxis : The z axis direction of the to coordinate system.
        Returns true if successful.
        """
        return bool()

    def set_to_rotate_to(self, _from: Vector3D, to: Vector3D, axis: Vector3D) -> bool:
        """
        Sets to the matrix of rotation that would align the 'from' vector with the 'to' vector. The optional
        axis argument may be used when the two vectors are perpendicular and in opposite directions to
        specify a specific solution, but is otherwise ignored
        from : The vector to rotate from.
        to : The vector to rotate to.
        axis : The optional axis vector to disambiguate the rotation axis.
        Returns true if successful.
        """
        return bool()

    def set_to_rotation(self, angle: float, axis: Vector3D, origin: Point3D) -> bool:
        """
        Sets this matrix to the matrix of rotation by the specified angle, through the specified origin, around the specified axis
        angle : The rotation angle in radians.
        axis : The axis of rotation.
        origin : The origin point of the axis of rotation.
        Returns true if successful.
        """
        return bool()

    @property
    def determinant(self) -> float:
        """
        Returns the determinant of the matrix.
        """
        return float()

    @property
    def translation(self) -> Vector3D:
        """
        Gets and sets the translation component of the matrix.
        """
        return Vector3D()

    @translation.setter
    def translation(self, value: Vector3D):
        """
        Gets and sets the translation component of the matrix.
        """
        pass


__all__ = ['Vector2D', 'Vector3D', 'Point2D', 'Point3D', 'Matrix2D', 'Matrix3D']
