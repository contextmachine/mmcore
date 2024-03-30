from __future__ import annotations

import numpy as np

from scipy.spatial.transform import Rotation as R

from mmcore.api._base import Base
from mmcore.api._base_vectors import BaseVector, BaseMatrix
from mmcore.geom.vec import angle, cross, unit, dist


class Vector2D(BaseVector):
    """
    2D vector. This object is a wrapper for 2D vector data and is used to
    pass vector data in and out of the API.
    They are created statically using the create method of the Vector2D class.
    """

    __dim__: int = 2

    def __repr__(self):
        return f"{self.__class__.__name__}(x={self.x}, y={self.y})"

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
            raise ValueError(f"{arg}")

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
        self._array[:] = unit(self._array)
        return True

    def set_with_array(self, coordinates: list[float]) -> bool:
        """
        Sets the definition of the vector by specifying an array containing the x and y coordinates.
        coordinates : An array that specifies the values for the x and y coordinates of the vector.
        Returns true if successful
        """
        self._array[:] = coordinates
        return True

    def subtract(self, vector: Vector2D) -> bool:
        """
        Subtract a vector from this vector.
        vector : The vector to subtract from this vector.
        Returns true if successful.
        """

        self._array -= vector._array
        return True

    def transform(self, matrix: Matrix2D) -> bool:
        """
        Transforms the vector by specifying a 2D transformation matrix.
        matrix : The Matrix2D object that defines the transformation.
        Returns true if successful.
        """
        self._array[:] = matrix._array.dot(self._array)

        return True

    def as_point(self) -> Point2D:
        """
        Return a point with the same x and y values as this vector.
        Returns the new point.
        """
        return Point2D.cast(self._array)

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

    def to_translation(self) -> Matrix2D:
        m = Matrix2D()
        m.translation = self
        return m


class Vector3D(BaseVector):
    """
    3D vector. This object is a wrapper over 3D vector data and is used as way to pass vector data
    in and out of the API and as a convenience when operating on vector data.
    They are created statically using the create method of the Vector3D class.
    """

    __dim__: int = 3

    def __repr__(self):
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z})"

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
            raise ValueError(f"{arg}")

    @classmethod
    def create(cls, x: float, y: float, z: float, **kwargs) -> Vector3D:
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

    def to_translation(self) -> Matrix3D:
        m = Matrix3D()
        m.translation = self
        return m

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


class Point2D(Vector2D):
    @classmethod
    def create(cls, x: float, y: float, **kwargs) -> Point2D:
        return cls((x, y))


class Point3D(Vector3D):
    @classmethod
    def create(cls, x: float, y: float, z: float, **kwargs) -> Point3D:
        return cls((x, y, z))


def _construct_affine2d(origin, xaxis, yaxis):
    import numpy as np

    # Define the vectors for rotation and scaling
    x = np.array([xaxis[0], xaxis[1]])  # x-axis components
    y = np.array([yaxis[0], yaxis[1]])  # y-axis components

    # Define the translation vector
    t = np.array(origin)  # translation along x, y and z axis

    # Construct the 4x4 affine matrix
    affine_matrix = np.array([np.append(x, 0), np.append(y, 0), np.append(t, 1)])

    return affine_matrix


def _construct_affine3d(origin, xaxis, yaxis, zaxis):
    import numpy as np

    # Define the vectors for rotation and scaling
    x = np.array([xaxis[0], xaxis[1], xaxis[2]])  # x-axis components
    y = np.array([yaxis[0], yaxis[1], yaxis[2]])  # y-axis components
    z = np.array([zaxis[0], zaxis[1], zaxis[2]])  # z-axis components

    # Define the translation vector
    t = np.array(origin)  # translation along x, y and z axis

    # Construct the 4x4 affine matrix
    affine_matrix = np.array(
        [np.append(x, 0), np.append(y, 0), np.append(z, 0), np.append(t, 1)]
    )

    return affine_matrix


class Matrix2D(BaseMatrix):
    """
    2D 3x3 matrix. This object is a wrapper over 2D matrix data and is used as way to pass matrix data
    in and out of the API and as a convenience when operating on matrix data.
    They are created statically using the create method of the Matrix2D class.
    """

    def __init__(self, arr=None):
        super().__init__()
        self._array = (
            np.zeros([3, 3])
            if not arr
            else (arr if isinstance(arr, np.ndarray) else np.array(arr))
        )
        self.set_to_identity()

    @staticmethod
    def cast(arg) -> Matrix2D:
        return Matrix2D(arg)

    @staticmethod
    def create() -> Matrix2D:
        """
        Creates a 2D matrix (3x3) object. It is initialized as an identity matrix.
        Returns the new matrix.
        """
        return Matrix2D()

    def set_with_array(self, cells: list[float]) -> bool:
        self._array[:] = np.array(cells).reshape((3, 3))
        return True

    def get_cell(self, row: int, column: int) -> float:
        return float(self._array[row, column])

    def set_cell(self, row: int, column: int, value: float) -> bool:
        self._array[row, column] = value
        return True

    def set_with_coordinate_system(
        self, origin: Point2D, xAxis: Vector2D, yAxis: Vector2D
    ) -> bool:
        """
        Reset this matrix to align with a specific coordinate system.
        origin : The origin point of the coordinate system.
        xAxis : The x axis direction of the coordinate system.
        yAxis : The y axis direction of the coordinate system.
        Returns true if successful.
        """
        self._array[:] = _construct_affine2d(
            origin.as_array(), xAxis.as_array(), yAxis.as_array()
        )
        return True

    def set_to_align_coordinate_systems(
        self,
        fromOrigin: Point2D,
        fromXAxis: Vector2D,
        fromYAxis: Vector2D,
        toOrigin: Point2D,
        toXAxis: Vector2D,
        toYAxis: Vector2D,
    ) -> bool:
        """
        Sets this matrix to be the matrix that maps from the 'from' coordinate system to the 'to' coordinate system.
        """
        from_ = self.set_with_coordinate_system(fromOrigin, fromXAxis, fromYAxis)
        to_ = self.set_with_coordinate_system(toOrigin, toXAxis, toYAxis)

        self._array = np.linalg.inv(from_).dot(to_)
        return True

    def set_to_rotate_to(self, _from: Vector2D, to: Vector2D) -> bool:
        """
        Sets this matrix so that _from vector will be rotationally aligned with to vector.
        """
        dot = _from.dot(to)
        det = np.linalg.det([_from, to])
        theta = np.arctan2(det, dot)  # Angle between _from and to
        self.set_to_rotation(theta, np.array([0.0, 0.0]))
        return True

    def set_to_rotation(self, angle: float, origin: Point2D = None) -> bool:
        """
        Sets this matrix to the matrix of rotation by the specified angle, through the specified origin.
        """
        self._array[:2, :2] = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        if origin is not None:
            # If the rotation center is not at the origin, translate
            self._array[:2, 2] = origin - origin @ self._array[:2, :2]
        return True

    def get_as_coordinate_system(self) -> tuple[Point2D, Vector2D, Vector2D]:
        """
        Gets the matrix data as the components that define a coordinate system.
        origin : The output origin point of the coordinate system.
        xAxis : The output x axis direction of the coordinate system.
        yAxis : The output y axis direction of the coordinate system.
        zAxis : The output z axis direction of the coordinate system.
        """

        origin = np.copy(self._array[:2, 2])
        xAxis = np.copy(self._array[:2, 0])
        yAxis = np.copy(self._array[:2, 1])

        return (
            Point2D(origin),
            Vector2D(xAxis),
            Vector2D(yAxis),
        )

    @property
    def translation(self) -> Vector2D:
        """
        Gets and sets the translation component of the matrix.
        """
        return Vector2D(self._array[:2, 2])

    @translation.setter
    def translation(self, value: Vector2D):
        """
        Gets and sets the translation component of the matrix.
        """
        if not isinstance(value, Vector2D):
            raise ValueError("Translation must be Vector3D instance")
        self._array[:2, 2] = value.as_array()


from ._typing import Self


class Matrix3D(BaseMatrix):
    """
    3D 4x4 matrix. This object is a wrapper over 3D matrix data and is used as way to pass matrix data
    in and out of the API and as a convenience when operating on matrix data.
    They are created statically using the create method of the Matrix3D class.
    """

    def __init__(self, arr=None):
        super().__init__()
        self._array = (
            np.zeros([4, 4])
            if not arr
            else (arr if isinstance(arr, np.ndarray) else np.array(arr))
        )
        self.set_to_identity()

    @classmethod
    def create(cls) -> Matrix3D:
        """
        Creates a 3d matrix object. It is initialized as an identity matrix and
        is created statically using the Matrix3D.create method.
        Returns the new matrix.
        """
        return Matrix3D()

    @classmethod
    def cast(cls, arg) -> Matrix3D:
        return Matrix3D(arg)

    def get_as_coordinate_system(self) -> tuple[Point3D, Vector3D, Vector3D, Vector3D]:
        """
        Gets the matrix data as the components that define a coordinate system.
        origin : The output origin point of the coordinate system.
        xAxis : The output x axis direction of the coordinate system.
        yAxis : The output y axis direction of the coordinate system.
        zAxis : The output z axis direction of the coordinate system.
        """

        origin = np.copy(self._array[:3, 3])
        xAxis = np.copy(self._array[:3, 0])
        yAxis = np.copy(self._array[:3, 1])
        zAxis = np.copy(self._array[:3, 2])

        return (
            Point3D(origin),
            Vector3D(xAxis),
            Vector3D(yAxis),
            Vector3D(zAxis),
        )

    def set_with_coordinate_system(
        self, origin: Point3D, xAxis: Vector3D, yAxis: Vector3D, zAxis: Vector3D
    ) -> bool:
        """
        Sets the matrix based on the components of a coordinate system.
        origin : The origin point of the coordinate system.
        xAxis : The x axis direction of the coordinate system.
        yAxis : The y axis direction of the coordinate system.
        zAxis : The z axis direction of the coordinate system.
        Returns true if successful.
        """

        self._array[:] = _construct_affine3d(
            origin.as_array(), xAxis.as_array(), yAxis.as_array(), zAxis.as_array()
        )

        return True

    def set_to_align_coordinate_systems(
        self,
        fromOrigin: Point3D,
        fromXAxis: Vector3D,
        fromYAxis: Vector3D,
        fromZAxis: Vector3D,
        toOrigin: Point3D,
        toXAxis: Vector3D,
        toYAxis: Vector3D,
        toZAxis: Vector3D,
    ) -> bool:
        """
        Sets this matrix to be the matrix that maps from the 'from' coordinate system to the 'to' coordinate system.
        """
        from_ = self.set_with_coordinate_system(
            fromOrigin, fromXAxis, fromYAxis, fromZAxis
        )
        to_ = self.set_with_coordinate_system(toOrigin, toXAxis, toYAxis, toZAxis)

        self._array = np.linalg.inv(from_).dot(to_)
        return True

    def set_to_rotate_to(
        self, _from: Vector3D, to: Vector3D, axis: Vector3D = None
    ) -> bool:
        """
        Sets to the matrix of rotation that would align the 'from' vector with the 'to' vector.
        """
        rotvec = _from.cross(to)
        if np.allclose(rotvec, 0):
            # Vectors are either parallel or opposite
            if _from.dot(to) < 0 and axis is not None:
                # Vectors are opposite, use given axis for rotation
                rotvec = axis
        else:
            return False

        ang = np.arccos(np.clip(_from.dot(to), -1.0, 1.0))
        r = R.from_rotvec(ang * rotvec.as_array())
        self._array[:3, :3] = r.as_matrix()
        return True

    def set_to_rotation(self, angle: float, axis: Vector3D, origin: Point3D) -> bool:
        """
        Sets this matrix to the matrix of rotation by the specified angle, through the specified origin, around the specified axis.
        """
        if not isinstance(axis, Vector3D) or not isinstance(origin, Point3D):
            raise ValueError(
                "Axis and origin must be Vector3D and Point3D instances respectively"
            )

        r = R.from_rotvec(np.radians(angle) * axis)
        self._array[:3, :3] = r.as_matrix()
        self._array[:3, 3] = origin
        return True

    @property
    def translation(self) -> Vector3D:
        """
        Gets and sets the translation component of the matrix.
        """
        return Vector3D(self._array[:3, 3])

    @translation.setter
    def translation(self, value: Vector3D):
        """
        Gets and sets the translation component of the matrix.
        """
        if not isinstance(value, Vector3D):
            raise ValueError("Translation must be Vector3D instance")
        self._array[:3, 3] = value


__all__ = ["Vector2D", "Vector3D", "Point2D", "Point3D", "Matrix2D", "Matrix3D"]
