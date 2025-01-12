from __future__ import annotations

from enum import IntEnum, Enum


class Curve2DTypes(IntEnum):
    """
    The different types of 2D curves.
    """

    Line2Dcurve_type = 0
    Arc2Dcurve_type = 1
    Circle2Dcurve_type = 2
    Ellipse2Dcurve_type = 3
    EllipticalArc2Dcurve_type = 4
    InfiniteLine2Dcurve_type = 5
    NurbsCurve2Dcurve_type = 6


class Curve3DTypes(IntEnum):
    """
    The different types of 3D curves.
    """

    Line3Dcurve_type = 0
    Arc3Dcurve_type = 1
    Circle3Dcurve_type = 2
    Ellipse3Dcurve_type = 3
    EllipticalArc3Dcurve_type = 4
    InfiniteLine3Dcurve_type = 5
    NurbsCurve3Dcurve_type = 6


class NurbsSurfaceProperties(IntEnum):
    """
    The different surface property types.
    """

    OpenNurbsSurface = 1
    ClosedNurbsSurface = 2
    PeriodicNurbsSurface = 4
    RationalNurbsSurface = 8


class SurfaceTypes(IntEnum):
    """
    The different types of surfaces.
    """

    Planesurface_type = 0
    Cylindersurface_type = 1
    Conesurface_type = 2
    Spheresurface_type = 3
    Torussurface_type = 4
    EllipticalCylindersurface_type = 5
    EllipticalConesurface_type = 6
    Nurbssurface_type = 7


class TextureTypes(IntEnum):
    """
    The different types of textures.
    """

    UnknownTexture = 0
    ImageTexture = 1
    CheckerTexture = 2
    GradientTexture = 3
    MarbleTexture = 4
    NoiseTexture = 5
    SpeckleTexture = 6
    TileTexture = 7
    WaveTexture = 8
    WoodTexture = 9


class IntersectionType(str, Enum):
    """
    The different types of intersections.
    Syntax:
    Type defined from string[4]
    string[0]: View type od first member
    string[1] {Geometry type of first member}
    string[2] {View type od second member}
    string[3] {Geometry type of second member}
    P - Parametric
    I - Implicit

    """

    PCPC = "PCPC"
    ICPC = "ICPC"
    ICIC = "ICIC"
    PSPS = "PSPS"
    ISPS = "ISPS"
    ISIS = "ISIS"
