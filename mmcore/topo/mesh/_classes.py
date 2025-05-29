from __future__ import annotations
from typing import Collection,TypedDict

import numpy as np


class Tessellation(TypedDict):
    vertices: np.ndarray
    segments: np.ndarray
    position: np.ndarray
    triangles: np.ndarray
class Mesh(TypedDict):
    position: np.ndarray
    faces: np.ndarray
    


def tess_to_mesh(tessellation:Tessellation)->Mesh:
    return Mesh(position=tessellation["position"],faces=tessellation["triangles"])