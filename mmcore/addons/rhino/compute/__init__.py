import os

import compute_rhino3d
import dotenv
from compute_rhino3d import AreaMassProperties, Brep, Curve, Mesh, NurbsCurve, SubD, Surface, Util

from mmcore.utils import EnvString

LOGGING = False
UPDATE = False
USE = "dotenv"

match USE:
    case "repo":

        try:
            Util.apiKey = os.getenv("RHINO_COMPUTE_APIKEY")

        except KeyError as err:
            print(f"Secrets Key Error {err}. \nDo not worry, default configuration from {__file__} will be used.")
            # Check yours default local host and port for compute.rhino3d or rhino.compute.
            # If you can't find compute.rhino3d or rhino.compute, you can build this from source.
            # We use a "JetBrains Rider" and this source code, but encourage you to read the solution and documentation
            # from the maintainer.
            # For more information, read original docs: https://developer.rhino3d.com/guides/compute/development/
            Util.url = "http://localhost:8081/"

    case "dotenv":
        dotenv.load_dotenv(dotenv.find_dotenv(".env", usecwd=True))

Util.url = EnvString("http://${RHINO_COMPUTE_URL}:${RHINO_COMPUTE_PORT}/")
from .request_models import ComputeRequest, GHRequest, SlimGHRequest, AdvanceGHRequest
from .geometry import surf_to_buffer_geometry, surf_to_buffer_mesh, surf_to_buffer, surface_closest_normals

__all__ = [
    "compute_rhino3d",
    "Util",
    "Surface",
    "Mesh",
    "Brep",
    "Curve",
    "NurbsCurve",
    "SubD",
    "AreaMassProperties",
    "ComputeRequest",
    "GHRequest",
    "SlimGHRequest",
    "AdvanceGHRequest",
    "geometry", "surf_to_buffer_geometry", "surf_to_buffer_mesh", "surf_to_buffer", "surface_closest_normals"
]
