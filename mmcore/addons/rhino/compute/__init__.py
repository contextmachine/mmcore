from compute_rhino3d import Mesh, Surface, Util

from setupsecrets import SecretsManager

with SecretsManager(logging=False, update=True) as secrets:
    try:
        Util.url = f"http://{secrets['RHINO_COMPUTE_URL']}:{secrets['RHINO_COMPUTE_PORT']}/"
        if secrets.get("RHINO_COMPUTE_AUTHTOKEN") is not None:
            Util.authToken = secrets.get("RHINO_COMPUTE_AUTHTOKEN")
        if secrets.get("RHINO_COMPUTE_APIKEY") is not None:
            Util.apiKey = secrets.get("RHINO_COMPUTE_APIKEY")

    except KeyError as err:
        print(f"Secrets Key Error {err}. \nDo not worry, default configuration from {__file__} will be used.")
        # Check yours default local host and port for compute.rhino3d or rhino.compute.
        # If you can't find compute.rhino3d or rhino.compute, you can build this from source.
        # We use a "JetBrains Rider" and this source code, but encourage you to read the solution and documentation
        # from the maintainer.
        # For more information, read original docs: https://developer.rhino3d.com/guides/compute/development/
        Util.url = "http://localhost:8081/"

from .request_models import ComputeRequest, GHRequest, SlimGHRequest, AdvanceGHRequest

__all__ = [
    "mmconfigs"
    "compute_rhino3d",
    "Util",
    "Surface",
    "Mesh",
    "ComputeRequest",
    "GHRequest",
    "SlimGHRequest",
    "AdvanceGHRequest"
    ]
