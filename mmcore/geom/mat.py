#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"

import pydantic


class TreeJsUserData(dict):
    ...


class TreeJsPhongMaterial(pydantic.BaseModel):
    uuid: str = "cb951742-2566-4fd9-89c8-c76cbcf9ff85",
    type: str = "MeshPhongMaterial",
    name: str = "c",
    color: int = 8026746,
    emissive: int = 0,
    specular: int = 1118481,
    shininess: int = 30,
    reflectivity: float = 1.8,
    refractionRatio: float = 0.98,
    side: int = 2,
    transparent: bool = True,
    depthFunc: int = 3,
    depthTest: bool = True,
    depthWrite: bool = True,
    colorWrite: bool = True,
    stencilWrite: bool = False,
    stencilWriteMask: int = 255,
    stencilFunc: int = 519,
    stencilRef: int = 0,
    stencilFuncMask: int = 255,
    stencilFail: int = 7680,
    stencilZFail: int = 7680,
    stencilZPass: int = 7680,
    alphaTest: int = 1,
    userData: TreeJsUserData = TreeJsUserData(dict())
