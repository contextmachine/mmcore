from __future__ import annotations

from os import PathLike

from pathlib import Path
from .step_writer import StepWriter
from mmcore.geom.nurbs import NURBSSurface
from mmcore.geom._nurbs_eval import NURBSSurfaceTuple

__all__=["to_step", "StepWriter"]

def to_step(fp: str | PathLike | Path, objects: list[NURBSSurface | NURBSSurfaceTuple], *args, **kwargs):
    writer = StepWriter()
    
    for obj in objects:
        if isinstance(obj, (NURBSSurface, NURBSSurfaceTuple)):
            writer.add_nurbs_surface(obj)
        else:
            raise ValueError(f"NURBSSurface or NURBSSurfaceTuple was expected, not {obj.__class__.__name__}.")
    with open(fp, 'w') as f:
        writer.step_file.write(f)
