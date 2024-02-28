import uvicorn
import rich
from fastapi import FastAPI
from dataclasses import asdict

from mmcore.geom.box import Box
from mmcore.common.viewer import DefaultGroupFabric
from mmcore.numeric import remove_dim
from mmcore.geom.vec import *
from mmcore.base.sharedstate import serve


def random_box_grid(x_count=10, y_count=10, x_range=(0, 100), y_range=(0, 100), min_uvh=np.array((2, 2, 2), int),
                    max_uvh=np.array((8, 8, 8), int)):
    vecs = unit(np.random.random((x_count * y_count, 3)))
    origins = remove_dim(np.dstack(np.meshgrid(np.linspace(*x_range, x_count), np.linspace(*y_range, y_count))), 1)
    uvh = remove_dim(np.random.randint(min_uvh, max_uvh, size=(x_count, y_count, 3)))
    for i in range(int(x_count * y_count)):
        box = Box(int(uvh[i, 0]), int(uvh[i, 1]), int(uvh[i, 2]))
        box.origin = origins[i]
        box.xaxis = vecs[i, :]
        box.refine(('y', 'z'))
        yield box


group = DefaultGroupFabric([bx.to_mesh() for bx in random_box_grid()], uuid='props-update-group')

app = FastAPI()
app.mount(os.getenv('MMCORE_API_PREFIX'), serve.app)

if __name__ == "__main__":
    rich.print("\nEntries:")
    rich.print([asdict(entry) for entry in group.entries])
    rich.print(f"\nlink: {os.getenv('MMCORE_ADDRESS')}{os.getenv('MMCORE_API_PREFIX')}fetch/{group.uuid}")
    uvicorn.run("main:app", host='0.0.0.0', port=7711)
