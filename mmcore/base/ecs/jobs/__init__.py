import time

PROFILE_JOBS = True
from mmcore.base.sharedstate import serve


class Job:
    """
>>> from mmcore.base.ecs.jobs import Job
>>> j1=Job('render')
>>> j2=Job('evaluate')
>>> j3=Job('update')
>>> j4=Job('generate dxf')
>>> j3.schedule(j2)
>>> j2.schedule(j1)
>>> j2.schedule(j4)
>>> j3.execute()
update (0.0, 9.5367431640625e-07)
evaluate (0.0, 1.6689300537109375e-06)
render (0.0, 9.5367431640625e-07)
generate dxf (0.0, 0.0)
>>> j5=Job('update gsheet')
>>> j6=Job('create snapshots')
>>> j1.schedule(j5,j6)
>>> j3.execute()
update (0.0, 2.1457672119140625e-06)
evaluate (0.0, 9.5367431640625e-07)
render (0.0, 0.0)
update gsheet (0.0, 9.5367431640625e-07)
create snapshots (0.0, 0.0)
generate dxf (0.0, 9.5367431640625e-07)

    """

    def __init__(self, name: str, data=None):
        self.name = name
        self.data = data
        self.result = None
        self.handles = []
        serve.jobs[self.name] = self

    def schedule(self, *handles):
        self.handles.extend(handles)

    def complete(self):
        for handle in self.handles:
            handle.execute()

    def execute(self):
        if PROFILE_JOBS:
            s = time.time()
            self()
            print(self.name, divmod(time.time() - s, 60))
        else:
            self()
        self.complete()

    def __call__(self):
        ...
