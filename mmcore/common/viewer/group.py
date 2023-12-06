import threading
import uuid as _uuid
from collections import deque
from queue import Queue

import time

from mmcore.base import AGroup, propsdict
from mmcore.base.ecs.components import request_component, request_component_type
from mmcore.base.userdata.props import Props, apply_props
from mmcore.common.viewer.mesh import Color, MeshSupport, PropsColor
from mmcore.geom.extrusion import Extrusion
from mmcore.geom.mesh import vertexMaterial
from mmcore.geom.vec import *

props_update_queue = Queue()


def mesh_support(cls, uuid=None, ref=None, **kws):
    if uuid is None:
        uuid = _uuid.uuid4().hex
    return cls(uuid, ref, **kws)


X = dict()


class UnitInterface:

    def __init__(self, uuid, rect, **kws):
        self.rect = rect
        self.uuid = uuid
        if self.uuid not in propsdict:
            propsdict[self.uuid] = Props()
        self.props = propsdict[self.uuid]
        self.props['u'] = self.rect.u
        self.props['v'] = self.rect.v
        apply_props(self.props, kws)
        self.mesh = None

        self.props['north'] = dot([0, 1, 0], unit(self.rect.yaxis)) + self.rect.origin[1]

        self.props['area'] = self.rect.area
        X[self.uuid] = self

    def sync(self):
        self.rect.v = self.props['u']
        self.rect.v = self.props['v']
        self.mesh = Extrusion(self.rect.corners, self.props['floor_h'] * self.props['floor']).to_mesh().amesh(
            uuid=self.uuid,
            material=vertexMaterial)


def solver(uid):
    props = request_component(Props.component_type, uid)
    if 'u' in props.keys():

        props['u'] = 'BEBEBEBEBEBEBEBE'
        props['reviewer'] = 'Sofiya'
    else:
        print("Sofiya miss u :(")


from mmcore.common.viewer._warn import props_update_support_warning

MeshComponentType = request_component_type(MeshSupport.component_type)
ColorComponentType = request_component_type(Color.component_type)
PropsComponentType = request_component_type(Props.component_type)
PropsColorComponentType = request_component_type(PropsColor.component_type)


class ViewerGroup(AGroup):

    def __new__(cls, items=None, /,
                uuid=None,
                name="ViewerGroup",
                entries_support=True,
                **kwargs):
        if 'props_update_support' in kwargs:
            del kwargs['props_update_support']
            props_update_support_warning(cls)
        return super().__new__(cls, items, uuid=uuid, name=name,
                               entries_support=entries_support,
                               props_update_support=True,
                               **kwargs
                               )

    def props_update(self, uuids: list[str], props: dict):
        s = time.time()
        self.make_dirty()

        for uid in uuids:
            print(obj)
            obj = adict[uid]
            obj.props.update(props)

            # propsdict[uid].update(props)
            # props_update_queue.put(uid)
        m, sec = divmod(time.time() - s, 60)
        print(f'updated at {m} min, {sec} sec')

        return True


RUN_MESH_UPDATER = True


class PropsUpdateLoop:
    def __init__(self, *jobs):
        self.jobs = deque(jobs)
        self.disabled = set()
        self.run = False
        self.thr = None

    def loop(self):

        while True:
            if not self.run:
                raise AssertionError("Stopping thread")
            if not props_update_queue.empty():

                uuid = props_update_queue.get()
                for i, j in enumerate(self.jobs):
                    if i not in self.disabled:
                        j(uuid)

    def start(self):
        self.thr = threading.Thread(target=self.loop, daemon=True)
        self.run = True
        self.thr.start()

    def disable(self, i):
        self.disabled.add(i)

    def enable(self, i):
        if i in self.disabled:
            self.disabled.remove(i)

    def bind(self, job):
        self.jobs.appendleft(job)
        return job


props_loop = PropsUpdateLoop()
