from queue import Queue

import time

from mmcore.base import AGroup
from mmcore.base.ecs.components import request_component
from mmcore.base.registry import adict, propsdict
from mmcore.base.userdata.props import Props
from mmcore.common.models.observer import Observable, Observer, observation

ENABLE_WARNINGS = True


def disable_warnings():
    global ENABLE_WARNINGS
    ENABLE_WARNINGS = False


def enable_warnings():
    global ENABLE_WARNINGS
    ENABLE_WARNINGS = True


def props_update_support_warning(cls):
    if ENABLE_WARNINGS:
        raise UserWarning(f"{cls} instance support this property by default . The user argument will be ignored!")



def solver(uid):
    props = request_component(Props.component_type, uid)
    if 'u' in props.keys():

        props['u'] = 'BEBEBEBEBEBEBEBE'
        props['reviewer'] = 'Sofiya'
    else:
        print("Sofiya miss u :(")


update_queue = Queue()


class ViewerGroup(AGroup):

    def __new__(cls, items=(), /,
                uuid=None,
                name="ViewerGroup",
                entries_support=True,
                **kwargs):
        if 'props_update_support' in kwargs:
            del kwargs['props_update_support']
            props_update_support_warning(cls)
        return super().__new__(cls, seq=items, uuid=uuid, name=name,
                               entries_support=entries_support,
                               props_update_support=True,
                               **kwargs
                               )

    def props_update(self, uuids: list[str], props: dict):
        s = time.time()
        self.make_dirty()
        for uid in uuids:
            print(uid, props)
            propsdict[uid].update(props)
        m, sec = divmod(time.time() - s, 60)
        print(f'updated at {m} min, {sec} sec')

        return True


class ViewerObservableGroup(ViewerGroup, Observable):
    """
    >>> group_observer=observation.init_observer(ViewerGroupObserver)
    >>> group=observation.init_observable(group_observer,
    ...                                   cls=lambda x:ViewerObservableGroup(x, items=(),uuid='fff'))
    """

    def __new__(cls, i, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)

        return self

    def __init__(self, i, *args, **kwargs):
        super().__init__(i)

    def props_update(self, uuids: list[str], props: dict):
        super().props_update(uuids, props)
        self.notify_observers(uuids=uuids, props=props)

        return True


class ViewerGroupObserver(Observer):
    def notify(self, observable: ViewerObservableGroup, uuids: list = None, props: list = None):
        for uid in uuids:
            print(uid, uuids)
            mesh = adict[uid]
            print(mesh)
            if hasattr(mesh, 'owner'):
                print(mesh.owner)
                mesh.owner.update_mesh()


group_observer = observation.init_observer(ViewerGroupObserver)


def create_group(uuid: str):
    return observation.init_observable(group_observer,
                                       cls=lambda x: ViewerObservableGroup(x, items=(), uuid=uuid))
