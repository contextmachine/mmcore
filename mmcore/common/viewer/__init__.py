from queue import Queue

import time

from mmcore.base import ACacheSupport, AGroup
from mmcore.base.ecs.components import request_component
from mmcore.base.registry import propsdict
from mmcore.base.userdata.props import Props

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


class ViewerModel(ACacheSupport):
    def __new__(cls, items=None, /,
                uuid=None,
                name="ViewerGroup",
                user_data_extras=None,
                entries_support=True,
                props_update_support=True,
                **kwargs):
        return super().__new__(cls, items, uuid=uuid, name=name,
                               _user_data_extras=user_data_extras,
                               entries_support=entries_support,
                               props_update_support=props_update_support,
                               **kwargs
                               )


def solver(uid):
    props = request_component(Props.component_type, uid)
    if 'u' in props.keys():

        props['u'] = 'BEBEBEBEBEBEBEBE'
        props['reviewer'] = 'Sofiya'
    else:
        print("Sofiya miss u :(")


update_queue = Queue()
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
            print(uid, props)
            propsdict[uid].update(props)
            solver(uid)
            update_queue.put((uid, props))
        m, sec = divmod(time.time() - s, 60)
        print(f'updated at {m} min, {sec} sec')

        return True
