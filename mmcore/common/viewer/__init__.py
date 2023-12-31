import uuid
from queue import Queue

import time

from mmcore.base import A, AGroup
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
        self.make_dirty()
        # super().props_update(uuids, props)
        self.notify_observers(uuids=uuids, props=props)


        return True


from mmcore.base.userdata.controls import decode_control_points, encode_control_points, CONTROL_POINTS_ATTRIBUTE_NAME, \
    set_points_in_controls


class ViewerControlPointsObserver(Observer):

    def notify(self, uuid: A, control_points: dict = None, **kwargs):
        self.notify_backward(uuid=uuid, control_points=control_points)
        self.notify_forward(uuid=uuid)

    def notify_backward(self, uuid, control_points: dict = None):
        mesh = adict.get(uuid)
        if hasattr(mesh, 'owner'):

            mesh.owner.control_points = decode_control_points(control_points)

    def notify_forward(self, uuid):
        mesh = adict.get(uuid)
        if hasattr(mesh, 'owner'):
            set_points_in_controls(mesh._controls, encode_control_points(mesh.owner.control_points))

            mesh.owner.update_mesh(no_back=True)





class ViewerGroupObserver(Observer):
    def notify(self, observable: ViewerObservableGroup, uuids: list = None, props: list = None):

        # Также это единственное место где мы знаем весь пулл обновлений
        # Нам нужно:
        # 1. Обновить значения атрибутов от представлений к родителям

        self.notify_backward(observable=observable, uuids=uuids, props=props)

        # Здесь значения атрибутов обновлены и родители могут произвести нужные вычисления
        # TODO: Вероятно нужен какой-то крючок оповещающий о том что все обновления получены

        # 2. Обновить значения атрибутов от родителей к их представлениям
        self.notify_forward(observable=observable)  # Здесь все представления и значения обновлены

    def notify_backward(self, observable: ViewerObservableGroup, uuids: list = None, props: list = None):
        """
               Здесь мы итерируемся строго по uuids чтобы не обновлять того что не должно было обновиться
               :param observable:
               :type observable:
               :return:
               :rtype:
        """
        for uid in uuids:
            print(uid, uuids)
            mesh = adict.get(uid, None)
            print(mesh)
            if hasattr(mesh, 'owner'):
                print(mesh.owner)
                mesh.owner.apply_backward(props)

    def notify_forward(self, observable: ViewerObservableGroup):
        """
        Здесь мы итерируемся строго по детям тк предполагаем что в результате обновления могли добавиться или
        удалиться некоторые представления, а также измениться сами uuid
        :param observable:
        :type observable:
        :return:
        :rtype:
        """
        for mesh in observable.children:
            print(mesh)
            if hasattr(mesh, 'owner'):
                print(mesh.owner)
                mesh.owner.update_mesh(no_back=True)





group_observer = observation.init_observer(ViewerGroupObserver)
control_points_observer = observation.init_observer(ViewerControlPointsObserver)


def create_group(uuid: str, obs=group_observer, cls=ViewerObservableGroup):
    return observation.init_observable(obs,
                                       cls=lambda x: cls(x, items=(), uuid=uuid))
