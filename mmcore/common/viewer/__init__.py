from typing import Type
from uuid import uuid4
from queue import Queue

import time

from mmcore.base import A, AGeom, AGroup, AMesh
from mmcore.base.ecs.components import request_component
from mmcore.base.registry import adict, propsdict
from mmcore.base.userdata.props import Props
from mmcore.common.models.observer import Observable, Observer, observation, Observation

ENABLE_WARNINGS = True


def disable_warnings():
    global ENABLE_WARNINGS
    ENABLE_WARNINGS = False


def enable_warnings():
    global ENABLE_WARNINGS
    ENABLE_WARNINGS = True


def props_update_support_warning(cls):
    if ENABLE_WARNINGS:
        raise UserWarning(f"{cls} instance support this property by default . The user argument will be ignored!"
                )


def solver(uid):
    props = request_component(Props.component_type, uid)
    if "u" in props.keys():
        props["u"] = "BEBEBEBEBEBEBEBE"
        props["reviewer"] = "Sofiya"
    else:
        print("Sofiya miss u :(")


update_queue = Queue()


class ViewerBaseGroup(AGroup):
    def __new__(cls, items=(), /, uuid=None, name="ViewerBaseGroup", entries_support=True, **kwargs):
        if "props_update_support" in kwargs:
            del kwargs["props_update_support"]
            props_update_support_warning(cls)
        return super().__new__(cls, items, uuid=uuid, name=name, entries_support=entries_support,
                               props_update_support=True, **kwargs
                               )

    def props_update(self, uuids: list[str], props: dict):
        s = time.time()
        self.make_dirty()
        for uid in uuids:
            print(uid, props)
            propsdict[uid].update(props)
        m, sec = divmod(time.time() - s, 60)
        print(f"updated at {m} min, {sec} sec")

        return True


class ViewerObservableGroup(ViewerBaseGroup, Observable):
    """
    >>> group_observer=observation.init_observer(ViewerGroupObserver)
    >>> group=observation.init_observable(group_observer,
    ...                                   cls=lambda x:ViewerObservableGroup(x, items=(),uuid='fff'))
    """
    i = 0

    def __new__(cls, items=(), uuid=None, name="ViewerGroup", **kwargs):
        self = super().__new__(cls, items, uuid=uuid, name=name, **kwargs)

        return self

    def __init__(self, i=0, *args, **kwargs):
        super().__init__(i)

    def props_update(self, uuids: list[str], props: dict):
        self.make_dirty()
        # super().props_update(uuids, props)
        self.notify_observers(uuids=uuids, props=props)
        # self.notify_observers_backward(uuids=uuids, props=props)

        # self.notify_observers_forward(uuids=uuids, props=props)

        return True


from mmcore.base.userdata.controls import (decode_control_points, encode_control_points, CONTROL_POINTS_ATTRIBUTE_NAME,
                                           set_points_in_controls,
    )


class ViewerControlPointsObserver(Observer):
    def notify(self, observable: A, control_points: dict = None, **kwargs):
        self.notify_backward(uuid=observable, control_points=control_points)

        self.notify_forward(uuid=observable, control_points=control_points, **kwargs)

    def notify_backward(self, uuid, control_points: dict = None, **kwargs):
        mesh = adict.get(uuid) if isinstance(uuid, str) else uuid
        print(mesh)
        if hasattr(mesh, "owner"):
            mesh.owner.control_points = decode_control_points(control_points)

    def notify_forward(self, uuid, **kwargs):
        mesh = adict.get(uuid) if isinstance(uuid, str) else uuid
        print(mesh)
        if hasattr(mesh, "owner"):
            print(mesh)
            set_points_in_controls(mesh._controls, encode_control_points(mesh.owner.control_points)
                    )
            if hasattr(mesh.owner, "parent"):
                mesh.owner.parent.solve()

            mesh.owner.update_mesh(no_back=True)


class ViewerGroupObserver(Observer):
    def notify(self, observable: ViewerObservableGroup, uuids: list = None, props: list = None, **kwargs, ):
        # Также это единственное место где мы знаем весь пулл обновлений
        # Нам нужно:
        # 1. Обновить значения атрибутов от представлений к родителям
        print(self, observable, uuids, props, kwargs)
        self.notify_backward(observable=observable, uuids=uuids, props=props)

        # Здесь значения атрибутов обновлены и родители могут произвести нужные вычисления
        # TODO: Вероятно нужен какой-то крючок оповещающий о том что все обновления получены

        # 2. Обновить значения атрибутов от родителей к их представлениям
        self.notify_forward(observable=observable, uuids=uuids
                )  # Здесь все представления и значения обновлены

    def notify_backward(self, observable: ViewerObservableGroup, uuids: list = None, props: list = None, **kwargs, ):
        """
        Здесь мы итерируемся строго по uuids чтобы не обновлять того что не должно было обновиться
        :param observable:
        :type observable:
        :return:
        :rtype:
        """
        for uid in uuids:
            mesh = adict.get(uid, None)
            print(mesh)
            mesh.properties.update(props)
            if hasattr(mesh, "owner"):
                print(mesh.owner)
                mesh.owner.apply_backward(props)

    def notify_forward(self, observable: AGroup, uuids=None, **kwargs):
        """
        Здесь мы итерируемся строго по детям тк предполагаем что в результате обновления могли добавиться или
        удалиться некоторые представления, а также измениться сами uuid
        :param observable:
        :type observable:
        :return:
        :rtype:
        """
        if hasattr(observable, "owner"):
            observable.owner.solve()

        for u in uuids:
            # print(mesh)

            mesh = adict[u]
            if isinstance(mesh, AGroup):
                if hasattr(mesh, "owner"):
                    mesh.owner.solve()


            elif isinstance(mesh, AMesh):
                if hasattr(mesh, "owner"):
                    # print(mesh.owner)
                    if hasattr(mesh.owner, "solve"):
                        mesh.owner.solve()

                    mesh.owner.update_mesh(no_back=True)

            else:
                print(mesh, "pass")


group_observer = observation.init_observer(ViewerGroupObserver)
control_points_observer = observation.init_observer(ViewerControlPointsObserver)
Group = ViewerObservableGroup


class GroupFabric:
    """
    >>> from mmcore.common.viewer import DefaultGroupFabric
    >>> vecs = unit(np.random.random((2, 4, 3)))
    >>> boxes = [Box(10, 20, 10), Box(5, 5, 5), Box(15, 5, 5), Box(25, 20, 2)]
    >>> for i in range(4):
    boxes[i].xaxis = vecs[0, i, :]
    boxes[i].origin = vecs[1, i, :] * np.random.randint(0, 20)
    boxes[i].refine(('y','z'))
from mmcore.common.viewer import DefaultGroupFabric
group = DefaultGroupFabric([bx.to_mesh() for bx in boxes], uuid='fabric-group')
    """

    def __init__(self, observer_fabric: Observation, default_group_cls: Type[ViewerBaseGroup] = ViewerObservableGroup,
                 observers: tuple[Observer] = ()):
        self.observer_fabric = observer_fabric
        self.default_group_cls = default_group_cls
        self.observers = list(observers)

    def __call__(self, items=(), *args, **kwargs):
        return self.observer_fabric.init_observable_obj(*self.observers,

                obj=self.prepare_group_ctor(items=items, *args, **kwargs
                                            )
                )

    def prepare_group_ctor(self, items=(), *args, uuid=None, **kwargs):

        return self.default_group_cls(items=items, uuid=self.uuid_generator(uuid), *args, **kwargs)

    def uuid_generator(self, value=None):
        if value is None:
            return uuid4().hex
        else:
            return value


ViewerGroup = ViewerObservableGroup

DefaultGroupFabric = GroupFabric(observation, ViewerObservableGroup, observers=(group_observer,))

group_fabric = DefaultGroupFabric

def create_group(uuid: str, *args, obs=group_observer, cls=ViewerObservableGroup, **kwargs):
    def ctor(x):
        obj = cls(*args, uuid=uuid, **kwargs)
        obj.i = x
        return obj

    return observation.init_observable(obs, cls=lambda x: ctor(x))


def group(*args, uuid: str = None, obs=group_observer, cls=ViewerObservableGroup, **kwargs):
    uuid = uuid4().hex if uuid is None else uuid
    return create_group(uuid, *args, obs=obs, cls=cls, **kwargs)
