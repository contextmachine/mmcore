from uuid import uuid4

from mmcore.base.userdata.props import Props
from mmcore.base.userdata.controls import Controls
from mmcore.common.models.fields import FieldMap

_architypes = dict()
_entities = dict()
_entities_views = dict()

from mmcore.common.models.observer import Observable, Notifier, Listener


class Entity:
    __field_map__: tuple[FieldMap] = ()
    __mmcore_views__ = (Props,)

    def update(self):
        print("updating", self.uuid)

    def __class_getitem__(cls, item):
        mmcore_views = cls.__mmcore_views__
        if isinstance(item, tuple):
            for it in item:
                if it not in mmcore_views:
                    mmcore_views += (it,)
        else:
            mmcore_views += (item,)
        if _architypes.get(mmcore_views) is None:
            _architypes[mmcore_views] = type(f'Entity[{", ".join(m.__name__ for m in mmcore_views)}]', (cls,),
                    dict(__mmcore_views__=mmcore_views), )

        return _architypes[mmcore_views]

    def __new__(cls, uuid=None, field_map=None):
        if uuid is None:
            uuid = uuid4().hex
        if uuid not in _entities:
            self = super().__new__(cls)
            self.uuid = uuid
            _entities_views[self.uuid] = dict()
            for view in cls.__mmcore_views__:
                view_name = getattr(view, "__view_name__", view.__name__)
                view_obj = view(uuid=uuid)

                _entities_views[self.uuid][view_name] = view_obj

            _entities[self.uuid] = self
            return self

        else:
            return _entities[uuid]

    @property
    def views(self):
        return _entities_views[self.uuid]

    @property
    def get_view(self, view: "str|type"):
        if isinstance(view, str):
            return _entities_views[self.uuid][view]
        else:
            return _entities_views[getattr(view, "__view_name__", view.__name__)]


ControllableEntity = Entity[Controls]
