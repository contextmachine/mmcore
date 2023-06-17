import uuid as _uuid

from mmcore.base import APoints, A
from mmcore.base.params import ParamGraphNode
from mmcore.geom.materials import ColorRGB


class Param:
    __param_item__ = None
    default = None

    def __init__(self, default=None):
        super().__init__()
        if default is None:
            default = self.__param_item__()
        self.default = default

    def __class_getitem__(cls, item):
        cls.__param_item__ = item

        return type(f"{cls.__name__}[{item.__qualname__}]", (cls,),
                    {"__qualname__": f'{cls.__name__}[{item.__qualname__}]', "__param_item__": item})


class NoParam(Param):
    ...


class ComponentType(type):
    def __new__(mcs, classname, bases, attrs=dict(), **kws):
        if len(bases) == 0:
            bases = bases + (object,)
        _attrs = dict()
        params = dict()
        print(attrs)
        if attrs.get('__annotations__') is not None:
            for k, v in attrs.get('__annotations__').items():

                if k in attrs.keys():
                    default = attrs[k]
                    if not issubclass(v, NoParam):
                        params[k] = default

                    _attrs[k] = attrs[k]
                else:
                    if not issubclass(v, NoParam):
                        default = v()
                        params[k] = default
                        print(k, default)
                        _attrs[k] = default
                    else:
                        _attrs[k] = v.__param_item__()

        attrs['__wrapped_new__'] = attrs.get("__new__") if attrs.get("__new__") \
                                                           is not None else lambda cls, *args: object.__new__(cls)

        attrs['__wrapped_call__'] = attrs.get("__call__") if attrs.get("__call__") is not None else lambda self, *args,
                                                                                                           **kwargs: self
        attrs['__component_params__'] = params
        attrs['__component_attrs__'] = _attrs
        attrs['__component_attrs_keys__'] = list(_attrs.keys())

        def new(cls, *args, uuid=None, **kwargs):

            if uuid is None:
                uuid = _uuid.uuid4().hex

            _args = dict(zip(cls.__component_attrs_keys__[:len(args)], args))
            inst = object.__new__(cls)
            inst.__call_count__ = 0
            inst.uuid = uuid
            inst.__component_attrs__ |= _args

            _args |= kwargs

            for key, vl in params.items():
                if hasattr(inst, key):
                    inst.__component_params__[key] = getattr(inst, key, None)
                else:
                    inst.__component_params__[key] = vl

            node = ParamGraphNode(inst.__component_params__, uuid=inst.uuid,
                                  name=inst.name if hasattr(inst, "name") else None, resolver=inst)
            inst.param_node = node
            inst.param_node(**_args)

            return node

        def call(slf, *ar, **kwar):
            slf.__call_count__ += 1
            for k, v in kwar.items():
                if v is not None:
                    setattr(slf, k, v)
            slf.__wrapped_call__(*ar, **kwar)
            slf.__repr3d__()

            return slf

        new.__name__ = '__new__'
        call.__name__ = '__call__'
        attrs['__new__'] = new
        attrs['__call__'] = call
        return super().__new__(mcs, classname, bases, attrs, **kws)


class Component(metaclass=ComponentType):

    def __repr3d__(self):
        if not hasattr(self, "_repr3d"):
            self._repr3d = A(uuid=self.uuid)
        if hasattr(self, "param_node"):
            self._repr3d.controls = self.param_node.todict()

        return self._repr3d

    def root(self):
        return self.__repr3d__().root()


from mmcore.base.models import gql


class ControlPoints(Component):
    x: float
    y: float
    z: float

    name: NoParam[str] = "ControlPoints"

    color: tuple = 157, 75, 75
    size: float = 1.0

    def __repr3d__(self):
        self._repr3d = APoints(geometry=[self.x, self.y, self.z],

                               material=gql.PointsMaterial(color=ColorRGB(*self.color).decimal,
                                                           size=self.size),
                               uuid=self.uuid,
                               name=self.name
                               )
        return super().__repr3d__()
