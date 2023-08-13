import copy
import dataclasses
import typing

import strawberry

from mmcore.base.utils import deep_merge, to_camel_case


class GenericList(list):
    def __class_getitem__(cls, item):

        _name = "Generic" + cls.__base__.__name__.capitalize() + "[" + item.__name__ + "]"

        def __new__(_cls, l):

            if l == []:
                return []
            elif l is None:
                return []
            else:
                ll = []
                for i in l:
                    if i is not None:
                        try:
                            ll.append(item(**i))
                        except TypeError:
                            # ##print(item, i)
                            ll.append(item(i))
                return ll

        __ann = typing.Optional[list[item]]

        return type(f'{__ann}', (list,), {"__new__": __new__, "__origin__": list[item]})


def traverse_annotations(schema):
    schema.generate_schema()
    _classes = [c.__name__ for c in schema.classes]

    def wrap(cls):
        if cls.__name__ in _classes:
            d = {}
            for k, v in cls.__annotations__.items():
                d[k] = wrap(v)
            return d
        else:
            return cls

    return wrap(schema.schema)


class DictSchema:
    """
    >>> import strawberry
    >>> from mmcore.base import  A, AGroup
    >>> from dataclasses import is_dataclass, asdict >>> a=A(name="A") >>> b = AGroup(name="B")
    >>> b.add(a) >>> dct = strawberry.asdict(b.get_child_three()) >>> ###print(dct) {'object': {'name': 'B',
    'uuid': 'bcd5e328-c5e5-4a8f-8381-bb97cb022708', 'userData': {'properties': {'name': 'B', 'children_count': 1},
    'gui': [{'key': 'name', 'id': 'name_chart_linechart_piechart', 'name': 'Name Chart', 'colors': 'default',
    'require': ('linechart', 'piechart')}, {'key': 'children_count', 'id': 'children_count_chart_linechart_piechart',
    'name': 'Children_count Chart', 'colors': 'default', 'require': ('linechart', 'piechart')}], 'params': None},
    'matrix': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 'layers': 1, 'type': 'Group', 'castShadow': True,
    'receiveShadow': True, 'children': [{'name': 'A', 'uuid': 'c4864663-67f6-44bb-888a-5f1a1a72e974', 'userData': {
    'properties': {'name': 'A', 'children_count': 0}, 'gui': None, 'params': None}, 'matrix': [1, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 1], 'layers': 1, 'type': 'Object3D', 'castShadow': True, 'receiveShadow': True,
    'children': []}]}, 'metadata': {'version': 4.5, 'type': 'Object', 'generator': 'Object3D.toJSON'}, 'materials': [
    ], 'geometries': []} >>> ds=DictSchema(dct) >>> tp=ds.get_init_default() >>> tp.object GenericObject(name='B',
    uuid='bcd5e328-c5e5-4a8f-8381-bb97cb022708', userData=GenericUserdata(properties=GenericProperties(name='B',
    children_count=1), gui=[GenericGui(key='name', id='name_chart_linechart_piechart', name='Name Chart',
    colors='default', require=('linechart', 'piechart')), GenericGui(key='children_count',
    id='children_count_chart_linechart_piechart', name='Children_count Chart', colors='default',
    require=('linechart', 'piechart'))], params=None), matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    layers=1, type='Group', castShadow=True, receiveShadow=True, children=[GenericChildren(name='A',
    uuid='c4864663-67f6-44bb-888a-5f1a1a72e974', userData=GenericUserdata(properties=GenericProperties(name='A',
    children_count=0), gui=None, params=None), matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], layers=1,
    type='Object3D', castShadow=True, receiveShadow=True, children=())]) >>> tp.object.children [GenericChildren(
    name='A', uuid='c4864663-67f6-44bb-888a-5f1a1a72e974', userData=GenericUserdata(properties=GenericProperties(
    name='A', children_count=0), gui=None, params=None), matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    layers=1, type='Object3D', castShadow=True, receiveShadow=True, children=())]
    """

    def bind(self, cls_name: str,
             fields: typing.Iterable,
             *args, **kwargs):
        if cls_name not in self.classes_dict.keys():
            new_class = dataclasses.make_dataclass(cls_name, fields, *args, **kwargs)
            self.classes_dict[cls_name] = new_class

        self.classes.add(self.classes_dict[cls_name])
        return self.classes_dict[cls_name]

    def __init__(self, dict_example):
        self.annotations = dict()
        self.cls = object
        self.root_class_name = "schema"
        self.dict_example = DeepDict(dict_example)
        self.classes = set()
        self.classes_dict = dict()

    def generate_schema(self, callback=lambda x: x):
        def wrap(name, obj):
            if obj is None:
                return name, typing.Optional[list[dict]], None
            elif isinstance(obj, dict):

                named_f = dict()
                for k, v in obj.items():
                    fld = wrap(k, v)

                    named_f[k] = fld

                # ###print(name, named_f)
                # ###print("Generic" + to_camel_case(name),

                dcls = callback(self.bind("Generic" + to_camel_case(name),
                                          list(named_f.values())))

                init = copy.deepcopy(dcls.__init__)

                def _init(slf, **kwargs):

                    kws = dict()
                    for nm in named_f.keys():

                        _name, tp, dflt = named_f[nm]

                        # ##print(_name, tp, dflt)
                        if nm in kwargs.keys():
                            if isinstance(kwargs[nm], dict):
                                kws[nm] = tp(**kwargs[nm])
                            elif isinstance(kwargs[nm], (GenericList, tuple)):
                                kws[nm] = tp(kwargs[nm])
                            else:
                                try:
                                    kws[nm] = tp(kwargs[nm])
                                except TypeError:
                                    kws[nm] = kwargs[nm]

                        else:
                            kws[nm] = tp(dflt)
                    init(slf, **kws)

                dcls.__init__ = _init
                return name, dcls, lambda: dcls(**obj)
            elif isinstance(obj, list):
                # ###print(name, type(obj), obj)
                *nt, = zip(*[wrap(name, o) for o in obj])
                # ##print(nt)
                if len(nt) == 0:
                    return name, tuple, lambda: []
                else:
                    g = GenericList[nt[1][0]]
                    if len(nt) == 3:
                        # ##print(g)
                        return name, g, lambda: nt[-1]
                    else:
                        return name, g, lambda: []
            elif obj is None:
                return name, typing.Optional[typing.Any], None
            else:
                return name, type(obj), lambda: obj

        return wrap(self.root_class_name, self.dict_example)[1]

    @property
    def schema(self):
        return self.generate_schema()

    def get_init_default(self):

        return self.schema(**self.dict_example)

    def get_init_default_strawberry(self):

        return self.get_strawberry()(**self.dict_example)

    def get_strawberry(self):

        return strawberry.type(self.schema)

    def init_partial(self, **kwargs):
        dct = copy.deepcopy(self.dict_example)
        dct |= kwargs
        return self.schema(**dct)

    def _get_new(self, kwargs):
        cp = DeepDict(copy.copy(self.dict_example))
        cp |= kwargs
        inst = self.schema(**cp)
        inst.__schema__ = self.schema
        inst.__classes__ = list(self.classes)
        return inst

    def __call__(self, **kwargs):

        return self._get_new(kwargs)

    def decorate(self, cls):
        self.cls = cls
        return self

    def traverse_annotations(self):
        self.annotations = traverse_annotations(self.schema)
        return self.annotations


class DictGqlSchema(DictSchema):
    bind = strawberry.type


class DeepDict(dict):
    def __ior__(self, other):
        deep_merge(self, other)
        return self
