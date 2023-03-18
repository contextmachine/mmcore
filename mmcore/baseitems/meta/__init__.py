import types

# try:
# import Rhino.Geometry as rg
# except:
from mmcore.addons import ModuleResolver
with ModuleResolver() as rsl:
    import rhino3dm
import rhino3dm as rg


# Create all the named tuple methods to be added to the class namespace
def create_method(func, namespace={}, typename="generic", arg_list=[], kwarg_dict={}, defaults=None):
    _namespace = {
        '_func': func,
        '__builtins__': {},
        '__name__': f'func_{typename}',
        }
    _namespace |= namespace
    code = f'lambda _cls, {arg_list}, {", ".join(kwarg_dict.keys())}: _tuple_new(_cls, ({arg_list}))'
    __func__ = eval(code, _namespace)
    __func__.__name__ = func.__name__
    __func__.__doc__ = f'Create new instance of {typename}({arg_list})'

    if defaults is not None:
        __func__.__defaults__ = defaults
    return __func__


class datatype(object):
    def __init__(self):
        object.__init__(self)

    def __get__(self, instance, owner):
        if instance is None:
            return "{}".format(owner.__name__)
        else:
            return "{}".format(type(instance).__name__)


class arraydatatype(datatype):
    def __init__(self):
        datatype.__init__(self)

    def __get__(self, instance, owner):
        return "{}[{}]".format(datatype.__get__(self, instance, owner),
                               datatype.__get__(self, None, instance.__param_type__))


Encodable = str, int, float, bytes, bool


class _Encodeble(object):
    datatype = ""

    def __init__(self, *args, **kwargs):
        object.__init__(self)

    def encode_extra(cls, o):
        """
        Override this method from custom for provide a custom serialise.
        :param o: Any object
        :return: dict like

        """
        raise TypeError("Could not find a way to serialise the {} object: \n\t{}".format(
            repr(o.__class__), repr(o)))

    @classmethod
    def decode_extra(cls, o):
        """
        Override this method from custom for provide a custom serialise.
        :param o: Any object
        :return: dict like

        """
        dct = {}
        for k, v in o.items():
            dct[k] = cls.decode(v)
        return cls(**dct)

    def encode(self, o):
        if isinstance(o, list):
            return [self.encode(v) for v in o]
        elif isinstance(o, _Encodeble):
            return o.asdict()
        elif isinstance(o, dict):
            return [{k: self.encode(v)} for k, v in o.items()]
        elif isinstance(o, Encodable):
            return o
        else:
            return self.encode_extra(o)

    def asdict(self):
        dct = {}
        for k in self.params:
            o = getattr(self, k)
            if isinstance(o, _Encodeble):
                dct[k] = o.asdict()
            else:
                dct[k] = self.encode(o)
        return {
            "type": self.datatype,
            "data": dct
            }

    @classmethod
    def decode(cls, obj):

        if isinstance(obj, dict):

            if (obj.get("type") is not None) and (obj.get("data") is not None):
                try:
                    cl = eval(obj.get("type"))

                except:
                    cl = encodable(obj.get("type"), params=tuple(obj.get("data").keys()))
                return cl.decode_extra(obj.get("data"))
            else:
                dct = {}
                for k, v in obj.items():
                    dct[k] = cls.decode(v)
                return dct
        elif isinstance(obj, (list, tuple, set)):
            lst = []
            for i in obj:
                lst.append(cls.decode(i))
            return lst
        elif isinstance(obj, Encodable):
            return obj

        else:
            print(obj)
            raise


class _EncodableArray(list):
    __param_type__ = None
    datatype = ""

    def asdict(self):
        dct = []
        for item in self:
            dct.append(item.asdict())
        return {
            "type": self.datatype,
            "data": dct
            }


def traverse(callback=lambda x: x):
    def wrp(seq):
        for i in seq:
            if isinstance(i, (_EncodableArray, list, set, tuple)):
                yield [wrp(j) for j in i]
            else:
                yield callback(i)

    return wrp


def traverse2(callback=lambda x: x):
    def wrp(seq):

        if isinstance(seq, dict):
            dct = {}
            for i, v in seq.items():
                dct[i] = wrp(v)
            return dct
        elif isinstance(seq, (list, tuple)):
            dt = []
            for i in seq:
                dt.append(wrp(i))
            return dt
        else:
            return callback(seq)

    return wrp


typecheck = traverse(callback=lambda x: x.__class__)


class alltypes(object):
    def __init__(self):
        object.__init__(self)
        self._types = {}

    def addtype(self, key, value):
        self._types.__setitem__(key, value)

    def __getattr__(self, key):
        if key in self._types.keys():
            return self._types.__getitem__(key)
        else:
            return object.__getattribute__(self, key)

    def keys(self):
        return self._types.keys()

    def items(self):
        return self._types.items()

    def values(self):
        return self._types.values()

    def __contains__(self, item):
        return self._types.__contains__(item)

    def __len__(self):
        return self._types.__len__()


class EncodableType(type):
    types = alltypes()

    def __new__(mcs, name, bases=(), attrs={}, params=None, initargs=None, **kws):
        if list in bases:
            class PreloadedArrayType(object):
                def __init__(self, *args, **kws):
                    self.args = args
                    self.kwargs = kws
                    object.__init__(self)

                def __call__(self, seq):
                    self.seq = seq
                    self.types = set(typecheck(self.seq))
                    self.__param_type__ = list(self.types)[0]
                    if len(self.types) == 1:

                        self.datatype = "EncodableTypes.{}[{}]".format(name, self.__param_type__.datatype)
                        bsb = list(bases)
                        bsb.remove(list)

                        attrs['__param_type__'] = self.__param_type__
                        attrs['datatype'] = self.datatype

                        cls = type(name + "[{}]".format(self.__param_type__.__name__), tuple(bsb) + (_EncodableArray,),
                                   attrs)
                        cls.superclass = (tuple(bsb) + (_EncodableArray,))[0]
                        return cls(seq, *self.args, **self.kwargs)
                    else:
                        raise TypeError("the array must contain values of the same type! \n\t{}\n{}".format(types, seq))


            return PreloadedArrayType
        else:

            def init(self, *args, **kwargs):
                self.params = params
                _args = list(args)
                self.param_values = []
                _Encodeble.__init__(self)
                for k in params:
                    if k in kwargs.keys():
                        self.param_values.append(kwargs[k])
                        continue
                    else:
                        try:
                            self.param_values.append(_args.pop(0))

                        except IndexError as err:
                            break

            def __getattr(self, k):
                if k in params:
                    return self.param_values[params.index(k)]
                else:
                    return self.__getattribute__(k)

            def __setattr(self, k, v):
                if k in params:
                    self.param_values[params.index(k)] = v
                else:
                    return _Encodeble.__setattr__(self, k, v)

            attrs["__init__"] = init
            attrs["__getattr__"] = __getattr
            attrs["__setattr__"] = __setattr

            attrs["__repr__"] = lambda self: "{}({})".format(self.asdict()["type"], "".join(
                [k + "=" + str(v) + ", " for k, v in self.asdict()["data"].items()])[:-2])
            attrs["__str__"] = attrs["__repr__"]
            attrs["GetString"] = attrs["__repr__"]
            if initargs is None:
                attrs["initargs"] = params
            else:
                attrs["initargs"] = initargs
            cls = type(name, bases + (_Encodeble,), attrs)
            cls._datatype = None
            cls.datatype = property(
                fget=lambda self: "{}".format(self._datatype) if self._datatype is not None else "{}".format(name))
            return cls


def encodable(name, params, extra=dict()):
    return EncodableType(name, (), extra, params)


EncodableTypes = EncodableType.types
EncodableArrayType = EncodableType("EncodableArrayType", (list,), {})
EncodableArray = EncodableArrayType()


def rhinotype(self):
    name = self.__class__.__name__.replace("Encodable", "")
    try:

        return eval("rg." + name)
    except ImportError as err:
        if rg.__name__ == 'rhino3dm':

            tp = encodable(name, params=self.params)
            tp.__doc__ = "It is stub"
            rg.__dict__[name] = tp
            return tp
        else:
            raise err


def fromrhino(cls, obj):
    d = []
    for k in cls.params:
        d.append(getattr(obj, k))
    return cls(*d)


class EncodableRhino(metaclass=EncodableType, params=[]):
    fromrhino = classmethod(fromrhino)
    rhtype = property(fget=rhinotype)

    def to_rhino(self):
        return self.rhtype(*self.param_values)

    def encode_extra(self, o):

        if o.__class__.__name__ in rg.__dict__:
            return eval("Encodable" + o.__class__.__name__ + ".fromrhino(o).asdict()")
        else:
            return self.__class__


class EncodablePoint3d(EncodableRhino, metaclass=EncodableType, params=("X", "Y", "Z")):
    def to_rhino(self):
        return super().to_rhino()


class EncodableVector3d(EncodablePoint3d, metaclass=EncodableType, params=("X", "Y", "Z")): ...


class EncodablePlane(EncodablePoint3d, metaclass=EncodableType, params=("Origin", "XAxis", "YAxis")): ...


class EncodableLine(EncodablePoint3d, metaclass=EncodableType, params=("From", "To")): ...


class EncodableCircle(EncodableRhino, metaclass=EncodableType, params=("Center", "Radius"),
                      initargs=("Center", "Radius")):
    def to_rhino(self):
        return self.rhtype(self.Center, self.Radius)


class EncodableRectangle3d(EncodableRhino, metaclass=EncodableType, params=("Center", "Radius", "Normal", "Plane"),
                           initargs=("Center", "Radius")): ...
