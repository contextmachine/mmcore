tables = dict()


def check_ref(ref):
    if isinstance(ref, (int, str, float)):
        return ref
    else:
        return ref.uuid


stack = []
stack_links = dict()
import abc


class AbsPtr:
    @abc.abstractmethod
    def deref(self):
        ...

    def __repr__(self):
        if isinstance(self._ref, str):
            return f'{self._parent_ptr}["{self._ref}"]'
        else:

            return f'{self._parent_ptr}[{self._ref}]'

    def __str__(self):
        return self.__repr__()

    def todict(self):

        if len(stack_links[self._address]) == 0:
            return self.deref()
        dct = dict()
        for k, v in stack_links[self._address].items():
            dct[k] = stack[v].todict()
        return dct


class RootPtr(AbsPtr):
    def __new__(cls, ref="tables"):
        ref = check_ref(ref)
        for s in stack:
            if s.__class__ == cls:
                return s
        self = super().__new__(cls)
        self._ref = ref
        stack.append(self)
        self._address = len(stack) - 1
        stack_links[self._address] = dict()

        return self

    def __iter__(self):
        return iter({self._ref: self.deref()}.items())

    @property
    def links(self):
        return stack_links[self._address]

    def trav_links(self):

        return dict((k, {self._ref: stack[v].trav_links()}) for k, v in stack_links[self._address].items())

    @property
    def parent(self):
        return globals()

    def deref(self):
        return self.parent[self._ref]

    def __repr__(self):

        return f'{self._ref}'

    @property
    def uuid(self):
        return f'mmcore_{self._ref}'

    def ref(self, k, resolver=None):
        if resolver is not None:
            return Ptr(k, self, resolver=resolver)

        return Ptr(k, self)


class StackPtr(RootPtr):
    def __new__(cls, ref="stack"):
        return super().__new__(cls, ref)


STACK_PTR = StackPtr("stack")

TABLES_ROOT_PTR = RootPtr("tables")


class Ptr(AbsPtr):

    def __new__(cls, ref, parent_ptr: 'typing.Union[AbsPtr, Ptr,RootPtr]' = TABLES_ROOT_PTR, ):

        ref = check_ref(ref)
        if ref in stack_links[parent_ptr._address].keys():
            return stack[stack_links[parent_ptr._address][ref]]

        self = super().__new__(cls)

        self._ref = ref
        stack.append(self)
        self._address = len(stack) - 1
        stack_links[self._address] = dict()
        stack_links[parent_ptr._address][ref] = self._address

        self._parent_ptr = parent_ptr

        return self

    def __iter__(self):

        return iter({self._ref: self.deref()}.items())

    @property
    def links(self):
        return dict((name, stack[link]) for name, link in stack_links[self._address].items())

    def setlink(self, name, v):
        stack[stack_links[self._address][name]].set(v)

    def set_index_link(self, name, i):
        stack_links[self._address][name] = i

    def set_ptr_link(self, name, ptr):
        stack_links[self._address][name] = ptr._address

    def getlink(self, name):
        return self.getlinkptr(name).deref()

    def getlinkptr(self, name):
        return stack[stack_links[self._address][name]]

    def trav_links(self):
        if len(stack_links[self._address]) == 0:
            return self.deref()
        return dict((k, {self._ref: stack[v].trav_links()}) for k, v in stack_links[self._address].items())

    @property
    def parent(self):
        print(self._parent_ptr)
        return self._parent_ptr.deref()

    def deref(self):
        print(self.parent)
        return self.parent[self._ref]

    def set(self, v):
        if isinstance(self.parent[self._ref], dict) and isinstance(v, dict):

            self.parent[self._ref] |= v
        else:
            self.parent[self._ref] = v

    def __hash__(self):

        return hash(self.__repr__())

    def mutate(self, val):
        self.set(val)

    def __getitem__(self, k):
        return self.ref(k)

    def __setitem__(self, k, v):
        self.ref(k).set(v)

    def __call__(self, **kwargs):

        self.set(kwargs)
        return self.deref()

    def ref(self, k, ptr_type=None):
        if ptr_type is None:
            ptr_type = Ptr

        return ptr_type(k, self)

    def refset(self, k, v, ptr_type=None):
        if ptr_type is None:
            ptr_type = Ptr
        p = ptr_type(k, self)
        p.set(v)
        return p

    @property
    def uuid(self):
        return f'{self._parent_ptr.uuid}_{self._ref}'

    def redis_key(self, pk):
        tags = self.uuid.split("_")
        app = ":".join(tags[1:-1])

        return f'api:mmcore:{pk}:{app}'


OBJECTS_PTR = Ptr("objects", TABLES_ROOT_PTR)


class PtrAttribute(Ptr):
    ...


class MultiPtr(Ptr):
    def parent(self):
        self._parent_ptr, self._ptrs

    def deref(self):
        return list(map(lambda x: x.deref(), self.parent))

    def set(self, v):
        return list(map(lambda x: x.deref(), self._ptrs))


class MethodPtr(Ptr):
    def __new__(cls, ref, method, parent, **kwargs):
        obj = super().__new__(cls, ref, parent)
        obj = obj(kwargs)
        obj.deref_method = method
        return obj

    def deref(self):
        return self.deref_method(**self.parent[self._ref])


class PtrPtr(Ptr):
    def __new__(cls, ref, parent_ptr: StackPtr = STACK_PTR):
        return super().__new__(cls, ref, parent_ptr=parent_ptr)


class MultiPtr:
    def __init__(self, *ptrs):
        self._ptrs = ptrs

    def deref(self):
        dct = dict()
        for ptr in self._ptrs:
            dct[ptr._ref] = ptr.deref()
        return dct

    def __iter__(self):
        return iter(self.deref().items())
