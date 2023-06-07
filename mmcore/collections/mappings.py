from .multi_description import MultiGetitem2


class MultiDict(dict):
    """"""

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __setitem__(self, k, v):

        try:
            item = dict.__getitem__(self, k)
            item.append(v)
        except KeyError:
            item = [v]
            dict.__setitem__(self, k, item)
        except Exception as err:
            #print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __getitem__(self, __k):
        return dict.__getitem__(self, __k)


class MDict(dict):
    __getitem__ = MultiGetitem2()


class pathdict(dict):

    def __getitem__(self, keys):
        if len(keys) == 1:
            #print("final: ", keys)
            return dict.__getitem__(self, keys[0])
        else:
            k = keys.pop(0)
            #print(k, keys)

            return self[k].__getitem__(keys)

    def __setitem__(self, keys, v):
        if len(keys) == 1:
            #print("final: ", keys)
            return dict.__setitem__(self, keys[0], v)
        else:
            k = list(keys).pop(0)
            #print(k)
            if self.get(k) is None:
                self[k] = pathdict({})
            self[k].__setitem__(keys, v)
