
from operator import attrgetter, itemgetter, methodcaller
objdict=dict()
def getitemattrs(*attrs):
    def wrap(obj):
        return [getitemattr(attr)(obj) for attr in attrs]

    return wrap
def getitemattr(attr):
    def wrp(obj):
        errs = []
        try:
            return attrgetter(attr)(obj)
        except AttributeError as err1:
            errs.append(err1)
            return itemgetter(attr)(obj)
        except ExceptionGroup("", [KeyError(1), TypeError(2)]) as err2:
            errs.append(err2)
            return methodcaller(attr)(obj)
        except Exception as err3:
            errs.append(err3)
            raise errs

    return wrp
