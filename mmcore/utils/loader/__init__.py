import os
import sys

sys.path.append(f"{os.getenv('HOME')}/PycharmProjects/mmcore")

import os
import pickle
import dill

GH_STATE_PATH = f"{os.getenv('HOME')}/.cxm/ghstate.pkl"


class Loader:
    settings = dict(grasshopper_out_path=f"{os.getenv('HOME')}/.cxm/ghout.pkl",
                    grasshopper_in_path=f"{os.getenv('HOME')}/.cxm/ghin.pkl",
                    grasshopper_state_path=GH_STATE_PATH
                    )

    def __init__(self, **kwargs):
        super().__init__()
        self.settings = kwargs

    def load(self, path: str):
        with open(path, 'rb') as f:
            try:
                return pickle.load(f, **self.settings)
            except pickle.PickleError as err:
                return dill.load(f, **self.settings)
            except Exception as err:
                raise Exception("Double Pickle Exception!", [err, err])

    def loads(self, buffer: bytes):
        try:
            return pickle.loads(buffer, **self.settings)
        except pickle.PickleError as err:
            return dill.load(buffer, **self.settings)
        except Exception as err:
            raise Exception("Double Pickle Exception!", [err, err])

    def dumps(self, obj, **options):

        try:
            return pickle.dumps(obj, **options)
        except pickle.PickleError as err:
            return dill.dumps(obj, **options)
        except Exception as err:
            raise Exception("Double Pickle Exception!", [err, err])

    def dump(self, obj, path, **options):
        with open(path, 'wb') as f:
            try:
                return pickle.dump(obj, f, **options)
            except pickle.PickleError as err:
                return dill.dump(obj, f, **options)
            except Exception as err:
                raise Exception("Double Pickle Exception!", [err, err])

    def load_from_gh(self):
        return self.load(self.settings.get("grasshopper_out_path"))

    def write_from_gh(self, obj, **options):
        return self.dump(obj, self.settings.get("grasshopper_in_path"), **options)


class GHContext:
    def __init__(self, state_path=GH_STATE_PATH, **settings):

        self._state_path = state_path
        if not os.path.exists(self._state_path):
            with open(self._state_path, "wb") as f:
                pickle.dump(dict(), f)

        self._last_check_update = self.last_update()

        self._loader = Loader(**settings)
        self._state = self._loader.load(self._state_path)

    def last_update(self):

        return os.lstat(self._state_path).st_atime

    def is_updated(self):
        return self._last_check_update != self.last_update()

    def pull(self):
        self._state = Loader().load(self._state_path)

    def keys(self):
        if self.is_updated():
            self.pull()
        return self._state.keys()

    def push(self):

        self._loader.dump(self._state, path=self._state_path)

    def __getitem__(self, item):
        if self.is_updated():
            self.pull()
        return self._state.__getitem__(item)

    def __setitem__(self, key, value):
        self._state.__setitem__(key, value)
        self.push()

    def __iter__(self):
        return self._state.__iter__()

    def __len__(self):
        return self._state.__len__()
