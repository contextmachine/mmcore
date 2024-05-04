from mmcore.utils.env import load_dotenv_from_path, load_dotenv_from_stream
load_dotenv_from_path('.env')
TOLERANCE = 1e-06

class VersionInfo(str):

    def __new__(cls, val):
        self = super().__new__(cls)
        self._value = val
        return self

    def __str__(self):
        return str.__str__(self._value)

    def __repr__(self):
        return str.__str__(self._value)

    def __call__(self):
        return self._value
__version__ = VersionInfo('0.25.9')