from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mmcore")
except PackageNotFoundError:  # source checkout, not installed (RESTRUCTURE.md §4.3)
    __version__ = "0.0.0+source"
