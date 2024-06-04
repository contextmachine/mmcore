try:
    from importlib.metadata import version
except ImportError:
    # For Python<3.8, use importlib-metadata package
    # noinspection PyUnresolvedReferences
    from importlib_metadata import version

__version__ = version("your-package-name")
