import dill

from mmcore.base.params import AssetType

try:
    with open(f'{"/".join(__file__.split("/")[:-1])}/asset.pkl', 'rb') as fl:
        _AssetExample = dill.load(fl)
except FileNotFoundError as err:
    import subprocess as sp

    proc = sp.Popen(["python3", f'{"/".join(__file__.split("/")[:-1])}/definition.py'], stdout=sp.PIPE, stderr=sp.PIPE)
    proc.communicate()
    with open(f'{"/".join(__file__.split("/")[:-1])}/asset.pkl', 'rb') as fl:
        _AssetExample = dill.load(fl)


class LineAssetExample(metaclass=AssetType):
    """
    >>> from mmcore.base.params.asset_example import LineAssetExample
    >>> detail1= LineAssetExample()
    >>> detail2= LineAssetExample()
    >>> detail2(t=0.3).to_dict()
{'t': 0.3, 'line': {'start': [1, 2, 3], 'end': [3, 2, 5]}}
    >>> detail1.todict()
{'t': 0.5, 'line': {'start': [1, 2, 3], 'end': [3, 2, 5]}}

detail2=AssetExample()
detailw(t=0.3)
    """
    __blueprint__ = _AssetExample


__all__ = ["LineAssetExample"]
