#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"

from typing import Any, Generic, TypeVar

from ..baseitems import Base, Identifiable, Item

SP = TypeVar("SP")  # descriptor
KT = TypeVar("KT")
VT = TypeVar("VT")  # descriptor
T1 = TypeVar("T1", bound=Base)
T2 = TypeVar("T2")


class D3(Generic[T1, SP, KT, VT], Base, dict[str, VT]):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dict[str, VT].__init__(self, **self.__dict__)

    def __getitem__(self, k: KT) -> VT: return dict[KT, VT].__getitem__(self, k)

    def __setitem__(self, k: KT, v: VT) -> None: dict[KT, VT].__setitem__(self, k, v)

    def __call__(self, **kwargs) -> Generic[T1, SP, KT, VT]:
        Base.__call__(self, **kwargs)
        self.update(kwargs)  # self: D3[T1, SP, KT, VT] kwargs:dict[str, VT]
        return self


DctBaseI = D3[Base, dict, str, Any]
DctIdentifiableI = D3[Identifiable, dict, str, Any]
DctItem = D3[Item, dict, str, Any]

from ...mm.geom.buffer import BufferGeometryOcc, TrimmingCone

BG_OCC = TypeVar("BG_OCC", bound=BufferGeometryOcc)

DctBuffer = D3[BG_OCC, dict, str, Any]
DctCone = DctBuffer[TrimmingCone]
