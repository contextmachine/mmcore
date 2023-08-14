import dataclasses
import typing

from collections import namedtuple

try:
    from itertools import pairwise
except Exception as err:
    from more_itertools import pairwise

import numpy as np

from mmcore.geom.parametric import Linear, PlaneLinear

from mmcore.geom.parametric.algorithms import global_to_custom, line_plane_intersection

ContourSliceResult = namedtuple("ContourSliceResult", ["slice_index", "segment_index", "t", "point"])


def even_filter(iterable, reverse=False):
    def even_filter_num(item):
        return reverse != ((iterable.index(item) % 2) == 0)

    return filter(even_filter_num, iterable)


def custom_line_plane_intersection(*args, **kwargs):
    return line_plane_intersection(*args, **kwargs, full_return=True)


@dataclasses.dataclass(unsafe_hash=True)
class ContourPlane(PlaneLinear):
    def in_plane_coords(self, pt):
        return global_to_custom(pt, self.origin, self.xaxis, self.yaxis, self.normal)

    def to_plane_linear(self):
        return PlaneLinear(origin=self.origin, xaxis=self.xaxis, yaxis=self.yaxis, normal=self.normal)


ns = dict()


@dataclasses.dataclass
class ContourSlice:
    _owner: dataclasses.InitVar[typing.Optional['ContourResult']]
    start: ContourSliceResult
    end: ContourSliceResult

    def __post_init__(self, _owner: typing.Optional['ContourResult']):

        self._owner_ptr = id(_owner)

    @property
    def owner(self):
        # return ns[self._owner_ptr]
        return ns[id(self._owner_ptr)]

    @property
    def points(self):
        return self.start.point.tolist(), self.end.point.tolist()

    @property
    def planes(self):
        if self.owner is not None:
            if self.owner.planes is not None:
                return self.get_planes(self.owner.planes)

    @property
    def segments(self):
        if self.owner is not None:
            if self.owner.segments is not None:
                return self.get_segments(self.owner.segments)

    def get_planes(self, planes):

        return planes[self.start.slice_index], planes[self.end.slice_index]

    def get_segments(self, segments):
        return segments[self.start.segment_index], segments[self.end.segment_index]

    def get_eval_segments(self, segments):

        return segments[self.start.segment_index].evaluate(self.start.t), segments[self.end.segment_index].evaluate(
            self.end.t)

    @property
    def params(self):
        return self.start.t, self.end.t

    @property
    def segment_indices(self):
        return self.start.segment_index, self.end.segment_index

    @property
    def slice_indices(self):
        return self.start.slice_index, self.end.slice_index

    def __iter__(self):
        return iter([self.start.point, self.end.point])


@dataclasses.dataclass
class ContourResult:
    segments: list
    slices: typing.Optional[list[list[ContourSlice]]] = None
    planes: typing.Optional[list[ContourPlane]] = None

    def __post_init__(self):
        ns[id(self)] = self
        self._segments_as_lines = None

    def line_segments(self):
        if self._segments_as_lines is None:
            self._segments_as_lines = [Linear.from_two_points(*i) for i in self.segments]
        return self._segments_as_lines

    def __iter__(self):
        return iter([list(ss) for ss in s] for s in self.slices)


TEST_CASE = dict(
    segments=[[(802.0719024679672, -2115.6971730783057, 350.7386903275634),
               (-607.77747825974, -1954.7618508065439, 305.73869032756295)],
              [(-607.77747825974, -1954.7618508065439, 305.73869032756295),
               (-1322.530384544842, -242.12639164870774, 32.73869032756218)],
              [(-1322.530384544842, -242.12639164870774, 32.73869032756218),
               (-506.56487614725256, 479.46409180026967, 28.689451929240477)],
              [(-506.56487614725256, 479.46409180026967, 28.689451929240477),
               (473.1669189275823, 479.19102170318274, 147.01101091963884)],
              [(473.1669189275823, 479.19102170318274, 147.01101091963884),
               (1236.2007813878104, 581.8903948009078, 146.73869032756275)],
              [(1236.2007813878104, 581.8903948009078, 146.73869032756275),
               (1771.5303845448425, -346.94504500382857, 304.7386903275632)],
              [(1771.5303845448425, -346.94504500382857, 304.7386903275632),
               (1384.7931246700518, -539.9683593900033, 144.7386903275631)],
              [(1384.7931246700518, -539.9683593900033, 144.7386903275631),
               (802.0719024679672, -2115.6971730783057, 350.7386903275634)],
              [(362.4598051390722, -342.79122703908456, 33.73869032756272),
               (-70.08364349561086, -500.2241673689615, 33.73869032756264)],
              [(-70.08364349561086, -500.2241673689615, 33.73869032756264),
               (65.04346577544663, -871.4828488473704, 33.73869032756281)],
              [(65.04346577544663, -871.4828488473704, 33.73869032756281),
               (497.5869144101305, -714.0499085174936, 33.73869032756289)],
              [(497.5869144101305, -714.0499085174936, 33.73869032756289),
               (362.4598051390722, -342.79122703908456, 33.73869032756272)],
              [(-590.9470030039327, -395.7892483075277, 31.738690327562438),
               (87.51106920349315, -148.85070482639523, 31.73869032756255)],
              [(87.51106920349315, -148.85070482639523, 31.73869032756255),
               (-47.61604006756514, 222.40797665201396, 31.73869032756241)],
              [(-47.61604006756514, 222.40797665201396, 31.73869032756241),
               (-726.0741122749903, -24.530566829118783, 31.738690327562267)],
              [(-726.0741122749903, -24.530566829118783, 31.738690327562267),
               (-590.9470030039327, -395.7892483075277, 31.738690327562438)],
              [(1005.7049427567675, -983.9451737685275, 33.73869032756315),
               (697.466029109475, -137.06571872534644, 32.73869032756275)],
              [(697.466029109475, -137.06571872534644, 32.73869032756275),
               (-184.15441226469054, -457.9493173061661, 33.73869032756258)],
              [(-184.15441226469054, -457.9493173061661, 33.73869032756258),
               (124.0855029714684, -1304.8284078008128, 149.7386891914858)],
              [(124.0855029714684, -1304.8284078008128, 149.7386891914858),
               (1005.7049427567675, -983.9451737685275, 33.73869032756315)]],
    start_plane=PlaneLinear(origin=(802.0719024679672, -2115.6971730783057, 350.7386903275634),
                            normal=[0., 0., 1.],
                            xaxis=[-0.73873079, -0.67400061, 0.],
                            yaxis=[0.67400061, -0.73873079, 0.]),
    max_high=10000.0,
    step=86.0
)


def contour(segments, start_plane, max_high, step, reverse=False):
    """

    Parameters
    ----------
    segments
    start_plane
    max_high
    step
    reverse

    Returns
    -------

    Example:
    >>> from mmcore.geom.parametric import PlaneLinear
    >>> segments = [[(802.0719024679672, -2115.6971730783057, 350.7386903275634), (-607.77747825974, -1954.7618508065439, 305.73869032756295)], [(-607.77747825974, -1954.7618508065439, 305.73869032756295), (-1322.530384544842, -242.12639164870774, 32.73869032756218)], [(-1322.530384544842, -242.12639164870774, 32.73869032756218), (-506.56487614725256, 479.46409180026967, 28.689451929240477)], [(-506.56487614725256, 479.46409180026967, 28.689451929240477), (473.1669189275823, 479.19102170318274, 147.01101091963884)], [(473.1669189275823, 479.19102170318274, 147.01101091963884), (1236.2007813878104, 581.8903948009078, 146.73869032756275)], [(1236.2007813878104, 581.8903948009078, 146.73869032756275), (1771.5303845448425, -346.94504500382857, 304.7386903275632)], [(1771.5303845448425, -346.94504500382857, 304.7386903275632), (1384.7931246700518, -539.9683593900033, 144.7386903275631)], [(1384.7931246700518, -539.9683593900033, 144.7386903275631), (802.0719024679672, -2115.6971730783057, 350.7386903275634)], [(362.4598051390722, -342.79122703908456, 33.73869032756272), (-70.08364349561086, -500.2241673689615, 33.73869032756264)], [(-70.08364349561086, -500.2241673689615, 33.73869032756264), (65.04346577544663, -871.4828488473704, 33.73869032756281)], [(65.04346577544663, -871.4828488473704, 33.73869032756281), (497.5869144101305, -714.0499085174936, 33.73869032756289)], [(497.5869144101305, -714.0499085174936, 33.73869032756289), (362.4598051390722, -342.79122703908456, 33.73869032756272)], [(-590.9470030039327, -395.7892483075277, 31.738690327562438), (87.51106920349315, -148.85070482639523, 31.73869032756255)], [(87.51106920349315, -148.85070482639523, 31.73869032756255), (-47.61604006756514, 222.40797665201396, 31.73869032756241)], [(-47.61604006756514, 222.40797665201396, 31.73869032756241), (-726.0741122749903, -24.530566829118783, 31.738690327562267)], [(-726.0741122749903, -24.530566829118783, 31.738690327562267), (-590.9470030039327, -395.7892483075277, 31.738690327562438)], [(1005.7049427567675, -983.9451737685275, 33.73869032756315), (697.466029109475, -137.06571872534644, 32.73869032756275)], [(697.466029109475, -137.06571872534644, 32.73869032756275), (-184.15441226469054, -457.9493173061661, 33.73869032756258)], [(-184.15441226469054, -457.9493173061661, 33.73869032756258), (124.0855029714684, -1304.8284078008128, 149.7386891914858)], [(124.0855029714684, -1304.8284078008128, 149.7386891914858), (1005.7049427567675, -983.9451737685275, 33.73869032756315)]]
    >>> plane = PlaneLinear(origin=(802.0719024679672, -2115.6971730783057, 350.7386903275634), normal=[0., 0., 1.], xaxis=[-0.73873079, -0.67400061,  0.        ], yaxis=[ 0.67400061, -0.73873079,  0.        ])
    >>> max_high = 10000.0
    >>> step = 86.0




    """
    result = ContourResult(segments=segments)
    lx, ly = 0.0, 0.0

    for i, segm in enumerate(segments):

        pt = start_plane.in_plane_coords(segm[0])
        if pt[0] <= lx:
            lx = pt[0]

        if pt[1] <= ly:
            ly = pt[1]

    cut_plane = ContourPlane(origin=np.array(start_plane.point_at((lx, ly, 0))),
                             xaxis=start_plane.normal,
                             yaxis=start_plane.yaxis)

    result.planes, result.slices = [], []

    for i in range(int(round(max_high / step, 0))):
        intersections = []

        next_cut_plane = ContourPlane(
            origin=tuple((np.array(cut_plane.origin) + np.array(cut_plane.normal) * i * (-step)).tolist()),
            xaxis=cut_plane.xaxis,
            yaxis=cut_plane.yaxis)
        result.planes.append(next_cut_plane)

        for j, segm in enumerate(segments):

            w, t, point = custom_line_plane_intersection(next_cut_plane, Linear.from_two_points(*segm))
            if point is not None:
                if 0 <= t <= 1:
                    intersections.append(ContourSliceResult(i, j, t, point.tolist()))
        intersections.sort(key=lambda x: next_cut_plane.in_plane_coords(x.point)[1])
        if len(intersections) > 0:
            result.slices.append(list(
                even_filter([ContourSlice(_owner=result, start=st, end=end, ) for st, end in pairwise(intersections)],
                            reverse=reverse)))

    return result


__all__ = ["ContourResult", "ContourSliceResult", "contour"]


def test():
    res = contour(**TEST_CASE)
    lnr = [Linear.from_two_points(*i) for i in TEST_CASE['segments']]
    check = []
    for s in res.slices:
        for ss in s:
            check.append(np.allclose(np.array(ss.points) - np.array(ss.get_eval_segments(lnr)), 0.0))

    return all(check)
