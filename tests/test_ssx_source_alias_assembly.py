"""Different exact source roots can have identical floating representatives."""
from types import SimpleNamespace

import numpy as np

from test_ssx_exact_root_separation import _source_events
from mmcore.numeric.intersection.ssx import _bez_ssx5 as bez
from mmcore.numeric.intersection.ssx import _nssx5 as nurbs


def source_arc():
    identity,points,boxes = _source_events(((3,.625),(2,.375),(0,.25)))
    boxes = identity.separate_source_boxes(points,boxes)
    order = sorted(range(3),key=lambda i:sum(boxes[i][0]))
    points = [points[i] for i in order]
    fragments = [bez._Fragment(a,b,np.array([a.stuv,b.stuv]),np.array([a.xyz,b.xyz]))
                 for a,b in zip(points,points[1:])]
    return identity,points,fragments


def test_registered_point_dedup_cannot_override_distinct_exact_sources():
    identity,points,_ = source_arc()
    assert not identity(points[0],points[-1],np.full(4,.001),.001)
    output = bez._deduplicate_ssx_points(
        [bez._registered_point(point) for point in (points[0],points[-1],points[0])],
        np.full(4,.001),.001,exact_topology=True,root_matcher=identity)
    assert len(output) == 2


def test_identical_displayed_ends_do_not_close_an_open_registered_arc():
    identity,points,fragments = source_arc()
    np.testing.assert_array_equal(points[0].stuv,points[-1].stuv)
    branch, = bez._assemble_fragments(fragments,root_matcher=identity)
    assert len(branch.curve[0]) == 3
    assert not branch.closed
    fragments.append(bez._Fragment(points[-1],points[0],
        np.array([points[-1].stuv,points[0].stuv]),
        np.array([points[-1].xyz,points[0].xyz])))
    cycle, = bez._assemble_fragments(fragments,root_matcher=identity)
    assert cycle.closed


def test_nurbs_collection_preserves_same_pair_source_alias_ownership():
    identity,points,fragments = source_arc()
    branch, = bez._assemble_fragments(fragments,root_matcher=identity)
    raw = nurbs._RawResults()
    nurbs._collect_pair(raw,dict(branches=[branch],
        points=[bez._registered_point(point) for point in (points[0],points[-1])]),
        tuple([0.,1.]*4),pair=(0,0))
    ctx = SimpleNamespace(lows=np.zeros(4),highs=np.ones(4),spans=np.ones(4),
                          ptol=np.full(4,.001),closed=np.zeros(4,dtype=bool),
                          source_patches=None)
    agg = nurbs._make_aggregate({},1)
    assert nurbs._build_chains(raw.frags,ctx,.001,agg) == [([(0,False)],False)]
    assert len(nurbs._assemble_points(raw.points,[],ctx,.001,agg)) == 2


def test_singularity_classification_does_not_consume_an_aliased_source_point():
    identity, points, _ = source_arc()
    feature = SimpleNamespace(kind='tangent_point', stuv=points[0].stuv,
                              xyz=points[0].xyz, _registered_root=points[0])
    first, distinct = (bez._registered_point(point) for point in (points[0], points[-1]))
    kept = bez._remove_source_classified_points(
        [first, distinct], [feature], identity, np.full(4, .001), .001)
    assert kept == [distinct]
    del feature._registered_root
    assert bez._remove_source_classified_points(
        [first, distinct], [feature], identity, np.full(4, .001), .001) == [first, distinct]
