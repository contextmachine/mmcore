"""Parent cut queries retain contacts and refine only unfinished local work."""

from types import SimpleNamespace

import numpy as np

from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx


def _root(t=.25, u=.5, v=.75):
    return dict(t=t, u=u, v=v, point=np.array([t, u, v]))


def test_complete_parent_cut_does_not_repeat_child_queries():
    calls = []
    root = _root()
    def query(curve, surface, **kwargs):
        calls.append((surface, kwargs))
        return dict(isolated=[root], overlaps=[], budget_exhausted=False)
    result = ssx._query_cut_face_contacts(
        query, 'curve', 'parent', ['left', 'right'], 0, [.3],
        1e-3, SimpleNamespace(exhausted=False))
    assert result == [root]
    assert [surface for surface, _ in calls] == ['parent']


def test_parent_depth_fallback_retains_and_remaps_all_contacts():
    calls = []
    parent_contact = _root()
    def query(curve, surface, **kwargs):
        calls.append((surface, kwargs))
        if surface == 'parent':
            return dict(isolated=[parent_contact], overlaps=[],
                        budget_exhausted=True, truncation_cause='depth')
        return dict(isolated=[_root(u=.5)], overlaps=[], budget_exhausted=False)
    result = ssx._query_cut_face_contacts(
        query, 'curve', 'parent', ['left', 'right'], 0, [.3],
        1e-3, SimpleNamespace(exhausted=False))
    assert [surface for surface, _ in calls] == ['parent', 'left', 'right']
    assert calls[0][1]['defer_local_depth']
    assert result[0] is parent_contact
    # These are arithmetic affine mappings, not CAD accuracy demands.
    np.testing.assert_allclose([root['u'] for root in result], [.5, .15, .65])


def test_resource_stop_does_not_start_child_retries():
    calls = []
    def query(curve, surface, **kwargs):
        calls.append(surface)
        return dict(isolated=[_root()], overlaps=[], budget_exhausted=True,
                    truncation_cause='cells')
    result = ssx._query_cut_face_contacts(
        query, 'curve', 'parent', ['left', 'right'], 1, [.3],
        1e-3, SimpleNamespace(exhausted=True))
    assert calls == ['parent']
    assert len(result) == 1


def test_paired_contact_object_is_shared_only_with_closed_incident_children(monkeypatch):
    monkeypatch.setattr(ssx, '_ssx_tangent_4d', lambda *a, **kw: (np.ones(4), None, None))
    cell = SimpleNamespace(box=((0., 1.),)*4,
                           g1=SimpleNamespace(surface=None),
                           g2=SimpleNamespace(surface=None))
    grid = [[[], []], [[], []]]
    ssx._register_cut_contacts(cell, [_root(u=.5)], 0, .4, 0, 2, [.5], grid)
    assert all(len(owner) == 1 for row in grid for owner in row)
    assert len({id(owner[0]) for row in grid for owner in row}) == 1
    exterior_grid = [[[], []], [[], []]]
    ssx._register_cut_contacts(cell, [_root(u=.5+1e-5)], 0, .4, 0, 2, [.5], exterior_grid)
    assert all(not row[0] and len(row[1]) == 1 for row in exterior_grid)


def test_narrow_parameter_interval_keeps_distinct_face_identities():
    low, high = .5, .5+5e-9
    assert ssx._on_axis_local(low, low, high) == 0
    assert ssx._on_axis_local(high, low, high) == 1
    assert ssx._on_axis_local(.5*(low+high), low, high) is None
