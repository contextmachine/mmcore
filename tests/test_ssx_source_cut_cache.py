import numpy as np

from mmcore.numeric.intersection.ssx._ssx_affine_constraints import AffineParameterConstraints
from mmcore.numeric.intersection.ssx._ssx_planar_cut import exact_source_planar_cut
from mmcore.numeric.intersection.ssx._ssx_root_identity import _pure_affine_coordinates
from mmcore.numeric.intersection.ssx._ssx_source_cut_cache import SourceFaceCensusCache


def _fixture(plane_x=(-.25,1.25)):
    graph = np.array([[[u,v,(v-.5)*z,1.] for v in (0.,1.)]
                      for u,z in zip((0.,.5,1.),(1.,1.,2.))])
    plane = np.array([[[u,v,0.,1.] for v in (0.,1.)] for u in plane_x])
    constraints = AffineParameterConstraints(tuple(_pure_affine_coordinates(s) for s in (graph,plane)))
    calls = []
    def solve(axis,value,box):
        calls.append((axis,value,box))
        return exact_source_planar_cut(graph,plane,axis,value,box,max_cells=1000)
    cache = SourceFaceCensusCache(solve,constraints,(id(graph),id(plane)),charge=lambda n:True)
    return cache,calls


def test_equivalent_faces_share_census_and_rebind_exact_metadata():
    cache,calls = _fixture()
    first = cache(0,.5,((0.,1.),)*4)
    second = cache(2,.5,((0.,1.),)*4)
    assert len(calls) == 1
    a, = first['isolated']
    b, = second['isolated']
    assert a['source_cut_certificate']['axis'] == 0
    assert b['source_cut_certificate']['axis'] == 2
    np.testing.assert_array_equal(a['stuv'],b['stuv'])
    np.testing.assert_array_equal(a['point'],b['point'])
    assert second['boundary_topology_complete']


def test_float_aliases_of_different_exact_pins_do_not_share_census():
    cache,calls = _fixture((0.,3.))
    first = cache(0,.5,((0.,1.),)*4)
    second = cache(2,1/6,((0.,1.),)*4)
    assert len(calls) == 2
    np.testing.assert_array_equal(first['isolated'][0]['stuv'],second['isolated'][0]['stuv'])
    assert first['isolated'][0]['source_cut_certificate']['pinned'] != second['isolated'][0]['source_cut_certificate']['pinned']


def test_foreign_source_certificate_cannot_be_reused():
    cache,calls = _fixture()
    result = cache(0,.5,((0.,1.),)*4)
    result['isolated'][0]['source_cut_certificate']['source_ids'] = (0,1)
    rejected = cache(2,.5,((0.,1.),)*4)
    assert not rejected['boundary_topology_complete']
    assert not rejected['isolated']


def test_changed_float_representative_requires_fresh_source_validation():
    cache,calls = _fixture()
    first = cache(0,.5,((0.,1.),)*4)
    first['isolated'][0]['stuv'][2] = np.nextafter(.5,1.)
    second = cache(2,.5,((0.,1.),)*4)
    assert len(calls) == 2
    assert second['isolated'][0]['stuv'][2] == .5


def test_partial_global_census_never_leaks_foreign_roots_into_local_owner():
    cache,calls = _fixture()
    full = cache(0,.5,((0.,1.),)*4)
    full['budget_exhausted'] = True
    # Mark the stored result partial (the return is intentionally a view
    # only of immutable root records, so result flags are copied).
    stored, = cache.censuses.values()
    stored['budget_exhausted'] = True
    stored['boundary_topology_complete'] = False
    cache.can_retry = lambda:False
    local = cache(0,.5,((0.,1.),(0.,.25),(0.,1.),(0.,1.)))
    assert not local['boundary_topology_complete']
    assert not local['isolated']
    assert local['unresolved_source_boxes']


def test_partial_equivalent_face_does_not_reuse_an_unvalidated_representative():
    cache, calls = _fixture()
    cache(0, .5, ((0., 1.),) * 4)
    stored, = cache.censuses.values()
    stored['budget_exhausted'] = True
    stored['boundary_topology_complete'] = False
    stored['isolated'][0]['stuv'][2] = np.nextafter(.5, 1.)
    cache.can_retry = lambda: False
    result = cache(2, .5, ((0., 1.),) * 4)
    assert not result['isolated']
    assert not result['boundary_topology_complete']
