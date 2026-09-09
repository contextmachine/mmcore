import numpy as np

from mmcore.numeric.intersection.ssx._ssx_planar_cut import exact_source_planar_cut


def test_identically_planar_source_cut_retains_a_localized_continuum_obligation():
    graph = np.array([[[s, t, s*(t-.5), 1.] for t in (0., 1.)] for s in (0., 1.)])
    plane = np.array([[[s, t, 0., 1.] for t in (0., 1.)] for s in (0., 1.)])
    result = exact_source_planar_cut(graph, plane, 1, .5, ((0., 1.),)*4)
    assert result is not None
    assert not result['boundary_topology_complete']
    assert result['truncation_cause'] == 'resolution'
    assert not result['isolated']
    obligation, = result['unresolved_source_boxes']
    assert obligation['reason'] == 'positive_dimensional_cut'
    np.testing.assert_array_equal(obligation['parameter_root_box'][1], [.5, .5])


def test_nonzero_source_polynomial_still_uses_isolated_root_census():
    graph = np.array([[[s, t, s-.5, 1.] for t in (0., 1.)] for s in (0., 1.)])
    plane = np.array([[[s, t, 0., 1.] for t in (0., 1.)] for s in (0., 1.)])
    result = exact_source_planar_cut(graph, plane, 1, .5, ((0., 1.),)*4)
    assert result['boundary_topology_complete']
    assert len(result['isolated']) == 1
