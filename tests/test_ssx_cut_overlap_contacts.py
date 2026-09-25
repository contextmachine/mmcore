"""CSX tolerance spans retain the point witnesses needed by SSX cuts."""

import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_curve, eval_surface
from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx
from mmcore.numeric.intersection.ssx._bez_ssx5 import _cut_face_contacts


@pytest.mark.parametrize("matrix", [np.eye(3),
    np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])])
def test_endpoint_touch_keeps_its_cut_contact(matrix):
    # C(t)=(t,0,(1-t)^2) meets S(u,v)=(u,v,0) only at t=u=1,v=0.
    # The near-contact interval starts at a normal-gap threshold, not a
    # domain entrance. Following v=0 does not turn it into an overlap.
    curve = np.array([[0., 0., 1.], [.5, 0., 0.], [1., 0., 0.]]) @ matrix.T
    surface = np.array([[[0., 0., 0.], [0., 1., 0.]],
                        [[1., 0., 0.], [1., 1., 0.]]]) @ matrix.T
    result = bez_csx(curve, surface, 1e-3, rational=False, max_results=8)

    assert not result['budget_exhausted']
    assert len(result['isolated']) == 1
    assert result['overlaps'] == []
    contacts = list(_cut_face_contacts(result))
    assert len(contacts) == 1
    _assert_endpoint_and_valid_contacts(curve, surface, contacts, matrix, 1e-3)


def _assert_endpoint_and_valid_contacts(curve, surface, contacts, matrix, atol):
    target = np.array([1., 0., 0.]) @ matrix.T
    assert any(np.linalg.norm(point['point']-target) <= atol for point in contacts)
    for point in contacts:
        parameters = np.array([point[k] for k in ('t', 'u', 'v')])
        assert np.isfinite(parameters).all()
        assert np.all((0. <= parameters) & (parameters <= 1.))
        assert np.linalg.norm(eval_curve(curve, point['t'], rational=False)
                              - point['point']) <= atol
        assert np.linalg.norm(eval_surface(surface, point['u'], point['v'],
                                            rational=False)-point['point']) <= atol


@pytest.mark.parametrize("matrix", [np.eye(3),
    np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])])
def test_whole_cad_overlap_retains_its_endpoint_cut_contact(matrix):
    # Every point has normal gap <=atol/2. Both ends are actual curve-domain
    # endpoints, so the whole span is a CAD overlap. Absorbing its isolated
    # roots must retain the exact right endpoint as a paired SSX seed.
    atol = 1e-3
    curve = np.array([[0., 0., .5*atol], [.5, 0., 0.], [1., 0., 0.]]) @ matrix.T
    surface = np.array([[[0., 0., 0.], [0., 1., 0.]],
                        [[1., 0., 0.], [1., 1., 0.]]]) @ matrix.T
    result = bez_csx(curve, surface, atol, rational=False)

    assert not result['budget_exhausted']
    assert result['isolated'] == []
    assert len(result['overlaps']) == 1
    overlap = result['overlaps'][0]
    for actual, expected in zip(overlap['t_range'], (0., 1.)):
        assert np.linalg.norm(eval_curve(curve, actual, rational=False)
                              -eval_curve(curve, expected, rational=False)) <= atol
    contacts = list(_cut_face_contacts(result))
    assert contacts and overlap['boundary_contacts']
    _assert_endpoint_and_valid_contacts(curve, surface, contacts, matrix, atol)


@pytest.mark.parametrize("max_results", [0, 1])
def test_preserved_contacts_respect_the_original_result_allowance(max_results):
    curve = np.array([[0., 0., 1.], [.5, 0., 0.], [1., 0., 0.]])
    surface = np.array([[[0., 0., 0.], [0., 1., 0.]],
                        [[1., 0., 0.], [1., 1., 0.]]])
    result = bez_csx(curve, surface, 1e-3, rational=False,
                     max_results=max_results)
    assert len(list(_cut_face_contacts(result))) <= max_results
    assert result['budget_exhausted']


def test_cut_range_boxes_cannot_invent_paired_contact_parameters():
    result = {'isolated': [], 'overlaps': [
        {'t_range': (0., 1.), 'u_range': (.2, .8), 'v_range': (.3, .7)}]}
    assert list(_cut_face_contacts(result)) == []


def test_cut_contacts_preserve_distinct_surface_preimages():
    first = dict(t=.5, u=.2, v=.3, point=np.zeros(3))
    second = dict(t=.5, u=.8, v=.7, point=np.zeros(3))
    result = {'isolated': [first], 'overlaps': [
        {'boundary_contacts': [second]}]}
    contacts = list(_cut_face_contacts(result))
    assert contacts[0] is first and contacts[1] is second
