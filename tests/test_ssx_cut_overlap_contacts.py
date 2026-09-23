"""CSX tolerance spans retain the point witnesses needed by SSX cuts."""

import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_curve, eval_surface
from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx
from mmcore.numeric.intersection.ssx._bez_ssx5 import _cut_face_contacts


@pytest.mark.parametrize("matrix", [np.eye(3),
    np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])])
def test_tolerance_span_keeps_its_polished_endpoint_contact(matrix):
    # C(t)=(t,0,(1-t)^2) meets S(u,v)=(u,v,0) only at t=u=1,v=0.
    # Its CAD tolerance neighborhood can be a valid CSX overlap span,
    # but that representation must not erase the SSX cut's point seed.
    curve = np.array([[0., 0., 1.], [.5, 0., 0.], [1., 0., 0.]]) @ matrix.T
    surface = np.array([[[0., 0., 0.], [0., 1., 0.]],
                        [[1., 0., 0.], [1., 1., 0.]]]) @ matrix.T
    result = bez_csx(curve, surface, 1e-3, rational=False, max_results=8)

    assert not result['budget_exhausted']
    assert result['isolated'] == []
    assert len(result['overlaps']) == 1
    contacts = list(_cut_face_contacts(result))
    assert len(contacts) == 1
    point = contacts[0]
    np.testing.assert_allclose([point[k] for k in ('t', 'u', 'v')],
                               [1., 1., 0.], atol=1e-6)
    np.testing.assert_allclose(eval_curve(curve, point['t'], rational=False),
                               point['point'], atol=1e-12)
    np.testing.assert_allclose(
        eval_surface(surface, point['u'], point['v'], rational=False),
        point['point'], atol=1e-12)


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
