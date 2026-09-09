"""Source root ownership survives rounded proposal equations only with accuracy."""
import numpy as np
import pytest

from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx
from mmcore.numeric.intersection.ssx._ssx_affine_path import (
    affine_path_representation_bounded,
)


def sources():
    first = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    second = np.array([[[u, v, 2*v-1] for v in (0., 1.)] for u in (0., 1.)])
    return first, second


def run_face(*, xyz=(0., .5, 0.), certified=True, charge=None, monkeypatch=None):
    original = sources()
    proposal = [net.copy() for net in original]
    # The registered source root is exact, while a rounded proposal can
    # fail the strict normalized residual bar at that same representative.
    proposal[1][..., 2] += 1e-5
    stuv = np.array([0., .5, 0., .5])
    callbacks = []

    def census(curve, surface, axis, value, **kwargs):
        roots = []
        if axis == 0 and value == 0.:
            roots = [dict(t=.5, u=0., v=.5, point=np.array(xyz),
                          parameter_root_box=((.5, .5), (0., 0.), (.5, .5)),
                          root_existence_certification='exact_source_identity')]
        return dict(isolated=roots, overlaps=[], parameter_fibers=[],
                    boundary_topology_complete=True, budget_exhausted=False,
                    _ssx_source_residual=certified)

    def representation(parameters, point):
        callbacks.append((parameters.copy(), point.copy()))
        return affine_path_representation_bounded(
            *original, parameters, parameters, np.array([point, point]),
            1e-3, rational=False, charge=charge)

    if monkeypatch is not None:
        monkeypatch.setattr(ssx, '_ssx_correct_fixed',
                            lambda *a, **k: (stuv.copy(), 1e-5, False))
    status = {}
    crossings, overlaps = ssx._find_ssx_boundary_zeros(
        *proposal, 1e-3, rational=False, face_csx_fn=census,
        census_sink=status, source_point_representation=representation)
    return crossings, overlaps, status, callbacks


def test_source_root_uses_original_accuracy_without_repolishing(monkeypatch):
    monkeypatch.setattr(ssx, '_ssx_correct_fixed',
                        lambda *a, **k: pytest.fail('source root must not be repolished'))
    crossings, overlaps, status, callbacks = run_face()
    assert len(crossings) == len(callbacks) == 1
    assert not overlaps
    assert status == dict(complete=True, boundary_obligations=[])
    root, = crossings
    np.testing.assert_array_equal(root.stuv, [0., .5, 0., .5])
    np.testing.assert_array_equal(root.xyz, [0., .5, 0.])
    assert root._source_root_box


@pytest.mark.parametrize('xyz', [(0., .5, .01), (0., .5, np.nan)])
def test_inaccurate_source_representative_retains_face_obligation(xyz):
    crossings, overlaps, status, callbacks = run_face(xyz=xyz)
    assert not crossings and not overlaps
    assert not status['complete']
    assert status['boundary_obligations'] == [((0., 0.), (0., 1.), (0., 1.), (0., 1.))]


def test_denied_source_accuracy_work_retains_face_obligation():
    charged = []
    def deny(amount):
        charged.append(amount)
        return False
    crossings, _, status, callbacks = run_face(charge=deny)
    assert len(charged) == len(callbacks) == 1 and charged[0] > 0
    assert not crossings and not status['complete']
    assert len(status['boundary_obligations']) == 1


def test_uncertified_candidate_cannot_use_accuracy_as_existence(monkeypatch):
    crossings, _, status, callbacks = run_face(certified=False, monkeypatch=monkeypatch)
    assert not callbacks and not crossings
    assert not status['complete']
