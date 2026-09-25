"""Shared curved-overlap corners and whole-chord CAD error bounds."""
import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_surface
from mmcore.numeric.intersection.ssx import _ssx5_overlap as overlap


ATOL = 1e-3


def _plane(x0=0., x1=1., y0=0., y1=1., z=0.):
    return np.array([[[x, y, z, 1.] for y in (y0, y1)] for x in (x0, x1)])


def _rim(owner, axis, side, parameters, surface):
    parameters = np.asarray(parameters, dtype=float)
    xyz = np.array([eval_surface(surface, *q[2*(owner-1):2*owner], rational=True)
                    for q in parameters])
    return dict(owner=owner, axis=axis, side=side, stuv=parameters,
                xyz=xyz, resid_max=0.)


def test_two_tolerance_fringes_become_one_paired_domain_corner():
    first, second = _plane(), _plane(.5, 1., -.5, .5)
    a = _rim(1, 1, 0., [[.4994, 0., 0., .5], [1., 0., 1., .5]], first)
    b = _rim(2, 0, 0., [[.5, 0., 0., .4994], [.5, .5, 0., 1.]], second)
    corner = overlap._joint_rim_corner(first, second, a, 0, b, 0,
                                       ATOL, np.full(4, ATOL), lambda n: True)
    assert corner is not None
    parameters, xyz, _ = corner
    assert np.linalg.norm(xyz-[.5, 0., 0.]) <= ATOL
    assert parameters[1] == parameters[2] == 0.  # fixed domain faces
    for surface, uv in ((first, parameters[:2]), (second, parameters[2:])):
        assert np.linalg.norm(eval_surface(surface, *uv, rational=True)-xyz) <= ATOL
    overlap._refine_curved_rim_corners([a, b], first, second, ATOL,
                                       np.full(4, ATOL), lambda n: True)
    # Shared object values establish the graph node; its accuracy is checked
    # geometrically above, rather than by requiring particular solver digits.
    np.testing.assert_array_equal(a['stuv'][0], b['stuv'][0])
    np.testing.assert_array_equal(a['xyz'][0], b['xyz'][0])


def test_parallel_nearby_edges_do_not_invent_a_unique_corner():
    first, second = _plane(), _plane(z=.5*ATOL)
    parameters = [[.2, 0., .2, 0.], [.8, 0., .8, 0.]]
    a, b = _rim(1, 1, 0., parameters, first), _rim(2, 1, 0., parameters, second)
    assert overlap._joint_rim_corner(first, second, a, 0, b, 0,
        ATOL, np.full(4, ATOL), lambda n: True) is None


def test_nearby_opposite_partner_faces_keep_their_parameter_identity():
    first, second = _plane(), _plane(.5, .5+.5*ATOL, -.5, .5)
    a = _rim(1, 1, 0., [[.5, 0., 0., .5], [.7, 0., 1., .5]], first)
    b = _rim(2, 0, 1., [[.5+.5*ATOL, 0., 1., .5], [.5+.5*ATOL, .5, 1., 1.]], second)
    # Even a poorly resolved partner axis cannot make its two domain faces
    # the same face. XYZ-only endpoint merging would accept this pair.
    assert overlap._joint_rim_corner(first, second, a, 0, b, 0,
        ATOL, np.ones(4), lambda n: True) is None


def test_corner_correction_cannot_jump_to_a_remote_contact_span():
    first, second = _plane(), _plane(.8, 1., -.5, .5)
    a = _rim(1, 1, 0., [[.1, 0., 0., .5], [.3, 0., 0., .5]], first)
    b = _rim(2, 0, 0., [[.8, 0., 0., .49], [.8, .5, 0., 1.]], second)
    assert overlap._joint_rim_corner(first, second, a, -1, b, 0,
        ATOL, np.full(4, ATOL), lambda n: True) is None


def test_homogeneous_chord_bound_catches_an_inflection_bulge():
    surface = np.array([[[s, t, z, 1.] for t in (0., 1.)]
                        for s, z in zip(np.linspace(0., 1., 4), (-.125, .125, -.125, .125))])
    start, end = np.array([0., 0.]), np.array([1., 0.])
    xyz = np.array([eval_surface(surface, *q, rational=True) for q in (start, end)])
    middle = eval_surface(surface, .5, 0., rational=True)
    assert np.linalg.norm(middle-xyz.mean(axis=0)) <= ATOL
    bound = overlap._surface_chord_error(surface, start, end, xyz)
    sampled = max(np.linalg.norm(eval_surface(surface, t, 0., rational=True)
                                  - ((1.-t)*xyz[0]+t*xyz[1]))
                  for t in np.linspace(0., 1., 257))
    assert sampled > ATOL and sampled <= bound


def test_corner_work_denial_keeps_the_original_rim_endpoints(monkeypatch):
    first, second = _plane(), _plane(.5, 1., -.5, .5)
    a = _rim(1, 1, 0., [[.4994, 0., 0., .5], [1., 0., 1., .5]], first)
    b = _rim(2, 0, 0., [[.5, 0., 0., .4994], [.5, .5, 0., 1.]], second)
    before = [(rim['stuv'].copy(), rim['xyz'].copy()) for rim in (a, b)]
    def forbidden(*args, **kwargs):
        raise AssertionError('corner derivative evaluation after work denial')
    monkeypatch.setattr(overlap, 'eval_surface_d1', forbidden)
    with pytest.raises(overlap._OverlapWorkStopped):
        overlap._refine_curved_rim_corners([a, b], first, second, ATOL,
                                           np.full(4, ATOL), lambda n: False)
    for rim, (parameters, xyz) in zip((a, b), before):
        np.testing.assert_array_equal(rim['stuv'], parameters)
        np.testing.assert_array_equal(rim['xyz'], xyz)


@pytest.mark.parametrize('gap_fraction', [.9, 1.])
def test_within_tolerance_offset_keeps_the_complete_paired_region(gap_fraction):
    # Degree elevation selects the general curved-rim path. Both entire
    # planes are within atol; different sampling stages must not impose a
    # stricter gap by mixing owner points with paired midpoint samples.
    first = np.array([[[i/2, j/2, gap_fraction*ATOL, 1.] for j in range(3)]
                      for i in range(3)])
    second = first.copy()
    second[..., 2] = 0.
    result = overlap.assemble_overlap_regions(
        first, second, atol=ATOL, ptol4=np.full(4, ATOL))
    assert len(result['regions']) == 1
    assert len(result['rim_branches']) == 4
    for branch in result['rim_branches']:
        parameters, points = branch.curve
        for fraction in (0., .25, .5, .75, 1.):
            for q, point in zip(
                    (1.-fraction)*parameters[:-1]+fraction*parameters[1:],
                    (1.-fraction)*points[:-1]+fraction*points[1:]):
                first_point = eval_surface(first, *q[:2], rational=True)
                second_point = eval_surface(second, *q[2:], rational=True)
                assert np.linalg.norm(first_point-point) <= ATOL
                assert np.linalg.norm(second_point-point) <= ATOL
                # The closed-boundary control has analytic gap exactly
                # atol; evaluating that constant may differ by one ULP.
                assert np.linalg.norm(first_point-second_point) <= ATOL+64*np.finfo(float).eps


def test_offset_above_tolerance_does_not_make_an_overlap_region():
    first = np.array([[[i/2, j/2, 1.01*ATOL, 1.] for j in range(3)]
                      for i in range(3)])
    second = first.copy()
    second[..., 2] = 0.
    result = overlap.assemble_overlap_regions(
        first, second, atol=ATOL, ptol4=np.full(4, ATOL))
    assert result['regions'] == []
