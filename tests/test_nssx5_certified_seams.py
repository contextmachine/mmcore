"""NURBS seams must join a unique root, without joining nearby roots."""
import numpy as np

from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
from mmcore.numeric.intersection.ssx._nssx5 import (
    _DomainCtx, _Frag, _assemble_branches, _make_aggregate,
)


def _patch(lo, hi, heights):
    nv = len(heights)
    cp = np.array([[[s, t, z] for t, z in zip(np.linspace(0., 1., nv), heights)]
                   for s in (lo, hi)])
    return NURBSSurfaceTuple(2, nv, np.array([lo, lo, hi, hi]),
                            np.r_[np.zeros(nv), np.ones(nv)], cp,
                            np.ones((2, nv)))


def _fragments(left_root, right_root):
    result = []
    for i, (lo, hi, root) in enumerate(((0., .5, left_root), (.5, 1., right_root))):
        stuv = np.array([[lo, root, lo, root], [hi, root, hi, root]])
        xyz = np.column_stack((stuv[:, :2], np.zeros(2)))
        result.append(_Frag(stuv, xyz, 'transversal', False,
                            pair=(i, 0), rect=(lo, hi, 0., 1., 0., 1., 0., 1.)))
    return result


def _context(heights):
    ctx = _DomainCtx(np.zeros(4), np.ones(4), np.ones(4),
                     np.full(4, .001), (False,) * 4)
    # Real decomposed source patches, used by the seam-root certificate.
    ctx.source_patches = ([_patch(0., .5, heights), _patch(.5, 1., heights)],
                          [_patch(0., 1., [0., 0.])])
    return ctx


def test_numerically_distinct_endpoints_join_at_one_certified_seam_root():
    # Intersection is exactly t=v=.5; the seam is exactly s=u=.5.
    frags = _fragments(.5 - 1e-12, .5 + 1e-12)
    branches = _assemble_branches(frags, _context([-.5, .5]), .001,
                                  _make_aggregate({}, 2))
    assert len(branches) == 1
    assert not branches[0].closed
    np.testing.assert_allclose(branches[0].curve[1][[0, -1], 0], [0., 1.])


def test_close_distinct_seam_roots_are_not_joined():
    a, b = .5 - 2.**-12, .5 + 2.**-12
    # Bernstein coefficients of (t-a)*(t-b), with two exact roots.
    heights = [a*b, a*b - (a+b)/2., (1.-a)*(1.-b)]
    branches = _assemble_branches(_fragments(a, b), _context(heights), .001,
                                  _make_aggregate({}, 2))
    assert len(branches) == 2
    assert not any(branch.closed for branch in branches)


def test_close_endpoints_without_patch_provenance_remain_separate():
    ctx = _context([-.5, .5])
    ctx.source_patches = None
    branches = _assemble_branches(_fragments(.5 - 1e-12, .5 + 1e-12), ctx,
                                  .001, _make_aggregate({}, 2))
    assert len(branches) == 2


def test_unique_but_absent_seam_root_does_not_authorize_joining():
    # z=t+2**-20 has an invertible seam Jacobian, but its zero is outside
    # t>=0. A tiny residual at t=0 must not replace root existence.
    delta = 2.**-20
    branches = _assemble_branches(_fragments(0., 1e-12),
                                  _context([delta, 1.+delta]), .001,
                                  _make_aggregate({}, 2))
    assert len(branches) == 2


def test_certified_seam_identity_survives_surface_swap():
    ctx = _context([-.5, .5])
    ctx.source_patches = ctx.source_patches[::-1]
    fragments = _fragments(.5-1e-12, .5+1e-12)
    for frag in fragments:
        frag.stuv = frag.stuv[:, [2, 3, 0, 1]]
        frag.pair = frag.pair[::-1]
        frag.rect = frag.rect[4:]+frag.rect[:4]
    branches = _assemble_branches(fragments, ctx, .001, _make_aggregate({}, 2))
    assert len(branches) == 1


def test_nearby_interior_curve_is_not_absorbed_by_boundary_owner():
    ctx = _context([0., 0.])
    root = .5+2.**-11
    branches = []
    for value, kind in ((root, 'transversal'), (.5, 'overlap')):
        stuv = np.array([[value, 0., value, 0.], [value, 1., value, 1.]])
        xyz = np.column_stack((stuv[:, :2], np.zeros(2)))
        branches.append(_Frag(stuv, xyz, kind, kind == 'overlap', pair=(1, 0),
                              rect=(.5, 1., 0., 1., 0., 1., 0., 1.)))
    result = _assemble_branches(branches, ctx, .001, _make_aggregate({}, 2))
    assert len(result) == 2


def test_boundary_retrace_requires_unique_target_preimage():
    from mmcore.numeric.intersection.ssx._ssx_boundary_identity import certified_boundary_retrace
    # C(t)=(1/16,t,0), S(u,v)=((u-.5)^2,v,0). Both u=.25
    # and u=.75 are valid preimages of every curve point.
    curve = np.array([[1/16, 0., 0., 1.], [1/16, 1., 0., 1.]])
    surface = np.array([[[x, v, 0., 1.] for v in (0., 1.)]
                        for x in (.25, -.25, .25)])
    source = np.array([[.5, 0., .75, 0.], [.5, 1., .75, 1.]])
    keeper = source.copy()
    keeper[:, 2] = .25
    xyz = curve[:, :3]
    assert not certified_boundary_retrace(
        source, xyz, keeper, xyz, curve, surface, 1, (2, 3), (0., 1.),
        np.full(4, .6), .001)
