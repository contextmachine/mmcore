from fractions import Fraction

import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._ssx_cofactor_bounds import SourceCofactorBounds
from mmcore.numeric.intersection.ssx._ssx_face_residual import SourceFaceResiduals


def _fraction_net(net):
    return np.array([Fraction(float(x)) for x in net.flat], dtype=object).reshape(net.shape)


def _blossom_restrict(net, axis, lo, hi):
    """Exact blossom formula, independent of sequential split/rescale code."""
    values = np.moveaxis(net, axis, 0)
    degree = len(values)-1
    out = []
    for index in range(degree+1):
        current = values.copy()
        for parameter in [lo]*(degree-index)+[hi]*index:
            current = (1-parameter)*current[:-1]+parameter*current[1:]
        out.append(current[0])
    return np.moveaxis(np.array(out, dtype=object), 0, axis)


@pytest.mark.parametrize('axis', range(4))
def test_transported_face_encloses_every_exact_source_coefficient(axis):
    rng = np.random.default_rng(392)
    first = rng.normal(size=(3, 2, 4))
    second = rng.normal(size=(2, 3, 4))
    first[..., 3] = rng.uniform(.5, 2., size=(3, 2))
    second[..., 3] = rng.uniform(.5, 2., size=(2, 3))
    source = SourceCofactorBounds(first, second, polynomial=False)
    faces = SourceFaceResiduals(source)
    box = ((.1, .7), (.2, .8), (.3, .9), (.15, .95))
    pin = .6
    net, error = faces(axis, pin, box)
    a, b = _fraction_net(first), _fraction_net(second)
    exact = (a[:, :, None, None, :3]*b[None, None, :, :, 3:]
             - b[None, None, :, :, :3]*a[:, :, None, None, 3:])
    for k, (lo, hi) in enumerate(box):
        if k == axis:
            lo = hi = pin
        exact = _blossom_restrict(exact, k, Fraction(lo), Fraction(hi))
    exact = np.take(exact, 0, axis=axis)
    remaining = [k for k in range(4) if k != axis]
    order = [axis ^ 1]+([2, 3] if axis < 2 else [0, 1])
    exact = np.transpose(exact, [remaining.index(k) for k in order]+[3])
    if axis >= 2:
        exact = -exact
    for index in np.ndindex(net.shape):
        assert abs(Fraction(float(net[index]))-exact[index]) <= Fraction(float(error[index[-1]]))
    assert faces(axis, pin, box) is faces(axis, pin, box)
    assert not net.flags.writeable


def test_face_construction_is_prepaid_and_denial_never_returns_rounded_geometry():
    first = np.array([[[s, t, s+t, 1.] for t in (0., 1.)] for s in (0., 1.)])
    source = SourceCofactorBounds(first, first, polynomial=False)
    faces = SourceFaceResiduals(source, charge=lambda units: False)
    assert faces(0, .5, ((0., 1.),)*4) is None
    assert faces.exhausted
    assert not faces.cache


def test_exact_face_root_uses_exact_affine_image_not_rounded_global_parameters():
    # x=3s-1 has the exact root s=1/3. A local t=1/2 on [0,2/3]
    # maps to the stored floating approximation of 1/3, not its exact root.
    first = np.array([[[3*s-1, t, 0., 1.] for t in (0., 1.)] for s in (0., 1.)])
    second = np.array([[[0., u, v, 1.] for v in (0., 1.)] for u in (0., 1.)])
    faces = SourceFaceResiduals(SourceCofactorBounds(first, second), sources=(first, second))
    face = (1, .5, ((0., 2/3), (0., 1.), (0., 1.), (0., 1.)))
    assert not faces.exact_root(face, (.5, .5, 0.))
    # A dyadic, exactly representable closed-boundary root is accepted.
    first[..., 0] += 1.
    exact_faces = SourceFaceResiduals(SourceCofactorBounds(first, second), sources=(first, second))
    assert exact_faces.exact_root(face, (0., .5, 0.))
