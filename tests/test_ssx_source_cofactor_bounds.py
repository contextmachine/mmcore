"""Cofactor signs must enclose the original source, including cancellation."""
import numpy as np

from mmcore.numeric.intersection.ssx._ssx_cofactor_bounds import SourceCofactorBounds


def _h(net):
    return np.concatenate((net, np.ones(net.shape[:-1]+(1,))), axis=-1)


def _circle():
    ca, cb, radius = 3/8, 5/8, 2.**-18
    za = [ca*ca, ca*ca-ca, (1-ca)**2]
    zb = [cb*cb, cb*cb-cb, (1-cb)**2]
    graph = np.array([[[i/2, j/2, za[i]+zb[j]-radius*radius]
                       for j in range(3)] for i in range(3)])
    plane = np.array([[[u,v,0.] for v in (0.,1.)] for u in (0.,1.)])
    box = tuple((c-2*radius, c+2*radius) for c in (ca, cb, ca, cb))
    return graph, plane, box


def test_interval_cofactors_reject_cancelled_closed_loop_sign():
    graph, plane, box = _circle()
    n = float(2**24)
    transform = np.array([[n,n-1,0.], [n+1,n,0.], [0.,0.,1.]])
    # det(transform)=1 exactly, so the true cofactors are unchanged.
    bounds = SourceCofactorBounds(_h(graph@transform.T), _h(plane@transform.T))
    lower, upper = bounds.bounds(box)
    assert np.all(lower <= 0.) and np.all(upper >= 0.)


def test_interval_cofactors_use_global_parameter_derivatives():
    graph, plane, _ = _circle()
    bounds = SourceCofactorBounds(_h(graph), _h(plane))
    box = ((.1,.2),(.2,.3),(.1,.2),(.2,.3))
    lower, upper = bounds.bounds(box)
    # The unsigned column minors are 2*(t-cb), 2*(s-ca), repeated.
    expected_lo = np.array([-.85,-.55,-.85,-.55])
    expected_hi = np.array([-.65,-.35,-.65,-.35])
    assert np.all(lower <= expected_lo) and np.all(upper >= expected_hi)
    assert np.max(upper-lower) < .21


def test_cofactor_certificate_budget_is_cached_and_prepaid(monkeypatch):
    graph, plane, box = _circle()
    charges = []
    bounds = SourceCofactorBounds(_h(graph), _h(plane), charge=lambda n: charges.append(n) or True)
    first = bounds.bounds(box)
    count = len(charges)
    assert first is bounds.bounds(box)
    assert len(charges) == count
    from mmcore.numeric.intersection.ssx import _ssx_cofactor_bounds as module
    monkeypatch.setattr(module, 'psi_vector_net', lambda *a: (_ for _ in ()).throw(AssertionError('unpaid allocation')))
    denied = SourceCofactorBounds(_h(graph), _h(plane), charge=lambda n: False)
    assert denied.bounds(box) is None
    assert denied.exhausted


def test_cofactor_bounds_allow_exact_fixed_coordinates():
    graph, plane, _ = _circle()
    bounds = SourceCofactorBounds(_h(graph), _h(plane))
    lower, upper = bounds.bounds(((.1,.1),(.2,.2),(.1,.1),(.2,.2)))
    expected = np.array([-.85,-.55,-.85,-.55])
    assert np.all(lower <= expected) and np.all(upper >= expected)
    assert np.max(upper-lower) < 1e-10


def test_nonuniform_rational_source_bounds_contain_exact_cofactors():
    from fractions import Fraction
    from itertools import permutations
    from mmcore.numeric.intersection._root_box_certificate import exact_bernstein_value
    graph, plane, _ = _circle()
    first, second = _h(graph), _h(plane)
    first *= np.array([[1.,2.,1.],[2.,1.,2.],[1.,2.,1.]])[...,None]
    second *= np.array([[1.,2.],[3.,4.]])[...,None]
    q = (.25,.75,.625,.375)
    a, b = exact_bernstein_value(first,q[:2]), exact_bernstein_value(second,q[2:])
    columns = []
    for owner, source in enumerate((first,second)):
        for axis in (0,1):
            # These dyadic source differences and degree-two products are
            # exactly representable; subsequent evaluation stays Fraction.
            derivative = (source.shape[axis]-1)*np.diff(source,axis=axis)
            d = exact_bernstein_value(derivative,q[2*owner:2*owner+2])
            columns.append(tuple(d[k]*b[-1]-b[k]*d[-1] if owner == 0 else
                                 a[k]*d[-1]-d[k]*a[-1] for k in range(3)))
    expected = []
    for excluded in range(4):
        columns_minor = [column for i,column in enumerate(columns) if i != excluded]
        determinant = Fraction(0)
        for order in permutations(range(3)):
            term = np.prod([columns_minor[column][row] for row,column in enumerate(order)])
            parity = sum(order[i] > order[j] for i in range(3) for j in range(i+1,3)) % 2
            determinant += -term if parity else term
        expected.append(determinant)
    lower,upper = SourceCofactorBounds(first,second).bounds(tuple((t,t) for t in q))
    for lo,hi,value in zip(lower,upper,expected):
        assert Fraction(float(lo)) <= value <= Fraction(float(hi))


def test_polynomial_cofactors_preserve_dependency_inside_surface_normal():
    # A=(s+st,t+st,0), B=(u,0,v). The third minor is -(1+s+t).
    # Entrywise Jacobian intervals instead contain the spurious value0
    # from (1+t)*(1+s)-s*t with independently chosen extrema.
    first = np.array([[[s+s*t, t+s*t, 0.] for t in (0.,1.)] for s in (0.,1.)])
    second = np.array([[[u,0.,v] for v in (0.,1.)] for u in (0.,1.)])
    box = ((0.,1.),)*4
    loose = SourceCofactorBounds(_h(first), _h(second), polynomial=False).bounds(box)
    tight = SourceCofactorBounds(_h(first), _h(second)).bounds(box)
    assert loose[1][2] >= 0.
    assert tight[0][2] <= -3. and tight[1][2] >= -1.
    assert tight[1][2] < -.99


def test_uniform_weight_polynomial_bounds_enclose_exact_random_determinants():
    from fractions import Fraction
    from itertools import permutations, product
    from mmcore.numeric.intersection._root_box_certificate import exact_bernstein_value
    random = np.random.default_rng(5319)
    for shapes in (((3,2),(2,3)), ((3,3),(3,2))):
        for scale in (2.**-100, 1., 2.**100):
            first, second = [_h(random.integers(-8,9,size=shape+(3,))*scale/8)
                             for shape in shapes]
            first *= 2.**-10
            second *= 2.**12
            certificate = SourceCofactorBounds(first, second)
            assert certificate.polynomial_nets is not None
            box = ((.125,.375),(.25,.75),(.375,.625),(.125,.875))
            lower, upper = certificate.bounds(box)
            for q in product(*[(lo, (lo+hi)/2, hi) for lo,hi in box]):
                a = exact_bernstein_value(first, q[:2])
                b = exact_bernstein_value(second, q[2:])
                columns = []
                for owner, source in enumerate((first, second)):
                    for axis in (0,1):
                        derivative = (source.shape[axis]-1)*np.diff(source,axis=axis)
                        d = exact_bernstein_value(derivative,q[2*owner:2*owner+2])
                        columns.append(tuple(d[k]*b[3] if owner == 0 else -d[k]*a[3]
                                             for k in range(3)))
                for excluded in range(4):
                    minor = [column for i,column in enumerate(columns) if i != excluded]
                    determinant = Fraction(0)
                    for order in permutations(range(3)):
                        term = np.prod([minor[column][row] for row,column in enumerate(order)])
                        odd = sum(order[i]>order[j] for i in range(3) for j in range(i+1,3)) % 2
                        determinant += -term if odd else term
                    assert Fraction(float(lower[excluded])) <= determinant <= Fraction(float(upper[excluded]))


def test_constant_foreign_axes_are_removed_without_changing_source_error():
    graph, plane, box = _circle()
    certificate = SourceCofactorBounds(_h(graph), _h(plane))
    assert all(net.shape[2:4] == (1,1) for net,error in certificate.polynomial_nets)
    lower, upper = certificate.bounds(box)
    assert np.all(lower <= 0.) and np.all(upper >= 0.)
