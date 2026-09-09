from fractions import Fraction

import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._ssx_affine_constraints import AffineParameterConstraints


def test_no_constraints_preserves_the_whole_box():
    box = ((0.,1.),(.25,.75),(.125,.5),(0.,.625))
    assert AffineParameterConstraints(({},{})).contract(box) == box


def test_exact_rational_constant_coordinate_fixes_opposite_parameter_only_in_constraints():
    from mmcore.numeric.intersection.ssx._ssx_affine_constraints import augment_constant_coordinates
    from mmcore.numeric.intersection.ssx._ssx_root_identity import _pure_affine_coordinates
    first = np.array([[[x,0.,5*w,w],[x,10*w,5*w,w]]
                      for x,w in ((0.,1.),(5.,1.5),(20.,2.))])
    second = np.array([[[10*u,10*v,10*u,1.] for v in (0.,1.)] for u in (0.,1.)])
    maps = tuple(_pure_affine_coordinates(net) for net in (first,second))
    augmented = augment_constant_coordinates((first,second),maps)
    assert (2,0) not in maps[0]
    assert augmented[0][2,0] == (Fraction(5),Fraction(0))
    box = AffineParameterConstraints(augmented).contract(((0.,1.),)*4)
    assert box[0] == box[2] == (.5,.5)
    assert box[1] == box[3] == (0.,1.)


def test_near_constant_source_coordinate_does_not_fix_a_parameter():
    from mmcore.numeric.intersection.ssx._ssx_affine_constraints import augment_constant_coordinates
    first = np.array([[[s,t,1.,1.] for t in (0.,1.)] for s in (0.,1.)])
    first[-1,-1,2] = np.nextafter(1.,2.)
    maps = ({},{})
    augmented = augment_constant_coordinates((first,first),maps)
    assert (2,0) not in augmented[0]


def test_constant_coordinate_proof_is_prepaid_before_fraction_work(monkeypatch):
    import mmcore.numeric.intersection.ssx._ssx_affine_constraints as module
    first = np.array([[[s,t,1.,1.] for t in (0.,1.)] for s in (0.,1.)])
    monkeypatch.setattr(module,'_fraction',lambda _: (_ for _ in ()).throw(AssertionError('unpaid')))
    assert module.augment_constant_coordinates((first,first),({},{ }),charge=lambda _:False) is None


def _mixed_curved_pair():
    first = np.array([[[i/2,j/2,i*j/4] for j in range(3)] for i in range(3)])
    second = first.copy()
    z = (1.,-1.,1.)
    second[...,2] += np.array([[z[i]+z[j]-.5 for j in range(3)] for i in range(3)])
    matrix = np.array([[1.,1.,0.],[0.,1.,1.],[1.,0.,1.]])
    return tuple(np.concatenate([net@matrix.T,np.ones((3,3,1))],axis=-1)
                 for net in (first,second))


def test_exact_projected_coordinates_restore_mixed_world_parameter_relations():
    from mmcore.numeric.intersection.ssx._ssx_affine_constraints import augment_projected_coordinates
    from mmcore.numeric.intersection.ssx._ssx_root_identity import _pure_affine_coordinates
    pair = _mixed_curved_pair()
    maps = tuple(_pure_affine_coordinates(net) for net in pair)
    copies = tuple(dict(mapping) for mapping in maps)
    augmented = augment_projected_coordinates(pair,maps)
    assert maps == copies
    assert any(coordinate > 2 for coordinate,axis in augmented[0])
    result = AffineParameterConstraints(augmented).contract(
        ((.25,.5),(.625,.75),(0.,1.),(0.,1.)))
    assert result == ((.25,.5),(.625,.75),(.25,.5),(.625,.75))


def test_projected_source_identity_is_exact_and_not_a_small_residual_gate():
    from mmcore.numeric.intersection.ssx._ssx_affine_constraints import augment_projected_coordinates
    first = np.array([[[i/2,j/2,i*j/4,1.] for j in range(3)] for i in range(3)])
    second = first.copy()
    first[-1,-1,0] = np.nextafter(first[-1,-1,0],np.inf)
    maps = augment_projected_coordinates((first,second),({},{}))
    result = AffineParameterConstraints(maps).contract(
        ((.25,.5),(.625,.75),(0.,1.),(0.,1.)))
    assert result[2] == (0.,1.)
    assert result[3] == (.625,.75)


def test_projected_coordinate_refuses_nonuniform_weights():
    from mmcore.numeric.intersection.ssx._ssx_affine_constraints import augment_projected_coordinates
    first,second = _mixed_curved_pair()
    first[1,1] *= 2.
    assert augment_projected_coordinates((first,second),({},{})) == ({},{})


def test_projected_affine_identity_handles_higher_degree_control_nets():
    from mmcore.numeric.intersection.ssx._ssx_affine_constraints import augment_projected_coordinates
    first = np.array([[[i/4,j/4,i*j/16] for j in range(5)] for i in range(5)])
    second = first.copy()
    second[...,2] += np.arange(25).reshape(5,5)**2/16.
    matrix = np.array([[1.,1.,0.],[0.,1.,1.],[1.,0.,1.]])
    pair = tuple(np.concatenate([net@matrix.T,np.ones((5,5,1))],axis=-1)
                 for net in (first,second))
    constraints = AffineParameterConstraints(augment_projected_coordinates(pair,({},{})))
    box = constraints.contract(((.25,.5),(.625,.75),(0.,1.),(0.,1.)))
    assert box == ((.25,.5),(.625,.75),(.25,.5),(.625,.75))


def test_projected_coordinate_proof_is_prepaid_before_fraction_conversion(monkeypatch):
    import mmcore.numeric.intersection.ssx._ssx_affine_constraints as module
    pair = _mixed_curved_pair()
    monkeypatch.setattr(module,'_fraction',lambda _: pytest.fail('unpaid source conversion'))
    assert module.augment_projected_coordinates(pair,({},{}),charge=lambda _:False) is None


def test_projected_coordinates_preserve_existing_synthetic_keys():
    from mmcore.numeric.intersection.ssx._ssx_affine_constraints import augment_projected_coordinates
    maps = ({(7,0):(Fraction(0),Fraction(1))},{(7,0):(Fraction(0),Fraction(1))})
    augmented = augment_projected_coordinates(_mixed_curved_pair(),maps)
    assert all(mapping[7,0] == (0,1) for mapping in augmented)
    assert any(coordinate > 7 for coordinate,axis in augmented[0])
    assert set(maps[0]) == {(7,0)}


def test_constraint_constructor_accepts_certified_projection_keys():
    result = AffineParameterConstraints((
        {(8,0):(Fraction(0),Fraction(1))},
        {(8,1):(Fraction(1),Fraction(-2))})).contract(
            ((.25,.5),(0.,1.),(0.,1.),(0.,1.)))
    assert result[3] == (.25,.375)


def test_proved_identity_contraction_does_not_spend_on_each_new_box():
    charges = []
    constraints = AffineParameterConstraints(({}, {}),
        charge=lambda n: charges.append(n) or len(charges) == 1)
    for width in (1., .5, .125, np.nextafter(0., 1.)):
        box = ((0., width),) * 4
        assert constraints.contract(box) == box
    assert len(charges) == 1
    assert not constraints.exhausted


def test_negative_slope_relation_contracts_both_source_parameters():
    maps = ({(0,0):(0,1)}, {(0,0):(1,-2)})
    result = AffineParameterConstraints(maps).contract(((.25,.5),(0.,1.),(0.,1.),(0.,1.)))
    assert result[0] == (.25,.5)
    assert result[2] == (.25,.375)


def test_nonbinary_fixed_pin_rounds_outward():
    maps = ({(0,0):(0,3)}, {(0,0):(0,1)})
    result = AffineParameterConstraints(maps).contract(((0.,1.),(0.,1.),(.5,.5),(0.,1.)))
    assert Fraction(result[0][0]) <= Fraction(1,6) <= Fraction(result[0][1])
    assert result[0][0] != result[0][1]


def test_consistent_identity_cycle_preserves_all_solutions():
    maps = ({(0,0):(0,1),(1,0):(0,2)}, {(0,0):(0,1),(1,0):(0,2)})
    result = AffineParameterConstraints(maps).contract(((.125,.5),(0.,1.),(.25,.75),(0.,1.)))
    assert result[0] == result[2] == (.25,.5)


def test_nonidentity_cycle_fixes_the_common_parameter():
    maps = ({(0,0):(0,1),(1,0):(0,2)}, {(0,0):(0,1),(1,0):(-1,4)})
    result = AffineParameterConstraints(maps).contract(((0.,1.),)*4)
    assert result[0] == result[2] == (.5,.5)


def test_inconsistent_cycle_proves_the_domain_empty():
    maps = ({(0,0):(0,1),(1,0):(0,1)}, {(0,0):(0,1),(1,0):(1,1)})
    assert AffineParameterConstraints(maps).contract(((0.,1.),)*4) is None


def test_disjoint_ranges_prove_empty_without_a_geometric_tolerance():
    maps = ({(0,0):(0,1)}, {(0,0):(0,1)})
    assert AffineParameterConstraints(maps).contract(((0.,.5),(0.,1.),
                         (np.nextafter(.5,np.inf),1.),(0.,1.))) is None


def test_work_denial_keeps_the_unknown_box_and_caching_spends_once():
    maps = ({(0,0):(0,1)}, {(0,0):(0,1)})
    box = ((0.,1.),)*4
    denied = AffineParameterConstraints(maps,charge=lambda n:False)
    assert denied.exhausted and denied.contract(box) == box
    charges = []
    constraints = AffineParameterConstraints(maps,charge=lambda n:charges.append(n) or True)
    first = constraints.contract(box)
    paid = sum(charges)
    assert constraints.contract(box) is first
    assert sum(charges) == paid


def test_random_consistent_affine_relations_retain_exact_solutions():
    random = np.random.default_rng(491)
    for _ in range(40):
        # Two equations force all four variables onto one affine component.
        alpha = [Fraction(int(random.integers(-3,4)),8) for _ in range(4)]
        beta = [Fraction(int(random.choice([-3,-2,-1,1,2,3])),4) for _ in range(4)]
        # Graph edges 0--2, 1--2, 1--3 in three world coordinates.
        maps = ({}, {})
        for coordinate,(x,y) in enumerate(((0,2),(1,2),(1,3))):
            maps[0][coordinate,x] = -alpha[x]/beta[x],1/beta[x]
            maps[1][coordinate,y-2] = -alpha[y]/beta[y],1/beta[y]
        constraints = AffineParameterConstraints(maps)
        box = ((-2.,2.),)*4
        result = constraints.contract(box)
        assert result is not None
        for root in (Fraction(-1),Fraction(0),Fraction(1)):
            values = [a+b*root for a,b in zip(alpha,beta)]
            for value,(lo,hi) in zip(values,result):
                assert Fraction(lo) <= value <= Fraction(hi)


def test_canonical_candidate_is_new_and_requires_exact_float_images():
    constraints = AffineParameterConstraints(({(0,0):(0,1)}, {(0,0):(0,2)}))
    point = np.array([.5,.625,np.nextafter(.25,np.inf),.75])
    result = constraints.canonical_candidate(point)
    np.testing.assert_array_equal(result,[.5,.625,.25,.75])
    assert point[2] != result[2]
    nonbinary = AffineParameterConstraints(({(0,0):(0,1)}, {(0,0):(0,3)}))
    assert nonbinary.canonical_candidate(point) is None


def test_exact_integer_offsets_are_not_rounded_before_cycle_comparison():
    large = 2**54
    maps = ({(0,0):(large,1),(1,0):(large,1)},
            {(0,0):(large,1),(1,0):(large+1,1)})
    assert AffineParameterConstraints(maps).contract(((0.,1.),)*4) is None


def test_equivalent_source_faces_share_exact_keys_and_float_aliases_do_not():
    constraints = AffineParameterConstraints(({(0,0):(0,1)}, {(0,0):(0,3)}))
    assert constraints.face_key(0,Fraction(1,2)) == constraints.face_key(2,Fraction(1,6))
    assert constraints.face_key(0,.5) != constraints.face_key(2,1/6)
    assert constraints.face_key(0,.5) != constraints.face_key(1,.5)


def test_exact_source_certificate_rebinding_preserves_polynomial_root_identity():
    from mmcore.numeric.intersection.ssx._ssx_planar_cut import exact_source_planar_cut
    from mmcore.numeric.intersection.ssx._ssx_root_identity import _pure_affine_coordinates
    graph = np.array([[[u,v,(v-.5)*z,1.] for v in (0.,1.)]
                      for u,z in zip((0.,.5,1.),(1.,1.,2.))])
    plane = np.array([[[u,v,0.,1.] for v in (0.,1.)] for u in (-.25,1.25)])
    constraints = AffineParameterConstraints(tuple(_pure_affine_coordinates(s) for s in (graph,plane)))
    result = exact_source_planar_cut(graph,plane,0,.5,((0.,1.),)*4,max_cells=1000)
    root, = result['isolated']
    certificate = root['source_cut_certificate']
    rebound = constraints.rebind_source_certificate(certificate,2,.5)
    assert rebound is not None
    assert rebound['polynomial'] == certificate['polynomial']
    assert rebound['interval'] == certificate['interval']
    assert rebound['axis'] == 2
    assert rebound['parameter_root_box'][2] == (('1','2'),('1','2'))
    assert certificate['axis'] == 0
    assert constraints.rebind_source_certificate(certificate,2,np.nextafter(.5,1.)) is None
