"""CAD-scale apex classification retains original chart and sheet identity."""
from types import SimpleNamespace

import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_surface, eval_surface_d1, restrict_net_axis_v
from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx

ATOL = 1e-3
UNIT_BOX = ((0., 1.),)*4


def _plane():
    return np.array([[[u, v, 0., 1.] for v in (0., 1.)] for u in (0., 1.)])


def _fan(scale=1.):
    # S(u,v)=(u*v,v,0), with every point of v=0 mapping to the apex.
    return np.array([[[scale*u*v, scale*v, 0., 1.]
                      for v in (0., 1.)] for u in (0., 1.)])


def _cell(sources, box=UNIT_BOX, original_context=True):
    children = []
    for owner, surface in enumerate(sources):
        child = surface.copy()
        for axis in (0, 1):
            child = restrict_net_axis_v(child, axis, *box[2*owner+axis], 0., 1.)
        children.append(SimpleNamespace(surface=child))
    cell = SimpleNamespace(g1=children[0], g2=children[1], box=box)
    if original_context:
        cell.source_surfaces = tuple(sources)
    return cell


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('apex_side', [0, 1])
def test_thin_child_uses_original_apex_chart(swap, transpose, apex_side):
    first, second = _fan(), _plane()
    near, far = 1e-12, 1e-8
    parameters = np.array([.35, .5*(near+far)])
    first_box = [(0., 1.), (near, far)]
    if apex_side == 1:
        first = first[:, ::-1].copy()
        parameters[1] = 1.-parameters[1]
        first_box[1] = (1.-far, 1.-near)
    if transpose:
        first = first.swapaxes(0, 1).copy()
        parameters = parameters[::-1].copy()
        first_box.reverse()
    point = eval_surface(first, *parameters, rational=True)
    assert np.linalg.norm(point) <= ATOL
    global_root = np.r_[parameters, point[:2]]
    sources = (first, second)
    box = tuple(first_box)+((0., 1.), (0., 1.))
    if swap:
        sources = sources[::-1]
        global_root = global_root[[2, 3, 0, 1]]
        box = box[2:]+box[:2]
    cell = _cell(sources, box)
    local_root = ssx._global_to_local(global_root, box)
    owner = 1 if swap else 0
    local_surface = (cell.g1.surface, cell.g2.surface)[owner]
    local_parameters = local_root[2*owner:2*owner+2]
    local_tolerance = ssx._cell_ptol4(cell, ATOL)[2*owner:2*owner+2]
    # Neither child boundary is collapsed: the original apex lies just
    # outside this child's interval. Its CAD neighborhood still belongs
    # to that original fiber, not to a new isolated C2 touch.
    assert not ssx._on_collapsed_boundary_fiber(
        local_surface, *local_parameters, param_tol=local_tolerance,
        distance_tol=ATOL)
    assert ssx._cell_root_on_collapsed_fiber(cell, local_root, ATOL,
                                            np.full(4, ATOL))


def test_parameter_nearness_does_not_hide_a_spatially_distant_point():
    sources = (_fan(scale=1000.), _plane())
    box = ((0., 1.), (0., 1e-3), (0., 1.), (0., 1.))
    global_root = np.array([.5, .5e-3, .25, .5])
    point = eval_surface(sources[0], *global_root[:2], rational=True)
    assert np.linalg.norm(point) > ATOL
    assert ssx._on_collapsed_boundary_fiber(sources[0], *global_root[:2],
                                            param_tol=np.full(2, ATOL))
    assert not ssx._cell_root_on_collapsed_fiber(
        _cell(sources, box), ssx._global_to_local(global_root, box),
        ATOL, np.full(4, ATOL))


def test_fixed_axis_keeps_its_own_parameter_tolerance():
    # f(v)=1000*v^3 has a small spatial image near zero but large global
    # speed. The u direction is independently scaled down by 1e-5.
    f = np.array([0., 0., 0., 1000.])
    first = np.array([[[1e-5*u*y, y, 0., 1.] for y in f] for u in (0., 1.)])
    sources = (first, _plane())
    box = ((0., 1.), (.004, .006), (0., 1.), (0., 1.))
    q = np.array([.5, .005])
    point = eval_surface(first, *q, rational=True)
    assert np.linalg.norm(point) < ATOL
    global_root = np.r_[q, point[:2]]
    tolerances = np.array([.1, 1e-6, ATOL, ATOL])
    assert not ssx._cell_root_on_collapsed_fiber(
        _cell(sources, box), ssx._global_to_local(global_root, box), ATOL, tolerances)


def test_regular_other_preimage_at_the_same_xyz_is_not_the_apex_fiber():
    # S(u,v)=(v*(u-.5), v*(v-.5), 0). The boundary v=0 is collapsed,
    # while (.5,.5) maps to the same XYZ with two independent derivatives.
    first = np.array([[[v*(u-.5), y, 0., 1.]
                       for v, y in zip((0., .5, 1.), (0., -.25, .5))]
                      for u in (0., 1.)])
    point, du, dv = eval_surface_d1(first, .5, .5, rational=True)
    assert np.linalg.norm(point) <= ATOL
    assert np.linalg.norm(np.cross(du, dv)) > .1
    box = ((.4, .6), (.4, .6), (0., 1.), (0., 1.))
    global_root = np.array([.5, .5, 0., 0.])
    assert not ssx._cell_root_on_collapsed_fiber(
        _cell((first, _plane()), box), ssx._global_to_local(global_root, box),
        ATOL, np.full(4, ATOL))


def test_child_boundary_can_expose_an_original_interior_collapsed_isoline():
    first = np.array([[[u*(v-.5), v-.5, 0., 1.]
                       for v in (0., 1.)] for u in (0., 1.)])
    box = ((0., 1.), (.5, 1.), (0., 1.), (0., 1.))
    global_root = np.array([.3, .5, 0., 0.])
    assert not ssx._on_collapsed_boundary_fiber(first, *global_root[:2],
                                                param_tol=np.full(2, ATOL),
                                                distance_tol=ATOL)
    assert ssx._cell_root_on_collapsed_fiber(
        _cell((first, _plane()), box), ssx._global_to_local(global_root, box),
        ATOL, np.full(4, ATOL))


def test_standalone_cell_keeps_local_collapsed_edge_evidence():
    cell = _cell((_fan(), _plane()), original_context=False)
    assert ssx._cell_root_on_collapsed_fiber(cell, np.array([.3, 0., 0., 0.]),
                                            ATOL, np.full(4, ATOL))
