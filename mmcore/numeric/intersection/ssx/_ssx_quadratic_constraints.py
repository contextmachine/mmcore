"""Complete affine zero sets of exact quadratic surface residuals.

Uniform positive weights give polynomial Cartesian charts. Linear residual
equations are eliminated by exact RREF. A semidefinite quadratic with zero
minimum vanishes exactly where its linear gradient vanishes, supplying more
linear constraints. A final one-dimensional affine set is clipped against
the four parameter domains and represented with certified curve chords.
Rationally factored quadratics give an exact union of affine constraints.
When their remaining source equations have linear consequences reducing
each factor to a line, the full line union and its shared singular
junctions are represented. Constant-image lines retain their nonimmersed
parameter fibers. Unfactored quadratics, remaining nonlinear equations,
higher-dimensional sets, and unsupported nonregular images remain the
general solver's responsibility.
"""
from fractions import Fraction
from itertools import permutations
from math import comb, isqrt, prod

import numpy as np

from mmcore.numeric._work_budget import (
    REASON_OUTPUT_CAP, REASON_PARAMETER_REPRESENTATION, REASON_WORK_BUDGET,
)
from mmcore.numeric.intersection._exact_univariate import _power, coefficient_build_work
from mmcore.numeric.intersection.ssx._ssx_affine_path import (
    _surface_affine_path, _restrict_exact_curve, _chord_error_bounded,
    affine_path_representation_bounded,
)
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch


class _Stopped(Exception):
    pass


def _clean(poly):
    return {power: value for power, value in poly.items() if value}


def _add(first, second, scale=1):
    result = dict(first)
    for power, value in second.items():
        result[power] = result.get(power, 0)+scale*value
    return _clean(result)


def _multiply(first, second, spend):
    spend(max(1, len(first)*len(second)))
    result = {}
    for a, x in first.items():
        for b, y in second.items():
            power = tuple(i+j for i,j in zip(a,b))
            result[power] = result.get(power, 0)+x*y
    return _clean(result)


def _substitute(poly, origin, matrix, spend):
    dimension = len(matrix[0])
    zero = (0,)*dimension
    linear = []
    for offset, row in zip(origin,matrix):
        item = {zero: offset} if offset else {}
        for axis, value in enumerate(row):
            if value:
                exponent = tuple(int(k == axis) for k in range(dimension))
                item[exponent] = value
        linear.append(item)
    result = {}
    for powers, coefficient in poly.items():
        term = {zero: coefficient}
        for axis, power in enumerate(powers):
            for _ in range(power):
                term = _multiply(term,linear[axis],spend)
        result = _add(result,term)
    return result


def _linear_solution(rows, dimension, spend):
    """Solve rows[0]+sum(rows[i+1]*x[i])=0; return affine RREF map."""
    spend(max(1,4*len(rows)*(dimension+1)**2))
    work = [list(row[1:])+[-row[0]] for row in rows]
    pivots = []
    for column in range(dimension):
        pivot = next((i for i in range(len(pivots),len(work)) if work[i][column]),None)
        if pivot is None:
            continue
        target = len(pivots)
        work[target],work[pivot] = work[pivot],work[target]
        scale = work[target][column]
        work[target] = [value/scale for value in work[target]]
        for i,row in enumerate(work):
            if i != target and row[column]:
                factor = row[column]
                work[i] = [a-factor*b for a,b in zip(row,work[target])]
        pivots.append(column)
    if any(not any(row[:dimension]) and row[-1] for row in work):
        return None
    free = [axis for axis in range(dimension) if axis not in pivots]
    origin = [Fraction(0)]*dimension
    matrix = [[Fraction(int(axis == free_axis)) for free_axis in free]
              for axis in range(dimension)]
    for row,pivot in zip(work,pivots):
        origin[pivot] = row[-1]
        matrix[pivot] = [-row[axis] for axis in free]
    return origin,matrix


def _psd(matrix, spend):
    """Exact symmetric LDL criterion, including zero pivots."""
    n = len(matrix)
    spend(max(1,4*n**3))
    work = [row.copy() for row in matrix]
    for i in range(n):
        pivot = work[i][i]
        if pivot < 0:
            return False
        if pivot == 0:
            if any(work[i][j] for j in range(i+1,n)):
                return False
            continue
        for j in range(i+1,n):
            for k in range(j,n):
                work[k][j] = work[j][k] = work[j][k]-work[i][j]*work[i][k]/pivot
    return True


def _quadratic_rows(poly, dimension, spend):
    """Return equivalent linear zero constraints, 'empty', or unsupported."""
    zero = (0,)*dimension
    matrix = [[Fraction(0)]*dimension for _ in range(dimension)]
    linear = [Fraction(0)]*dimension
    for power,value in poly.items():
        axes = [i for i,p in enumerate(power) for _ in range(p)]
        if len(axes) == 1:
            linear[axes[0]] = value
        elif len(axes) == 2:
            i,j = axes
            if i == j:
                matrix[i][i] = value
            else:
                matrix[i][j] = matrix[j][i] = value/2
    sign = 1
    if not _psd(matrix,spend):
        matrix = [[-value for value in row] for row in matrix]
        if not _psd(matrix,spend):
            return None
        linear = [-value for value in linear]
        sign = -1
    rows = [[linear[i]]+[2*value for value in row] for i,row in enumerate(matrix)]
    critical = _linear_solution(rows,dimension,spend)
    if critical is None:
        return None
    point = critical[0]
    minimum = sign*poly.get(zero,0)+sum(a*b for a,b in zip(linear,point))
    minimum += sum(point[i]*matrix[i][j]*point[j]
                   for i in range(dimension) for j in range(dimension))
    if minimum > 0:
        return 'empty'
    return rows if minimum == 0 else None


def _affine_factor_rows(poly, dimension, spend):
    """Exact rational affine factors of a quadratic, or unsupported.

    Completing the square in one variable gives discriminant L²-4AQ.
    It factors over the rationals precisely when that discriminant is
    the square of a rational affine polynomial. A reversible shear handles
    mixed-only quadratics such as x*y without a squared-coordinate term.
    Every proposed factorization is checked by exact polynomial equality.
    """
    zero = (0,)*dimension
    basis = [tuple(int(k == axis) for k in range(dimension)) for axis in range(dimension)]
    square = [tuple(2*value for value in power) for power in basis]
    pivot = next((i for i,power in enumerate(square) if poly.get(power)),None)
    shear = None
    if pivot is None:
        mixed = next((power for power in poly if sum(power) == 2),None)
        if mixed is None:
            return None
        i,j = [axis for axis,power in enumerate(mixed) if power]
        transform = [[Fraction(int(a == b)) for b in range(dimension)] for a in range(dimension)]
        transform[j][i] = Fraction(1)  # x_j = y_j+y_i.
        poly = _substitute(poly,[Fraction(0)]*dimension,transform,spend)
        pivot,shear = i,(i,j)
    coefficient = poly[square[pivot]]
    linear, remainder = {}, {}
    for power,value in poly.items():
        if power[pivot] == 1:
            exponent = list(power); exponent[pivot] = 0
            linear[tuple(exponent)] = value
        elif not power[pivot]:
            remainder[power] = value
    discriminant = _add(_multiply(linear,linear,spend),remainder,-4*coefficient)
    augmented = [zero]+basis
    diagonal = [discriminant.get(tuple(2*x for x in power),Fraction(0)) for power in augmented]
    index = next((i for i,value in enumerate(diagonal) if value),None)
    if index is None:
        if discriminant:
            return None
        root = [Fraction(0)]*(dimension+1)
    else:
        value = diagonal[index]
        if value < 0:
            return None
        spend(max(1,value.numerator.bit_length()+value.denominator.bit_length()))
        numerator,denominator = isqrt(value.numerator),isqrt(value.denominator)
        if numerator*numerator != value.numerator or denominator*denominator != value.denominator:
            return None
        root = [Fraction(0)]*(dimension+1)
        root[index] = Fraction(numerator,denominator)
        for k,power in enumerate(augmented):
            if k != index:
                exponent = tuple(a+b for a,b in zip(power,augmented[index]))
                root[k] = discriminant.get(exponent,Fraction(0))/(2*root[index])
        root_poly = {power:value for power,value in zip(augmented,root) if value}
        if _multiply(root_poly,root_poly,spend) != discriminant:
            return None
    center = [linear.get(power,Fraction(0)) for power in augmented]
    center[pivot+1] += 2*coefficient
    factors = [[a+sign*b for a,b in zip(center,root)] for sign in (-1,1)]
    if _multiply(*[{power:value for power,value in zip(augmented,row) if value}
                   for row in factors],spend) != {power:4*coefficient*value for power,value in poly.items()}:
        return None
    if shear is not None:
        i,j = shear
        for row in factors:
            row[i+1] -= row[j+1]  # y_j = x_j-x_i.
    return factors


def _surface_polynomials(net, owner, rational, spend):
    m,n = net.shape[:2]
    if rational:
        spend(max(1,(m*n+127)//128))
        if net[0,0,3] <= 0 or not np.all(net[...,3] == net[0,0,3]):
            return None
    spend(net.size+3*(n*coefficient_build_work(m)+m*coefficient_build_work(n)))
    weight = Fraction(float(net[0,0,3])) if rational else Fraction(1)
    result = []
    for coordinate in range(3):
        values = [[Fraction(float(net[i,j,coordinate]))/weight for j in range(n)]
                  for i in range(m)]
        along_first = [_power([values[i][j] for i in range(m)]) for j in range(n)]
        poly = {}
        for i in range(m):
            coefficients = _power([row[i] if i < len(row) else Fraction(0)
                                   for row in along_first])
            for j,value in enumerate(coefficients):
                if value:
                    exponent = [0]*4
                    exponent[2*owner:2*owner+2] = i,j
                    poly[tuple(exponent)] = value
        result.append(poly)
    return result


def _differentiate(poly, axis):
    result = {}
    for power,value in poly.items():
        if power[axis]:
            exponent = list(power)
            exponent[axis] -= 1
            result[tuple(exponent)] = power[axis]*value
    return result


def _one_sign(poly):
    degree = max((power[0] for power in poly),default=0)
    values = [sum(poly.get((j,),0)*Fraction(comb(i,j),comb(degree,j))
                  for j in range(i+1)) for i in range(degree+1)]
    return all(value > 0 for value in values) or all(value < 0 for value in values)


def _regular_tangent_line(surfaces, origin, direction, spend, require_tangency=True):
    matrix = [[value] for value in direction]
    columns = [[_substitute(_differentiate(poly,axis),origin,matrix,spend)
                for poly in surfaces[axis//2]] for axis in range(4)]
    for owner in range(2):
        a,b = columns[2*owner:2*owner+2]
        normal = [_add(_multiply(a[(i+1)%3],b[(i+2)%3],spend),
                       _multiply(a[(i+2)%3],b[(i+1)%3],spend),-1)
                  for i in range(3)]
        if not any(_one_sign(poly) for poly in normal):
            return False
    for omitted in (range(4) if require_tangency else ()):
        chosen = [column for axis,column in enumerate(columns) if axis != omitted]
        determinant = {}
        for order in permutations(range(3)):
            term = {(0,):Fraction(1)}
            for column,coordinate in enumerate(order):
                term = _multiply(term,chosen[column][coordinate],spend)
            parity = sum(order[i] > order[j] for i in range(3) for j in range(i+1,3))%2
            determinant = _add(determinant,term,-1 if parity else 1)
        if determinant:
            return False
    path = [_substitute(poly,origin,matrix,spend) for poly in surfaces[0]]
    return any(_one_sign(_differentiate(poly,0)) for poly in path)


def exact_quadratic_constraint_ssx(first,second,atol,rational,budget):
    """Return supported source curves/the empty set, else None.

    Original source nets and world tolerance are required. Returned XYZ is
    already in world coordinates. Nonlinear equalities are replaced only
    by equivalent semidefinite gradient constraints or a verified union
    of rational affine factors. Interrupted representation remains partial.
    """
    nets = tuple(np.asarray(net,dtype=float) for net in (first,second))
    if any(net.ndim != 3 or net.shape[-1] != (4 if rational else 3)
           or min(net.shape[:2]) < 2 or not np.all(np.isfinite(net)) for net in nets):
        return None
    def spend(amount,source='quadratic_constraints'):
        if not budget.charge_cells(max(1,int(amount)),source):
            raise _Stopped
    result = dict(branches=[],points=[],singularities=[],overlap_regions=[],unresolved_regions=[])
    try:
        surfaces = [_surface_polynomials(net,owner,rational,spend) for owner,net in enumerate(nets)]
        if any(surface is None for surface in surfaces):
            return None
        residuals = [_add(a,b,-1) for a,b in zip(*surfaces)]
        if any(sum(power) > 2 for poly in residuals for power in poly):
            return None
        origin = [Fraction(0)]*4
        matrix = [[Fraction(int(i == j)) for j in range(4)] for i in range(4)]
        while True:
            dimension = len(matrix[0])
            reduced = [_substitute(poly,origin,matrix,spend) for poly in residuals]
            nonzero = [poly for poly in reduced if poly]
            if not nonzero:
                break
            rows = []
            for poly in nonzero:
                if max(map(sum,poly)) <= 1:
                    zero = (0,)*dimension
                    rows.append([poly.get(zero,Fraction(0))]+[
                        poly.get(tuple(int(i == axis) for i in range(dimension)),Fraction(0))
                        for axis in range(dimension)])
            if not rows:
                for poly in nonzero:
                    derived = _quadratic_rows(poly,dimension,spend)
                    if derived == 'empty':
                        return result
                    if derived is not None:
                        rows.extend(derived)
                if not rows:
                    return _factored_affine_zero_set(
                        nets,surfaces,residuals,nonzero,origin,matrix,
                        atol,rational,budget,spend)
            solution = _linear_solution(rows,dimension,spend)
            if solution is None:
                return result
            offset, transform = solution
            if len(transform[0]) >= dimension:
                return None
            origin = [a+sum(x*y for x,y in zip(row,offset)) for a,row in zip(origin,matrix)]
            matrix = [[sum(row[k]*transform[k][j] for k in range(dimension))
                       for j in range(len(transform[0]))] for row in matrix]
        if len(matrix[0]) == 0:
            return _represent_isolated_source_root(
                nets,surfaces,tuple(origin),atol,rational,budget,spend)
        if len(matrix[0]) != 1:
            return None
        direction = [row[0] for row in matrix]
        low,high = None,None
        for offset,slope in zip(origin,direction):
            if not slope:
                if not 0 <= offset <= 1:
                    return result
                continue
            a,b = sorted((-offset/slope,(1-offset)/slope))
            low = a if low is None else max(low,a)
            high = b if high is None else min(high,b)
        if high < low:
            return result
        if high == low:
            return None
        start = tuple(a+low*d for a,d in zip(origin,direction))
        end = tuple(a+high*d for a,d in zip(origin,direction))
        direction = tuple(b-a for a,b in zip(start,end))
        if not _regular_tangent_line(surfaces,start,direction,spend):
            return None
        construction = sum((1+sum(n-1 for n in net.shape[:2]))*net.size for net in nets)
        spend((12*construction+127)//128,'quadratic_curve')
        curve = _surface_affine_path(nets[0],start[:2],end[:2],rational)
    except _Stopped:
        return None

    boundary = any(a == b and a in (0,1) for a,b in zip(start,end))
    return _represent_affine_component(nets,start,end,curve,atol,rational,budget,
                                       'overlap' if boundary else 'tangential')


def _represent_isolated_source_root(nets,surfaces,point,atol,rational,budget,spend):
    """Publish the complete singleton proved by zero-free-parameter RREF."""
    result = dict(branches=[],points=[],singularities=[],overlap_regions=[],unresolved_regions=[])
    if any(value < 0 or value > 1 for value in point):
        return result
    def partial(reason):
        budget.mark_incomplete(reason)
        budget.append_output(result['unresolved_regions'],dict(
            stuv_min=(0.,)*4,stuv_max=(1.,)*4,reason=reason,
            exact_dimension=0,exact_stuv=tuple(map(str,point)),
            source_certificate='exact_quadratic_singleton'), 'unresolved_region')
        return result
    if budget.output_items >= budget.max_output_items:
        return partial(REASON_OUTPUT_CAP)
    def value(poly):
        spend(max(1,sum(5+sum(power) for power in poly)),'quadratic_point')
        return sum(coefficient*prod(x**degree for x,degree in zip(point,power))
                   for power,coefficient in poly.items())
    actual = [tuple(value(poly) for poly in surface) for surface in surfaces]
    if actual[0] != actual[1]:
        return None  # Defensive: the eliminated original equations must agree.
    from mmcore.numeric.intersection.ssx._ssx_singular_candidate import _cross
    normals = []
    for owner in range(2):
        spend(max(1,sum(len(poly) for poly in surfaces[owner])),'quadratic_point')
        columns = [tuple(value(_differentiate(poly,2*owner+axis))
                         for poly in surfaces[owner]) for axis in (0,1)]
        normals.append(_cross(*columns))
    try:
        stuv = np.asarray(tuple(map(float,point)))
        xyz = np.asarray(tuple(map(float,actual[0])))
        tolerance = Fraction(float(atol))
        if (not np.all(np.isfinite(stuv)) or not np.all(np.isfinite(xyz))
                or sum((a-Fraction(float(b)))**2 for a,b in zip(actual[0],xyz))
                > (tolerance/2)**2):
            return partial(REASON_PARAMETER_REPRESENTATION)
        if not affine_path_representation_bounded(
                *nets,stuv,stuv,np.array([xyz,xyz]),atol,rational=rational,
                charge=lambda n: budget.charge_cells(n,'quadratic_point')):
            return partial(REASON_WORK_BUDGET if budget.exhausted else REASON_PARAMETER_REPRESENTATION)
    except (ValueError,OverflowError,ZeroDivisionError):
        return partial(REASON_PARAMETER_REPRESENTATION)
    from mmcore.numeric.intersection.ssx._bez_ssx5 import SSXSingularity,SSXPoint
    degenerate = [owner for owner,normal in enumerate(normals) if not any(normal)]
    if degenerate:
        for owner in degenerate:
            if not budget.append_output(result['singularities'],SSXSingularity(
                    kind='cusp',stuv=stuv,xyz=xyz,surface=owner+1),'singularity'):
                return partial(REASON_OUTPUT_CAP)
    elif not any(_cross(*normals)):
        budget.append_output(result['singularities'],SSXSingularity(
            kind='tangent_point',stuv=stuv,xyz=xyz),'singularity')
    else:
        budget.append_output(result['points'],SSXPoint(stuv,xyz),'point')
    return result


def _represent_affine_component(nets,start,end,curve,atol,rational,budget,kind,
                                fiber_surfaces=()):
    """Represent one already proved exact affine component, preserving prefixes."""
    direction = tuple(b-a for a,b in zip(start,end))
    result = dict(branches=[],points=[],singularities=[],overlap_regions=[],unresolved_regions=[])
    def spend(amount,source='quadratic_curve'):
        if not budget.charge_cells(max(1,int(amount)),source):
            raise _Stopped
    def partial(reason,remaining=Fraction(0)):
        budget.mark_incomplete(reason)
        budget.append_output(result['unresolved_regions'],{
            'stuv_min': (0.,)*4,'stuv_max': (1.,)*4,'reason':reason,
            'exact_dimension':1,'exact_stuv':tuple(tuple(str(x) for x in p) for p in (start,end)),
            'remaining_path_interval':(str(remaining),'1'),
            'source_certificate':'exact_quadratic_affine_zero_set',
        },'unresolved_region')
        return result
    if budget.output_items >= budget.max_output_items:
        return partial(REASON_OUTPUT_CAP)
    tolerance = Fraction.from_float(float(atol))
    cache = {}
    def vertex(value):
        if value not in cache:
            spend(max(1,6*len(curve)),'quadratic_curve')
            location = tuple(a+value*d for a,d in zip(start,direction))
            restricted = _restrict_exact_curve(curve,value,value)
            actual = tuple(x/restricted[0][3] for x in restricted[0][:3])
            stuv = tuple(float(x) for x in location)
            xyz = tuple(float(x) for x in actual)
            if not all(np.isfinite(x) for x in stuv+xyz):
                raise ValueError('nonrepresentable source endpoint')
            exact_xyz = tuple(Fraction(x) for x in xyz)
            if sum((a-b)**2 for a,b in zip(actual,exact_xyz)) > (tolerance/2)**2:
                raise ValueError('nonrepresentable source endpoint')
            # The original affine set and the float STUV chord are separate
            # objects. Verify the published endpoint on both source charts.
            if not affine_path_representation_bounded(*nets,stuv,stuv,np.array([xyz,xyz]),
                    atol,rational=rational,charge=lambda n: budget.charge_cells(n,'quadratic_curve')):
                if budget.exhausted:
                    raise _Stopped
                raise ValueError('nonrepresentable parameter endpoint')
            cache[value] = (stuv,xyz)
        return cache[value]
    accepted = []
    accepted_end = Fraction(0)
    stack = [(Fraction(0),Fraction(1))]
    reason,remaining = None,Fraction(0)
    try:
        while stack:
            lo,hi = stack.pop()
            remaining = lo
            a,b = vertex(lo),vertex(hi)
            if any(a[0][2*owner:2*owner+2] == b[0][2*owner:2*owner+2]
                   and start[2*owner:2*owner+2] != end[2*owner:2*owner+2]
                   for owner in range(2)):
                raise ValueError('distinct parameter endpoints alias')
            spend(max(1,6*len(curve)**2),'quadratic_curve')
            local_curve = _restrict_exact_curve(curve,lo,hi)
            chord = [tuple(Fraction(x) for x in item[1]) for item in (a,b)]
            true_path = _chord_error_bounded(local_curve,chord,tolerance/2)
            public_path = true_path and affine_path_representation_bounded(
                *nets,a[0],b[0],np.array([a[1],b[1]]),atol,rational=rational,
                charge=lambda n: budget.charge_cells(n,'quadratic_curve'))
            if budget.exhausted:
                raise _Stopped
            if public_path:
                if not accepted:
                    accepted.append(a)
                accepted.append(b)
                accepted_end = hi
            else:
                midpoint = (lo+hi)/2
                if float(midpoint) in (float(lo),float(hi)):
                    raise ValueError('source curve exceeds parameter resolution')
                stack.extend(((midpoint,hi),(lo,midpoint)))
    except _Stopped:
        reason = REASON_WORK_BUDGET
    except (OverflowError,ValueError,ZeroDivisionError):
        reason = REASON_PARAMETER_REPRESENTATION
    if len(accepted) >= 2:
        stuv = np.array([item[0] for item in accepted])
        xyz = np.array([item[1] for item in accepted])
        endpoints = (
            tuple(start),tuple(a+accepted_end*d for a,d in zip(start,direction)))
        if fiber_surfaces:
            from mmcore.numeric.intersection.ssx._bez_ssx5 import SSXSingularity
            for owner in fiber_surfaces:
                feature = SSXSingularity(kind='cusp_curve',stuv=stuv[0],xyz=xyz[0],
                                         samples=stuv.copy(),surface=owner+1)
                feature._source_parameter_path = endpoints
                if not budget.append_output(result['singularities'],feature,'singularity'):
                    reason = REASON_OUTPUT_CAP
                    break
        else:
            branch = SSXBranch((stuv,xyz),kind=kind,closed=False,overlap=kind == 'overlap')
            branch._exact_endpoint_keys = endpoints
            # This is the actual zero affine path established by elimination,
            # not the rounded output polyline; downstream ownership can use it.
            branch._source_parameter_path = endpoints
            budget.append_output(result['branches'],branch,'branch')
    return partial(reason,remaining) if reason else result


def _factor_line(residuals,origin,matrix,row,spend):
    """Restrict one exact factor, then eliminate its linear consequences."""
    rows = [row]
    while rows:
        dimension = len(matrix[0])
        solved = _linear_solution(rows,dimension,spend)
        if solved is None:
            return ()  # This factor has no source solutions.
        offset,transform = solved
        origin = tuple(a+sum(x*y for x,y in zip(old,offset)) for a,old in zip(origin,matrix))
        matrix = [[sum(old[k]*transform[k][j] for k in range(dimension))
                   for j in range(len(transform[0]))] for old in matrix]
        dimension = len(matrix[0])
        reduced = [_substitute(poly,origin,matrix,spend) for poly in residuals]
        nonzero = [poly for poly in reduced if poly]
        if not nonzero:
            return (origin,tuple(old[0] for old in matrix)) if dimension == 1 else None
        rows = []
        for poly in nonzero:
            if max(map(sum,poly)) <= 1:
                rows.append([poly.get((0,)*dimension,Fraction(0))]+[
                    poly.get(tuple(int(i == axis) for i in range(dimension)),Fraction(0))
                    for axis in range(dimension)])
    return None  # A remaining nonlinear equation is not discharged.


def _line_strata(surfaces,start,direction,spend):
    """Prove a monotone image or a constant-image fiber; classify normals."""
    matrix = [[d] for d in direction]
    normals = []
    for owner in range(2):
        columns = [[_substitute(_differentiate(poly,2*owner+axis),start,matrix,spend)
                    for poly in surfaces[owner]] for axis in (0,1)]
        a,b = columns
        normals.append([_add(_multiply(a[(i+1)%3],b[(i+2)%3],spend),
                             _multiply(a[(i+2)%3],b[(i+1)%3],spend),-1)
                        for i in range(3)])
    path = [_substitute(poly,start,matrix,spend) for poly in surfaces[0]]
    degenerate = tuple(owner for owner,normal in enumerate(normals) if not any(normal))
    if not any(_differentiate(poly,0) for poly in path):
        return (degenerate,()) if degenerate else None
    if not any(_one_sign(_differentiate(poly,0)) for poly in path):
        return None
    endpoints = []
    for owner,normal in enumerate(normals):
        # Nonnegative, nonzero Bernstein coefficients imply strict sign
        # on the open interval. Endpoint normal zeros remain C1 events.
        def interior_sign(poly):
            degree = max((power[0] for power in poly),default=0)
            spend(max(1,(degree+1)**2),'quadratic_strata')
            values = [sum(poly.get((j,),0)*Fraction(comb(i,j),comb(degree,j))
                          for j in range(i+1)) for i in range(degree+1)]
            return any(values) and (all(v >= 0 for v in values) or all(v <= 0 for v in values))
        if not any(interior_sign(poly) for poly in normal):
            return None
        for endpoint in (0,1):
            if not any(sum(coefficient*endpoint**power[0] for power,coefficient in poly.items())
                       for poly in normal):
                endpoints.append((endpoint,owner))
    return (),tuple(endpoints)


def _factored_affine_zero_set(nets,surfaces,residuals,reduced,origin,matrix,
                              atol,rational,budget,spend):
    """Complete factor unions whose linear consequences are affine lines."""
    dimension = len(matrix[0])
    if dimension < 2:
        return None
    factors = next((rows for poly in reduced
                    if (rows := _affine_factor_rows(poly,dimension,spend)) is not None),None)
    if factors is None:
        return None
    segments,strata = [],[]
    for row in factors:
        solved = _factor_line(residuals,origin,matrix,row,spend)
        if solved is None:
            return None
        if not solved:
            continue
        anchor,direction = solved
        low,high = None,None
        for value,slope in zip(anchor,direction):
            if not slope:
                if not 0 <= value <= 1:
                    low,high = Fraction(1),Fraction(0)
                    break
                continue
            a,b = sorted((-value/slope,(1-value)/slope))
            low = a if low is None else max(low,a)
            high = b if high is None else min(high,b)
        if high < low:
            continue
        if high == low:
            return None  # A clipped singleton needs point classification.
        start = tuple(a+low*d for a,d in zip(anchor,direction))
        end = tuple(a+high*d for a,d in zip(anchor,direction))
        segment = tuple(sorted((start,end)))
        if segment in segments:
            continue
        delta = tuple(b-a for a,b in zip(*segment))
        character = _line_strata(surfaces,segment[0],delta,spend)
        if character is None:
            return None
        segments.append(segment)
        strata.append(character)

    cuts = [{Fraction(0),Fraction(1)} for _ in segments]
    junctions = set()
    for i,(a,b) in enumerate(segments):
        first_direction = tuple(y-x for x,y in zip(a,b))
        for j,(c,d) in enumerate(segments[:i]):
            second_direction = tuple(y-x for x,y in zip(c,d))
            rows = [[x-y,u,-v] for x,y,u,v in zip(a,c,first_direction,second_direction)]
            crossing = _linear_solution(rows,2,spend)
            if crossing is None:
                continue
            if crossing[1][0]:
                return None  # Distinct records must not overlap on an interval.
            u,v = crossing[0]
            if 0 <= u <= 1 and 0 <= v <= 1:
                cuts[i].add(u);cuts[j].add(v)
                junctions.add(tuple(x+u*y for x,y in zip(a,first_direction)))

    result = dict(branches=[],points=[],singularities=[],overlap_regions=[],unresolved_regions=[])
    construction = sum((1+sum(n-1 for n in net.shape[:2]))*net.size for net in nets)
    try:
        for (start,end),parameters,(fiber_surfaces,_) in zip(segments,cuts,strata):
            direction = tuple(b-a for a,b in zip(start,end))
            ordered = [Fraction(0),Fraction(1)] if fiber_surfaces else sorted(parameters)
            for lo,hi in zip(ordered,ordered[1:]):
                a = tuple(x+lo*d for x,d in zip(start,direction))
                b = tuple(x+hi*d for x,d in zip(start,direction))
                spend((12*construction+127)//128,'quadratic_curve')
                curve = _surface_affine_path(nets[0],a[:2],b[:2],rational)
                boundary = any(x == y and x in (0,1) for x,y in zip(a,b))
                part = _represent_affine_component(
                    nets,a,b,curve,atol,rational,budget,'overlap' if boundary else 'transversal',
                    fiber_surfaces=fiber_surfaces)
                for key in result:
                    result[key].extend(part[key])
                if part['unresolved_regions']:
                    # The whole remaining factor union is still an obligation;
                    # a represented prefix cannot discharge the other factor.
                    reason = part['unresolved_regions'][0]['reason']
                    budget.append_output(result['unresolved_regions'],dict(
                        stuv_min=(0.,)*4,stuv_max=(1.,)*4,reason=reason,
                        source_certificate='exact_quadratic_affine_union'), 'unresolved_region')
                    return result
        # Distinct exact component endpoints must remain distinguishable
        # in the published lifted data, including across component records.
        published,aliases = {},set()
        for branch in result['branches']+result['singularities']:
            spend(2,'quadratic_curve')
            path = branch.curve[0] if hasattr(branch,'curve') else branch.samples
            for key,end in zip(branch._source_parameter_path,(0,len(path)-1)):
                floating = tuple(path[end])
                if floating in published and published[floating] != key:
                    aliases.add(floating)
                published[floating] = key
        if aliases:
            # Preserve both known components, but their float endpoints
            # cannot express the distinct source incidences. This early
            # tier returns a representation partial without joining them.
            budget.mark_incomplete(REASON_PARAMETER_REPRESENTATION)
            budget.append_output(result['unresolved_regions'],dict(
                stuv_min=(0.,)*4,stuv_max=(1.,)*4,reason=REASON_PARAMETER_REPRESENTATION,
                source_certificate='exact_quadratic_affine_union'), 'unresolved_region')
        from mmcore.numeric.intersection.ssx._bez_ssx5 import SSXSingularity
        from mmcore.numeric.intersection.ssx._ssx_singular_candidate import _cross
        def on_path(point,path):
            spend(16,'quadratic_junction')
            a,b = path
            candidates = [(p-x)/(y-x) for p,x,y in zip(point,a,b) if y != x]
            return (bool(candidates) and 0 <= candidates[0] <= 1
                    and all(value == candidates[0] for value in candidates)
                    and all(x != y or p == x for p,x,y in zip(point,a,b)))
        def endpoint_links(point):
            spend(max(1,2*len(result['branches'])),'quadratic_junction')
            return [(i,end) for i,branch in enumerate(result['branches'])
                    for key,end in zip(branch._exact_endpoint_keys,(0,len(branch.curve[0])-1))
                    if key == point and tuple(branch.curve[0][end]) not in aliases]
        fibers = list(result['singularities'])
        for fiber in fibers:
            fiber.branch_links = [(i,end) for i,branch in enumerate(result['branches'])
                                  for key,end in zip(branch._exact_endpoint_keys,(0,len(branch.curve[0])-1))
                                  if tuple(branch.curve[0][end]) not in aliases
                                  and on_path(key,fiber._source_parameter_path)]
        for segment,(_,endpoints) in zip(segments,strata):
            for endpoint,owner in endpoints:
                point = segment[endpoint]
                if any(fiber.surface == owner+1 and on_path(point,fiber._source_parameter_path)
                       for fiber in fibers):
                    continue  # The represented C1 fiber owns this endpoint.
                links = endpoint_links(point)
                if links:
                    index,end = links[0]
                    branch = result['branches'][index]
                    budget.append_output(result['singularities'],SSXSingularity(
                        kind='cusp',stuv=branch.curve[0][end],xyz=branch.curve[1][end],
                        surface=owner+1,branch_links=links),'singularity')
        def value(poly,point):
            spend(max(1,len(poly)*9),'quadratic_junction')
            return sum(coefficient*np.prod([x**power for x,power in zip(point,exponent)])
                       for exponent,coefficient in poly.items())
        for point in sorted(junctions):
            if tuple(map(float,point)) in aliases:
                continue
            normals = []
            for owner in range(2):
                jets = [tuple(value(_differentiate(poly,2*owner+axis),point)
                              for poly in surfaces[owner]) for axis in (0,1)]
                normals.append(_cross(*jets))
            if not all(any(normal) for normal in normals) or any(_cross(*normals)):
                continue
            links = endpoint_links(point)
            if not links:
                continue
            index,end = links[0]
            branch = result['branches'][index]
            budget.append_output(result['singularities'],SSXSingularity(
                kind='tangent_point',stuv=branch.curve[0][end],xyz=branch.curve[1][end],
                branch_links=links),'singularity')
    except _Stopped:
        budget.mark_incomplete(REASON_WORK_BUDGET)
        budget.append_output(result['unresolved_regions'],dict(
            stuv_min=(0.,)*4,stuv_max=(1.,)*4,reason=REASON_WORK_BUDGET,
            source_certificate='exact_quadratic_affine_union'), 'unresolved_region')
    return result
