"""Generate exact source singular candidates from necessary affine relations.

Changing a numerical candidate does not identify it with its replacement.
The returned point is independently checked against the original source
residual and all four Jacobian minors using exact binary-rational arithmetic.
No existence or absence statement is made about the old candidate's box.
"""
from fractions import Fraction

import numpy as np


def _evaluate(net, parameters):
    work = net
    for parameter in parameters:
        while len(work) > 1:
            work = (1-parameter)*work[:-1]+parameter*work[1:]
        work = work[0]
    return tuple(work)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


class SourceSingularCandidates:
    def __init__(self, first, second, constraints, charge=None):
        self.sources = first, second
        self.constraints = constraints
        self.charge = charge
        self.exact = None
        self.cache = {}
        self.isolated_regular_cache = {}

    def __call__(self, candidate):
        corrected = self.constraints.canonical_candidate(candidate)
        if corrected is None or np.any(corrected < 0.) or np.any(corrected > 1.):
            return None
        key = tuple(corrected)
        if key in self.cache:
            return corrected if self.cache[key] else None
        work = sum(net.size*(1+sum(n-1 for n in net.shape[:2])) for net in self.sources)
        if self.charge is not None and not self.charge(max(1, (8*work+127)//128)):
            return None
        if self.exact is None:
            self.exact = []
            for source in self.sources:
                net = np.empty(source.shape, dtype=object)
                for index in np.ndindex(source.shape):
                    net[index] = Fraction.from_float(float(source[index]))
                self.exact.append(net)
        parameters = tuple(Fraction.from_float(float(x)) for x in corrected)
        values = [_evaluate(net, parameters[2*owner:2*owner+2])
                  for owner, net in enumerate(self.exact)]
        valid = all(value[3] > 0 for value in values) and all(
            values[0][k]*values[1][3] == values[1][k]*values[0][3] for k in range(3))
        if valid:
            columns = []
            for owner, (net, value) in enumerate(zip(self.exact, values)):
                for axis in (0, 1):
                    degree = net.shape[axis]-1
                    derivative = (_evaluate(degree*np.diff(net, axis=axis), parameters[2*owner:2*owner+2])
                                  if degree else (Fraction(0),)*4)
                    columns.append(tuple(derivative[k]*value[3]-value[k]*derivative[3]
                                         for k in range(3)))
            for omitted in range(4):
                a, b, c = [column for k, column in enumerate(columns) if k != omitted]
                if sum(x*y for x, y in zip(_cross(a, b), c)):
                    valid = False
                    break
        self.cache[key] = valid
        return corrected if valid else None

    def isolated_regular(self, candidate):
        """A source-verified regular-chart singular root with rank-four DΔ.

This sufficient local isolation test distinguishes a saddle junction from
arbitrary samples on a one-dimensional tangential locus. Failure is only
inconclusive; higher-order singularities keep their other search owners.
        """
        corrected = self(candidate)
        if corrected is None:
            return None
        key = tuple(corrected)
        if key in self.isolated_regular_cache:
            return corrected if self.isolated_regular_cache[key] else None
        if any(min(net.shape[:2]) < 2 for net in self.sources):
            self.isolated_regular_cache[key] = False
            return None
        work = sum(net.size*(1+sum(n-1 for n in net.shape[:2])) for net in self.sources)
        if self.charge is not None and not self.charge(max(1, (40*work+127)//128)):
            return None
        parameters = tuple(Fraction.from_float(float(x)) for x in corrected)
        columns, changes, values = [], [], []
        for owner, net in enumerate(self.exact):
            uv = parameters[2*owner:2*owner+2]
            value = _evaluate(net, uv)
            values.append(value)
            first_nets = [(net.shape[axis]-1)*np.diff(net, axis=axis) for axis in (0, 1)]
            first = [_evaluate(derivative, uv) for derivative in first_nets]
            for axis in (0, 1):
                derivative = first[axis]
                columns.append(tuple(derivative[k]*value[3]-value[k]*derivative[3] for k in range(3)))
                by_parameter = [(Fraction(0),)*3 for _ in range(4)]
                for other_axis in (0, 1):
                    degree = first_nets[axis].shape[other_axis]-1
                    second = (_evaluate(degree*np.diff(first_nets[axis], axis=other_axis), uv)
                              if degree else (Fraction(0),)*4)
                    other = first[other_axis]
                    by_parameter[2*owner+other_axis] = tuple(
                        second[k]*value[3]+derivative[k]*other[3]
                        -other[k]*derivative[3]-value[k]*second[3] for k in range(3))
                changes.append(by_parameter)
        if not any(_cross(columns[0], columns[1])) or not any(_cross(columns[2], columns[3])):
            self.isolated_regular_cache[key] = False
            return None
        matrix = [[columns[j][k]*(1 if j < 2 else -1)/values[j//2][3]**2
                   for j in range(4)] for k in range(3)]
        for omitted in range(4):
            indices = [j for j in range(4) if j != omitted]
            row = []
            for parameter in range(4):
                total = Fraction(0)
                for varied in indices:
                    a, b, c = [changes[j][parameter] if j == varied else columns[j] for j in indices]
                    total += sum(x*y for x, y in zip(_cross(a, b), c))
                row.append(total)
            matrix.append(row)
        rank = 0
        for column in range(4):
            pivot = next((i for i in range(rank, len(matrix)) if matrix[i][column]), None)
            if pivot is None:
                continue
            matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
            scale = matrix[rank][column]
            matrix[rank] = [value/scale for value in matrix[rank]]
            for i in range(rank+1, len(matrix)):
                scale = matrix[i][column]
                matrix[i] = [a-scale*b for a, b in zip(matrix[i], matrix[rank])]
            rank += 1
        self.isolated_regular_cache[key] = rank == 4
        return corrected if rank == 4 else None
