"""Python 3 with Numpy"""
# requirements: numpy, shapely

import Rhino.Geometry as rg

import rhinoscriptsyntax as rs


def offset(coordinates):
    x1, y1, z1 = coordinates.__next__()
    points = []

    for x2, y2, z2 in coordinates:

        # tangential slope approximation
        try:
            slope = (y2 - y1) / (x2 - x1)
            # perpendicular slope
            pslope = -1 / slope  # (might be 1/slope depending on direction of travel)
        except ZeroDivisionError:
            continue
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        sign = ((pslope > 0) == (x1 > x2)) * 2 - 1

        # if z is the distance to your parallel curve,
        # then your delta-x and delta-y calculations are:
        #   z**2 = x**2 + y**2
        #   y = pslope * x
        #   z**2 = x**2 + (pslope * x)**2
        #   z**2 = x**2 + pslope**2 * x**2
        #   z**2 = (1 + pslope**2) * x**2
        #   z**2 / (1 + pslope**2) = x**2
        #   z / (1 + pslope**2)**0.5 = x

        delta_x = sign * z1 / ((1 + pslope ** 2) ** 0.5)
        delta_y = pslope * delta_x
        points.append((mid_x + delta_x, mid_y + delta_y, 0))
        x1, y1, z1 = x2, y2, z2
    return points


def add_semicircle(x_origin, y_origin, radius, num_x=50):
    points = []
    for index in range(num_x):
        x = radius * index / num_x
        y = (radius ** 2 - x ** 2) ** 0.5
        points.append((x, -y))
    points += [(x, -y) for x, y in reversed(points)]
    return [(x + x_origin, y + y_origin) for x, y in points]


def round_data(data):
    # Add infinitesimal rounding of the envelope
    assert data[-1] == data[0]
    x0, y0 = data[0]
    x1, y1 = data[1]
    xe, ye = data[-2]

    x = x0 - (x0 - x1) * .01
    y = y0 - (y0 - y1) * .01
    yn = (x - xe) / (x0 - xe) * (y0 - ye) + ye
    data[0] = x, y
    data[-1] = x, yn
    data.extend(add_semicircle(x, (y + yn) / 2, abs((y - yn) / 2)))
    del data[-18:]


def control_points_curve(points, degree=3):
    return rg.NurbsCurve.CreateControlPointCurve(list(map(lambda x: rg.Point3d(*x), points)),
                                                 degree=degree)


def GetCurve():
    "Create a interpolated curve based on a parametric equation."
    aa = rg.Curve()
    t0 = rs.coercecurve(rs.GetCurveObject("GetCurveObject", aa))
    if t0 is not None:
        return t0[0]


# crv = control_points_curve()


# rs.AddCurve(np.random.random((15,3)))

# ##print(rs.GetCurveObject.__doc__)

t0 = GetCurve()
if t0 is not None:
    ##print("Succsess ", t0)
else:
    ##print(t0)
