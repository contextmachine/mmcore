from mmcore.numeric.interval import Interval,IntervalND,interval_newton_nd

def f3(p):
    u, v, t = p
    return (u-0.5)**2 + (v+1)**2 + (t-0.2)**2


def grad3_interval(B: IntervalND):
    u, v, t = B.iv
    return [ 2*(u-0.5),  2*(v+1),  2*(t-0.2) ]   # three Interval objects

search_domain = IntervalND([Interval(-1, 2), Interval(-3, 1), Interval(-2, 3)])

roots = interval_newton_nd(
            f     = f3,

            grad_interval  = grad3_interval,
            domain         = search_domain,
            tol            = 1e-3,    # < 1 mm in CAD terms
            max_depth      = 32)

print("Certified IntervalNDes:", len(roots))
for b in roots:
    print(b.as_tuple())

# Result:

# Certified IntervalNDes: 4
# ((0.4970703125, 0.5), (-1.001953125, -1.0), (0.19970703125, 0.2021484375))
# ((0.5, 0.5029296875), (-1.001953125, -1.0), (0.19970703125, 0.2021484375))
# ((0.4970703125, 0.5), (-1.0, -0.998046875), (0.19970703125, 0.2021484375))