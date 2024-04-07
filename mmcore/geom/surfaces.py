import numpy as np


class BaseSurface:
    def __init__(self):
        super().__init__()
        self.evaluate_multi = np.vectorize(self.evaluate, signature='(i)->(j)')

    def evaluate(self, uv):
        ...

    def __call__(self, uv):
        if uv.ndim == 1:
            return self.evaluate(uv)
        else:
            return self.evaluate_multi(uv)


def blossom(b, s, t):
    bs0 = (1 - s) * b[0] + s * b[1]
    #b1t = (1 - t) * b[1] + t * b[2]
    bs1 = (1 - s) * b[1] + s * b[2]
    bst = (1 - t) * bs0 + t * bs1
    #bts = (1 - s) * b0t + s * b1t
    return bst


def evaluate_bilinear(uv, b00, b01, b10, b11):
    return np.array([1 - uv[1], uv[1]]).dot(
        np.array([1 - uv[0], uv[0]]).dot(np.array([[b00, b01], [b10, b11]])))


"""
Ψ"""


class BiLinear(BaseSurface):
    def __init__(self, a, b, c, d):
        super().__init__()
        self.b00, self.b01, self.b11, self.b10 = np.array([a, b, c, d], dtype=float)

    def evaluate(self, uv):
        return np.array([1 - uv[1], uv[1]]).dot(
            np.array([1 - uv[0], uv[0]]).dot(np.array([[self.b00, self.b01], [self.b10, self.b11]])))




class Ruled(BaseSurface):
    def __init__(self, c1, c2):
        super().__init__()
        self.c1, self.c2 = c1, c2

    def evaluate(self, uv):
        return (1 - uv[1]) * self.c1(uv[0]) + uv[1] * self.c2(uv[0])
def ruled(c1,c2):
    xu0, xu1 = lambda u: c1(u), lambda u: c2(u)

    def inner(uv):
        return (1 - uv[1])*xu0(uv[0]) + uv[1]*xu1( uv[0])
    return inner

class Coons(BaseSurface):
    """
    import numpy as np
def cubic_spline(p0, c0, c1, p1):
    p0, c0, c1, p1 = np.array([p0, c0, c1, p1])
    def inner(t):
        return ((p0 * ((1 - t) ** 3)
                 + 3 * c0 * t * ((1 - t) ** 2)
                 + 3 * c1 * (t ** 2) * (1 - t))
                + p1 * (t ** 3))
    return np.vectorize(inner, signature='()->(i)')
from mmcore.geom.surfaces import BiLinear
a,b,c,d=np.array([
    [
        (-25.632193861977559, -25.887792238151487, -8.9649174298769161),
        (-7.6507873591044131, -28.580781837412534, -4.7727445980947056),
        (3.1180460594601840, -31.620627096247443, 11.245007095153923),
        (33.586827711309354, -30.550809492847861, 0.0)],
    [
        (33.586827711309354, -30.550809492847861, 0.0),
        (23.712213781367616, -20.477792480394431, -13.510455008728185),
        (23.624609526477588, -7.8543655761938815, -12.449036305764146),
        (27.082667168033424, 5.5380493986617410, 0.0)],
   [
        (27.082667168033424, 5.5380493986617410, 0.0),
        (8.6853191615639460, -2.1121318577726527, -10.580957050242024),
        (-3.6677924590213919, -2.9387254504549816, -13.206225703752022),
        (-20.330418684651349, 3.931006353774948, 0.0)],
[
        (-20.330418684651349, 3.931006353774948, 0.0),
        (-22.086936165417491, -5.8423256715423690, 0.0),
        (-23.428753995169622, -15.855467779623531, -7.9942325520491337),
        (-25.632193861977559, -25.887792238151487, -8.9649174298769161)
    ]
])
spls=cubic_spline(*a),cubic_spline(*b),cubic_spline(*reversed(c)),cubic_spline(*reversed(d))



from mmcore.geom.surfaces import Coons
cns=Coons(*spls)

ress=[]
for u in np.linspace(0.,1.,10):
    for v in np.linspace(0., 1., 10):
        ress.append(cns.evaluate(np.array([u,v])).tolist())

    """
    def __init__(self, c1,  d1, c2, d2):
        super().__init__()
        self.c1, self.c2 = c1, c2
        self.d1, self.d2 = d1, d2

        self.xu0,  self.xu1,  self.x0v,  self.x1v=_bl( c1,  c2, d1, d2)
        self._rc,self._rd=ruled(self.c1,self.c2),ruled(self.d2,self.d1)
        self._rcd= BiLinear(self.xu0(0), self.xu0(1), self.xu1(1), self.x1v(1))

    def evaluate(self, uv):
        return self._rc(uv)+ self._rd(np.array([uv[1], uv[0]])) - self._rcd(uv)

def _bl(c1, c2, d1, d2):
    return lambda u: c1(u), lambda u: c2(u), lambda v: d1(v), lambda v: d2(v)

"""
rc, rd, rcd = Ruled(spls[0], spls[2]), ruled(spls[1], spls[3]), BiLinear(xu0(0), xu0(1), xu1(1), x1v(1))
l = []
dd = []
rcdd = []
ress = []
xu0, xu1, x0v, x1v = bl(*spls)
for i in np.linspace(0., 1., 10):
    for j in np.linspace(0., 1., 10):
        r1, r2, r3 = rc(np.array([i, j])), rd(np.array([j, i])), rcd(np.array([i, j]))
        l.append(r1.tolist())
        dd.append(r2.tolist())
        rcdd.append(r3.tolist())
        ress.append((r1 + r2 - r3).tolist())
"""