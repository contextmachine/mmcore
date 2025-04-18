import math
from mmcore.numeric.interval._interval import Interval
# ---------- elementary transcendental enclosures ----------
PI_2=2*math.pi
PI_05=math.pi/2


def sin_interval(I:Interval):
    """tight enclosure of sin(I)"""
    L,H=I.low, I.upp
    if H-L>=PI_2:
        return Interval(-1.,1.)
    # reduce range to 0..2π for easier reasoning
    k=math.floor(L/(PI_2))
    Lr, Hr = L-PI_2*k, H-PI_2*k
    pts=[Lr,Hr]
    # critical points at π/2+πn
    for n in range(int(math.floor((Lr-PI_05)/math.pi))-1,
                   int(math.ceil((Hr-PI_05)/math.pi))+2):
        cp=n*math.pi+PI_05
        if L<=cp<=H: pts.append(cp)
    # sin is periodic, evaluate sin of original points
    vals=[math.sin(p) for p in pts]
    return Interval(min(vals), max(vals))
def cos_interval(I:Interval):
    return sin_interval(I+PI_05)
# patch into math-like namespace for convenience
def sin(X): return sin_interval(X) if isinstance(X,Interval) else math.sin(X)
def cos(X): return cos_interval(X) if isinstance(X,Interval) else math.cos(X)
