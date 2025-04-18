from mmcore.numeric.interval import Interval, interval_newton,cos,sin
import math

# ---------------------------- DEMONSTRATION ----------------------------
# Complex problem: find ALL real roots of f(x)=sin(x)-0.1*x in [-100, 100]

def f_scalar(x): return math.sin(x)-0.1*x
def df_interval_f(I): return cos(I)-0.1      # cos enclosure minus constant
domain=Interval(-100,100)
roots=interval_newton(f_scalar, df_interval_f, domain, tol=1e-7, max_depth=40)
print("Number of certified root intervals found:", len(roots))
print("First ten intervals:")
for root in roots:
    print(root)

# Result:

# Number of certified root intervals found: 11
# First ten intervals:
# Interval(-4.81256, -4.81256)
# Interval(-2.85234, -2.85234)
# Interval(-1.47063, -1.47063)
# Interval(-3.80315e-09, 3.80315e-09)
# Interval(1.47063, 1.47063)
# Interval(2.85234, 2.85234)
# Interval(4.81256, 4.81256)
# Interval(7.06817, 7.06817)
# Interval(7.75381, 7.75381)
# Interval(8.4232, 8.4232)
# Interval(11.0957, 11.0957)