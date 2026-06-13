"""case 8: one branch with segments dropping out

Same s1 ribbon as case 7. s2 is the same bilinear patch but pulled down
in z so the intersection becomes a single open branch. Currently returns
2 branches — a true branch is being split somewhere (chaining failure or
double-discovery of the same fragment).
"""

import numpy as np

from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx


S1 = np.array([[[33.05079627, -57.09987394, 0.],
                [29.5295466, -63.44484237, 6.7646494],
                [21.73708777, -71.24200956, 0.]],
               [[40.28725776, -58.67948118, 0.],
                [32.51384961, -66.4481508, 9.37051336],
                [28.97354926, -69.99318967, 0.]],
               [[43.28107855, -61.67330197, 0.],
                [35.49815262, -69.45145922, 9.37051336],
                [28.73859119, -79.68670826, 0.]],
               [[45.10433572, -68.20265667, 0.],
                [41.48244052, -72.46428996, 4.71678541],
                [38.71855016, -76.6579268, 0.]]])

S2 = np.array([[[40.25282656, -76.40733562, 2.23739797],
                [45.30378577, -65.64948729, 3.66374609]],
               [[23.11248642, -70.28329548, 3.75950495],
                [30.39942473, -58.97443598, 2.23739797]]])


def bez_ssx_case8():
    res = bez_ssx(S1, S2, 1e-3, rational=False)
    print(res)
    return res


if __name__ == "__main__":
    bez_ssx_case8()
