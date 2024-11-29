
import sys

import numpy as np

try:
    from google.colab.patches import cv2_imshow
except ModuleNotFoundError as err:
    pass  # We do not in colab
except ImportError as err:
    pass  # We do not in colab

# @title Check trace_skeleton {"run":"auto","vertical-output":true,"display-mode":"form"}

"""## Start coding here"""

import trace_skeleton
import sys
from pathlib import Path
import rhino3dm as rg

sys.path.extend([Path(__file__).parent.__str__()])


def next_name(name, i=0):
    """
    Changes name to name-i if name already exists in the directory.For example:
    If untitled.txt exists, returns untitled-1.txt
    If untitled-1.txt also exists, returns untitled-2.txt.
    and so on ad infinitum
    """
    name_path = Path(name)
    if i == 0:
        pth = (name_path.parent / f'{name_path.stem}{"".join(name_path.suffixes)}').resolve()
    else:
        pth = (name_path.parent / f'{name_path.stem}-{i}{"".join(name_path.suffixes)}').resolve()
    if pth.exists():
        return next_name(name, i + 1)
    return pth.__str__()


from mmcore.geom.nurbs import NURBSCurve, greville_abscissae
from mmcore.numeric import evaluate_curvature
from mmcore.numeric.vectors import norm  # High performance vectorized norm


def pts_curves(pts):
    if len(pts) == 1:
        pts = pts[0]

    if isinstance(pts, (tuple, list)) and isinstance(pts[0], (float, int)):
        return pts

    elif isinstance(pts, (tuple, list)):

        lst = [pts_curves(pt) for pt in pts]

        if len(lst) == 1:
            return lst

        if isinstance(lst[0], (tuple, list, np.ndarray)) and len(lst[0]) == 2 and isinstance(lst[0][0], (int, float)):

            degree = max(min(len(lst) - 1, 3), 1)

            crv = NURBSCurve(np.array([(*pts, 0.) for pts in lst], dtype=float), degree)

            return crv

        else:

            return lst
    else:
        raise ValueError(f'Unknown {pts}')


def curvature_vector(crv, t):
    return evaluate_curvature(crv.derivative(t), crv.second_derivative(t))[1]


def curvature(curvature_vector):
    return np.linalg.norm(curvature_vector)


def reduce_control_points(curve, threshold=0.1):
    """
        :param curve: Curve
        :param threshold:maximum curvature value at which the curve will be approximated into a straight line
        Experiment with this parameter to find the best option for your case
     """
    params = np.array(greville_abscissae(curve.knots, curve.degree))
    curvature_in_params = np.array(norm(
        np.array([evaluate_curvature(curve.derivative(param), curve.second_derivative(param))[1] for param in params])))
    mask = ~(curvature_in_params < threshold)
    mask[0] = True
    mask[-1] = True
    reduced_cpts = curve.control_points[mask]
    pts, indices = list(zip(*sorted(zip(*np.unique(reduced_cpts, axis=0, return_index=True)), key=lambda x: x[1])))
    return np.array(pts)


def reduce_nurbs_curve(curve, threshold=0.1):
    """
    :param curve: Curve
    :param threshold:maximum curvature value at which the curve will be approximated into a straight line
    Experiment with this parameter to find the best option for your case
    """
    cpts = reduce_control_points(curve, threshold)
    degree = max(min(len(cpts) - 1, 3), 1)

    return NURBSCurve(cpts, degree)


def mmcore_curve_to_rhino(curve: NURBSCurve) -> rg.NurbsCurve:
    rcrv = rg.NurbsCurve.CreateControlPointCurve([rg.Point3d(*pt) for pt in curve.control_points
                                                  ], degree=curve.degree).ToNurbsCurve()
    for i in range(1, len(curve.knots) - 2):
        rcrv.Knots[i - 1] = curve.knots[i]
    if not rcrv.IsValid:
        raise ValueError('Invalid Rhino Curve created.')
    return rcrv



import cv2

path = Path(__file__).parent / "data/image-25.png"
im = cv2.imread(path.__str__(), 0)
blurred = cv2.GaussianBlur(im, (3, 3), 0)
denoised = cv2.bilateralFilter(blurred, 3, 75, 75)
_, im = cv2.threshold(denoised, 90, 255, cv2.THRESH_BINARY);

csizeCustom = 1
maxIterCustom = 75000

polys = trace_skeleton.from_numpy(im, csize=csizeCustom, maxIter=maxIterCustom);

# Creating NURBS Curves
curves = pts_curves(
    polys
)

model = rg.File3dm()

for curve in curves:

    cpts_count=len(curve.control_points)
    # Reduce NURBS Curve control points
    reduced_curve=reduce_nurbs_curve(curve)
    print(f"Reduce control points count from {cpts_count} to {len(reduced_curve.control_points)}")

    # Convert mmcore NURBSCurve to Rhino NurbsCurve
    rhino_curve = mmcore_curve_to_rhino(reduced_curve)
    # Add Rhino curve in 3dm
    model.Objects.AddCurve(rhino_curve)

# Write 3dm file
nm = next_name("drawings.3dm")

model.Write(nm, version=7)
print("\nSaved to:",nm)

# success, model, curves = write_rhino(polys)
