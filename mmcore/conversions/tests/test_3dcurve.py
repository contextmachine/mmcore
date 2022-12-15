"""
    Examples for the NURBS-Python Package
    Released under MIT License
    Developed by Onur Rauf Bingol (c) 2018
    3-dimensional B-Spline curve
"""

import io

from geomdl import BSpline
from geomdl import utilities
# from gmdl.visualization import VisMPL
from geomdl.visualization import VisPlotly
from mm.conversions.gmdl import txt

if __name__ == "__main__":
    def test():
        ss = io.StringIO(
            """5,15,0
        10,25,5
        20,20,10
        15,-5,15
        7.5,10,20
        12.5,15,25
        15,0,30
        5,-10,35
        10,15,40
        5,15,30""")

        # Create a B-Spline curve instance
        curve = BSpline.Curve()

        # Set up curve
        curve.degree = 3
        curve.ctrlpts = txt.import_txt(ss)

        # Auto-generate knot vector
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))

        # Set evaluation delta
        curve.delta = 0.01

        # Evaluate curve
        curve.evaluate()

        # Plot the control point polygon and the evaluated curve
        vis_comp = VisPlotly.VisCurve3D()
        curve.vis = vis_comp
        curve.render()


    test()
