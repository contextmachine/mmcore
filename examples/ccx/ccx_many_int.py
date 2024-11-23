import argparse
from pathlib import Path

import numpy as np
import json
import sys

from mmcore.geom.nurbs import NURBSCurve
from mmcore.numeric.intersection.ccx import ccx
from mmcore.renderer.renderer2d import Renderer2D
def parse_control_points(cp_str):
    try:
        cp = json.loads(cp_str)
        if isinstance(cp, list) and all(isinstance(pt, list) for pt in cp):
            return cp
        else:
            raise ValueError
    except json.JSONDecodeError:
        try:
            with open(cp_str, 'r') as f:
                cp = json.load(f)
            if isinstance(cp, list) and all(isinstance(pt, list) for pt in cp):
                return cp
            else:
                raise ValueError
        except Exception as e:
            print(f"Error parsing control points: {e}")
            sys.exit(1)
    except ValueError:
        print("Control points must be a JSON list of lists or a valid file containing such a JSON structure.")
        sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
        This script computes the intersections between two NURBS curves and visualizes the results. 
        For visualize used 2D rendering, i.e. you will see the projection of the passed curves and intersection points in Top View.

        You can customize the control points, degrees of the curves, intersection tolerance, 
        and visualization settings via command-line arguments.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Control Points
    parser.add_argument(
        '--curve1_points', type=str, default=None,
        help="""  Control points for the first NURBS curve.
        Provide as a JSON string or a path to a JSON file.
        Example (inline):
                '[ [x1, y1, z1], [x2, y2, z2], ... ]'
        Example (file):        
        The file should contain a list of [x, y, z] coordinates.         
        [
            [-6.1666, 37.8919, 0.0],
            [14.2155, 34.2951, 0.0],
            ...
        ]
        If not provided, default control points will be used.
        """
    )

    parser.add_argument(
        '--curve2_points', type=str, default=None,
        help="""  Control points for the second NURBS curve.
        Provide as a JSON string or a path to a JSON file.
        Example (inline):
                '[ [x1, y1, z1], [x2, y2, z2], ... ]'
        Example (file):        
        The file should contain a list of [x, y, z] coordinates.         
        [
            [-6.1666, 37.8919, 0.0],
            [14.2155, 34.2951, 0.0],
            ...
        ]
        If not provided, default control points will be used.
        """
    )

    # Degrees
    parser.add_argument(
        '--degree1', type=int, default=3,
        help="Degree of the first NURBS curve (default: 3)."
    )

    parser.add_argument(
        '--degree2', type=int, default=3,
        help="Degree of the second NURBS curve (default: 3)."
    )

    # Intersection Parameters
    parser.add_argument(
        '--tolerance', type=float, default=0.001,
        help="Tolerance for intersection calculation (default: 0.001)."
    )

    # Visualization Options
    parser.add_argument(
        '--marker_color', type=str, default='red',
        help="Color of the intersection markers (default: 'red')."
    )

    parser.add_argument(
        '--marker_size', type=int, default=5,
        help="Size of the intersection markers (default: 5)."
    )

    parser.add_argument(
        '--display_ctrlpts', action='store_true',
        help="If set, control points of the curves will be displayed."
    )

    parser.add_argument(
        '--output_image', type=str, default=None,
        help="Path to save the rendered image. If not provided, the image will be displayed on screen."
    )

    return parser.parse_args()

def load_control_points(file_path_or_inline, default_points):
    if file_path_or_inline:
        try:
            points=parse_control_points(file_path_or_inline)
            points = np.array(points)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("Control points must be a list of [x, y, z] coordinates.")
            return points
        except Exception as e:
            print(f"Error loading control points from {file_path_or_inline}: {e}")
            sys.exit(1)
    else:
        return np.array(default_points)

def main():
    args = parse_arguments()

    # Default control points for Curve 1
    #default_pt1 = [
    #    [22.158641527416805, -41.265945906519704, 0.0],
    #    [38.290860468167494, -12.153299366618626, 0.0],
    #    [-4.337633425866585, -26.514161725191443, 0.0],
    #    [15.0519385376206, 19.088976634204428, 0.0],
    #    [-32.020429607822194, -4.4209634623771734, 0.0],
    #    [-14.840038397145449, 35.28715780512934, 0.0],
    #    [-44.548538735168648, 25.263858719823133, 0.0]
    #]

    # Default control points for Curve 2
    #default_pt2 = [
    #    [-6.1666326988875966, 37.89197602192263, 0.0],
    #    [14.215544129727249, 34.295146382508435, 0.0],
    #    [40.163941122493227, 14.352347571166774, 0.0],
    #    [43.540592939134157, -9.415717002272153, 0.0],
    #    [16.580183934742877, -30.210021970020129, 0.0],
    #    [-10.513217234303696, -21.362760866641814, 0.0],
    #    [-26.377549521918183, -1.2133457261141416, 0.0],
    #    [-9.3086771658378353, 19.974390832869219, 0.0],
    #    [6.3667708626935706, 27.313795735872205, 0.0],
    #    [22.990902897521764, 11.683487552065344, 0.0],
    #    [26.711915155435108, 0.6064494223866177, 0.0],
    #    [19.37450960261674, -11.611372227389872, 0.0],
    #    [8.2582629104155956, -16.234752290968999, 0.0],
    #    [-3.0903039985573031, -11.940646639020102, 0.0],
    #    [-10.739472285742522, -2.2469933680379199, 0.0],
    #    [2.2509778312197994, 7.9168038191384795, 0.0],
    #    [14.498391690318186, -0.17203316116128065, 0.0]
    #]
    d=Path(__file__).parent


    with open(    d/'cxx_many_int-curve1.json') as f:
        default_pt1=json.load(f)
    with open(d / 'cxx_many_int-curve2.json') as f:
        default_pt2 = json.load(f)
    # Load control points
    pt1 = load_control_points(args.curve1_points, default_pt1)
    pt2 = load_control_points(args.curve2_points, default_pt2)

    # Create NURBS curves
    curve1 = NURBSCurve(pt1, degree=args.degree1)
    curve2 = NURBSCurve(pt2, degree=args.degree2)


    # Compute intersections
    intersections = ccx(curve1, curve2, args.tolerance)
    print(f"Intersections count: {len(intersections)}")
    print("Intersection parameter pairs (curve1_s, curve2_t):")
    for idx, (s, t) in enumerate(intersections):
        print(f"{idx + 1}: ({s}, {t})")

    # Evaluate intersection points
    intersection_points = np.array([curve1.evaluate(s) for s, t in intersections])

    # Initialize Renderer
    renderer = Renderer2D()

    # Add intersection markers
    if intersection_points.size > 0:
        renderer.add_marker(intersection_points, color=args.marker_color, size=args.marker_size)

    # Prepare objects to render
    objects_to_render = [curve1, curve2]

    # Render the scene
    rendered_image = renderer(
        objects_to_render, 
        display_ctrlpts=args.display_ctrlpts
    )

    # Save or display the image
    if args.output_image:
        try:
            rendered_image.write_image(args.output_image)
            print(f"Rendered image saved to {args.output_image}")
        except Exception as e:
            print(f"Error saving image to {args.output_image}: {e}")
    else:
        rendered_image.show()

if __name__ == "__main__":
    main()
    [[22.158641527416805, -41.265945906519704, 0.0],[38.290860468167494, -12.153299366618626, 0.0],[-4.337633425866585, -26.514161725191443, 0.0],[15.0519385376206, 19.088976634204428, 0.0],[-32.020429607822194, -4.4209634623771734, 0.0],[-14.840038397145449, 35.28715780512934, 0.0],[-44.548538735168648, 25.263858719823133, 0.0]]
