import math


def is_close(a, b, tol=1e-9):
    """Return True if a and b are within spt."""
    return abs(a - b) < tol


def cross(o, a, b):
    """
    Compute the cross product of the vectors OA and OB.
    A positive value indicates a counter-clockwise turn,
    negative indicates a clockwise turn, and values near 0 (within spt)
    indicate that the points are nearly collinear.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points, tol=1e-9):
    """
    Computes the convex hull of a set of 2D points.

    Parameters:
      points: A list of (x, y) tuples.
      tol: Tolerance for determining collinearity and point equality.

    Returns:
      A list of points representing the convex hull in counter-clockwise order.
      In degenerate cases (e.g. all points collinear) the hull is reduced accordingly.
    """
    # Sort the points lexicographically (first by x, then by y)
    # and remove nearly duplicate points.
    sorted_points = sorted(points)
    unique_points = []
    for p in sorted_points:
        if not unique_points or not (is_close(p[0], unique_points[-1][0], tol) and is_close(p[1], unique_points[-1][1], tol)):
            unique_points.append(p)

    points = unique_points
    if len(points) <= 1:
        return points

    # Build the lower part of the hull.
    lower = []
    for p in points:
        # While the last two points in 'lower' and point p do not make a
        # counter-clockwise turn (or are nearly collinear), remove the last point.
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < tol:
            lower.pop()
        lower.append(p)

    # Build the upper part of the hull.
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < tol:
            upper.pop()
        upper.append(p)

    # The last point of each list is omitted because it is repeated at the beginning of the other list.
    hull = lower[:-1] + upper[:-1]
    return hull


import math



def points_are_close(p, q, tol=1e-9):
    """Return True if points p and q are nearly identical."""
    return is_close(p[0], q[0], tol) and is_close(p[1], q[1], tol)



def is_point_on_segment(p, a, b, tol=1e-9):
    """Return True if point p lies on segment ab (including endpoints) within spt."""
    # Check collinearity
    if abs(cross(a, b, p)) > tol:
        return False
    # Check if p is between a and b in both x and y (with tolerance)
    if (min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol and
            min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol):
        return True
    return False


def point_in_convex_polygon(p, poly, tol=1e-9):
    """
    For a convex polygon (with >= 3 vertices) given in counter-clockwise order,
    return True if p is inside or on the edge.
    For degenerate cases (point or segment) the corresponding tests are applied.
    """
    n = len(poly)
    if n == 0:
        return False
    if n == 1:
        return points_are_close(p, poly[0], tol)
    if n == 2:
        return is_point_on_segment(p, poly[0], poly[1], tol)

    # For a proper polygon, p is inside if it is not to the right of any edge.
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if cross(a, b, p) < -tol:
            return False
    return True


def segments_intersect(a, b, c, d, tol=1e-9):
    """
    Return True if segment ab intersects segment cd.
    Uses orientation tests and handles collinear cases.
    """

    def orientation(p, q, r):
        val = cross(p, q, r)
        if abs(val) < tol:
            return 0
        return 1 if val > 0 else 2

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    if o1 == 0 and is_point_on_segment(c, a, b, tol): return True
    if o2 == 0 and is_point_on_segment(d, a, b, tol): return True
    if o3 == 0 and is_point_on_segment(a, c, d, tol): return True
    if o4 == 0 and is_point_on_segment(b, c, d, tol): return True

    return False


def segment_polygon_intersect(seg, poly, tol=1e-9):
    """
    Returns True if the segment (list/tuple of two points) intersects the convex polygon.
    Works for degenerate or full polygons.
    """
    # Check if either endpoint is inside the polygon.
    if point_in_convex_polygon(seg[0], poly, tol) or point_in_convex_polygon(seg[1], poly, tol):
        return True

    # Check for intersections with polygon edges.
    n = len(poly)
    if n >= 2:
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            if segments_intersect(seg[0], seg[1], a, b, tol):
                return True
    return False


def polygon_intersect(poly1, poly2, tol=1e-9):
    """
    Returns True if two (non-degenerate) convex polygons intersect.
    Checks if any vertex of one is inside the other or if any edges intersect.
    """
    # Check if a vertex of poly1 is inside poly2.
    for p in poly1:
        if point_in_convex_polygon(p, poly2, tol):
            return True
    # Check if a vertex of poly2 is inside poly1.
    for p in poly2:
        if point_in_convex_polygon(p, poly1, tol):
            return True
    # Otherwise, check edge intersections.
    for i in range(len(poly1)):
        a = poly1[i]
        b = poly1[(i + 1) % len(poly1)]
        for j in range(len(poly2)):
            c = poly2[j]
            d = poly2[(j + 1) % len(poly2)]
            if segments_intersect(a, b, c, d, tol):
                return True
    return False


def bounding_box(poly):
    """Return the bounding box ((min_x, min_y), (max_x, max_y)) for a set of points."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys)), (max(xs), max(ys))


def bounding_boxes_intersect(poly1, poly2, tol=1e-9):
    """
    Quick rejection test using bounding boxes.
    Returns True if the bounding boxes of poly1 and poly2 intersect.
    """
    (min1, max1) = bounding_box(poly1)
    (min2, max2) = bounding_box(poly2)

    if max1[0] < min2[0] - tol or max2[0] < min1[0] - tol:
        return False
    if max1[1] < min2[1] - tol or max2[1] < min1[1] - tol:
        return False
    return True


def convex_hulls_intersect(hull1, hull2, tol=1e-9):
    """
    Determine if two convex hulls (each a list of points in counter-clockwise order)
    intersect. The hulls may be degenerate (a point or a segment).
    """
    # If either hull is empty, they don't intersect.
    if not hull1 or not hull2:
        return False

    # Quick bounding box test.
    if not bounding_boxes_intersect(hull1, hull2, tol):
        return False

    n1, n2 = len(hull1), len(hull2)

    # Both hulls are points.
    if n1 == 1 and n2 == 1:
        return points_are_close(hull1[0], hull2[0], tol)

    # One hull is a point, the other is a segment.
    if n1 == 1 and n2 == 2:
        return is_point_on_segment(hull1[0], hull2[0], hull2[1], tol)
    if n1 == 2 and n2 == 1:
        return is_point_on_segment(hull2[0], hull1[0], hull1[1], tol)

    # One hull is a point and the other is a proper polygon.
    if n1 == 1 and n2 >= 3:
        return point_in_convex_polygon(hull1[0], hull2, tol)
    if n1 >= 3 and n2 == 1:
        return point_in_convex_polygon(hull2[0], hull1, tol)

    # Both hulls are segments.
    if n1 == 2 and n2 == 2:
        return segments_intersect(hull1[0], hull1[1], hull2[0], hull2[1], tol)

    # One hull is a segment, the other is a polygon.
    if n1 == 2 and n2 >= 3:
        return segment_polygon_intersect(hull1, hull2, tol)
    if n1 >= 3 and n2 == 2:
        return segment_polygon_intersect(hull2, hull1, tol)

    # Both hulls are proper polygons.
    return polygon_intersect(hull1, hull2, tol)


# Example usage:
if __name__ == '__main__':
    # Example convex hulls (each provided in counter-clockwise order)
    # Hull A: a non-degenerate polygon (a square)
    hull_A = [(0, 0), (2, 0), (2, 2), (0, 2)]

    # Hull B: a degenerate hull (a segment)
    hull_B = [(1, -1), (1, 1)]

    # Hull C: a point
    hull_C = [(3, 3)]

    # Hull D: another polygon (triangle)
    hull_D = [(1, 1), (3, 1), (2, 3)]

    print("Hull A and Hull B intersect?", convex_hulls_intersect(hull_A, hull_B))
    print("Hull A and Hull C intersect?", convex_hulls_intersect(hull_A, hull_C))
    print("Hull A and Hull D intersect?", convex_hulls_intersect(hull_A, hull_D))
    print("Hull B and Hull D intersect?", convex_hulls_intersect(hull_B, hull_D))
# Example usage:
if __name__ == "__main__":
    # Define a set of points, including collinear and nearly degenerate cases.
    pts = [
        (0, 0),
        (1, 1),
        (2, 2),  # Collinear points along a diagonal
        (2, 0),
        (1, -1),
        (0, -2),
        (-1, -1),
        (-2, 0),
        (-1, 1),
        (0, 0.000000001),  # Nearly duplicate point (with a tiny difference)
    ]

    hull = convex_hull(pts)
    print("Convex Hull:")
    for point in hull:
        print(point)
