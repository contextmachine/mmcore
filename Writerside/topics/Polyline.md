# PolyLine

This doc provides a detailed explanation of the `mmcore.geom.polyline` module, which contains several
functions and classes for handling and manipulating polylines. The module contains several functions and classes, which
are explained in detail below.

<note title="Compound Parametric Range">
The main distinguishing feature of these parametric objects is the particular range <code>t</code> used in the estimation. 
The 
integer part of
   <code>t</code> indicates the segment index, while the fractional part indicates the <code>t</code> on that segment. For example, if 
<code> t = 3.1</code>,
segment <code>3</code> will be evaluated with <code> t = 0.1</code>.
</note>

---

## Functions

1. `evaluate_polyline(corners, t: float)`: Evaluate the position on a polyline at a given parametric value.

   Parameters:
    - `corners`: The corners of the polyline.
    - `t`: The parametric value.

2. `polyline_to_lines(pln_points)`: Converts a polyline to a set of connected lines.

   Parameters:
    - `pln_points`: The points of the polyline.

3. `trim_polyline(pln, t1, t2)`: Trims a polyline based on the given start and end parameters.

   Parameters:
    - `pln`: The polyline to be trimmed.
    - `t1`: The start parameter value for trimming.
    - `t2`: The end parameter value for trimming.

4. `insert_polyline_points(pln, ixs, pts)`: Inserts points into a polyline at specified indices.

   Parameters:
    - `pln`: A NumPy array representing a polyline.
    - `ixs`: A NumPy array containing the indices at which to insert the new points.
    - `pts`: A NumPy array containing the new points to be inserted into the polyline.

5. `split_closed_polyline(pln, t1, t2)`: Splits a closed polyline into two parts by specifying two parameter values.

   Parameters:
    - `pln`: A closed polyline represented as a numpy array of 3D points.
    - `t1`: The parameter value between 0 and 1 indicating where to split the polyline.
    - `t2`: The parameter value between 0 and 1 indicating where to split the polyline.
6. `polyline_intersection` : Intersect two polylines. The intersection points will be returned.

   Parameters:
    - `a`:  A NumPy array representing a first polyline.
    - `b`:  A NumPy array representing a second polyline.

---

## Classes

1. `PolyLine`: Represents a polyline in a 3D space. Provides methods for evaluating the polyline at specific
   parameter values, adding chamfers to corners of the polyline, and accessing the corners of the polyline.


