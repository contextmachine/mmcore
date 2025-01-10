


def control_points_grid_shape(n_knots_u,
                              n_knots_v,
                              degree_u,
                              degree_v,
                              n_cpts):
    """
    Computes the number of control points in the u- and v-directions based on
    the knot vectors and degrees for a B-spline/NURBS surface.

    Parameters
    ----------
    n_knots_u : int
        Number of knots in the u-direction.
    n_knots_v : int
        Number of knots in the v-direction.
    degree_u : int
        Polynomial degree in the u-direction.
    degree_v : int
        Polynomial degree in the v-direction.
    n_cpts : int
        Total number of control points.

    Returns
    -------
    (u_count, v_count) : tuple of int
        The shape (dimensions) of the control point grid.

    Raises
    ------
    ValueError
        If the computed grid shape does not match the given total number of control points.
    """

    # Compute number of control points in each direction
    u_count = n_knots_u - degree_u - 1
    v_count = n_knots_v - degree_v - 1

    # Check that the total matches the provided n_cpts
    if u_count * v_count != n_cpts:
        raise ValueError(
            f"The provided total number of control points (n_cpts={n_cpts}) "
            f"does not match the computed shape {u_count} x {v_count} "
            f"(which equals {u_count * v_count})."
        )

    return u_count, v_count

