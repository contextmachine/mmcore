from __future__ import annotations

import numpy as np
from numpy._typing import NDArray
from scipy.spatial import KDTree

from mmcore.geom._nurbs_eval import NURBSSurfaceTuple, evaluate_nurbs_surface
from mmcore.numeric.intersection.ssx.boundary_intersection import IntersectionPoint
from mmcore.numeric.intersection.ssx.refine import refine_intersection_point


def normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of v."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def det3(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """Compute the determinant of three 3D vectors."""
    return np.linalg.det(np.column_stack((v1, v2, v3)))

def intersection_ode(x: np.ndarray, surf1: NURBSSurfaceTuple, surf2: NURBSSurfaceTuple, tol: float = 1e-8) -> np.ndarray:
    """
    Compute the derivative of the intersection curve in parametric space.
    The state vector x = [sigma, t, u, v] corresponds to parameters on surf1 and surf2.
    Depending on whether the intersection is transversal or tangential, different formulas are used.

    For transversal intersections, the marching direction is computed using the cross product
    of the surface normals. For tangential intersections, second order derivatives and the
    corresponding fundamental form coefficients are used to resolve the marching direction.
    """
    sigma, t, u, v = x
    # Evaluate surfaces with second derivatives for robustness.
    p = evaluate_nurbs_surface(surf1, sigma, t, d_order=2)
    q = evaluate_nurbs_surface(surf2, u, v, d_order=2)

    # Compute first derivatives for surface p
    p_sigma = np.array(p["Su"])
    p_t = np.array(p["Sv"])
    # Normal vector for p (P)
    P = np.cross(p_sigma, p_t)

    # Compute first derivatives for surface q
    q_u = np.array(q["Su"])
    q_v = np.array(q["Sv"])
    # Normal vector for q (Q)
    Q = np.cross(q_u, q_v)

    # Compute the cross product of the normals
    cross_normals = np.cross(P, Q)
    norm_cross = np.linalg.norm(cross_normals)

    # ----------------------------------------------------------------
    # Case 1: Transversal Intersection
    # ----------------------------------------------------------------
    if norm_cross > tol:
        # Unit tangent vector of the intersection curve in model space.
        c_tangent = cross_normals / norm_cross
        # Compute derivatives for surf1 using determinants:
        P_dot = np.dot(P, P)
        if abs(P_dot) < tol:
            sigma_prime = 0.0
            t_prime = 0.0
        else:
            sigma_prime = det3(c_tangent, p_t, P) / P_dot
            t_prime = det3(p_sigma, c_tangent, P) / P_dot

        # Compute derivatives for surf2:
        Q_dot = np.dot(Q, Q)
        if abs(Q_dot) < tol:
            u_prime = 0.0
            v_prime = 0.0
        else:
            u_prime = det3(c_tangent, q_v, Q) / Q_dot
            v_prime = det3(q_u, c_tangent, Q) / Q_dot

        return np.array([sigma_prime, t_prime, u_prime, v_prime])

    # ----------------------------------------------------------------
    # Case 2: Tangential Intersection
    # ----------------------------------------------------------------
    else:
        # Obtain second derivatives for surface p.
        p_ss = np.array(p["Suu"])
        p_st = np.array(p["Suv"])
        p_tt = np.array(p["Svv"])
        # Compute (or approximate) p's normal vector.
        if np.linalg.norm(P) < tol:
            p_normal = np.zeros_like(P)
        else:
            p_normal = normalize(P)

        # First and second fundamental form coefficients for p.
        L_p = np.dot(p_ss, p_normal)
        M_p = np.dot(p_st, p_normal)
        N_p = np.dot(p_tt, p_normal)

        # Obtain second derivatives for surface q.
        q_uu = np.array(q["Suu"])
        q_uv = np.array(q["Suv"])
        q_vv = np.array(q["Svv"])
        if np.linalg.norm(Q) < tol:
            q_normal = np.zeros_like(Q)
        else:
            q_normal = normalize(Q)

        # First and second fundamental form coefficients for q.
        L_q = np.dot(q_uu, q_normal)
        M_q = np.dot(q_uv, q_normal)
        N_q = np.dot(q_vv, q_normal)

        # For tangential intersections, the common normal is taken (using p_normal).
        N_vec = p_normal
        denom = np.dot(Q, N_vec)
        if abs(denom) < tol:
            # Fallback: unable to compute; return zero derivative.
            return np.zeros(4)

        # Compute the coefficients a11, a12, a21, a22.
        a11 = np.dot(np.cross(p_sigma, q_v), N_vec) / denom
        a12 = np.dot(np.cross(p_t, q_v), N_vec) / denom
        a21 = np.dot(np.cross(q_u, p_sigma), N_vec) / denom
        a22 = np.dot(np.cross(q_u, p_t), N_vec) / denom

        # Compute the coefficients b1, b12, b22 using the second fundamental forms.
        b1 = (a11 ** 2) * L_q + 2 * a11 * a21 * M_q + (a21 ** 2) * N_q - L_p
        b12 = a11 * a12 * L_q + (a11 * a22 + a12 * a21) * M_q + a21 * a22 * N_q - M_p
        b22 = (a12 ** 2) * L_q + 2 * a12 * a22 * M_q + (a22 ** 2) * N_q - N_p

        # Discriminant of the quadratic equation.
        disc = b12 ** 2 - b1 * b22

        if disc < 0:
            # Isolated tangential contact point.
            return np.zeros(4)
        elif abs(disc) < tol and (abs(b1) < tol and abs(b12) < tol and abs(b22) < tol):
            # Intersection cannot be evaluated by this method.
            return np.zeros(4)
        else:
            # Select the branch according to the coefficients.
            if abs(b1) > tol:
                sigma_ratio = -b12 / b1
                t_ratio = 1.0  # by convention
            elif abs(b22) > tol:
                t_ratio = -b12 / b22
                sigma_ratio = 1.0
            else:
                sigma_ratio = 0.0
                t_ratio = 0.0

            # --- Fix: Use the normalized marching direction ---
            # Compute the marching direction in the parametric domain of surf1.
            dir_p = sigma_ratio * p_sigma + t_ratio * p_t
            norm_dir_p = np.linalg.norm(dir_p)
            if norm_dir_p < tol:
                sigma_prime = 0.0
                t_prime = 0.0
            else:
                sigma_prime = sigma_ratio / norm_dir_p
                t_prime = t_ratio / norm_dir_p

            u_prime = a11 * sigma_prime + a12 * t_prime
            v_prime = a21 * sigma_prime + a22 * t_prime

            return np.array([sigma_prime, t_prime, u_prime, v_prime])


_DEFAULT_ANGLE_TOL=0.0523


def _project_point_to_segment_nd(p:NDArray[float], a:NDArray[float], b:NDArray[float], tol:float)->tuple[float,bool,float]:
    """
    Projects point p onto the line defined by segment endpoints a and b in n-dimensional space.

    Parameters:
        p (array-like): The point to project, e.g. [x, y, z, ...].
        a (array-like): The first endpoint of the segment, e.g. [x, y, z, ...].
        b (array-like): The second endpoint of the segment, e.g. [x, y, z, ...].

    Returns:
        distance (float): The Euclidean distance between p and its projection on the line.
        is_on_segment (bool): True if the projection lies within the segment [a, b], False otherwise.
    """

    # Compute the vector from a to b and from a to p
    ab = b - a
    ap = p - a

    # Compute the squared length of the segment
    ab_squared = np.dot(ab, ab)

    # Handle degenerate segment (a and b are identical)
    if ab_squared == 0:
        distance = np.linalg.norm(ap)
        return distance, False ,-1# or False, depending on how you want to handle degenerate segments

    # Compute the projection scalar 't'
    t = np.dot(ap, ab) / ab_squared

    # Compute the projection of p onto the line
    projection = a + t * ab

    # Calculate the distance from p to the projection
    distance = np.linalg.norm(p - projection)

    # Check if the projection lies within the segment boundaries (0 <= t <= 1)
    is_on_segment = ((0-tol) <= t <= (1.+tol))

    return distance, bool(is_on_segment),t


def check_boundary_intersections_condition(x0,x, interval_u_1,interval_v_1,interval_u_2,interval_v_2):
    x=x
    x0=x0
    first_condition=(x[0] < interval_u_1[0] or x[0] > interval_u_1[1] or
     x[1] < interval_v_1[0] or x[1] > interval_v_1[1] or
     x[2] < interval_u_2[0] or x[2] > interval_u_2[1] or
     x[3] < interval_v_2[0] or x[3] > interval_v_2[1])
    second_condition = not (x0[0] < interval_u_1[0] or x0[0] > interval_u_1[1] or
                       x0[1] < interval_v_1[0] or x0[1] > interval_v_1[1] or
                       x0[2] < interval_u_2[0] or x0[2] > interval_u_2[1] or
                       x0[3] < interval_v_2[0] or x0[3] > interval_v_2[1])
    return first_condition and second_condition


def check_boundary_intersections(boundary_intersection_points:list[IntersectionPoint], interval_u_1,interval_v_1,interval_u_2,interval_v_2, surface1,surface2,current, prev, tol,spt, use_spt=True, eps_n=None,angle_tol=_DEFAULT_ANGLE_TOL):
        x=current
        x0=prev


        if check_boundary_intersections_condition(x0, x, interval_u_1, interval_v_1, interval_u_2, interval_v_2):
            x_stack = [((x0, None), (x, None), list(range(len(boundary_intersection_points))))]
            while x_stack:
                new_candidates=[]
                (x0,pt0), (x,pt),   candidates = x_stack.pop(-1)
                if check_boundary_intersections_condition(x0,x,interval_u_1,interval_v_1,interval_u_2,interval_v_2):
                    #print("CHECK:", x0,x)
                    for i in range(len(candidates) ):
                        ix=candidates[i]
                        intersection_point=boundary_intersection_points[ix]
                        dist1, in_segment1,t1= _project_point_to_segment_nd( intersection_point.stuv[:2], x0[:2], x[:2], tol)
                        dist2, in_segment2 ,t2= _project_point_to_segment_nd(intersection_point.stuv[2:], x0[2:], x[2:],tol)

                        #print(    dist1, in_segment1,t1,intersection_point.stuv[:2], x0[:2], x[:2], spt)
                        if in_segment1 and in_segment2 and (dist1<tol) and(dist2<tol):
                            #print('FIND BOUNDARY INTERSECTION POINT:', intersection_point.point.tolist(),dist1,dist2,in_segment1,in_segment2,spt)
                            return True, ix
                        elif (in_segment1 and in_segment2):
                            if use_spt:
                                if pt0 is None:
                                    pt0=evaluate_nurbs_surface(
                                    surface1, x0[0],x0[1],0)["S"]
                                if pt is None:
                                    pt = evaluate_nurbs_surface(
                                    surface1,x[0],x[1], 0)["S"]
                                dist, in_segment,_=_project_point_to_segment_nd(intersection_point.point,pt0,pt,spt)

                                if dist<spt:
                                    #print('FIND BOUNDARY INTERSECTION POINT (SPT):', intersection_point.point.tolist(), dist,
                                    #      in_segment, spt)
                                    return True, ix

                            new_candidates.append(candidates[i])

                            #print("IN_SEGM")
                            #print(dist1,dist2,in_segment1,in_segment2,spt)
                            #print([intersection_point.point.tolist(), intersection_point.stuv.tolist(),x.tolist(), x0.tolist()])
                        else:
                            ...
                            #print('FAIL',x,x0,dist1,dist2, in_segment1,in_segment2,t1,t2,intersection_point.stuv)
                    if len(new_candidates) == 0:
                        continue
                    x_mid=(x + x0) / 2

                    x_mid, pt1,pt2,_= refine_intersection_point(x_mid, surface1, surface2, spt=spt, max_iter=100, angle_tol=angle_tol, eps_n=eps_n
                                                                )




                    x_stack.append(((x0,pt0),(x_mid,pt1['S']),new_candidates))
                    x_stack.append(((x_mid,pt1['S']), (x,pt),new_candidates))


                else:
                    continue
            pt0 = evaluate_nurbs_surface(
                surface1, x0[0], x0[1], 0)["S"]
            pt = evaluate_nurbs_surface(
                surface1, x[0], x[1], 0)["S"]
            #print([pt0.tolist(),pt.tolist(),[ b.point.tolist() for b in boundary_intersection_points]])



            raise ValueError(
                "The area boundary has been reached, but the boundary intersection point has not been found: "+(f"\n\nboundary intersection points:\n{[pt.point.tolist() for pt in boundary_intersection_points]}\n"
             f"surfaces control points:\n{ [surface1.control_points.tolist(),surface2.control_points.tolist()]}\n"
             f"last marching step (xyz, next_xyz):\n{np.array([pt0, pt]).tolist()}\n"))

        else:


            #pt_prev=evaluate_nurbs_surface(surface1,x0[0],x0[1], 0)['S']
            #pt_next=evaluate_nurbs_surface(surface1, x[0], x[1], 0)['S']
            #print([pt_prev.tolist(),pt_next.tolist()])
            #raise ValueError("The area boundary has been reached, but the boundary intersection point has not been found")

            return False, -1


def validated_ode_solver(
        f,
        x0: np.ndarray,
        surf1: NURBSSurfaceTuple,
        surf2: NURBSSurfaceTuple,
        s_max: float,
        h_initial: float,
        tol: float,
        spt:float,
        boundary_intersections=None,
        context:dict|None=None,
        boundary_check_spt=True,
        angle_tol:float=_DEFAULT_ANGLE_TOL,
        eps_n:float=None,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """
    A validated ODE solver that marches along the intersection curve by solving the ODE system:
        x' = f(x, surf1, surf2, spt)
    using an adaptive step size strategy and step doubling to produce a validated enclosure
    at each step.

    Parameters:
        f       : Function computing the derivative (the ODE system) in parametric space.
        x0      : Initial state (4-vector: [sigma, t, u, v]).
        surf1   : First NURBS surface.
        surf2   : Second NURBS surface.
        s_max   : Total arc length (or parameter length) to integrate.
        h_initial: Initial step size.
        spt     : Tolerance for local error and validation.

    Returns:
        A tuple containing:
         - A numpy array of states along the marching path.
         - A list of interval enclosures (each as a 2x4 numpy array: lower and upper bounds)
           for the corresponding state.
           :param eps_n:
           :param angle_tol:
    """
    interval_u_1=surf1.knot_u[surf1.order_u-1],surf1.knot_u[len(surf1.control_points)+1]
    interval_v_1 = surf1.knot_v[surf1.order_v-1],surf1.knot_v[len(surf1.control_points[0])+1]
    interval_u_2 = surf2.knot_u[surf2.order_u - 1], surf2.knot_u[len(surf2.control_points) + 1]
    interval_v_2 = surf2.knot_v[surf2.order_v - 1], surf2.knot_v[len(surf2.control_points[0]) + 1]
    s = 0.0
    h =h_initial


    x = np.array(x0, dtype=float)
    initial=x
    #print(initial)
    p_initial = evaluate_nurbs_surface(surf1, initial[0],initial[1],1)
    q_initial = evaluate_nurbs_surface(surf2, initial[2], initial[3], 1)

    initial_point=p_initial['S']
    pN = np.cross(p_initial["Su"], p_initial["Sv"])
    qN = np.cross(q_initial["Su"], q_initial["Sv"])
    pN/=np.linalg.norm(pN)
    qN/=np.linalg.norm(qN)
    initial_tangent=np.cross(pN,qN)
    #print("INITIAL_POINT",initial_point,initial.tolist(), h_initial,context, interval_u_1,interval_v_1,interval_u_2,interval_v_2)
    initial_tangent/=np.linalg.norm(initial_tangent)

    solution = [x.copy()]
    enclosures = [
        np.vstack((x, x))]  # Each enclosure is a 2x4 array: first row = lower bounds, second row = upper bounds


    check_tree=False
    if context is not None:
        check_tree = context.get("init_points_tree") is not None
    check_smax=s_max!=-1
    termination_reason=0
    iteration=-1
    while s < s_max if check_smax else True:
        iteration+=1

        # Attempt a full step of size h using Euler's method.
        f_x = f(x, surf1, surf2, tol)

        x_full = x + h * f_x

        if (iteration == 0) and (x_full[0] < interval_u_1[0] or x_full[0] > interval_u_1[1] or
                                 x_full[1] < interval_v_1[0] or x_full[1] > interval_v_1[1] or
                                 x_full[2] < interval_u_2[0] or x_full[2] > interval_u_2[1] or
                                 x_full[3] < interval_v_2[0] or x_full[3] > interval_v_2[1]):
            h=-h
            continue
        # Perform step doubling: two half-steps.
        f_x_half = f(x, surf1, surf2, tol)
        x_half = x + (h / 2.0) * f_x_half
        f_x_half2 = f(x_half, surf1, surf2, tol)
        x_half2 = x_half + (h / 2.0) * f_x_half2


        # Estimate local error.
        error_estimate = np.linalg.norm(x_half2 - x_full)
        if error_estimate > tol:
            # Reduce step size and try again.
            h /= 2.0
            if abs(h) < 1e-10:
                raise RuntimeError("Step size reduced below minimum threshold; singular point encountered.")
            continue

        # Accept the step with the more accurate two-half-step result.
        x_new = x_half2.copy()
        # Apply iterative point refinement until ||S0 - S1|| < spt.
        x_new,p_eval,q_eval,error = refine_intersection_point(x_new, surf1, surf2, spt=spt, max_iter=100, eps_n=eps_n, angle_tol=angle_tol)




        # Construct an interval enclosure for x_new.
        enclosure_lower = x_new - error_estimate
        enclosure_upper = x_new + error_estimate
        enclosure = np.vstack((enclosure_lower, enclosure_upper))
        solution.append(x_new)
        enclosures.append(enclosure)
        habs=np.abs(h)


        s += habs
        x_prev=x.copy()
        x = x_new.copy()


        success, bp_index = check_boundary_intersections(boundary_intersections, interval_u_1,interval_v_1,interval_u_2,interval_v_2,surf1,surf2,x,x_prev, tol=tol, spt=spt, use_spt=boundary_check_spt,eps_n=eps_n,angle_tol=angle_tol)


        if success:
                #print("B",[boundary_intersections[bp_index].stuv.tolist(), x_prev.tolist()])
                if np.allclose(boundary_intersections[bp_index].stuv,x_prev):
                    #print('reverse')
                    solution.pop(-1)
                    enclosures.pop(-1)
                    x=x_prev
                    h=-h
        else:
                    solution.pop(-1)
                    enclosures.pop(-1)
                    termination_reason = 1
                    solution.append(boundary_intersections[bp_index].stuv.copy())
                    x_new= boundary_intersections[bp_index].stuv.copy()
                    x=x_new.copy()
                    del boundary_intersections[bp_index]

        h = np.copysign(h,min(np.abs(h * 2.), s_max - s)) if check_smax else h*2.

        # Terminate if any parametric coordinate goes outside the [0, 1] domain.
        if check_tree:

            tree = context["init_points_tree"]
            if isinstance(tree,KDTree):
                pt = x_new
                ixs = tree.query_ball_point(pt,habs, return_sorted=True
                                            )
                if len(ixs)>0:
                    #print(tree.data[ixs].tolist())
                    pts = np.delete(tree.data, ixs,axis=0).reshape((-1,4))
                    #print(pts)
                    #print(f'Cull initial points: {ixs}')
                    if pts.size == 0:
                        check_tree = False
                        context["init_points_tree"] = None
                    elif len(pts)==1:
                        #print('pts',pts)
                        context["init_points_tree"] = pts
                        check_tree=True
                    else:
                        context["init_points_tree"] = KDTree(pts)
            elif isinstance(tree,np.ndarray):
                pts=context["init_points_tree"]
                pt = x_new

                if np.linalg.norm(pt-pts[0])<=habs:
                    context["init_points_tree"] = None
                    check_tree = False


        if termination_reason == 1:
            break
        if s>habs:
            pt = p_eval["S"]
            d=initial_point-pt
            sdist=np.linalg.norm(d)
            #print("s>habs", initial_point,   pt.tolist(), sdist, habs)


            if (sdist<=habs):
                n1=np.cross(p_eval['Su'],p_eval['Sv'])
                n2 = np.cross(q_eval['Su'],q_eval['Sv'])
                n1/=np.linalg.norm(n1)
                n2 /= np.linalg.norm(n2)
                current_tangent=np.cross(n1,n2)
                current_tangent/=np.linalg.norm(current_tangent)

                if ((1.-abs(np.dot(initial_tangent,current_tangent)))<=0.01):
                    termination_reason=2
                    break


    return np.array(solution), enclosures,termination_reason
