from __future__ import annotations



import numpy as np

from typing import Callable

from mmcore.numeric.bvh.lbvh import BVH, AABB, build_bvh


class HalfEdgeSegm:
    __slots__ = ('origin','twin','next','face','angle','ccw_prev','idx')
    def __init__(self, origin, angle):
        self.origin   = origin    # vertex‐index at the tail
        self.angle    = angle     # for sorting around the vertex
        self.twin     = None      # opposite direction on same fragment
        self.next     = None      # next around the face
        self.face     = None      # face ID on its left
        self.ccw_prev = None      # helper for embedding
        self.idx      = None      # will store its index in the half_edges list

def polygonize_segments(vertices, edges):
    """
    Input:
      vertices: list of (x,y) points
      edges:    list of (u,v) pairs of indices into `vertices`
                (these are your already‐noded curve‐fragments)
    Returns:
      half_edges: list of HalfEdge objects (size = 2 * len(edges))
      regions:    list of outer‐boundary loops, each a list of half_edge indices
    """
    # 1) build the 2 directed half-edges for each fragment
    half_edges = []
    varr=np.array(vertices)
    for (u, v) in edges:
        x1, y1 = vertices[u]
        x2, y2 = vertices[v]
        uv=varr[v] - vertices[u]
        vu=varr[u] - varr[v]

        N=np.linalg.norm( uv)
        uv /= N
        vu/=N

        # h_uv = HalfEdgeSegm(u, math.atan2(y2-y1, x2-x1))
        # h_vu = HalfEdgeSegm(v, math.atan2(y1-y2, x1-x2))
        h_uv = HalfEdgeSegm(u, np.atan2(uv[1],uv[0]))
        h_vu = HalfEdgeSegm(v, np.atan2(vu[1],vu[0]))
        h_uv.twin = h_vu
        h_vu.twin = h_uv
        half_edges.extend([h_uv, h_vu])

    # assign each half-edge an index for easy reference
    for i, h in enumerate(half_edges):
        h.idx = i

    # 2) sort outgoing half-edges CCW around each vertex
    by_origin = defaultdict(list)
    for h in half_edges:
        by_origin[h.origin].append(h)
    for hs in by_origin.values():
        hs.sort(key=lambda h: h.angle)
        m = len(hs)
        for i in range(m):
            # the next CCW half-edge’s ccw_prev points back here
            hs[(i+1) % m].ccw_prev = hs[i]

    # 3) link h.next so that walking →h.next keeps the face on your left
    for h in half_edges:
        h.next = h.twin.ccw_prev

    # 4) walk every half-edge to extract faces
    faces = []
    for h in half_edges:
        if h.face is None:
            curr = h
            fid = len(faces)
            cycle = []
            while True:
                curr.face = fid
                cycle.append(curr)
                curr = curr.next
                if curr is h:
                    break
            faces.append(cycle)

    # 5) drop the unbounded face (largest abs‐area)
    def signed_area(cycle):
        pts = [vertices[h.origin] for h in cycle]

        return 0.5 * np.sum([x1*y2 - x2*y1 for (x1,y1),(x2,y2) in zip(pts, pts[1:]+pts[:1])])


    areas = [signed_area(c) for c in faces]
    unbounded = max(range(len(faces)), key=lambda i: np.abs(areas[i]))

    # 6) build adjacency of bounded faces
    adj = defaultdict(set)
    for h in half_edges:
        f1, f2 = h.face, h.twin.face
        if f1 != f2 and f1!=unbounded and f2!=unbounded:
            adj[f1].add(f2)
            adj[f2].add(f1)

    # 7) flood‐fill to group faces into connected “regions”
    comp_of = {}
    comp_id = 0
    for fid in range(len(faces)):
        if fid==unbounded or fid in comp_of:
            continue
        stack = [fid]
        while stack:
            f = stack.pop()
            if f in comp_of:
                continue
            comp_of[f] = comp_id
            for nb in adj[f]:
                if nb not in comp_of:
                    stack.append(nb)
        comp_id += 1

    # 8) for each region, collect its boundary half-edges and trace one loop
    comps = defaultdict(list)
    for f, c in comp_of.items():
        comps[c].append(f)

    regions = []
    for c, flist in comps.items():
        # all half-edges whose twin is outside this component
        boundary = [h for h in half_edges
                    if h.face in flist and comp_of.get(h.twin.face,-1)!=c]

        # map tail‐vertex → an outgoing boundary half-edge
        origin_map = {h.origin: h for h in boundary}

        visited = set()
        loops = []
        for h in boundary:
            if h in visited:
                continue
            loop = []
            cur = h
            while True:
                visited.add(cur)
                loop.append(cur)
                nxt = origin_map.get(cur.twin.origin)
                if nxt is None or nxt is h:
                    break
                cur = nxt
            loops.append(loop)

        # pick the single outer loop by max |area|
        best = None
        best_area = -1
        for loop in loops:
            pts = [vertices[h.origin] for h in loop]
            A = np.abs(np.sum(x1*y2 - x2*y1
                        for (x1,y1),(x2,y2) in zip(pts, pts[1:]+pts[:1]))*0.5)
            if A > best_area:
                best_area = A
                best = loop

        # record its half-edge indices
        if best:
            regions.append([h.idx for h in best])

    return half_edges, regions

import math
from collections import defaultdict
from mmcore.numeric.aabb import aabb_intersect_fast_3d,aabb,aabb_segm3d
import math
from collections import defaultdict
from mmcore.numeric.intersection.ccx.segment import segment_intersection
from mmcore.geom._nurbs_eval import NURBSCurveTuple,evaluate_nurbs_curve,_curve_interval
from mmcore.geom._nurbs_knots import split_curve,trim_curve,reverse_curve
import math
from collections import defaultdict
from typing import List, Tuple

Point   = Tuple[float, float]
Segment = Tuple[Point, Point]


def merge_overlapping_segments(
    segments,
    eps: float = 1e-9
) :
    """
    Merge all approximately colinear, overlapping segments into their union.

    Args:
        segments: list of ((x1, y1), (x2, y2))
        eps:       maximum perpendicular distance to consider two segments
                   as lying on the same line *and* the gap tolerance
                   when merging scalar intervals.

    Returns:
        list of merged segments
    """
    # Each group represents one “strip” (approximate line)
    # and stores its unit direction (ux,uy), normal (nx,ny),
    # offset = n·P for any point P on that line, and a list of
    # scalar intervals along (ux,uy).
    groups = []

    for (x1, y1), (x2, y2) in segments:
        dx, dy = x2 - x1, y2 - y1
        # skip degenerate
        if abs(dx) < eps and abs(dy) < eps:
            continue

        # unit direction
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        # canonicalize orientation
        if ux < -eps or (abs(ux) < eps and uy < -eps):
            ux, uy = -ux, -uy
        # normal vector
        nx, ny = -uy, ux
        # signed offset from origin
        offset = nx * x1 + ny * y1

        # try to place this segment into an existing group
        placed = False
        for g in groups:
            # check perpendicular distance of both endpoints to g’s line
            d1 = g['nx'] * x1 + g['ny'] * y1 - g['offset']
            d2 = g['nx'] * x2 + g['ny'] * y2 - g['offset']
            if abs(d1) <= eps and abs(d2) <= eps:
                # project onto g’s direction to get [a,b]
                t1 = g['ux'] * x1 + g['uy'] * y1
                t2 = g['ux'] * x2 + g['uy'] * y2
                a, b = min(t1, t2), max(t1, t2)
                g['intervals'].append((a, b))
                placed = True
                break

        if not placed:
            # start a new group
            t1 = ux * x1 + uy * y1
            t2 = ux * x2 + uy * y2
            a, b = min(t1, t2), max(t1, t2)
            groups.append({
                'ux': ux, 'uy': uy,
                'nx': nx, 'ny': ny,
                'offset': offset,
                'intervals': [(a, b)]
            })

    # Now merge intervals in each group and reconstruct 2D segments
    merged: List[Segment] = []
    for g in groups:
        ux, uy = g['ux'], g['uy']
        nx, ny = g['nx'], g['ny']
        offset = g['offset']
        # base point on the line (closest to origin)
        base_x, base_y = nx * offset, ny * offset

        intervals = sorted(g['intervals'], key=lambda iv: iv[0])
        cur_a, cur_b = intervals[0]
        for a, b in intervals[1:]:
            if a <= cur_b + eps:
                # overlapping or within gap tolerance
                cur_b = max(cur_b, b)
            else:
                # flush current
                p0 = (base_x + ux * cur_a, base_y + uy * cur_a)
                p1 = (base_x + ux * cur_b, base_y + uy * cur_b)
                merged.append((p0, p1))
                cur_a, cur_b = a, b

        # flush last
        p0 = (base_x + ux * cur_a, base_y + uy * cur_a)
        p1 = (base_x + ux * cur_b, base_y + uy * cur_b)
        merged.append((p0, p1))

    return merged




class CurveFragment:

    def __init__(self, curve:NURBSCurveTuple, t0:float, t1:float):
        """
        curve: any object with .trim(t0,t1) → new curve (same param domain),
               .evaluate(t), .derivative(t), .bbox(), .reversed()
        t0, t1: parameters where this fragment begins/ends on the original curve
        """
        # store the trimmed piece; param domain is unchanged

        self.curve = trim_curve(curve, t0, t1)
        self.t0 = t0
        self.t1 = t1
        self._build()
    def _build(self):
        _start=evaluate_nurbs_curve(self.curve,self.t0,1)
        self.start=_start['C']
        self.start_der=_start['C1']
        _end = evaluate_nurbs_curve(self.curve, self.t1, 1)
        self.end = _end["C"]
        self.end_der = _end["C1"]
        self._bbox=np.array(aabb(np.array(self.curve.control_points)))
    def evaluate(self, t):
        return evaluate_nurbs_curve(self.curve,t,0)['C']

    def derivative(self, t):
        return evaluate_nurbs_curve(self.curve,t,1)['C1']

    def bbox(self):
        return self.curve._bbox

    def reversed(self):
        # build a new fragment that traverses the same trimmed curve backwards
        rev = CurveFragment.__new__(CurveFragment)
        rev.curve = self.curve.reversed()

        rev.t0, rev.t1 = self.t1, self.t0
        rev._build()
        return rev


class HalfEdge:
    __slots__ = (
        'origin',    # (x,y) tuple
        'twin',      # opposite half-edge
        'next',      # next around the face
        'face',      # face id on its left
        'angle',     # for CCW sorting
        'ccw_prev',  # helper link at each vertex
        'idx',       # index in the half_edges list
        'frag_index',# which fragment this came from
        'forward'    # True if along increasing t
    )
    def __init__(self, origin, angle, frag_index, forward):
        self.origin = origin
        self.angle = angle
        self.frag_index = frag_index
        self.forward = forward
        self.twin = None
        self.next = None
        self.face = None
        self.ccw_prev = None
        self.idx = None


def polygonize_fragments(fragments):
    """
    Input:
      fragments: list of CurveFragment (already trimmed at t0,t1)
    Output:
      half_edges: list of HalfEdge objects (length = 2 * len(fragments))
      regions:    list of regions, each a list of half-edge indices giving
                  the single outer‐boundary loop in CCW order
    """
    half_edges = []
    # ─── 1) create half-edges with analytic angles ─────────────────────────
    for i, frag in enumerate(fragments):
        # endpoints
        p0 = frag.evaluate(frag.t0)
        p1 = frag.evaluate(frag.t1)
        # derivatives
        dx0, dy0 = frag.derivative(frag.t0)
        dx1, dy1 = frag.derivative(frag.t1)

        # forward half-edge (tail = p0, tangent = +d/dt)
        ang0 = math.atan2(dy0, dx0)
        h_fwd = HalfEdge(origin=p0, angle=ang0, frag_index=i, forward=True)

        # backward half-edge (tail = p1, tangent = –d/dt)
        ang1 = math.atan2(-dy1, -dx1)
        h_bwd = HalfEdge(origin=p1, angle=ang1, frag_index=i, forward=False)

        h_fwd.twin = h_bwd
        h_bwd.twin = h_fwd
        half_edges.extend([h_fwd, h_bwd])

    # assign each half-edge its index
    for idx, h in enumerate(half_edges):
        h.idx = idx

    # ─── 2) sort outgoing half-edges CCW at each unique origin ───────────────
    by_origin = defaultdict(list)
    for h in half_edges:
        by_origin[h.origin].append(h)
    for hs in by_origin.values():
        hs.sort(key=lambda h: h.angle)
        m = len(hs)
        for j in range(m):
            hs[(j + 1) % m].ccw_prev = hs[j]

    # ─── 3) set next pointers so walking h→h.next keeps the face on your left ─
    for h in half_edges:
        h.next = h.twin.ccw_prev

    # ─── 4) walk every half-edge to enumerate faces ───────────────────────────
    faces = []
    for h in half_edges:
        if h.face is None:
            fid = len(faces)
            curr = h
            cycle = []
            while True:
                curr.face = fid
                cycle.append(curr)
                curr = curr.next
                if curr is h:
                    break
            faces.append(cycle)

    # ─── 5) detect the unbounded face *topologically* ────────────────────────
    # pick the vertex with minimal (y,x)
    extreme = min(by_origin.keys(), key=lambda p: (p[1], p[0]))
    # among its outgoing half-edges, the one with smallest angle
    h_ext = min(by_origin[extreme], key=lambda h: h.angle)
    unbounded = h_ext.twin.face

    # ─── 6) build adjacency among the *bounded* faces ────────────────────────
    adj = defaultdict(set)
    for h in half_edges:
        f1, f2 = h.face, h.twin.face
        if f1 != f2 and f1 != unbounded and f2 != unbounded:
            adj[f1].add(f2)
            adj[f2].add(f1)

    # ─── 7) flood-fill to group bounded faces into connected components ────────
    comp_of = {}
    comp_id = 0
    for fid in range(len(faces)):
        if fid == unbounded or fid in comp_of:
            continue
        stack = [fid]
        while stack:
            f = stack.pop()
            if f in comp_of:
                continue
            comp_of[f] = comp_id
            for nbr in adj[f]:
                if nbr not in comp_of:
                    stack.append(nbr)
        comp_id += 1

    # ─── 8) for each component, extract the *outer* boundary loop ────────────
    regions = []
    for c in range(comp_id):
        # all faces in this component
        flist = [f for f, cid in comp_of.items() if cid == c]

        # boundary half-edges are those whose twin.face ∉ this component
        boundary = [
            h for h in half_edges
            if comp_of[h.face] == c and comp_of.get(h.twin.face, -1) != c
        ]

        # build a map tail‐point → one outgoing boundary half-edge
        origin_map = {h.origin: h for h in boundary}

        visited = set()
        loops = []
        for h in boundary:
            if h in visited:
                continue
            cur = h
            loop = []
            while True:
                visited.add(cur)
                loop.append(cur)
                nxt = origin_map.get(cur.twin.origin)
                if nxt is None or nxt is h:
                    break
                cur = nxt
            loops.append(loop)

        # pick the CCW loop of *maximum* area as the outer boundary
        best_loop = None
        best_area = -1
        for loop in loops:
            pts = [h.origin for h in loop]
            A = abs(sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1])
            ) * 0.5)
            if A > best_area:
                best_area = A
                best_loop = loop

        if best_loop:
            regions.append([h.idx for h in best_loop])

    return half_edges, regions


def build_graph(per_shape_ts, shapes, eval_pt=lambda shape,t: (shape[0]+t*(shape[1]-shape[0])), shape_interval_fun =lambda s: (0.,1.), eps=1e-8):
    """
    per_shape_ts: dict mapping each shape → list of (t, other_shape)
                  from your Option 1
    shapes:       dict mapping each shape → (t_start, t_end, shape.eval(t)->(x,y))

    Returns:
      V: list of unique (x,y) points
      E: list of (i, j) index‐pairs into V
    """

    # 1) For each shape, collect & sort its split‐parameters
    splits = {}
    for shape, hits in per_shape_ts.items():
        #t0, t1, shp = shapes[shape]    # your curve’s domain & eval fn
        #t0, t1=shape_interval_fun(shapes[shape] )
        ts =  [t for t,_ in hits]
        ts = sorted(set(ts))
        splits[shape] = ts

    # 2) Evaluate all split‐points, dedupe into V
    V = []               # list of (x,y)
    vid_of = {}          # quantized (x,y) → vid
    def get_vid(pt):
        # bucket by rounding to eps to merge duplicates
        key = (round(pt[0]/eps), round(pt[1]/eps))
        if key not in vid_of:
            vid_of[key] = len(V)
            V.append(pt)
        return vid_of[key]

    # store per‐shape, per‐t which vertex index we got
    vid_at = defaultdict(dict)
    for shape, ts in splits.items():

        for t in ts:
            pt = eval_pt(shapes[shape],t)                  # (x, y)
            vid_at[shape][t] = get_vid(pt)

    # 3) For each shape, connect consecutive splits
    E = []
    for shape, ts in splits.items():
        for t0, t1 in zip(ts, ts[1:]):
            u = vid_at[shape][t0]
            v = vid_at[shape][t1]
            if u != v:
                E.append((u, v))

    return V, E
def extract_region_fragments(sub_shapes, E, half_edges, regions, reverse_edge_fun=lambda e: [e[1], e[0]]):
    """
    Parameters
    ----------
    sub_shapes : list
        your curve‐fragment objects, in the same order as `edges`
    E : list of (u,v)
        the index‐pairs you passed to polygonize_edges
    half_edges : list of HalfEdge
        the 2×len(edges) half-edges returned by polygonize_edges
    regions : list of list of int
        each region is a list of indices into half_edges

    Returns
    -------
    List[List[curve_fragment]]
        for each region, the CCW‐oriented list of fragments from E
    """
    region_frags = []
    for region in regions:
        frags = []
        for he_idx in region:
            # which original fragment is this?
            frag_idx = he_idx // 2
            # half-edge tail
            tail_v = half_edges[he_idx].origin
            # the original edge endpoints
            u, v = E[frag_idx]
            # pick and orient
            if tail_v == u:
                # half-edge goes u→v, so keep fragment as-is
                frags.append(sub_shapes[frag_idx])
            else:
                # half-edge goes v→u, so reverse the fragment
                # assume E[frag_idx].reversed() exists;
                # otherwise for simple (u,v) do: frags.append((v,u))
                frags.append(reverse_edge_fun(sub_shapes[frag_idx]))
        region_frags.append(frags)
    return region_frags

from typing import Any,TypeVar
from numpy.typing import NDArray
ShapeType=TypeVar('ShapeType')
def _shape_ints(bvh:BVH,  shapes:list[ShapeType]|NDArray[ShapeType], int_fun:Callable[[ShapeType,ShapeType],list[tuple[float,float]]|tuple[float,float]|None], shape_interval_fun =lambda s: (0.,1.),):

    res=bvh.find_intersecting_leaves2(exact=True)

    ints = dict()
    visited = dict()

    for node_i, ixs in res.items():
        i = bvh.nodes[node_i].object
        segm1 = shapes[i]


        i_ints = []
        for node_j in ixs:
            j = bvh.nodes[node_j].object
            k = (i, j)
            if k not in visited:
                segm2 = shapes[j]

                result = int_fun(segm1, segm2)

                visited[(i, j)] = visited[(j, i)] = result
                if isinstance(result,tuple):

                    visited[(j, i)] = (result[1], result[0])
                elif isinstance(result,list):



                    visited[(j, i)] = [(s, t) for t, s in result]

            else:
                result = visited[(i, j)]
            if result is None:
                pass
            elif isinstance(result,tuple):
                i_ints.append((result[0],j))
            else:
                for t,_ in result:
                    i_ints.append((t, j))

        if len(i_ints) > 0:
            ints[i] = i_ints







    return ints,visited


def segment_boolean(segments, bvh=None, tol=1e-3):
    def _segm_int_fun(segm1, segm2):
        p1 = segm1[0]
        p2 = segm1[1]
        q1 = segm2[0]
        q2 = segm2[1]
        return segment_intersection(p1, p2, q1, q2)

    segments_arr= np.array(merge_overlapping_segments((np.array(segments)[...,:-1]).tolist(),eps=tol))

    if bvh is None:
        bvh = build_bvh([AABB.from_points(segments_arr[i]) for i in range(segments_arr.shape[0])])
    ints,visited = _shape_ints(bvh, segments_arr, int_fun= _segm_int_fun)
    V,E=build_graph(ints, segments_arr)
    hs,regions=polygonize_segments(V, E)
    sub_shapes=(np.array(V)[np.array(E,int)]).tolist()
    return extract_region_fragments(sub_shapes,E,half_edges=hs,regions=regions,reverse_edge_fun=lambda e: [e[1], e[0]])
