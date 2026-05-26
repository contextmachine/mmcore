"""
Rectangle subtraction via strip decomposition.

Given a bounding rectangle and a set of rectangles to subtract,
returns the remaining area as a minimal set of non-overlapping rectangles.

Uses a sweep-line approach: collect all unique edge coordinates along one axis,
creating strips. Within each strip, merge the covered intervals and return the gaps.
Automatically picks the axis (horizontal vs vertical strips) that yields fewer output rectangles.
"""

from __future__ import annotations
from typing import List, Tuple,NamedTuple

__all__ = ['subtract_intervals_2d']

class Rect(NamedTuple):
    """Axis-aligned rectangle defined by [x0, x1) x [y0, y1)."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)

    def is_valid(self) -> bool:
        return self.x0 < self.x1 and self.y0 < self.y1

    def intersects(self, other: Rect) -> bool:
        return (self.x0 < other.x1 and self.x1 > other.x0 and
                self.y0 < other.y1 and self.y1 > other.y0)

    def clamp(self, bounds: Rect) -> Rect | None:
        """Clamp this rect to bounds. Returns None if no overlap."""
        r = Rect(
            max(self.x0, bounds.x0), max(self.y0, bounds.y0),
            min(self.x1, bounds.x1), min(self.y1, bounds.y1),
        )
        return r if r.is_valid() else None
    def as_bounds(self):
        return (self.x0,self.y0), (self.x1,self.y1)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge a list of (lo, hi) intervals into non-overlapping sorted intervals."""
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _gap_intervals(
        merged: List[Tuple[float, float]], lo: float, hi: float
) -> List[Tuple[float, float]]:
    """Return the uncovered portions of [lo, hi] given merged covered intervals."""
    gaps = []
    cursor = lo
    for a, b in merged:
        if a > cursor:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < hi:
        gaps.append((cursor, hi))
    return gaps


# ---------------------------------------------------------------------------
# Strip-based subtraction along a single axis
# ---------------------------------------------------------------------------

def _subtract_strips_y(
        bounds: Rect, holes: List[Rect]
) -> List[Rect]:
    """
    Sweep along Y (horizontal strips).
    Collect unique Y coords → strips; per strip find uncovered X intervals.
    """
    # Clamp holes to bounds and discard empties
    clamped: List[Rect] = []
    for h in holes:
        c = h.clamp(bounds)
        if c is not None:
            clamped.append(c)

    # Unique Y coordinates (strip boundaries)
    ys: set[float] = {bounds.y0, bounds.y1}
    for h in clamped:
        ys.add(h.y0)
        ys.add(h.y1)
    ys_sorted = sorted(ys)

    result: List[Rect] = []
    for i in range(len(ys_sorted) - 1):
        strip_y0 = ys_sorted[i]
        strip_y1 = ys_sorted[i + 1]
        if strip_y0 < bounds.y0 or strip_y1 > bounds.y1:
            continue
        if strip_y0 >= strip_y1:
            continue

        # Collect X intervals covered by holes in this strip
        covered: List[Tuple[float, float]] = []
        for h in clamped:
            if h.y0 <= strip_y0 and h.y1 >= strip_y1:
                covered.append((h.x0, h.x1))

        merged = _merge_intervals(covered)
        gaps = _gap_intervals(merged, bounds.x0, bounds.x1)

        for gx0, gx1 in gaps:
            result.append(Rect(gx0, strip_y0, gx1, strip_y1))

    return result


def _subtract_strips_x(
        bounds: Rect, holes: List[Rect]
) -> List[Rect]:
    """
    Sweep along X (vertical strips).
    Same logic, transposed axis.
    """
    clamped: List[Rect] = []
    for h in holes:
        c = h.clamp(bounds)
        if c is not None:
            clamped.append(c)

    xs: set[float] = {bounds.x0, bounds.x1}
    for h in clamped:
        xs.add(h.x0)
        xs.add(h.x1)
    xs_sorted = sorted(xs)

    result: List[Rect] = []
    for i in range(len(xs_sorted) - 1):
        strip_x0 = xs_sorted[i]
        strip_x1 = xs_sorted[i + 1]
        if strip_x0 < bounds.x0 or strip_x1 > bounds.x1:
            continue
        if strip_x0 >= strip_x1:
            continue

        covered: List[Tuple[float, float]] = []
        for h in clamped:
            if h.x0 <= strip_x0 and h.x1 >= strip_x1:
                covered.append((h.y0, h.y1))

        merged = _merge_intervals(covered)
        gaps = _gap_intervals(merged, bounds.y0, bounds.y1)

        for gy0, gy1 in gaps:
            result.append(Rect(strip_x0, gy0, strip_x1, gy1))

    return result


# ---------------------------------------------------------------------------
# Merge adjacent rectangles to reduce count
# ---------------------------------------------------------------------------

def _merge_adjacent(rects: List[Rect]) -> List[Rect]:
    """
    Merge rectangles that share an edge and form a larger rectangle.
    Greedily merge pairs that share the same X-span with touching Y,
    or same Y-span with touching X.
    """
    if not rects:
        return rects

    changed = True
    pool = list(rects)
    while changed:
        changed = False
        merged_set: set[int] = set()
        new_pool: List[Rect] = []
        # Index by x-span for horizontal merging
        for i in range(len(pool)):
            if i in merged_set:
                continue
            r = pool[i]
            for j in range(i + 1, len(pool)):
                if j in merged_set:
                    continue
                s = pool[j]
                # Same X span, adjacent Y
                if (r.x0 == s.x0 and r.x1 == s.x1):
                    if abs(r.y1 - s.y0) < 1e-12:
                        r = Rect(r.x0, r.y0, r.x1, s.y1)
                        merged_set.add(j)
                        changed = True
                    elif abs(s.y1 - r.y0) < 1e-12:
                        r = Rect(r.x0, s.y0, r.x1, r.y1)
                        merged_set.add(j)
                        changed = True
                # Same Y span, adjacent X
                elif (r.y0 == s.y0 and r.y1 == s.y1):
                    if abs(r.x1 - s.x0) < 1e-12:
                        r = Rect(r.x0, r.y0, s.x1, r.y1)
                        merged_set.add(j)
                        changed = True
                    elif abs(s.x1 - r.x0) < 1e-12:
                        r = Rect(s.x0, r.y0, r.x1, r.y1)
                        merged_set.add(j)
                        changed = True
            new_pool.append(r)
        pool = new_pool

    return pool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def subtract_intervals_2d(intervs, bounds=((0.,0.),(1.,1.)),merge:bool=True):
    return [item.as_bounds() for item in subtract_rects(Rect(*bounds[0],*bounds[1]),  [ Rect(*i[0],*i[1]) for i in intervs],merge=merge,direction='auto')]

def subtract_rects(
        bounds: Rect,
        holes: List[Rect],
        *,
        direction: str = "auto",
        merge: bool = True,
) -> List[Rect]:
    """
    Subtract *holes* from *bounds* and return the remaining area
    as a list of non-overlapping rectangles.

    Parameters
    ----------
    bounds : Rect
        The outer bounding rectangle.
    holes : list[Rect]
        Rectangles to subtract. May overlap each other and may extend
        beyond *bounds* (they are clamped automatically).
    direction : str
        "y" for horizontal strips, "x" for vertical strips,
        "auto" to pick whichever produces fewer output rectangles.
    merge : bool
        If True, greedily merge adjacent result rectangles that form
        a larger rectangle (reduces count further).

    Returns
    -------
    list[Rect]
        Non-overlapping rectangles covering exactly the remaining area.
    """
    if direction == "y":
        result = _subtract_strips_y(bounds, holes)
    elif direction == "x":
        result = _subtract_strips_x(bounds, holes)
    else:
        ry = _subtract_strips_y(bounds, holes)
        rx = _subtract_strips_x(bounds, holes)
        result = ry if len(ry) <= len(rx) else rx

    if merge:
        result = _merge_adjacent(result)

    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def debug_visualize(
        bounds: Rect,
        holes: List[Rect],
        result: List[Rect],
        title: str = "",
        save_path: str | None = None,
):
    """Render bounds, holes, and result rectangles with matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="black")
    if title:
        fig.suptitle(title, color="white", fontsize=14)

    for ax in axes:
        ax.set_facecolor("black")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlim(bounds.x0 - 5, bounds.x1 + 5)
        ax.set_ylim(bounds.y1 + 5, bounds.y0 - 5)
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_color("gray")

    # Left: bounds (white) + holes (blue)
    ax = axes[0]
    ax.set_title("Input: bounds + holes", color="white")
    ax.add_patch(patches.Rectangle(
        (bounds.x0, bounds.y0), bounds.width, bounds.height,
        linewidth=1, edgecolor="white", facecolor="none",
    ))
    for h in holes:
        ax.add_patch(patches.Rectangle(
            (h.x0, h.y0), h.width, h.height,
            linewidth=1, edgecolor="dodgerblue", facecolor="none",
        ))

    # Right: result (red)
    ax = axes[1]
    ax.set_title(f"Result: {len(result)} rectangles", color="white")
    ax.add_patch(patches.Rectangle(
        (bounds.x0, bounds.y0), bounds.width, bounds.height,
        linewidth=1, edgecolor="white", facecolor="none",
    ))
    for r in result:
        ax.add_patch(patches.Rectangle(
            (r.x0, r.y0), r.width, r.height,
            linewidth=1, edgecolor="red", facecolor="none",
        ))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor="black")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Demo matching the uploaded images
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Bounding box
    bounds = Rect(0, 0, 100, 100)

    # Blue rectangles from Image 1 (approximate)
    holes = [
        Rect( 49.5, 0,50.5,100 ),     # top-right

    ]

    result = subtract_rects(bounds, holes, direction="auto")

    print(f"Input:  {len(holes)} holes")
    print(f"Output: {len(result)} result rectangles")
    print()
    for i, r in enumerate(result):
        print(f"  R{i}: x=[{r.x0:.1f}, {r.x1:.1f}]  y=[{r.y0:.1f}, {r.y1:.1f}]  "
              f"area={r.area:.1f}")

    # Total area check
    total_result_area = sum(r.area for r in result)
    print(f"\nBounds area:       {bounds.area:.1f}")
    print(f"Result total area: {total_result_area:.1f}")

    debug_visualize(bounds, holes, result, title="Rectangle Subtraction", save_path="demo.png")
