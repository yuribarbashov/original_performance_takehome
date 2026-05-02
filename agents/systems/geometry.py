"""Pure geometric helpers for 2D Cartesian coordinates.

All coordinates are interpreted as ``(x, y)`` in a flat Euclidean plane.
"""

from __future__ import annotations

from math import hypot
from typing import Iterable, Sequence, Tuple

Point = Tuple[float, float]
Circle = Tuple[Point, float]


def distance(a: Point, b: Point) -> float:
    """Return Euclidean distance between two ``(x, y)`` points."""
    return hypot(a[0] - b[0], a[1] - b[1])


def _orientation(a: Point, b: Point, c: Point) -> float:
    """Return signed orientation/cross-product for triangle ``a->b->c``."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    """Return True when point ``p`` lies on closed segment ``ab``."""
    return (
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """Return True when two closed line segments intersect or overlap."""
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    if o1 == 0 and _on_segment(a1, a2, b1):
        return True
    if o2 == 0 and _on_segment(a1, a2, b2):
        return True
    if o3 == 0 and _on_segment(b1, b2, a1):
        return True
    if o4 == 0 and _on_segment(b1, b2, a2):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def segment_intersects_circle(start: Point, end: Point, center: Point, radius: float) -> bool:
    """Return True when closed segment ``start->end`` intersects/touches a circle."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    if dx == 0 and dy == 0:
        return distance(start, center) <= radius

    t = ((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    closest = (start[0] + t * dx, start[1] + t * dy)
    return distance(closest, center) <= radius


def path_hits_planet_or_sun(path: Sequence[Point], hazards: Iterable[Circle]) -> bool:
    """Return True if any path segment intersects/touches a hazard circle."""
    pts = list(path)
    if len(pts) < 2:
        return False
    circles = list(hazards)
    for i in range(len(pts) - 1):
        for center, radius in circles:
            if segment_intersects_circle(pts[i], pts[i + 1], center, radius):
                return True
    return False
