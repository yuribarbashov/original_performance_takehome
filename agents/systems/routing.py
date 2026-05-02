"""Movement-targeting and interception utilities.

These helpers are intentionally stateless: all needed positions and velocities are
passed as parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Optional, Sequence

from agents.systems.geometry import distance, path_hits_planet_or_sun

Point = tuple[float, float]
Vector = tuple[float, float]


@dataclass(frozen=True)
class RoutingDecision:
    heading: Vector
    speed: float
    eta: Optional[float]
    intercept_point: Optional[Point]
    confidence: float


def choose_heading_aim_vector(
    source: Point,
    target: Point,
    max_speed: float,
) -> RoutingDecision:
    """Choose normalized heading from ``source`` to ``target`` with bounded speed."""
    if max_speed <= 0:
        return RoutingDecision((0.0, 0.0), 0.0, None, None, 0.0)

    dx = target[0] - source[0]
    dy = target[1] - source[1]
    dist = distance(source, target)

    if dist == 0:
        return RoutingDecision((0.0, 0.0), 0.0, 0.0, target, 1.0)

    heading = (dx / dist, dy / dist)
    speed = min(max_speed, dist)
    eta = estimate_eta(source, target, speed)
    return RoutingDecision(heading, speed, eta, target, 1.0)


def estimate_eta(source: Point, target: Point, speed: float) -> Optional[float]:
    """Estimate arrival time from ``source`` to ``target`` at constant ``speed``."""
    dist = distance(source, target)
    if dist == 0:
        return 0.0
    if speed <= 0:
        return None
    return dist / speed


def predict_moving_target_intercept(
    source: Point,
    target_position: Point,
    target_velocity: Vector,
    projectile_speed: float,
    hazards: Sequence[tuple[Point, float]] = (),
) -> RoutingDecision:
    """Predict interception point for a moving target.

    Returns low-confidence decision when no physical interception is possible.
    """
    if projectile_speed <= 0:
        return RoutingDecision((0.0, 0.0), 0.0, None, None, 0.0)

    rx = target_position[0] - source[0]
    ry = target_position[1] - source[1]
    tvx, tvy = target_velocity

    # Solve |r + v t| = s t -> (v·v - s^2)t^2 + 2(r·v)t + r·r = 0
    a = tvx * tvx + tvy * tvy - projectile_speed * projectile_speed
    b = 2.0 * (rx * tvx + ry * tvy)
    c = rx * rx + ry * ry

    times: list[float] = []
    eps = 1e-12
    if abs(a) < eps:
        if abs(b) < eps:
            t = 0.0 if c == 0 else None
            if t is not None:
                times.append(t)
        else:
            t = -c / b
            if t >= 0:
                times.append(t)
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            root = disc**0.5
            t1 = (-b - root) / (2 * a)
            t2 = (-b + root) / (2 * a)
            if t1 >= 0:
                times.append(t1)
            if t2 >= 0:
                times.append(t2)

    if not times:
        return RoutingDecision((0.0, 0.0), projectile_speed, None, None, 0.0)

    eta = min(times)
    intercept = (target_position[0] + tvx * eta, target_position[1] + tvy * eta)
    dist = distance(source, intercept)
    if dist == 0:
        heading = (0.0, 0.0)
    else:
        heading = ((intercept[0] - source[0]) / dist, (intercept[1] - source[1]) / dist)

    blocked = path_hits_planet_or_sun([source, intercept], hazards)
    confidence = 0.0 if blocked else 1.0
    return RoutingDecision(heading, projectile_speed, eta, intercept, confidence)
