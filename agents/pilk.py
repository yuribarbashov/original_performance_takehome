"""Pilk agent helpers.

Geometry and routing helpers are delegated to :mod:`agents.systems` modules.
"""

from agents.systems.geometry import (
    distance,
    path_hits_planet_or_sun,
    segment_intersects_circle,
    segments_intersect,
)
from agents.systems.routing import (
    RoutingDecision,
    choose_heading_aim_vector,
    estimate_eta,
    predict_moving_target_intercept,
)

from agents.systems.world import WorldState

__all__ = [
    "distance",
    "segments_intersect",
    "segment_intersects_circle",
    "path_hits_planet_or_sun",
    "RoutingDecision",
    "choose_heading_aim_vector",
    "estimate_eta",
    "predict_moving_target_intercept",
    "WorldState",
]
