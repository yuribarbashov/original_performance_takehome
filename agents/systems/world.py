"""Canonical world-state access layer.

The :class:`WorldState` wrapper normalizes common schema variants and provides a
stable interface for agents and routing systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class PlanetView:
    """Normalized planet representation."""

    id: Any
    raw: dict[str, Any]

    @property
    def owner(self) -> Any:
        for key in ("owner", "owner_id", "player", "controller"):
            if key in self.raw:
                return self.raw[key]
        return None


@dataclass(frozen=True)
class FleetView:
    """Normalized fleet representation."""

    id: Any
    raw: dict[str, Any]

    @property
    def owner(self) -> Any:
        for key in ("owner", "owner_id", "player", "controller"):
            if key in self.raw:
                return self.raw[key]
        return None

    @property
    def destination(self) -> Any:
        for key in ("destination", "to", "target", "target_planet_id", "destination_id"):
            if key in self.raw:
                return self.raw[key]
        return None


class WorldState:
    """Normalized accessor around parsed turn state."""

    def __init__(self, turn_state: dict[str, Any] | None):
        self._raw = turn_state or {}
        self._planets_by_id: dict[Any, dict[str, Any]] = {}
        self._fleets: list[dict[str, Any]] = []
        self._ownership_history: dict[Any, list[dict[str, Any]]] = {}
        self._parse_and_normalize()

    @property
    def raw(self) -> dict[str, Any]:
        return self._raw

    def _parse_and_normalize(self) -> None:
        planets_container = self._first_present(
            self._raw,
            "planets",
            "planet_states",
            "planetState",
            "world",
        )
        planets = self._coerce_collection(planets_container)
        for idx, planet in enumerate(planets):
            if not isinstance(planet, dict):
                continue
            planet_id = self._planet_id(planet, fallback=idx)
            self._planets_by_id[planet_id] = planet

        fleets_container = self._first_present(
            self._raw,
            "fleets",
            "fleet_states",
            "ships",
            "moving_fleets",
        )
        self._fleets = [f for f in self._coerce_collection(fleets_container) if isinstance(f, dict)]

        history_container = self._first_present(
            self._raw,
            "ownership_history",
            "planet_ownership_history",
            "history",
        )
        if isinstance(history_container, dict):
            for pid, entries in history_container.items():
                self._ownership_history[pid] = self._normalize_timeline(entries)

        for pid, planet in self._planets_by_id.items():
            p_hist = self._first_present(planet, "ownership_history", "history", "owner_history")
            if p_hist is not None:
                self._ownership_history[pid] = self._normalize_timeline(p_hist)

    @staticmethod
    def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    @staticmethod
    def _coerce_collection(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            if "planets" in value:
                return WorldState._coerce_collection(value["planets"])
            if "items" in value and isinstance(value["items"], list):
                return value["items"]
            return list(value.values())
        return []

    @staticmethod
    def _planet_id(planet: dict[str, Any], fallback: Any) -> Any:
        for key in ("id", "planet_id", "planetId"):
            if key in planet:
                return planet[key]
        return fallback

    @staticmethod
    def _normalize_timeline(entries: Any) -> list[dict[str, Any]]:
        rows = WorldState._coerce_collection(entries)
        norm: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            if isinstance(row, dict):
                turn = row.get("turn", row.get("t", i))
                owner = row.get("owner", row.get("owner_id", row.get("player", row.get("controller"))))
            else:
                turn = i
                owner = row
            norm.append({"turn": turn, "owner": owner})
        norm.sort(key=lambda r: (r["turn"] is None, r["turn"]))
        return norm

    def planet_by_id(self, id: Any) -> PlanetView | None:
        planet = self._planets_by_id.get(id)
        return PlanetView(id=id, raw=planet) if planet is not None else None

    def fleets(self) -> Iterator[FleetView]:
        for idx, fleet in enumerate(self._fleets):
            fleet_id = fleet.get("id", fleet.get("fleet_id", fleet.get("fleetId", idx)))
            yield FleetView(id=fleet_id, raw=fleet)

    def fleets_by_owner(self, owner: Any) -> Iterator[FleetView]:
        for fleet in self.fleets():
            if fleet.owner == owner:
                yield fleet

    def fleets_targeting_planet(self, planet_id: Any) -> Iterator[FleetView]:
        for fleet in self.fleets():
            if fleet.destination == planet_id:
                yield fleet

    def current_owner(self, planet_id: Any) -> Any:
        planet = self.planet_by_id(planet_id)
        if planet is not None:
            owner = planet.owner
            if owner is not None:
                return owner
        hist = self.ownership_timeline(planet_id)
        return hist[-1]["owner"] if hist else None

    def is_contested(self, planet_id: Any) -> bool:
        incumbent = self.current_owner(planet_id)
        contenders = {fleet.owner for fleet in self.fleets_targeting_planet(planet_id)}
        contenders.discard(None)
        if incumbent is None:
            return len(contenders) > 1
        return any(owner != incumbent for owner in contenders)

    def ownership_timeline(self, planet_id: Any) -> list[dict[str, Any]]:
        return list(self._ownership_history.get(planet_id, ()))

    def owner_at_turn(self, planet_id: Any, turn: int) -> Any:
        timeline = self.ownership_timeline(planet_id)
        owner = None
        for event in timeline:
            event_turn = event.get("turn")
            if event_turn is None or event_turn <= turn:
                owner = event.get("owner")
            else:
                break
        if owner is None:
            return self.current_owner(planet_id)
        return owner


__all__ = ["WorldState", "PlanetView", "FleetView"]
