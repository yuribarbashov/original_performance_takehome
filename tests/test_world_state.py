import unittest

from agents.systems.world import WorldState


class WorldStateTests(unittest.TestCase):
    def test_planet_lookup_and_owner(self):
        snapshot = {
            "planets": [
                {"planet_id": "p1", "owner_id": "A"},
                {"id": "p2", "owner": "B"},
            ]
        }
        world = WorldState(snapshot)
        self.assertEqual(world.planet_by_id("p1").owner, "A")
        self.assertEqual(world.current_owner("p2"), "B")

    def test_fleet_filters_and_contested(self):
        snapshot = {
            "planets": [{"id": 1, "owner": "A"}],
            "fleets": [
                {"id": "f1", "owner": "A", "destination": 1},
                {"id": "f2", "owner": "B", "destination_id": 1},
                {"id": "f3", "owner": "B", "destination": 2},
            ],
        }
        world = WorldState(snapshot)
        self.assertEqual(len(list(world.fleets_by_owner("B"))), 2)
        self.assertEqual(len(list(world.fleets_targeting_planet(1))), 2)
        self.assertTrue(world.is_contested(1))

    def test_timeline_resolution(self):
        snapshot = {
            "planets": [{"id": "p1", "owner": "C"}],
            "ownership_history": {
                "p1": [
                    {"turn": 1, "owner": "A"},
                    {"turn": 5, "owner": "B"},
                    {"turn": 9, "owner": "C"},
                ]
            },
        }
        world = WorldState(snapshot)
        self.assertEqual(world.owner_at_turn("p1", 0), "C")
        self.assertEqual(world.owner_at_turn("p1", 5), "B")
        self.assertEqual(world.owner_at_turn("p1", 99), "C")
        self.assertEqual(world.ownership_timeline("p1")[-1]["owner"], "C")


if __name__ == "__main__":
    unittest.main()
