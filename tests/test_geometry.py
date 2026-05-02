import unittest

from agents.systems.geometry import (
    distance,
    path_hits_planet_or_sun,
    segment_intersects_circle,
    segments_intersect,
)


class GeometryTests(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(distance((0, 0), (3, 4)), 5)

    def test_segments_intersect_cross(self):
        self.assertTrue(segments_intersect((0, 0), (2, 2), (0, 2), (2, 0)))

    def test_segments_intersect_overlap(self):
        self.assertTrue(segments_intersect((0, 0), (3, 0), (1, 0), (4, 0)))

    def test_segment_intersects_circle(self):
        self.assertTrue(segment_intersects_circle((0, 0), (2, 0), (1, 0), 0.1))
        self.assertFalse(segment_intersects_circle((0, 1), (2, 1), (1, 0), 0.1))

    def test_path_hits_planet_or_sun(self):
        path = [(0, 0), (2, 0), (4, 0)]
        hazards = [((1, 0), 0.2), ((10, 10), 1.0)]
        self.assertTrue(path_hits_planet_or_sun(path, hazards))
        self.assertFalse(path_hits_planet_or_sun(path, [((10, 10), 1.0)]))


if __name__ == "__main__":
    unittest.main()
