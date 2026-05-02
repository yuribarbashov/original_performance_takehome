import unittest

from agents.systems.routing import (
    choose_heading_aim_vector,
    estimate_eta,
    predict_moving_target_intercept,
)


class RoutingTests(unittest.TestCase):
    def test_stationary_target_intercept(self):
        decision = predict_moving_target_intercept((0, 0), (10, 0), (0, 0), 5)
        self.assertEqual(decision.intercept_point, (10.0, 0.0))
        self.assertAlmostEqual(decision.eta, 2.0)
        self.assertEqual(decision.confidence, 1.0)

    def test_same_position_source_target(self):
        decision = choose_heading_aim_vector((5, 5), (5, 5), 3)
        self.assertEqual(decision.heading, (0.0, 0.0))
        self.assertEqual(decision.speed, 0.0)
        self.assertEqual(decision.eta, 0.0)

    def test_impossible_intercept(self):
        decision = predict_moving_target_intercept((0, 0), (10, 0), (5, 0), 1)
        self.assertIsNone(decision.intercept_point)
        self.assertIsNone(decision.eta)
        self.assertEqual(decision.confidence, 0.0)

    def test_max_speed_constraints(self):
        decision = choose_heading_aim_vector((0, 0), (10, 0), 3)
        self.assertEqual(decision.speed, 3)
        self.assertAlmostEqual(decision.eta, 10 / 3)

    def test_blocked_intercept_confidence(self):
        decision = predict_moving_target_intercept(
            (0, 0),
            (10, 0),
            (0, 0),
            5,
            hazards=[((5, 0), 1.0)],
        )
        self.assertEqual(decision.confidence, 0.0)

    def test_zero_speed_eta(self):
        self.assertIsNone(estimate_eta((0, 0), (10, 10), 0))


if __name__ == "__main__":
    unittest.main()
