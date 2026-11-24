"""Regression tests for safety/event semantics and deterministic environment reuse."""
from dataclasses import replace
import unittest
import numpy as np
from extreme_avoidance.scenario import Scenario, Road, Obstacle, rectangle, separation, road_margin
from extreme_avoidance.environment import AvoidanceEnv
from extreme_avoidance.experiment import run
from extreme_avoidance.prediction import rollout


class GeometryTests(unittest.TestCase):
    def test_touching_and_rotated_rectangles(self):
        a=rectangle(0,0,0,4,2)
        self.assertLessEqual(separation(a,rectangle(4,0,0,4,2)),0)
        self.assertGreater(separation(a,rectangle(4.1,0,0,4,2)),0)
        self.assertLess(separation(a,rectangle(1,0,np.pi/4,4,2)),0)

    def test_road_projection_and_boundaries(self):
        for k in (-.02,0,.02):
            road=Road(curvature=k)
            x,y,yaw=road.pose(30,2)
            np.testing.assert_allclose(road.project(x,y),(30,2,yaw),atol=1e-10)
        self.assertAlmostEqual(road_margin(Road(width=10),rectangle(0,4,0,4,2)),0)

    def test_swept_body_catches_tunnelling(self):
        env=AvoidanceEnv(Scenario(obstacles=(Obstacle(s=10,reveal_distance=1),)))
        before=dict(env.backend.state,x_m=5.)
        after=dict(env.backend.state,x_m=15.)
        env._check_geometry(before,after)
        self.assertTrue(env.collision)


class EnvironmentTests(unittest.TestCase):
    def test_force_scale_observation_has_explicit_name_and_legacy_alias(self):
        env = AvoidanceEnv(Scenario.load("straight_critical"))
        obs = env.reset()
        self.assertEqual(obs["force_scale_estimate"], obs["scale_estimate"])
        self.assertEqual(obs["force_scale_estimate"], [0.6] * 4)

    def test_reset_clears_actuators_and_reproduces_trajectory(self):
        env=AvoidanceEnv(Scenario(obstacles=(),duration=.4))
        for _ in range(10): env.step((.03,-3000))
        first=[dict(row) for row in env.rows]
        env.reset()
        self.assertEqual(env.rows[0]["force_executed_N"],0)
        for _ in range(10): env.step((.03,-3000))
        self.assertEqual(first,env.rows)

    def test_hidden_obstacle_is_physical_and_terminal(self):
        env=AvoidanceEnv(Scenario(obstacles=(Obstacle(s=10,reveal_distance=1),),duration=1))
        self.assertEqual(env.observe()["obstacles"],[])
        while env.status=="running": env.step((0,0))
        self.assertEqual(env.status,"collision")
        self.assertEqual(env.observe()["obstacles"],[])
        with self.assertRaises(RuntimeError): env.step((0,0))

    def test_hazard_clearance_is_not_recovery(self):
        _,m,_=run(Scenario.load("straight_low_grip"),"joint_legacy")
        self.assertTrue(m["avoidance_success"])
        self.assertTrue(m["recovery_failed"])
        self.assertFalse(m["success"])
        self.assertEqual(m["status"],"offroad")

    def test_safe_stop_and_time_limit(self):
        _,m,_=run(Scenario.load("straight_popup"),"aeb")
        self.assertTrue(m["avoidance_success"])
        self.assertTrue(m["stable_recovery"])
        self.assertEqual(m["status"],"stopped_safe")
        env=AvoidanceEnv(Scenario(obstacles=(),duration=.025))
        env.step((0,0));env.step((0,0))
        self.assertEqual(env.status,"timeout")
        self.assertAlmostEqual(env.backend.time,.025)
        self.assertIsNone(env.metrics()["avoidance_success"])

    def test_invalid_inputs(self):
        for dt in (0,float("nan"),.003,.0007):
            with self.assertRaises(ValueError): Scenario(dt=dt)
        env=AvoidanceEnv(Scenario())
        with self.assertRaises(ValueError): env.step((float("nan"),0))


class PredictorTests(unittest.TestCase):
    def test_batch_consistency(self):
        initial=[0,0,0,15,0,0,0,0]
        actions=np.zeros((2,20,2));actions[1,:,0]=.03
        batch=rollout(initial,actions)
        self.assertEqual(batch.shape,(2,21,8))
        for i in range(2):
            np.testing.assert_allclose(batch[i],rollout(initial,actions[i:i+1])[0])
        self.assertGreater(batch[1,-1,1],0)

if __name__=="__main__":
    unittest.main()
