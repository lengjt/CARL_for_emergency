"""Planner-specific physics, information boundary, timing fallback and closed-loop regression."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from extreme_avoidance.scenario import Scenario,Road,Obstacle,ROOT
from extreme_avoidance.backend import OCDBackend,PREFIX
from extreme_avoidance.environment import AvoidanceEnv
from extreme_avoidance.planner import RollingPlanner,predictor_parameters
from extreme_avoidance.experiment import run


class PlannerTests(unittest.TestCase):
    def test_internal_scaling_matches_fixed_tire_parameters_without_compounding(self):
        sc=Scenario(speed=15,base_scale=.6,friction_mode="internal_mu",obstacles=())
        original=OCDBackend(sc)
        config=json.loads((ROOT/"configs/vehicle.json").read_text())
        for k in config["ocd_parameters"]:
            if k.endswith((".LMUX",".LMUY")):config["ocd_parameters"][k]*=.6
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"vehicle.json";path.write_text(json.dumps(config))
            fixed=OCDBackend(replace(sc,base_scale=1.,friction_mode="post_force"),path)
            for i in range(400):
                if i%20==0:
                    original.request(.04,-1200);fixed.request(.04,-1200)
                original.step();fixed.step()
            for key in original.state:
                np.testing.assert_allclose(original.state[key],fixed.state[key],rtol=1e-12,atol=1e-12)

    def test_internal_reset_restores_reproducible_tire_parameters(self):
        b=OCDBackend(Scenario(base_scale=.5,friction_mode="internal_mu",obstacles=()))
        def segment():
            b.request(.02,-2000)
            for _ in range(200):b.step()
            return dict(b.state)
        a=segment();b.reset();self.assertEqual(a,segment())

    def test_no_access_to_unrevealed_obstacles(self):
        outputs=[]
        for ob in (Obstacle(s=60,reveal_distance=20),Obstacle(s=100,reveal_distance=10)):
            env=AvoidanceEnv(Scenario(obstacles=(ob,),friction_mode="internal_mu"))
            planner=RollingPlanner(env.scenario.road,env.backend,20,.02)
            outputs.append(planner(env.observe()))
            self.assertEqual(planner.journal,[])
            self.assertIsNone(planner.obstacle)
        self.assertEqual(outputs[0],outputs[1])

    def test_search_repeats_from_new_state_and_preserves_both_sides(self):
        sc=Scenario(obstacles=(Obstacle(s=40,reveal_distance=50),),friction_mode="internal_mu")
        env=AvoidanceEnv(sc);planner=RollingPlanner(sc.road,env.backend,20,.02)
        for _ in range(7):env.step(planner(env.observe()))
        self.assertEqual(len(planner.journal),2)
        for e in planner.journal:
            sides={np.sign(z["parameters"][0]) for z in e["candidates"]}
            self.assertTrue({-1,1}.issubset(sides))
        self.assertGreater(planner.journal[1]["t"],planner.journal[0]["t"])

    def test_no_candidate_and_late_results_use_aeb(self):
        env=AvoidanceEnv(Scenario(obstacles=(Obstacle(s=40,reveal_distance=50),),friction_mode="internal_mu"))
        planner=RollingPlanner(env.scenario.road,env.backend,20,.02)
        def infeasible(*args):
            a=np.zeros((len(args[1]),9));a[:,1]=1e9;return a
        with patch("extreme_avoidance.planner.screen_maneuvers",side_effect=infeasible):
            action=planner(env.observe())
        self.assertEqual(action[1],-10000)
        self.assertTrue(planner.journal[-1]["fallback"])
        planner=RollingPlanner(env.scenario.road,env.backend,20,.02)
        with patch("extreme_avoidance.planner.time.perf_counter",side_effect=[0.,1.]):
            action=planner(env.observe())
        self.assertEqual(action[1],-10000)
        self.assertTrue(planner.journal[-1]["deadline_missed"])

    def test_critical_avoidance_and_recovery(self):
        sc=replace(Scenario.load("straight_critical"),friction_mode="internal_mu")
        _,m,_=run(sc,"rolling")
        self.assertTrue(m["avoidance_success"])
        self.assertTrue(m["stable_recovery"])
        self.assertGreater(m["planner_calls"],10)

    def test_curve_is_explicitly_unsupported(self):
        env=AvoidanceEnv(Scenario(road=Road(curvature=.005),obstacles=()))
        with self.assertRaises(ValueError):RollingPlanner(env.scenario.road,env.backend,20,.02)

    def test_integrated_steering_rate_and_success_constraints(self):
        sc=Scenario(speed=15,base_scale=.5,friction_mode="internal_mu",obstacles=(Obstacle(s=60,reveal_distance=30),))
        _,m,_=run(sc,"joint")
        self.assertLessEqual(m["max_abs_steering_rate_radps"],.501)
        self.assertFalse(m["actuator_limit_violation"])
        env=AvoidanceEnv(Scenario(obstacles=()))
        env.status="recovered";env.max_steer_rate=1.
        self.assertTrue(env.metrics()["stable_recovery"])
        self.assertFalse(env.metrics()["success"])

if __name__=="__main__":unittest.main()
