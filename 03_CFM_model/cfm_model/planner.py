"""CFM and nearest-demonstration candidate sources using the shared screening backend."""

import time
import numpy as np

from .numpy_model import NumpyCFM
from .schema import CONDITION_FIELDS


def condition_from_observation(obs, road_width, initial_speed, obstacle):
    state = obs["state"]
    values = {
        "relative_obstacle_s_m": obstacle.s - obs["s"],
        "relative_obstacle_d_m": obstacle.d - obs["d"],
        "road_width_m": road_width,
        "capability_scale": min(obs["capability_scale_estimate"]),
        "v_x_mps": state["v_x_mps"],
        "v_y_mps": state["v_y_mps"],
        "yaw_rate_radps": state["psi_dot_radps"],
        "steering_rad": state["steering_rad"],
        "actuator_force_kN": obs["actuator_force_N"] / 1000.0,
        "lateral_d_m": obs["d"],
        "heading_error_rad": state["psi_rad"] - obs["road_heading"],
        "initial_speed_mps": initial_speed,
    }
    return np.asarray([values[name] for name in CONDITION_FIELDS], dtype=np.float64)


class GeneratedCandidateMixin:
    """Override RollingPlanner candidate enumeration while retaining all downstream logic."""
    source_name = "generated"

    def __call__(self, obs):
        self.current_observation = obs
        return super().__call__(obs)

    def _candidate_rows(self):
        raise NotImplementedError

    def _candidates(self):
        rows = []
        if self.selected is not None:
            rows.append(tuple(self.selected))
        for offset, ramp, brake_kn in self._candidate_rows():
            if abs(offset) < 0.25:
                offset = 0.25 if offset >= 0 else -0.25
            rows.append((float(offset), float(ramp), float(brake_kn * 1000.0),
                         float(self.obstacle.s), float(self.config["return_length_m"]), 0.0))
        ramp = max(5.0, self.obstacle.s - self.obstacle.length / 2 - self.length / 2 - 2 - self.trigger)
        rows.append((0.0, ramp, 0.0, float(self.obstacle.s),
                     float(self.config["return_length_m"]), 1.0))
        unique = list(dict.fromkeys(rows))
        return unique

    def metrics(self):
        result = super().metrics()
        result["candidate_source"] = self.source_name
        result["candidate_generation_p95_ms"] = (float(np.percentile(self.generation_times_ms, 95))
                                                    if self.generation_times_ms else 0.0)
        return result


def make_cfm_planner(base_class):
    class CFMPlanner(GeneratedCandidateMixin, base_class):
        source_name = "cfm"

        def __init__(self, *args, model_path, model_seed=0, candidate_count=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = NumpyCFM(model_path)
            self.model_seed = int(model_seed)
            self.generated_candidate_count = int(candidate_count or self.model.metadata["candidate_count"])
            self.generation_calls = 0
            self.generation_times_ms = []

        def _candidate_rows(self):
            started = time.perf_counter()
            condition = condition_from_observation(self.current_observation, self.road_width,
                                                   self.speed, self.obstacle)
            count = self.generated_candidate_count
            steps = int(self.model.metadata["sampling_steps"])
            rows = self.model.sample(condition, count, steps,
                                     self.model_seed * 1000003 + self.generation_calls)
            self.generation_calls += 1
            self.generation_times_ms.append((time.perf_counter() - started) * 1000.0)
            return rows
    return CFMPlanner


def make_hybrid_planner(base_class):
    CFMPlanner = make_cfm_planner(base_class)

    class HybridPlanner(CFMPlanner):
        source_name = "half_cfm_half_lattice"

        def _candidates(self):
            rows = []
            if self.selected is not None:
                rows.append(tuple(self.selected))
            generated = self._candidate_rows()
            for offset, ramp, brake_kn in generated:
                if abs(offset) < 0.25:
                    offset = 0.25 if offset >= 0 else -0.25
                rows.append((float(offset), float(ramp), float(brake_kn * 1000.0),
                             float(self.obstacle.s), float(self.config["return_length_m"]), 0.0))
            lattice = base_class._candidates(self)
            lattice_nonstop = [tuple(value) for value in lattice if value[5] < 0.5
                               and (self.selected is None or tuple(value) != tuple(self.selected))]
            rows.extend(lattice_nonstop[:len(generated)])
            ramp = max(5.0, self.obstacle.s - self.obstacle.length / 2 - self.length / 2 - 2 - self.trigger)
            rows.append((0.0, ramp, 0.0, float(self.obstacle.s),
                         float(self.config["return_length_m"]), 1.0))
            return list(dict.fromkeys(rows))

    return HybridPlanner


def make_retrieval_planner(base_class):
    class RetrievalPlanner(GeneratedCandidateMixin, base_class):
        source_name = "nearest_training_demonstrations"

        def __init__(self, *args, conditions, targets, condition_mean, condition_std,
                     candidate_count=36, **kwargs):
            super().__init__(*args, **kwargs)
            self.retrieval_conditions = np.asarray(conditions, dtype=np.float64)
            self.retrieval_targets = np.asarray(targets, dtype=np.float64)
            self.retrieval_mean = np.asarray(condition_mean, dtype=np.float64)
            self.retrieval_std = np.asarray(condition_std, dtype=np.float64)
            self.retrieval_count = int(candidate_count)
            self.generation_times_ms = []

        def _candidate_rows(self):
            started = time.perf_counter()
            condition = condition_from_observation(self.current_observation, self.road_width,
                                                   self.speed, self.obstacle)
            delta = (self.retrieval_conditions - condition) / self.retrieval_std
            order = np.argsort(np.sum(delta * delta, axis=1))
            rows = []
            seen = set()
            for index in order:
                value = self.retrieval_targets[index]
                key = tuple(np.round(value, 5))
                if key not in seen:
                    rows.append(value)
                    seen.add(key)
                if len(rows) == self.retrieval_count:
                    break
            self.generation_times_ms.append((time.perf_counter() - started) * 1000.0)
            return rows
    return RetrievalPlanner
