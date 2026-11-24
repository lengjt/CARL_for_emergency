"""SI units, right-handed world: x forward, y left, yaw counterclockwise."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Road:
    width: float = 10.0
    length: float = 130.0
    curvature: float = 0.0

    def pose(self, s, d=0.0):
        k = self.curvature
        if abs(k) < 1e-12:
            return float(s), float(d), 0.0
        a = k * s
        return math.sin(a)/k-d*math.sin(a), (1-math.cos(a))/k+d*math.cos(a), a

    def project(self, x, y):
        k = self.curvature
        if abs(k) < 1e-12:
            return float(x), float(y), 0.0
        a = math.atan2(k*x, 1-k*y)
        d = (1-math.hypot(k*x, 1-k*y))/k
        return a/k, d, a

@dataclass(frozen=True)
class Obstacle:
    s: float = 45.0
    d: float = 0.0
    length: float = 4.0
    width: float = 1.8
    reveal_distance: float = 35.0  # arc distance from ego CG; present from t=0

@dataclass(frozen=True)
class Patch:
    s_start: float
    s_end: float
    d_min: float = -100.0
    d_max: float = 100.0
    scale: float = 0.5
    t_start: float = 0.0
    t_end: float = 1e9

@dataclass(frozen=True)
class Scenario:
    name: str = "straight_popup"
    seed: int = 0
    speed: float = 20.0
    initial_s: float = 0.0
    initial_d: float = 0.0
    initial_heading_error: float = 0.0
    initial_vy: float = 0.0
    initial_yaw_rate: float = 0.0
    dt: float = 0.001
    control_dt: float = 0.02
    duration: float = 12.0
    base_scale: float = 1.0
    friction_mode: str = "post_force"  # preserved for old JSON; new benchmark explicitly uses internal_mu
    scale_estimate_factor: float = 1.0
    scale_estimate_delay: float = 0.0
    road: Road = field(default_factory=Road)
    obstacles: tuple = field(default_factory=lambda: (Obstacle(),))
    patches: tuple = ()
    recovery_duration: float = 0.8
    recovery_lateral: float = 0.35
    recovery_heading: float = 0.05
    recovery_beta: float = 0.05
    recovery_yaw_error: float = 0.08

    def __post_init__(self):
        if self.friction_mode not in ("post_force", "internal_mu"):
            raise ValueError("friction_mode must be post_force or internal_mu")
        scalars = [self.speed, self.dt, self.control_dt, self.duration, self.base_scale,
                   self.scale_estimate_factor, self.scale_estimate_delay, self.road.width,
                   self.road.length, self.road.curvature, self.initial_s, self.initial_d,
                   self.initial_heading_error, self.initial_vy, self.initial_yaw_rate,
                   self.recovery_duration, self.recovery_lateral, self.recovery_heading,
                   self.recovery_beta, self.recovery_yaw_error]
        if not all(math.isfinite(x) for x in scalars):
            raise ValueError("Scenario contains nonfinite numbers")
        if min(self.dt, self.control_dt, self.duration, self.base_scale, self.road.width,
               self.road.length, self.scale_estimate_factor, self.recovery_duration,
               self.recovery_lateral, self.recovery_heading, self.recovery_beta,
               self.recovery_yaw_error) <= 0 or self.speed < 0 or self.scale_estimate_delay < 0:
            raise ValueError("Invalid scenario ranges")
        if abs(round(self.control_dt/self.dt)*self.dt-self.control_dt) > 1e-10:
            raise ValueError("control_dt must be an integer multiple of dt")
        if self.dt > 0.002 or abs(self.road.curvature)*self.road.length > 2.5:
            raise ValueError("Use dt <= 0.002 s and a non-wrapping circular road")
        for ob in self.obstacles:
            if not all(math.isfinite(v) for v in asdict(ob).values()) or min(ob.length, ob.width, ob.reveal_distance) <= 0:
                raise ValueError("Invalid obstacle")
        for p in self.patches:
            if not all(math.isfinite(v) for v in asdict(p).values()) or not (
                    p.s_end > p.s_start and p.d_max > p.d_min and p.scale > 0 and p.t_end > p.t_start):
                raise ValueError("Invalid road patch")

    def scale_at(self, x, y, t):
        s, d, _ = self.road.project(x, y)
        scale = self.base_scale
        # Last matching patch wins. This ordering is persisted in run configuration.
        for p in self.patches:
            if p.s_start <= s < p.s_end and p.d_min <= d < p.d_max and p.t_start <= t < p.t_end:
                scale = p.scale
        return scale

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["road"] = Road(**data.get("road", {}))
        data["obstacles"] = tuple(Obstacle(**o) for o in data.get("obstacles", []))
        data["patches"] = tuple(Patch(**p) for p in data.get("patches", []))
        return cls(**data)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            path = ROOT / "configs" / "scenarios" / (str(path) + ".json")
        return cls.from_dict(json.loads(path.read_text()))

def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

def rectangle(x, y, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[length/2, width/2], [-length/2, width/2],
                      [-length/2, -width/2], [length/2, -width/2]])
    return local @ np.array([[c, s], [-s, c]]) + [x, y]

def separation(a, b):
    """Maximum signed SAT axis gap; <= 0 means overlap (not Euclidean distance)."""
    best = -math.inf
    for polygon in (a, b):
        for i in (0, 1):
            e = polygon[i+1]-polygon[i]
            axis = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            pa, pb = a @ axis, b @ axis
            best = max(best, float(pb.min()-pa.max()), float(pa.min()-pb.max()))
    return best

def road_margin(road, polygon):
    """Exact lateral strip margin of a rectangle on a straight/circular road."""
    points = list(polygon)
    if abs(road.curvature) > 1e-12:
        center = np.array([0.0, 1/road.curvature])
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            e = b-a
            t = float(np.clip((center-a) @ e / (e @ e), 0, 1))
            points.append(a+t*e)
    return road.width/2 - max(abs(road.project(*p)[1]) for p in points)
