"""TMM supplement Table VI defaults; no environment-dependent policy overrides."""
from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class ETConfig:
    alpha: float = 1.0
    tau_s: float = 1.0
    k_min: int = 5
    k_max: int = 50
    m_min: float = 0.60
    m_max: float = 0.95
    gamma_o: float = 0.05
    lambda_u: float = 0.05
    lambda_b: float = 0.80
    lambda_fa: float = 1.0
    lambda_fr: float = 1.0
    lambda_d1: float = 0.35
    lambda_d0: float = 0.35
    beta_b: float = 0.1
    k_boundary: int = 12
    k_reject: int = 1
    k_targets: int = 2
    max_views: int = 4
    crop_ratio: float = 0.68
    epsilon_js: float = 0.12
    lambda_m: float = 1.0
    lambda_o: float = 0.15
    lambda_w: float = 0.20
    epsilon_gain: float = 0.005
    rho_max: float = 0.30
    eta: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("k_min", "k_max", "k_boundary", "k_reject", "k_targets", "max_views"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not (self.k_min <= self.k_max and 0 < self.m_min <= self.m_max <= 1):
            raise ValueError("Invalid adaptive candidate bounds")
        if self.tau_s <= 0 or self.lambda_fa <= 0 or self.lambda_fr <= 0:
            raise ValueError("Temperature and false-action costs must be positive")
        for name in ("alpha", "lambda_u", "lambda_b", "lambda_d0", "lambda_d1", "beta_b",
                     "epsilon_js", "lambda_m", "lambda_o", "lambda_w", "epsilon_gain", "eta"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not (0 <= self.gamma_o <= 1 and 0 <= self.rho_max <= 1 and 0 < self.crop_ratio <= 1):
            raise ValueError("Invalid gate, fusion bound or crop ratio")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown policy keys: {sorted(unknown)}")
        return cls(**values)

