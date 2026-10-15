"""Validate the fixed TMM parameter profile and label configured variants."""

from .config import ETConfig


# Keep this table independent of ETConfig defaults so accidental default changes
# are detected by the profile validator.
TABLE_VI = {
    "alpha": 1.0,
    "tau_s": 1.0,
    "k_min": 5,
    "k_max": 50,
    "m_min": 0.60,
    "m_max": 0.95,
    "gamma_o": 0.05,
    "lambda_u": 0.05,
    "lambda_b": 0.80,
    "lambda_fa": 1.0,
    "lambda_fr": 1.0,
    "lambda_d1": 0.35,
    "lambda_d0": 0.35,
    "beta_b": 0.1,
    "k_boundary": 12,
    "k_reject": 1,
    "k_targets": 2,
    "max_views": 4,
    "crop_ratio": 0.68,
    "epsilon_js": 0.12,
    "lambda_m": 1.0,
    "lambda_o": 0.15,
    "lambda_w": 0.20,
    "epsilon_gain": 0.005,
    "rho_max": 0.30,
    "eta": 1.0,
}

METHODS = frozenset({"etscd", "regular", "vcd"})
BENCHMARKS = frozenset({"pope", "mme", "chair"})
PROFILES = frozenset({"tmm", "custom"})


def validate_alignment(config, *, benchmark, profile="tmm", allow_subset=False,
                       max_new_tokens=64, method="etscd"):
    """Return run-profile metadata after validating parameters and task names.

    The ``tmm`` profile fixes Table VI parameters and the 64-token generation
    limit. The ``custom`` profile accepts valid alternative parameters and
    reports each changed name. Dataset content validation is separate.
    """
    if not isinstance(config, ETConfig):
        raise TypeError("config must be an ETConfig")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {benchmark!r}")
    if method not in METHODS:
        raise ValueError(f"Unsupported decoding method: {method!r}")
    if profile not in PROFILES:
        raise ValueError(f"Unsupported parameter profile: {profile!r}")
    if not isinstance(allow_subset, bool):
        raise ValueError("allow_subset must be a bool")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")

    values = config.to_dict()
    policy_changes = sorted(name for name in set(TABLE_VI) | set(values)
                            if values.get(name) != TABLE_VI.get(name))
    changed = policy_changes + (["max_new_tokens"] if max_new_tokens != 64 else [])
    if profile == "tmm" and changed:
        details = ", ".join(
            f"{name}={max_new_tokens!r} (expected 64)" if name == "max_new_tokens"
            else f"{name}={values.get(name)!r} (expected {TABLE_VI.get(name)!r})"
            for name in changed
        )
        raise ValueError(f"TMM profile parameters differ: {details}; use profile='custom' for variants")

    return {
        "profile": profile,
        "benchmark": benchmark,
        "method": method,
        "policy_matches_table_vi": not policy_changes,
        "generation_limit_matches_paper": max_new_tokens == 64,
        "changed_parameter_names": changed,
        "diagnostic_subset": allow_subset,
        "max_new_tokens": max_new_tokens,
    }


def label_method(method, report):
    """Give variants and ET-SCD diagnostic subsets distinct result labels."""
    if method not in METHODS:
        raise ValueError(f"Unsupported decoding method: {method!r}")
    if report.get("method") != method:
        raise ValueError("Method differs from the validated run profile")
    if report.get("profile") not in PROFILES:
        raise ValueError("Missing or unsupported parameter profile")
    if report["profile"] == "custom":
        return f"{method}-custom"
    if method == "etscd" and report.get("diagnostic_subset"):
        return "etscd-diagnostic"
    return method
