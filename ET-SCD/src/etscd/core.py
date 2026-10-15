"""ET-SCD Eqs. (2)-(15) and supplementary Algorithm S1.

Score ties use ascending token ID; view ties retain generation order. Local
intervals use original full-vocabulary entropy. Inactive steps maximize the
full-vocabulary contrastive score.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Callable, Mapping
import numpy as np
from .config import ETConfig


def _vector(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError("Expected a nonempty finite 1D logit/evidence vector")
    return result


def softmax(logits):
    logits = _vector(logits)
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def normalized_entropy(logits):
    p = softmax(logits)
    if p.size == 1:
        return 0.0
    positive = p > 0
    return float(np.clip(-np.sum(p[positive] * np.log(p[positive])) / np.log(p.size), 0, 1))


def candidate_set(logits, cfg=ETConfig()):
    p = softmax(logits)
    u = normalized_entropy(logits)
    order = np.lexsort((np.arange(p.size), -p))
    mass = cfg.m_min + (cfg.m_max - cfg.m_min) * u
    cap = int(np.floor(cfg.k_min + (cfg.k_max - cfg.k_min) * u))
    required = int(np.searchsorted(np.cumsum(p[order]), mass, side="left")) + 1
    return [int(i) for i in order[:min(required, cap, p.size)]]


@dataclass
class Evidence:
    support: np.ndarray
    width: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def to_dict(self):
        return {name: getattr(self, name).tolist() for name in ("support", "width", "lower", "upper")}


def evidence(delta, u, cfg=ETConfig()):
    if not np.isfinite(u) or not 0 <= u <= 1:
        raise ValueError("Entropy must be normalized to [0,1]")
    s = softmax(_vector(delta) / cfg.tau_s)
    w = cfg.lambda_u * u + cfg.lambda_b * s * (1 - s)
    return Evidence(s, w, np.maximum(0, s - w), np.minimum(1, s + w))


def risks(lower, upper, cfg=ETConfig()):
    if not 0 <= lower <= upper <= 1:
        raise ValueError("Invalid evidence interval")
    d_l = cfg.lambda_d0 + (cfg.lambda_d1 - cfg.lambda_d0) * lower
    d_u = cfg.lambda_d0 + (cfg.lambda_d1 - cfg.lambda_d0) * upper
    return {"Accept": cfg.lambda_fa * (1 - lower),
            "Reject": cfg.lambda_fr * upper, "Defer": max(d_l, d_u)}


def choose_action(lower, upper, cfg=ETConfig()):
    values = risks(lower, upper, cfg)
    return min(("Accept", "Reject", "Defer"), key=values.__getitem__)


def _best(candidates, scores):
    return min(candidates, key=lambda i: (-scores[i], i))


def boundary_set(candidates, leader, rejected, ev, cfg=ETConfig()):
    pos = {token: j for j, token in enumerate(candidates)}
    j = pos[leader]
    other = [i for i in candidates if i != leader and i not in rejected]
    overlap = [i for i in other if max(ev.lower[j], ev.lower[pos[i]]) <= min(ev.upper[j], ev.upper[pos[i]])]
    fallback = [i for i in other if i not in overlap]
    priority = lambda i: (-(ev.support[pos[i]] + cfg.beta_b * ev.width[pos[i]]), i)
    return [leader] + (sorted(overlap, key=priority) + sorted(fallback, key=priority))[:cfg.k_boundary - 1]


def _summary(ev):
    support = np.sort(ev.support)[::-1]
    n = support.size
    if n < 2:
        raise ValueError("Local disambiguation requires at least two candidates")
    overlaps = sum(max(ev.lower[i], ev.lower[j]) <= min(ev.upper[i], ev.upper[j])
                   for i in range(n) for j in range(i + 1, n))
    return float(support[0] - support[1]), float(2 * overlaps / (n * (n - 1))), float(ev.width.mean())


def js_divergence(p, q):
    p, q = _vector(p), _vector(q)
    if p.shape != q.shape or (p < 0).any() or (q < 0).any() or not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1):
        raise ValueError("JS inputs must be probability vectors on the same candidates")
    m = .5 * (p + q)
    def kl(a):
        active = a > 0
        return np.sum(a[active] * np.log(a[active] / m[active]))
    return float(.5 * (kl(p) + kl(q)))


@dataclass
class StepResult:
    token_id: int
    trace: dict
    scores: np.ndarray


def decode_step(clean_logits, weak_logits, *, config=ETConfig(), binary_ids=None,
                object_ids=frozenset(), target_for_token: Mapping[int, str] | None = None,
                question_target=None, query_views: Callable | None = None):
    le, la = _vector(clean_logits), _vector(weak_logits)
    if le.shape != la.shape:
        raise ValueError("Original and weakened logits must share vocabulary")
    cfg = config
    delta = le - la
    base = le + cfg.alpha * delta
    u = normalized_entropy(le)
    if binary_ids is not None:
        if len(binary_ids) != 2 or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in binary_ids):
            raise ValueError("Binary candidates require two distinct token IDs")
        candidates = sorted(int(i) for i in binary_ids)
        if len(set(candidates)) != 2 or min(candidates) < 0 or max(candidates) >= le.size:
            raise ValueError("Invalid binary candidates")
        ratio, activated = None, True
    else:
        candidates = candidate_set(le, cfg)
        ratio = sum(i in object_ids for i in candidates) / len(candidates)
        activated = ratio >= cfg.gamma_o
    trace = {"entropy": u, "candidates": candidates, "object_ratio": ratio,
             "activated": activated, "route": [], "rejected": [], "boundary": [],
             "targets": [], "views": [], "requery_rounds": 0, "view_forwards": 0,
             "rho": 0.0, "updated": False, "base_token_id": int(_best(candidates, base)),
             "base_candidate_scores": base[candidates].tolist(),
             "candidate_delta": delta[candidates].tolist()}

    def finish(token, reason, scores=base):
        trace["reason"] = reason
        trace["token_id"] = int(token)
        trace["changed_from_base"] = token != trace["base_token_id"]
        return StepResult(int(token), trace, scores.copy())

    if not activated:
        token = _best(range(le.size), base)
        trace["base_token_id"] = int(token)
        return finish(token, "object_gate_bypass")

    ev = evidence(delta[candidates], u, cfg)
    trace["evidence"] = ev.to_dict()
    positions = {token: j for j, token in enumerate(candidates)}
    rejected = set()

    def fallback(reason):
        admissible = [i for i in candidates if i not in rejected] or candidates
        trace["rejected"] = sorted(rejected)
        return finish(_best(admissible, base), reason)

    attempts = 0
    while [i for i in candidates if i not in rejected] and attempts < cfg.k_reject:
        leader = _best([i for i in candidates if i not in rejected], base)
        j = positions[leader]
        action = choose_action(float(ev.lower[j]), float(ev.upper[j]), cfg)
        trace["route"].append({"leader": leader, "action": action,
                               "risks": risks(float(ev.lower[j]), float(ev.upper[j]), cfg)})
        if action == "Accept":
            trace["rejected"] = sorted(rejected)
            return finish(leader, "accept")
        if action == "Reject":
            rejected.add(leader)
            attempts += 1
            continue

        boundary = boundary_set(candidates, leader, rejected, ev, cfg)
        trace["boundary"] = boundary
        if len(boundary) < 2:
            return fallback("invalid_boundary")
        if binary_ids is not None:
            targets = [str(question_target).strip()] if question_target and str(question_target).strip() else []
        else:
            targets = []
            for i in boundary:
                concept = (target_for_token or {}).get(i)
                if concept and str(concept).strip() and str(concept).strip() not in targets:
                    targets.append(str(concept).strip())
            targets = targets[:cfg.k_targets]
        trace["targets"] = targets
        if not targets:
            return fallback("invalid_target")
        if query_views is None:
            return fallback("no_view_provider")

        # Re-normalize pre-query evidence within B; do not slice C-normalized S.
        local0 = evidence(delta[boundary], u, cfg)
        m0, o0, w0 = _summary(local0)
        trace["local_before"] = local0.to_dict()
        best = None
        trace["requery_rounds"] = 1
        # islice avoids accidentally evaluating a fifth lazy model forward.
        for view_id, view_logits in islice(query_views(tuple(targets)), cfg.max_views):
            lq = _vector(view_logits)
            if lq.shape != le.shape:
                raise ValueError("View logits must share original vocabulary")
            dq = le[boundary] - lq[boundary]
            localq = evidence(dq, u, cfg)
            mq, oq, wq = _summary(localq)
            gain = cfg.lambda_m * (mq - m0) + cfg.lambda_o * (o0 - oq) + cfg.lambda_w * (w0 - wq)
            shift = js_divergence(localq.support, local0.support)
            admissible = shift <= cfg.epsilon_js
            trace["views"].append({"id": str(view_id), "gain": gain, "js": shift,
                                   "admissible": admissible, "margin": mq, "overlap": oq,
                                   "mean_width": wq, "evidence": localq.to_dict()})
            trace["view_forwards"] += 1
            if admissible and (best is None or gain > best[0]):
                best = (gain, str(view_id), dq)
        if best is None:
            return fallback("no_admissible_view")
        gain, view_id, dq = best
        trace["selected_view"] = view_id
        trace["gain"] = gain
        if gain < cfg.epsilon_gain:
            return fallback("gain_gate_failed")
        rho = float(np.clip(cfg.eta * (max(gain, 0) + .5 * w0), 0, cfg.rho_max))
        refined = delta.copy()
        refined[boundary] = (1 - rho) * delta[boundary] + rho * dq
        scores = le + cfg.alpha * refined
        trace.update({"rho": rho, "updated": True, "rejected": sorted(rejected),
                      "refined_boundary_delta": refined[boundary].tolist()})
        return finish(_best([i for i in boundary if i not in rejected], scores), "defer_fused", scores)
    return fallback("reject_budget_exhausted")
