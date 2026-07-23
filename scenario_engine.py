"""
scenario_engine.py
==================
Blackbox stress-testing engine for weekly-rebalancing portfolio models.

16 anomaly classes are implemented across 4 layers. Use `enabled_types` in
ScenarioSampler or main() to run with a minimal subset during initial testing.

Realistic vs adversarial
------------------------
Two banks are generated. The realistic ("core") bank is meant to contain only
perturbations with a defensible generating mechanism — things that could
plausibly happen to a real curve. The adversarial bank additionally contains
estimator probes: inputs no trading process produces, kept because they usefully
attack the Layer 1 estimators.

The split is enforced by ADVERSARIAL_ONLY_TYPES, which the ScenarioSampler
filters out whenever density == "realistic". Do not add a type to the pattern
layer without deciding which side of that line it falls on.

CALIBRATION_TODO
----------------
Every ParamSpec range in this module is hand-set, not measured. The mechanisms
are grounded; the numbers are not. Once real fund data lands, each range should
be refit against the empirical distribution of the corresponding statistic on
the real panel (per curve class — algo vs market), e.g.:
    vol_spike.magnitude      <- realised vol-of-vol ratios
    ar1_injection.phi        <- empirical ACF(1)
    merton_jump.lam/sigma_j  <- tail-day frequency and size
    *_drawdown.depth/days    <- realised max-DD depth and duration
    *_drawdown.path_jitter   <- realised roughness of historical DD paths
    regime_persistence.*     <- fitted 2-state switching frequency and drift
Until then, "realistic" claims the mechanism is right, not that the parameters
match the fund.

Data convention throughout: returns arrays are (T x N)
  T = trading days (rows), N = assets (columns)
  Matches nco_engine: ledoit_wolf_cov and denoise_cov both expect (T x n).

Usage
-----
python scripts/scenario_engine.py
    [--data-run-id <uuid>]
    [--n-scenarios-core 50]
    [--n-scenarios-adv  100]
    [--seed 42]
    [--enabled-types vol_spike contagion vol_regime_swap regime_persistence]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp, log
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy import stats
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from quant.utils.db import RunRegistry, get_connection



# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

WEEK_DAYS   = 5
LN2         = np.log(2)
AR_MAX_LAGS = 5

DENSITY_PRESETS = {
    "realistic":   {"max_concurrent": 2},
    "adversarial": {"max_concurrent": 5},
}

LAYER_ORDER    = ["univariate", "cross_curve", "regime", "pattern"]
RECOVERY_SHAPES = ["V", "U", "L", "W", "sqrt"]

# One per layer — used as the default minimal test set.
# Layer IV entry must be a realistic type, or the pattern layer drops out
# entirely when density == "realistic" (see ADVERSARIAL_ONLY_TYPES).
MINIMAL_TEST_SET = [
    "vol_spike",           # Layer I
    "contagion",           # Layer II
    "vol_regime_swap",     # Layer III
    "regime_persistence",  # Layer IV
]


# ─────────────────────────────────────────────────────────────────────────────
# 0b.  MODULE-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def cov2corr(cov: np.ndarray) -> np.ndarray:
    """Covariance → correlation matrix."""
    std  = np.sqrt(np.diag(cov))
    # Guard against zero-variance columns (degenerate assets)
    std  = np.where(std < 1e-12, 1e-12, std)
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr

    
def corr2cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Correlation matrix + std vector → covariance matrix."""
    return corr * np.outer(std, std)


def denoise_cov(cov: np.ndarray, q: float) -> np.ndarray:
    """
    De-noise a covariance matrix by clipping noise eigenvalues.

    Eigenvalues below the MP upper bound are replaced by their mean
    (preserving trace), leaving signal eigenvalues untouched.

    q = T/n ratio for this block.
    """
    corr          = cov2corr(cov)
    std           = np.sqrt(np.diag(cov))
    e_vals, e_vecs = np.linalg.eigh(corr)

    # eigh returns ascending order — reverse to descending
    e_vals  = e_vals[::-1]
    e_vecs  = e_vecs[:, ::-1]

    # σ²=1 exactly for a correlation matrix, so the theoretical
    # MP bound is used directly — no KDE fitting needed.
    e_max = (1.0 + q**-0.5) ** 2

    # Number of signal factors: eigenvalues strictly above e_max
    n_signal = int(np.sum(e_vals > e_max))
    n_signal = max(n_signal, 1)   # keep at least 1 signal factor

    # Clip: replace noise eigenvalues with their mean
    e_clip            = e_vals.copy()
    noise_mean        = e_clip[n_signal:].mean()
    e_clip[n_signal:] = noise_mean

    # Reconstruct correlation matrix
    corr_dn = e_vecs @ np.diag(e_clip) @ e_vecs.T
    corr_dn = cov2corr(corr_dn)   # re-normalise numerical drift

    return corr2cov(corr_dn, std), n_signal

def _active_days(zero_mask: np.ndarray, curve_idx: int,
                 start: int, end: int) -> np.ndarray:
    """T-axis indices of active trading days in [start, end) for one curve."""
    rel = np.where(~zero_mask[start:end, curve_idx])[0]
    return start + rel


# Below this, a measured std is treated as a stale/frozen-price artifact
# rather than a real (if quiet) market — three orders of magnitude below the
# 0.01 fallback below is not "low vol", it's a flat line with rounding noise.
_MIN_SANE_DAILY_VOL = 1e-4


def _local_vol(returns: np.ndarray, zero_mask: np.ndarray,
               curve_idx: int, before: int, n: int = 20) -> float:
    """Annualised-equivalent daily vol using up to n active days before `before`."""
    pre = np.where(~zero_mask[:before, curve_idx])[0]
    if len(pre) >= 2:
        v = float(np.std(returns[pre[-n:], curve_idx]))
        if v > _MIN_SANE_DAILY_VOL:
            return v
    # pre-window std is zero, near-flat, or insufficient — fall back to
    # global history
    avail = np.where(~zero_mask[:, curve_idx])[0]
    if len(avail) >= 2:
        v = float(np.std(returns[avail, curve_idx]))
        if v > _MIN_SANE_DAILY_VOL:
            return v
    return 0.01


def _annualised_sharpe(daily: np.ndarray) -> float:
    # Annualise a daily return series with sqrt(252).
    if len(daily) < 2:
        return 0.0
    return float(daily.mean() / (daily.std() + 1e-12) * np.sqrt(252))


def _max_drawdown(daily: np.ndarray) -> float:
    if len(daily) == 0:
        return 0.0
    cum  = np.cumprod(1 + daily)
    peak = np.maximum.accumulate(cum)
    return float((cum / peak - 1).min())


def _to_serialisable(obj):
    """Recursively convert numpy scalars/arrays to JSON-safe Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [_to_serialisable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    return obj


def _serialise_param_specs(param_specs: dict) -> dict:
    out = {}
    for k, v in param_specs.items():
        choices = None
        if v.choices is not None:
            choices = [json.dumps(c) if isinstance(c, list) else c
                       for c in v.choices]
        out[k] = {"value": v.value, "dist": v.dist,
                  "low": v.low, "high": v.high, "choices": choices}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARAMETER SPEC & SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParamSpec:
    value:   object = None
    dist:    str    = None
    low:     float  = None
    high:    float  = None
    choices: list   = None

    def sample(self, rng: np.random.Generator):
        if self.value is not None:
            return self.value
        if self.dist == "uniform":
            return float(rng.uniform(self.low, self.high))
        if self.dist == "loguniform":
            return float(exp(rng.uniform(log(self.low), log(self.high))))
        if self.dist == "normal":
            return float(rng.normal(self.low, self.high))
        if self.dist == "choice":
            idx = int(rng.integers(0, len(self.choices)))
            return self.choices[idx]
        raise ValueError(f"Unknown dist: {self.dist!r}")


class ParameterSampler:
    @staticmethod
    def sample(param_specs: dict, rng: np.random.Generator) -> dict:
        return {k: v.sample(rng) for k, v in param_specs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BASE ANOMALY CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InjectionWindow:
    start_idx: int
    end_idx:   int

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class AnomalyRecord:
    anomaly_id:       str
    anomaly_type:     str
    layer:            str
    affected_curves:  list
    window:           InjectionWindow
    params:           dict
    param_specs:      dict


class BaseAnomaly(ABC):
    layer:               str
    name:                str
    layer_priority:      int = 0
    default_param_specs: dict = field(default_factory=dict)
    targeting_policy:    dict = field(default_factory=dict)

    def __init__(self, param_specs=None):
        self.param_specs = param_specs or self.default_param_specs

    @abstractmethod
    def apply(self,
              returns:   np.ndarray,
              zero_mask: np.ndarray,
              curves:    list,
              window:    InjectionWindow,
              params:    dict,
              rng:       np.random.Generator) -> np.ndarray:
        ...

    def required_length(self, params: dict) -> int:
        return int(params.get("window_length", 4 * WEEK_DAYS))

    def sample_window(self,
                      T:            int,
                      active_mask:  np.ndarray,
                      required_len: int,
                      rng:          np.random.Generator,
                      valid_start_min: int = 20) -> InjectionWindow:
        if required_len <= 0:
            required_len = 1
        # Rolling sum to find runs of `required_len` consecutive True values
        conv = np.convolve(active_mask.astype(int),
                           np.ones(required_len, dtype=int), "valid")
        candidates = np.where(conv == required_len)[0]
        candidates = candidates[candidates >= valid_start_min]
        if len(candidates) == 0:
            raise ValueError(
                f"No valid window of length {required_len} found "
                f"(T={T}, valid_start_min={valid_start_min})"
            )
        s = int(rng.choice(candidates))
        return InjectionWindow(start_idx=s, end_idx=s + required_len)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SHARED PATH UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def build_path_segment(shape: str, depth: float,
                       n_days: int, rng: np.random.Generator,
                       jitter: float = 0.0) -> np.ndarray:
    """
    Returns (n_days,) daily log-return magnitudes for one path leg.
    V / U / sqrt / L: all non-negative (caller applies sign via += or -=).
    W: signed (net = +depth over n_days, models a dip-then-recovery leg).

    `jitter` (>= 0) applies multiplicative day-to-day roughness to the leg so
    the injected path is not a ruler-straight ramp. Real drawdowns and
    recoveries are jagged: they retrace, stall, and overshoot. The noise is
    renormalised so the leg still sums exactly to its target, i.e. `depth`
    remains the semantic parameter it always was.

    Multiplicative (not additive) noise is deliberate: it leaves structural
    zeros structurally zero (the L shape's flat leg, the U shape's floor),
    where the day-to-day variation is meant to come from the underlying real
    return the caller adds this path on top of — not from the injected shape.
    """
    if n_days <= 0:
        return np.zeros(0)
    d = float(depth)

    if shape == "V":
        out = np.full(n_days, d / n_days)

    elif shape == "U":
        floor_days = max(1, n_days // 4)
        move_days  = n_days - floor_days
        out = np.zeros(n_days)
        if move_days > 0:
            out[:move_days] = d / move_days

    elif shape == "L":
        out = np.zeros(n_days)

    elif shape == "W":
        # Recovery W: up 0.5d — dip 0.25d — recover 0.75d; net = d
        leg = max(1, n_days // 4)
        out = np.zeros(n_days)
        out[0:leg]       =  (0.5  * d) / leg   # rise
        out[leg:2*leg]   = -(0.25 * d) / leg   # dip
        out[2*leg:3*leg] =  (0.75 * d) / leg   # recover to full
        # 4th leg: flat

    elif shape == "sqrt":
        k = np.arange(n_days, dtype=float)
        cum = d * np.sqrt((k + 1) / n_days)
        out = np.diff(cum, prepend=0.0)

    else:
        raise ValueError(f"Unknown shape: {shape!r}")

    return _roughen_path(out, jitter, rng)


def _roughen_path(path: np.ndarray, jitter: float,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Apply multiplicative day-to-day roughness to a path leg, renormalised so
    the leg total is unchanged. A zero-sum leg (e.g. the flat 'L' shape) is
    returned untouched: there is no total to preserve and no shape to roughen.
    """
    if jitter <= 0 or path.size < 2:
        return path
    target = float(path.sum())
    if abs(target) < 1e-12:
        return path
    rough = path * rng.normal(1.0, jitter, path.size)
    s = float(rough.sum())
    if abs(s) < 1e-12:
        return path
    return rough * (target / s)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LAYER I — UNIVARIATE PERTURBATIONS
# ─────────────────────────────────────────────────────────────────────────────

class VolSpike(BaseAnomaly):
    layer          = "univariate"
    name           = "vol_spike"
    layer_priority = 0
    default_param_specs = {
        "magnitude":            ParamSpec(dist="uniform", low=2.0,  high=6.0),
        "n_days":               ParamSpec(dist="choice",  choices=[1, 2, 3]),
        "direction":            ParamSpec(dist="choice",  choices=["up", "down", "neutral"]),
        "asymmetry":            ParamSpec(dist="choice",  choices=["symmetric", "asymmetric"]),
        "asymmetry_multiplier": ParamSpec(dist="uniform", low=1.2, high=2.0),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (1, 10)}

    def required_length(self, params):
        return int(params["n_days"])

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out       = returns.copy()
        mag       = params["magnitude"]
        n_days    = int(params["n_days"])
        direction = params["direction"]
        asymmetry = params["asymmetry"]
        asym_mult = params["asymmetry_multiplier"]

        for c in curves:
            active = _active_days(zero_mask, c, window.start_idx, window.end_idx)
            active = active[:n_days]
            if len(active) == 0:
                continue
            lv     = _local_vol(returns, zero_mask, c, window.start_idx)
            spikes = rng.normal(0.0, mag * lv, size=len(active))

            if direction == "up":
                spikes = np.abs(spikes)
            elif direction == "down":
                spikes = -np.abs(spikes)
            # "neutral": keep signed

            if asymmetry == "asymmetric":
                spikes[spikes < 0] *= asym_mult

            out[active, c] += spikes
        return out


class VolClusterBurst(BaseAnomaly):
    layer          = "univariate"
    name           = "vol_cluster_burst"
    layer_priority = 0
    default_param_specs = {
        "burst_vol_multiplier": ParamSpec(dist="uniform", low=1.5,  high=4.0),
        "decay_halflife":       ParamSpec(dist="uniform", low=3.0,  high=15.0),
        "window_length":        ParamSpec(dist="uniform", low=2*WEEK_DAYS, high=8*WEEK_DAYS),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (1, 10)}

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out        = returns.copy()
        mult       = params["burst_vol_multiplier"]
        halflife   = params["decay_halflife"]
        win_len    = int(params["window_length"])

        ramp_days    = max(1, round(0.20 * win_len))
        sustain_days = max(1, round(0.30 * win_len))

        for c in curves:
            active = _active_days(zero_mask, c, window.start_idx, window.end_idx)
            for k, t in enumerate(active):
                if k < ramp_days:
                    env = 1.0 + (mult - 1.0) * (k / ramp_days)
                elif k < ramp_days + sustain_days:
                    env = mult
                else:
                    t_d = k - ramp_days - sustain_days
                    env = mult * np.exp(-t_d * LN2 / halflife)
                out[t, c] *= env
        return out


class PersistentVolShift(BaseAnomaly):
    layer          = "univariate"
    name           = "persistent_vol_shift"
    layer_priority = 0
    default_param_specs = {
        "vol_multiplier": ParamSpec(dist="loguniform", low=0.3,  high=3.0),
        "window_length":  ParamSpec(dist="uniform",    low=4*WEEK_DAYS, high=24*WEEK_DAYS),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (1, 10)}

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out = returns.copy()
        vm  = params["vol_multiplier"]
        for c in curves:
            out[window.start_idx:window.end_idx, c] *= vm
        return out


class ArtificialDrawdown(BaseAnomaly):
    layer          = "univariate"
    name           = "artificial_drawdown"
    layer_priority = 0
    default_param_specs = {
        "depth":          ParamSpec(dist="uniform", low=0.05, high=0.40),
        "shape":          ParamSpec(dist="choice",  choices=["V", "U", "L", "W", "sqrt"]),
        "drawdown_days":  ParamSpec(dist="uniform", low=WEEK_DAYS,   high=8*WEEK_DAYS),
        "recovery_days":  ParamSpec(dist="uniform", low=WEEK_DAYS,   high=24*WEEK_DAYS),
        # Roughness of the injected leg. NOT calibrated — placeholder range
        # pending real fund data; see CALIBRATION_TODO.
        "path_jitter":    ParamSpec(dist="uniform", low=0.20,        high=0.50),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (1, 10)}

    def required_length(self, params):
        if params["shape"] == "L":
            return int(params["drawdown_days"])
        return int(params["drawdown_days"] + params["recovery_days"])

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out      = returns.copy()
        depth    = params["depth"]
        shape    = params["shape"]
        dd_days  = int(params["drawdown_days"])
        rec_days = int(params["recovery_days"])
        jitter   = float(params.get("path_jitter", 0.0))

        for c in curves:
            active = _active_days(zero_mask, c, window.start_idx, window.end_idx)
            if len(active) == 0:
                continue

            # Drawdown leg
            dd_n    = min(dd_days, len(active))
            dd_path = build_path_segment("V", depth, dd_n, rng, jitter=jitter)
            for k in range(dd_n):
                out[active[k], c] -= dd_path[k]

            # Recovery leg
            if shape != "L" and len(active) > dd_n:
                rec_active = active[dd_n:]
                rec_n      = min(rec_days, len(rec_active))
                rec_path   = build_path_segment(shape, depth, rec_n, rng,
                                                jitter=jitter)
                for k in range(rec_n):
                    out[rec_active[k], c] += rec_path[k]
        return out


class DriftInjection(BaseAnomaly):
    layer          = "univariate"
    name           = "drift_injection"
    layer_priority = 0
    default_param_specs = {
        "drift_per_day":     ParamSpec(dist="normal",  low=0.0,  high=0.002),
        "drift_type":        ParamSpec(dist="choice",  choices=["linear", "mean_reverting"]),
        "mean_revert_speed": ParamSpec(dist="uniform", low=0.02, high=0.20),
        "sigma_ou":          ParamSpec(dist="uniform", low=5e-4, high=3e-3),
        "window_length":     ParamSpec(dist="uniform", low=4*WEEK_DAYS, high=20*WEEK_DAYS),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (1, 10)}

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out        = returns.copy()
        d0         = params["drift_per_day"]
        dtype      = params["drift_type"]
        speed      = params["mean_revert_speed"]
        sigma_ou   = params["sigma_ou"]

        for c in curves:
            active = _active_days(zero_mask, c, window.start_idx, window.end_idx)
            if len(active) == 0:
                continue
            if dtype == "linear":
                for t_idx, t in enumerate(active):
                    out[t, c] += d0 * t_idx
            else:
                drift = d0
                for t in active:
                    out[t, c] += drift
                    drift = (1 - speed) * drift + sigma_ou * rng.standard_normal()
        return out


class TrendReversal(BaseAnomaly):
    layer          = "univariate"
    name           = "trend_reversal"
    layer_priority = 0
    default_param_specs = {
        "flip_fraction":  ParamSpec(dist="uniform", low=0.5, high=1.0),
        "trend_lookback": ParamSpec(dist="choice",  choices=[3, 5, 10]),
        "window_length":  ParamSpec(dist="uniform", low=WEEK_DAYS, high=6*WEEK_DAYS),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (1, 10)}

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out      = returns.copy()
        flip     = params["flip_fraction"]
        lookback = int(params["trend_lookback"])

        for c in curves:
            active = _active_days(zero_mask, c, window.start_idx, window.end_idx)
            # All active days up to (but not including) this window for rolling mean
            pre_all = np.where(~zero_mask[:window.start_idx, c])[0]

            for t in active:
                prev = pre_all[pre_all < t] if len(pre_all) else np.array([], dtype=int)
                if len(prev) == 0:
                    trend = 0.0
                else:
                    trend = float(np.mean(returns[prev[-lookback:], c]))
                out[t, c] = returns[t, c] - (1 + flip) * trend
        return out


class HeavyTailSubstitution(BaseAnomaly):
    layer          = "univariate"
    name           = "heavy_tail_sub"
    layer_priority = 0
    default_param_specs = {
        "tail_df":       ParamSpec(dist="uniform", low=3.0, high=8.0),
        "window_length": ParamSpec(dist="uniform", low=2*WEEK_DAYS, high=12*WEEK_DAYS),
        "preserve_vol":  ParamSpec(dist="choice",  choices=[True, False]),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (1, 10)}

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out          = returns.copy()
        df_param     = params["tail_df"]
        preserve_vol = params["preserve_vol"]

        for c in curves:
            active = _active_days(zero_mask, c, window.start_idx, window.end_idx)
            if len(active) == 0:
                continue
            lv = _local_vol(returns, zero_mask, c, window.start_idx)

            rng_seed = int(rng.integers(0, 2**31))
            new_ret  = stats.t.rvs(df=df_param, size=len(active),
                                   random_state=rng_seed) * lv

            if preserve_vol:
                orig_std = np.std(returns[active, c])
                gen_std  = np.std(new_ret)
                if gen_std > 1e-8:
                    ratio   = float(np.clip(orig_std / gen_std, 0.5, 2.0))
                    new_ret = new_ret * ratio

            out[active, c] = new_ret
        return out


class AR1Injection(BaseAnomaly):
    layer          = "univariate"
    name           = "ar1_injection"
    layer_priority = 0
    default_param_specs = {
        "phi":           ParamSpec(dist="uniform", low=-0.6, high=0.6),
        "window_length": ParamSpec(dist="uniform", low=4*WEEK_DAYS, high=16*WEEK_DAYS),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (1, 10)}

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out = returns.copy()
        phi = params["phi"]

        for c in curves:
            active = _active_days(zero_mask, c, window.start_idx, window.end_idx)
            if len(active) == 0:
                continue
            original_vol = np.std(returns[active, c])
            original_vol = max(float(original_vol), 1e-8)

            # Initialise from the last active return before the window
            pre = np.where(~zero_mask[:window.start_idx, c])[0]
            r_prev = float(returns[pre[-1], c]) if len(pre) > 0 else 0.0

            scale = float(np.sqrt(max(1.0 - phi**2, 1e-8)))
            gen   = np.empty(len(active))
            for k in range(len(active)):
                r_t    = phi * r_prev + scale * rng.standard_normal()
                gen[k] = r_t
                r_prev = r_t

            gen_vol = np.std(gen)
            if gen_vol > 1e-8:
                gen = gen * (original_vol / gen_vol)

            out[active, c] = gen
        return out


class MertonJumpInjection(BaseAnomaly):
    layer          = "univariate"
    name           = "merton_jump"
    layer_priority = 0
    default_param_specs = {
        "lam":           ParamSpec(dist="uniform", low=0.01,  high=0.08),
        "mu_j":          ParamSpec(dist="normal",  low=-0.01, high=0.01),
        "sigma_j":       ParamSpec(dist="uniform", low=0.01,  high=0.05),
        "mu_j_down":     ParamSpec(dist="normal",  low=-0.03, high=0.005),
        "mu_j_up":       ParamSpec(dist="normal",  low=0.005, high=0.02),
        "asymmetric":    ParamSpec(dist="choice",  choices=[True, False]),
        "window_length": ParamSpec(dist="uniform", low=4*WEEK_DAYS, high=24*WEEK_DAYS),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (1, 10)}

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out        = returns.copy()
        lam        = params["lam"]
        mu_j       = params["mu_j"]
        sigma_j    = params["sigma_j"]
        mu_j_down  = params["mu_j_down"]
        mu_j_up    = params["mu_j_up"]
        asymmetric = params["asymmetric"]

        for c in curves:
            active = _active_days(zero_mask, c, window.start_idx, window.end_idx)
            for t in active:
                if rng.uniform() < lam:
                    if asymmetric:
                        mu_used = mu_j_down if rng.uniform() < 0.5 else mu_j_up
                    else:
                        mu_used = mu_j
                    out[t, c] += rng.normal(mu_used, sigma_j)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LAYER II — CROSS-CURVE PERTURBATIONS
# ─────────────────────────────────────────────────────────────────────────────

class DecorrelationInjection(BaseAnomaly):
    layer          = "cross_curve"
    name           = "decorrelation"
    layer_priority = 0
    default_param_specs = {
        "decorr_strength": ParamSpec(dist="uniform", low=0.3,  high=1.0),
        "window_length":   ParamSpec(dist="uniform", low=2*WEEK_DAYS, high=12*WEEK_DAYS),
    }
    targeting_policy = {"mode": "correlated_cluster"}

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out = returns.copy()
        ds  = params["decorr_strength"]

        win_mask = ~zero_mask[window.start_idx:window.end_idx, :][:, curves]

        # Drop sparse curves (< 50% active in window) so all() can find rows
        active_frac = win_mask.mean(axis=0)
        liquid_idx  = np.where(active_frac >= 0.5)[0]
        if len(liquid_idx) < 2:
            return out
        liquid_curves = [curves[i] for i in liquid_idx]

        # Days where ALL liquid curves are simultaneously active
        active_rows = np.where(
            (~zero_mask[window.start_idx:window.end_idx, :][:, liquid_curves]).all(axis=1)
        )[0] + window.start_idx

        if len(active_rows) < 2:
            return out

        R_act  = returns[active_rows, :][:, liquid_curves]  # (T_act, n_liq)
        r_mean = R_act.mean(axis=1)                         # (T_act,)
        norm2  = float(r_mean @ r_mean)

        if norm2 < 1e-12:
            return out

        for i, c in enumerate(liquid_curves):
            col    = R_act[:, i]
            proj   = float(col @ r_mean) / norm2
            r_orth = col - ds * proj * r_mean

            pre_vol  = np.std(col)
            post_vol = np.std(r_orth)
            if post_vol > 1e-8:
                r_orth = r_orth * (pre_vol / post_vol)
            else:
                r_orth = col

            out[active_rows, c] = r_orth
        return out


class ContagionInjection(BaseAnomaly):
    layer          = "cross_curve"
    name           = "contagion"
    layer_priority = 0
    default_param_specs = {
        "contagion_strength": ParamSpec(dist="uniform", low=0.3,  high=0.9),
        "common_shock_vol":   ParamSpec(dist="uniform", low=0.8,  high=2.0),
        "window_length":      ParamSpec(dist="uniform", low=WEEK_DAYS, high=8*WEEK_DAYS),
        "n_groups":           ParamSpec(dist="choice",  choices=[2, 3, 4]),
    }
    targeting_policy = {"mode": "inter_cluster_sample"}

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out      = returns.copy()
        strength = params["contagion_strength"]
        csv      = params["common_shock_vol"]

        active_rows = np.where(
            (~zero_mask[window.start_idx:window.end_idx, :][:, curves]).all(axis=1)
        )[0] + window.start_idx

        if len(active_rows) == 0:
            return out

        vols = [max(_local_vol(returns, zero_mask, c, window.start_idx), 1e-8)
                for c in curves]

        Z = rng.standard_normal(size=len(active_rows))

        for i, c in enumerate(curves):
            eps  = csv * vols[i] * Z
            r_in = returns[active_rows, c]
            out[active_rows, c] = (np.sqrt(1 - strength) * r_in +
                                   np.sqrt(strength) * eps)
        return out


class SynchronisedDrawdownWithRecovery(BaseAnomaly):
    layer          = "cross_curve"
    name           = "sync_drawdown_recovery"
    layer_priority = 1  # after Decorrelation / Contagion
    default_param_specs = {
        "depth":            ParamSpec(dist="uniform", low=0.05, high=0.30),
        "drawdown_days":    ParamSpec(dist="uniform", low=WEEK_DAYS,   high=8*WEEK_DAYS),
        "synchrony":        ParamSpec(dist="uniform", low=0.5,  high=1.0),
        "lag_max":          ParamSpec(value=3),
        "recovery_days":    ParamSpec(dist="uniform", low=2*WEEK_DAYS, high=24*WEEK_DAYS),
        "per_curve_shape":  ParamSpec(dist="choice",  choices=[True, False]),
        "recovery_shape":   ParamSpec(dist="choice",  choices=RECOVERY_SHAPES),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (3, 20)}

    def required_length(self, params):
        return int(params["drawdown_days"] + params["recovery_days"] + params["lag_max"])

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out         = returns.copy()
        depth       = params["depth"]
        dd_days     = int(params["drawdown_days"])
        synchrony   = params["synchrony"]
        lag_max     = int(params["lag_max"])
        rec_days    = int(params["recovery_days"])
        per_curve   = params["per_curve_shape"]
        rec_shape_g = params["recovery_shape"]

        dd_path = build_path_segment("V", depth, dd_days, rng)

        for c in curves:
            active = _active_days(zero_mask, c, window.start_idx, window.end_idx)
            if len(active) == 0:
                continue

            lag = 0
            if synchrony < 1.0 and rng.uniform() >= synchrony:
                lag = int(rng.integers(1, lag_max + 1))

            # Drawdown
            dd_a = active[lag: lag + dd_days]
            for k, t in enumerate(dd_a):
                out[t, c] -= dd_path[k] if k < len(dd_path) else 0.0

            # Recovery
            shape_i  = rng.choice(RECOVERY_SHAPES) if per_curve else rec_shape_g
            rec_start = lag + dd_days
            if shape_i != "L" and rec_start < len(active):
                rec_a = active[rec_start: rec_start + rec_days]
                if len(rec_a) > 0:
                    rec_path = build_path_segment(shape_i, depth, len(rec_a), rng)
                    for k, t in enumerate(rec_a):
                        out[t, c] += rec_path[k]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LAYER III — REGIME PERTURBATIONS
# ─────────────────────────────────────────────────────────────────────────────

class VolRegimeSwap(BaseAnomaly):
    layer          = "regime"
    name           = "vol_regime_swap"
    layer_priority = 0
    default_param_specs = {
        "source_window_length":  ParamSpec(dist="uniform", low=6*WEEK_DAYS, high=16*WEEK_DAYS),
        "target_window_length":  ParamSpec(dist="uniform", low=6*WEEK_DAYS, high=16*WEEK_DAYS),
        "preserve_sign_pattern": ParamSpec(dist="choice",  choices=[True, False]),
        "ar_order":              ParamSpec(dist="choice",  choices=[1, 2, 3]),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (10, 60)}

    def required_length(self, params):
        return int(params["target_window_length"])

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out        = returns.copy()
        src_len    = int(params["source_window_length"])
        preserve   = params["preserve_sign_pattern"]
        ar_order   = int(params["ar_order"])
        T          = returns.shape[0]

        for c in curves:
            target_active = _active_days(zero_mask, c,
                                         window.start_idx, window.end_idx)
            if len(target_active) == 0:
                continue

            # Find valid donor windows (same curve, non-overlapping)
            valid_donors = []
            for s in range(0, T - src_len + 1):
                if s + src_len <= window.start_idx or s >= window.end_idx:
                    n_active = int(np.sum(~zero_mask[s:s + src_len, c]))
                    if n_active >= max(20, src_len // 3):
                        valid_donors.append(s)

            if not valid_donors:
                continue

            ds      = int(rng.choice(valid_donors))
            d_act   = _active_days(zero_mask, c, ds, ds + src_len)
            R_donor = returns[d_act, c]

            donor_vol  = max(float(np.std(R_donor)), 1e-8)
            target_vol = max(float(np.std(returns[target_active, c])), 1e-10)

            # Vol transplant
            out[target_active, c] *= (donor_vol / target_vol)

            if not preserve and len(R_donor) > ar_order + 1:
                Y     = R_donor[ar_order:]
                X_lag = np.column_stack(
                    [R_donor[ar_order - k - 1: len(R_donor) - k - 1]
                     for k in range(ar_order)]
                )
                try:
                    phi_vec, _, _, _ = np.linalg.lstsq(X_lag, Y, rcond=None)
                    poly   = np.concatenate([[1.0], -phi_vec])
                    roots  = np.roots(poly)
                    sr     = float(np.max(np.abs(roots)))
                    if sr >= 1.0:
                        phi_vec = phi_vec * (0.95 / sr)

                    residuals  = Y - X_lag @ phi_vec
                    sigma_res  = float(np.std(residuals))
                    R_tgt      = out[target_active, c].copy()
                    for k in range(ar_order, len(R_tgt)):
                        ar_term  = float(sum(phi_vec[j] * R_tgt[k - j - 1]
                                             for j in range(ar_order)))
                        R_tgt[k] = ar_term + sigma_res * rng.standard_normal()
                    out[target_active, c] = R_tgt

                    new_std = float(np.std(out[target_active, c]))
                    if new_std > 1e-10:
                        out[target_active, c] *= (donor_vol / new_std)
                except Exception:
                    pass  # keep vol-swapped result on AR failure
        return out


class DrawdownRecoveryVariants(BaseAnomaly):
    layer          = "regime"
    name           = "drawdown_recovery_var"
    layer_priority = 0
    default_param_specs = {
        "depth":                ParamSpec(dist="uniform", low=0.05, high=0.35),
        "drawdown_days":        ParamSpec(dist="uniform", low=WEEK_DAYS, high=4*WEEK_DAYS),
        "recovery_shape_pool":  ParamSpec(dist="choice",
                                           choices=[["V", "V"],
                                                    ["L", "L"],
                                                    ["V", "U", "L"],
                                                    ["W", "sqrt", "V"]]),
        "recovery_days":        ParamSpec(dist="uniform", low=2*WEEK_DAYS, high=8*WEEK_DAYS),
        # Not calibrated — placeholder range pending real fund data.
        "path_jitter":          ParamSpec(dist="uniform", low=0.20, high=0.50),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (10, 80)}

    def required_length(self, params):
        return int(params["drawdown_days"] + params["recovery_days"])

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out        = returns.copy()
        depth      = params["depth"]
        dd_days    = int(params["drawdown_days"])
        shape_pool = params["recovery_shape_pool"]
        rec_days   = int(params["recovery_days"])
        jitter     = float(params.get("path_jitter", 0.0))

        # The *shape* of a regime-level drawdown is common across curves — that
        # is what makes it a regime event. The roughness around it is not:
        # sharing one jittered path would make the day-to-day noise perfectly
        # correlated across every affected curve. So build the common leg once,
        # then roughen it independently per curve.
        dd_base = build_path_segment("V", depth, dd_days, rng)

        for c in curves:
            active = _active_days(zero_mask, c, window.start_idx, window.end_idx)
            if len(active) == 0:
                continue

            dd_path = _roughen_path(dd_base, jitter, rng)
            for k in range(min(dd_days, len(active))):
                out[active[k], c] -= dd_path[k] if k < len(dd_path) else 0.0

            shape_c = shape_pool[int(rng.integers(0, len(shape_pool)))]
            if shape_c != "L" and len(active) > dd_days:
                rec_a    = active[dd_days: dd_days + rec_days]
                if len(rec_a) > 0:
                    rec_path = build_path_segment(shape_c, depth, len(rec_a),
                                                  rng, jitter=jitter)
                    for k, t in enumerate(rec_a):
                        out[t, c] += rec_path[k]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 7.  LAYER IV — PATTERN PERTURBATIONS
# ─────────────────────────────────────────────────────────────────────────────

class RegimePersistence(BaseAnomaly):
    """
    REALISTIC. A two-state (risk-on / risk-off) drift regime that persists and
    switches with low probability per day.

    This is the realistic half of the old `oscillating_pattern`. Two things
    differ from that class's "markov" mode, both deliberate:

    1. ADDITIVE, not wholesale replacement. The regime adds a drift term on top
       of the curve's real return; the original day-to-day texture (vol, tails,
       whatever the curve actually does) survives. The old mode overwrote the
       return entirely with `N(±A, noise*A)`, discarding the real series.

    2. PERSISTENT, not anti-persistent. `p_flip` is low, so mean regime
       duration is 1/p_flip ≈ 7-50 days. The old mode drew p_flip from
       [0.55, 0.90] — i.e. it flipped MORE often than not, giving a mean regime
       duration of 1.1-1.8 days and ACF(1) down to -0.81. That is an
       oscillator, not a regime model, regardless of the "markov" label; it now
       lives in OscillatingPattern with its siblings.

    `drift_amp` is expressed in units of the curve's own local vol, so a regime
    moves the mean by a fraction of a daily sigma — the drift accumulates into
    a visible trend over the regime's life without producing an implausible
    per-day Sharpe.
    """
    layer          = "pattern"
    name           = "regime_persistence"
    layer_priority = 0
    default_param_specs = {
        # Regime drift as a fraction of the curve's local daily vol.
        # NOT calibrated — placeholder pending real fund data.
        "drift_amp":     ParamSpec(dist="uniform", low=0.05, high=0.30),
        # Per-day switch probability; mean regime duration = 1/p_flip.
        # 0.02 -> ~50 trading days, 0.15 -> ~7 trading days.
        "p_flip":        ParamSpec(dist="uniform", low=0.02, high=0.15),
        "window_length": ParamSpec(dist="uniform", low=4*WEEK_DAYS, high=16*WEEK_DAYS),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (1, 5)}

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out       = returns.copy()
        drift_amp = params["drift_amp"]
        p_flip    = params["p_flip"]

        for c in curves:
            active = _active_days(zero_mask, c, window.start_idx, window.end_idx)
            if len(active) == 0:
                continue
            lv    = _local_vol(returns, zero_mask, c, window.start_idx)
            mu    = drift_amp * lv
            state = int(rng.choice([-1, 1]))
            for t in active:
                out[t, c] += state * mu
                if rng.uniform() < p_flip:
                    state = -state
        return out


class OscillatingPattern(BaseAnomaly):
    """
    ADVERSARIAL ONLY. Deterministic / stochastic sign oscillation at a fixed or
    near-fixed frequency.

    No trading process generates this: it is a signal-processing probe, kept to
    test whether the Layer 1 estimators can be fooled by a strongly
    anti-persistent input. It is excluded from the realistic bank by
    ADVERSARIAL_ONLY_TYPES.

    The three modes are one family — all oscillators, differing only in how the
    sign flips:
      deterministic : flips every day (period-2 square wave)
      markov        : flips with probability p_flip, drawn HIGH (>0.5) so the
                      series is anti-persistent — the stochastic counterpart of
                      the square wave, not a regime model
      sinusoidal    : smooth sine at a fixed period

    Note on p_flip's range: since this class is now explicitly the oscillator
    family, p_flip runs to 1.0, where markov mode converges exactly on
    deterministic mode. The low-p_flip (persistent) half of the old range moved
    to RegimePersistence, so there is no longer a gap between the two classes —
    together they cover p_flip in (0, 1].
    """
    layer          = "pattern"
    name           = "oscillating_pattern"
    layer_priority = 0
    default_param_specs = {
        "amplitude":     ParamSpec(dist="uniform", low=1.0,  high=3.0),
        "noise_level":   ParamSpec(dist="uniform", low=0.0,  high=0.5),
        "window_length": ParamSpec(dist="uniform", low=2*WEEK_DAYS, high=8*WEEK_DAYS),
        "oscillation_type": ParamSpec(dist="choice",
                                      choices=["deterministic", "markov", "sinusoidal"]),
        "p_flip":        ParamSpec(dist="uniform", low=0.55, high=1.0),
        "period":        ParamSpec(dist="uniform", low=4.0,  high=20.0),
    }
    targeting_policy = {"mode": "random_subset", "n_curves_range": (1, 5)}

    def apply(self, returns, zero_mask, curves, window, params, rng):
        out        = returns.copy()
        amplitude  = params["amplitude"]
        noise_lvl  = params["noise_level"]
        oscillation_type = params["oscillation_type"]
        p_flip     = params["p_flip"]

        for c in curves:
            active = _active_days(zero_mask, c, window.start_idx, window.end_idx)
            if len(active) == 0:
                continue
            lv = _local_vol(returns, zero_mask, c, window.start_idx)
            A  = amplitude * lv

            if oscillation_type == "deterministic":
                for k, t in enumerate(active):
                    sign      = 1 if k % 2 == 0 else -1
                    out[t, c] = rng.normal(sign * A, noise_lvl * A + 1e-10)
            elif oscillation_type == "markov":
                state = int(rng.choice([-1, 1]))
                for t in active:
                    out[t, c] = rng.normal(state * A, noise_lvl * A + 1e-10)
                    if rng.uniform() < p_flip:
                        state = -state
            else:
                period = params["period"]
                for k, t in enumerate(active):
                    sign      = np.sin(2 * np.pi * k / period)
                    out[t, c] = rng.normal(sign * A, noise_lvl * A + 1e-10)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ANOMALY REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

ANOMALY_REGISTRY: dict[str, type] = {
    # Layer I
    "vol_spike":              VolSpike,
    "vol_cluster_burst":      VolClusterBurst,
    "persistent_vol_shift":   PersistentVolShift,
    "artificial_drawdown":    ArtificialDrawdown,
    "drift_injection":        DriftInjection,
    "trend_reversal":         TrendReversal,
    "heavy_tail_sub":         HeavyTailSubstitution,
    "ar1_injection":          AR1Injection,
    "merton_jump":            MertonJumpInjection,
    # Layer II
    "decorrelation":          DecorrelationInjection,
    "contagion":              ContagionInjection,
    "sync_drawdown_recovery": SynchronisedDrawdownWithRecovery,
    # Layer III
    "vol_regime_swap":        VolRegimeSwap,
    "drawdown_recovery_var":  DrawdownRecoveryVariants,
    # Layer IV
    "regime_persistence":     RegimePersistence,
    "oscillating_pattern":    OscillatingPattern,
}

# Types that are structurally implausible as real market/EA behaviour. They are
# kept as estimator probes but must never enter the realistic core bank; the
# ScenarioSampler filters them out whenever density == "realistic".
ADVERSARIAL_ONLY_TYPES: set[str] = {
    "oscillating_pattern",  # fixed-frequency / anti-persistent sign oscillation
}


# ─────────────────────────────────────────────────────────────────────────────
# 9b.  EMPIRICAL CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
# Fits the numeric bounds of a curated, defensible subset of ParamSpecs against
# the empirical distribution of the corresponding statistic on the real panel,
# pooled by timeframe bucket (tf). Everything not listed in PARAM_ESTIMATORS
# keeps its hand-set class default and is reported in the `uncalibrated`
# manifest — the tiering is explicit, not implied.
#
# Granularity: the statistic is measured PER CURVE; the ParamSpec is set PER
# (anomaly_type × param × tf-bucket × density) from a percentile band of the
# pooled per-curve values. Never per curve — that would overfit and has nowhere
# to live in the random-subset injection flow.
#
# Provenance: the fit is persisted to the `scenario_calibration` table stamped
# with the data_run_id it was fit against, and reloaded on subsequent bank
# builds. "Is this bank calibrated?" is answerable by inspecting that table.

CALIB_MIN_CURVES_PER_BUCKET = 20   # below this, fall back to the pooled-all band
CALIB_MIN_OBS_PER_CURVE     = 50   # a curve needs this many active days to fit

# Percentile bands: realistic keeps the body, adversarial reaches the tails.
CALIB_PCT = {
    "realistic":   (10.0, 90.0),
    "adversarial": (1.0,  99.0),
}


def _active_returns(returns: np.ndarray, zero_mask: np.ndarray,
                    c: int) -> np.ndarray:
    """1-D active-day returns for one curve (drops inactive/zero days)."""
    idx = np.where(~zero_mask[:, c])[0]
    return returns[idx, c]


# ── per-curve estimators (each returns a scalar, a tuple, or a list) ──────────

def _est_acf1(r: np.ndarray) -> float:
    x = r - r.mean()
    denom = float(x @ x)
    if denom < 1e-18:
        return np.nan
    return float((x[:-1] @ x[1:]) / denom)


def _est_extreme_vol_units(r: np.ndarray) -> float:
    """Tail-day size in units of the curve's own daily vol (vol_spike.magnitude,
    regime_persistence.drift_amp live in these units)."""
    sd = r.std()
    if sd < _MIN_SANE_DAILY_VOL:
        return np.nan
    return float(np.percentile(np.abs(r) / sd, 99))


def _est_tail_df(r: np.ndarray) -> float:
    """Student-t dof on the vol-standardised series (tail heaviness, vol-free)."""
    sd = r.std()
    if sd < _MIN_SANE_DAILY_VOL:
        return np.nan
    try:
        df, _, _ = stats.t.fit(r / sd, floc=0.0)
    except Exception:
        return np.nan
    return float(np.clip(df, 2.5, 30.0))


def _est_vol_ratio(r: np.ndarray, win: int = 20,
                   hi_q: float = 0.90, base_q: float = 0.50) -> float:
    """High-vol-window std / typical-window std (burst/persistent multipliers)."""
    if r.size < 3 * win:
        return np.nan
    roll = pd.Series(r).rolling(win).std().dropna().to_numpy()
    if roll.size < 5:
        return np.nan
    base = float(np.quantile(roll, base_q))
    if base < _MIN_SANE_DAILY_VOL:
        return np.nan
    return float(np.quantile(roll, hi_q) / base)


def _est_drawdowns(r: np.ndarray) -> list:
    """Peak-to-trough episodes on arithmetic fixed-notional equity.
    Returns [(depth_frac, duration_days), ...]. NOT cumprod — that would
    reintroduce the compounding the pipeline is built to avoid."""
    eq   = 1.0 + np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / np.where(peak > 1e-12, peak, 1e-12)
    episodes, in_dd, start, trough = [], False, 0, 0.0
    for t in range(dd.size):
        if not in_dd and dd[t] > 1e-9:
            in_dd, start, trough = True, t, dd[t]
        elif in_dd:
            trough = max(trough, dd[t])
            if dd[t] <= 1e-9:
                episodes.append((trough, t - start))
                in_dd = False
    if in_dd:
        episodes.append((trough, dd.size - start))
    return [e for e in episodes if e[0] > 0.005]   # ignore trivial dips


def _est_jumps(r: np.ndarray, k: float = 4.0) -> Optional[tuple]:
    """Threshold jumps at k robust-sigma. Returns (lam, mu, sigma_j, mu_down,
    mu_up) in raw arithmetic-return units, matching MertonJumpInjection."""
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
    if sigma < _MIN_SANE_DAILY_VOL:
        return None
    z     = (r - np.median(r)) / sigma
    jmask = np.abs(z) > k
    lam   = float(jmask.mean())
    js    = r[jmask]
    if js.size == 0:
        return (lam, np.nan, np.nan, np.nan, np.nan)
    down, up = js[js < 0], js[js > 0]
    return (
        lam,
        float(js.mean()),
        float(js.std()) if js.size > 1 else float(abs(js[0])),
        float(down.mean()) if down.size else np.nan,
        float(up.mean())   if up.size   else np.nan,
    )


def _est_drift_per_day(r: np.ndarray) -> float:
    """Mean daily arithmetic return (drift_injection.drift_per_day, a normal spec
    whose loc/scale we set from the cross-curve mean/std of this)."""
    return float(r.mean())


# ── estimator wiring ──────────────────────────────────────────────────────────
# mode:
#   "band"   -> low/high = (pct_lo, pct_hi) of the pooled per-curve values
#   "normal" -> low/high = (mean, std) of the pooled per-curve values
#              (for ParamSpecs whose dist == "normal"; low=loc, high=scale)
# The dist type is taken from the class default_param_specs and preserved; only
# the numbers are replaced. `pick` extracts the scalar this param wants out of
# an estimator that returns a tuple/list.

@dataclass
class _Est:
    stat:  str            # key into the per-curve stat cache
    mode:  str            # "band" | "normal"
    pick:  object = None  # index into a tuple stat, or None for scalar
    floor: float  = None  # clamp low
    ceil:  float  = None  # clamp high


# stat producers: name -> (estimator_fn, "scalar"|"tuple"|"list")
_STAT_PRODUCERS = {
    "acf1":        (_est_acf1,             "scalar"),
    "vol_units":   (_est_extreme_vol_units, "scalar"),
    "tail_df":     (_est_tail_df,          "scalar"),
    "vol_ratio":   (_est_vol_ratio,        "scalar"),
    "drawdowns":   (_est_drawdowns,        "list"),    # list of (depth, dur)
    "jumps":       (_est_jumps,            "tuple"),   # (lam,mu,sig,mdn,mup)
    "drift":       (_est_drift_per_day,    "scalar"),
}

# (anomaly_type, param) -> _Est.  Anything absent stays at its class default.
PARAM_ESTIMATORS: dict[tuple, _Est] = {
    ("vol_spike",            "magnitude"):            _Est("vol_units", "band", floor=1.5),
    ("persistent_vol_shift", "vol_multiplier"):       _Est("vol_ratio", "band", floor=0.2),
    ("vol_cluster_burst",    "burst_vol_multiplier"): _Est("vol_ratio", "band", floor=1.0),
    ("artificial_drawdown",  "depth"):                _Est("drawdowns", "band", pick=0, floor=0.01),
    ("artificial_drawdown",  "drawdown_days"):        _Est("drawdowns", "band", pick=1, floor=WEEK_DAYS),
    ("sync_drawdown_recovery","depth"):               _Est("drawdowns", "band", pick=0, floor=0.01),
    ("drawdown_recovery_var","depth"):                _Est("drawdowns", "band", pick=0, floor=0.01),
    ("heavy_tail_sub",       "tail_df"):              _Est("tail_df",   "band", floor=2.5, ceil=30.0),
    ("ar1_injection",        "phi"):                  _Est("acf1",      "band"),
    ("merton_jump",          "lam"):                  _Est("jumps",     "band", pick=0, floor=0.0),
    ("merton_jump",          "sigma_j"):              _Est("jumps",     "band", pick=2, floor=1e-4),
    ("merton_jump",          "mu_j"):                 _Est("jumps",     "normal", pick=1),
    ("drift_injection",      "drift_per_day"):        _Est("drift",     "normal"),
    ("regime_persistence",   "drift_amp"):            _Est("vol_units", "band", floor=0.02),
    ("contagion",            "common_shock_vol"):     _Est("vol_units", "band", floor=0.5),
}


@dataclass
class Calibration:
    """Fitted bounds + the tf-bucket index map. `bands` is
    density -> tf -> anomaly_type -> param -> (low, high).
    `buckets` (tf -> [curve_idx]) and `tf_of` are rebuilt each run from
    algo_meta; only `bands` is persisted."""
    bands:    dict
    buckets:  dict
    tf_of:    list
    run_id:   str

    def bucket_of(self, curve_idx: int) -> str:
        return self.tf_of[curve_idx] if 0 <= curve_idx < len(self.tf_of) else "_pooled"

    def majority_tf(self, curves: list) -> str:
        if not curves:
            return "_pooled"
        tfs = [self.bucket_of(c) for c in curves]
        return max(set(tfs), key=tfs.count)

    def spec_for(self, anomaly_type: str, tf: str, density: str) -> dict:
        """Return a param_specs dict for this type: class defaults with the
        calibrated params' bounds substituted. Falls back to the pooled bucket
        for a tf that wasn't fit, and to the class default for any param the fit
        skipped or couldn't estimate."""
        base = {k: ParamSpec(**vars(v))
                for k, v in ANOMALY_REGISTRY[anomaly_type]().default_param_specs.items()}
        dens = self.bands.get(density, {})
        fitted = dens.get(tf) or dens.get("_pooled") or {}
        tvals  = fitted.get(anomaly_type, {})
        for param, (lo, hi) in tvals.items():
            if param in base and lo is not None and hi is not None:
                base[param].low, base[param].high = lo, hi
                base[param].value = None
        return base


def _pooled_scalars(stat_cache: dict, curve_idxs: list, stat: str,
                    pick) -> np.ndarray:
    """Collect the (optionally picked) scalar for `stat` across curves. For
    'list' stats (drawdowns), flattens every episode's picked field."""
    out = []
    for c in curve_idxs:
        v = stat_cache[c].get(stat)
        if v is None:
            continue
        if isinstance(v, list):                    # drawdown episodes
            out.extend(e[pick] for e in v if np.isfinite(e[pick]))
        elif isinstance(v, tuple):
            x = v[pick]
            if x is not None and np.isfinite(x):
                out.append(x)
        else:
            if np.isfinite(v):
                out.append(v)
    return np.asarray(out, float)


def build_calibration(returns: np.ndarray, zero_mask: np.ndarray,
                      tf_of: list, run_id: str) -> Calibration:
    """Fit all bands for both densities from the real panel."""
    N = returns.shape[1]

    # buckets: tf -> [curve idx], only curves with enough active history
    buckets: dict = {}
    stat_cache: dict = {}
    for c in range(N):
        r = _active_returns(returns, zero_mask, c)
        if r.size < CALIB_MIN_OBS_PER_CURVE:
            continue
        cache = {}
        for name, (fn, _) in _STAT_PRODUCERS.items():
            try:
                cache[name] = fn(r)
            except Exception:
                cache[name] = None
        stat_cache[c] = cache
        buckets.setdefault(tf_of[c], []).append(c)

    all_fit = [c for c in stat_cache]   # curves that cleared the obs gate

    def fit_group(curve_idxs: list, pct_lo: float, pct_hi: float) -> dict:
        types: dict = {}
        for (atype, param), est in PARAM_ESTIMATORS.items():
            vals = _pooled_scalars(stat_cache, curve_idxs, est.stat, est.pick)
            if vals.size < 5:
                continue
            if est.mode == "band":
                lo, hi = np.percentile(vals, [pct_lo, pct_hi])
            else:  # normal -> (loc, scale)
                lo, hi = float(vals.mean()), float(vals.std() + 1e-9)
            if est.mode == "band":
                if est.floor is not None:
                    lo = max(lo, est.floor)
                if est.ceil is not None:
                    hi = min(hi, est.ceil)
                if hi <= lo:
                    hi = lo * 1.5 + 1e-9
            types.setdefault(atype, {})[param] = (float(lo), float(hi))
        return types

    bands: dict = {}
    for density, (plo, phi) in CALIB_PCT.items():
        per_tf = {"_pooled": fit_group(all_fit, plo, phi)}
        for tf, idxs in buckets.items():
            group = idxs if len(idxs) >= CALIB_MIN_CURVES_PER_BUCKET else all_fit
            if len(idxs) < CALIB_MIN_CURVES_PER_BUCKET:
                print(f"    [calib] tf={tf}: {len(idxs)} curves "
                      f"(< {CALIB_MIN_CURVES_PER_BUCKET}) — using pooled band")
            per_tf[tf] = fit_group(group, plo, phi)
        bands[density] = per_tf

    return Calibration(bands=bands, buckets=buckets, tf_of=tf_of, run_id=run_id)


# ── persistence ───────────────────────────────────────────────────────────────

def _ensure_calibration_table(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS scenario_calibration (
            run_id        VARCHAR,
            density       VARCHAR,
            tf            VARCHAR,
            anomaly_type  VARCHAR,
            param         VARCHAR,
            low           DOUBLE,
            high          DOUBLE,
            calibrated_at TIMESTAMP
        )
    """)


def persist_calibration(con, registry, calib: Calibration) -> None:
    _ensure_calibration_table(con)
    con.execute("DELETE FROM scenario_calibration WHERE run_id = ?", [calib.run_id])
    now  = datetime.now(timezone.utc)
    rows = []
    for density, per_tf in calib.bands.items():
        for tf, types in per_tf.items():
            for atype, params in types.items():
                for param, (lo, hi) in params.items():
                    rows.append([calib.run_id, density, tf, atype, param,
                                 lo, hi, now])
    if rows:
        df = pd.DataFrame(rows, columns=["run_id", "density", "tf",
                                         "anomaly_type", "param", "low",
                                         "high", "calibrated_at"])
        registry.log_dataframe(calib.run_id, "scenario_calibration", df,
                               if_exists="append")


def load_calibration(con, data_run_id: str, tf_of: list,
                     buckets: dict) -> Optional[Calibration]:
    try:
        df = con.execute(
            "SELECT density, tf, anomaly_type, param, low, high "
            "FROM scenario_calibration WHERE run_id = ?", [data_run_id]
        ).df()
    except Exception:
        return None
    if df is None or df.empty:
        return None
    bands: dict = {}
    for _, row in df.iterrows():
        (bands.setdefault(row["density"], {})
              .setdefault(row["tf"], {})
              .setdefault(row["anomaly_type"], {}))[row["param"]] = (
            float(row["low"]), float(row["high"]))
    return Calibration(bands=bands, buckets=buckets, tf_of=tf_of,
                       run_id=data_run_id)


def resolve_or_build_calibration(con, registry, data_run_id: str,
                                 returns: np.ndarray, zero_mask: np.ndarray,
                                 tf_of: list, recalibrate: bool = False):
    """Load the persisted fit for this run, or fit + persist if absent.
    Returns a Calibration, or None if calibration is unavailable (caller then
    runs on hand-set defaults)."""
    buckets: dict = {}
    for c in range(returns.shape[1]):
        if _active_returns(returns, zero_mask, c).size >= CALIB_MIN_OBS_PER_CURVE:
            buckets.setdefault(tf_of[c], []).append(c)

    if not recalibrate:
        cached = load_calibration(con, data_run_id, tf_of, buckets)
        if cached is not None:
            print("  Calibration      : loaded from scenario_calibration "
                  f"(run {data_run_id[:8]})")
            return cached

    print("  Calibration      : fitting from panel"
          f"{' (--recalibrate)' if recalibrate else ' (none on record)'}...",
          flush=True)
    calib = build_calibration(returns, zero_mask, tf_of, data_run_id)
    persist_calibration(con, registry, calib)
    print(f"  Calibration      : fit + persisted for run {data_run_id[:8]}")
    return calib


def validate_calibration(results: list, calib: Calibration) -> dict:
    """Containment check: every calibrated parameter drawn for the realistic
    bank must sit inside the empirical envelope (the wider adversarial p1–p99
    band) for the bucket it was drawn against. A realistic draw landing outside
    p1–p99 of the real panel means the fit or the sampling is wrong.

    Returns {(type, param): (n_checked, n_in_envelope)} and prints a summary.
    The per-type injected-vs-empirical *plot* belongs in run_stress_test.py,
    where the plotting infrastructure lives; this is the numeric core.
    """
    adv = calib.bands.get("adversarial", {})
    # Only band-mode params define a containment interval. normal-mode params
    # store (loc, scale), which is not an interval — a Gaussian draw is not
    # bounded by its own mean and sd — so they're excluded from this check.
    band_params = {k for k, e in PARAM_ESTIMATORS.items() if e.mode == "band"}
    tally: dict = {}
    for res in results:
        for rec in res.get("anomalies", []):
            tf     = calib.majority_tf(rec.affected_curves)
            fitted = (adv.get(tf) or adv.get("_pooled") or {}).get(rec.anomaly_type, {})
            for param, (lo, hi) in fitted.items():
                if param not in rec.params:
                    continue
                if (rec.anomaly_type, param) not in band_params:
                    continue
                val = rec.params[param]
                if not isinstance(val, (int, float)):
                    continue
                n, ok = tally.get((rec.anomaly_type, param), (0, 0))
                inside = (lo - 1e-9) <= val <= (hi + 1e-9)
                tally[(rec.anomaly_type, param)] = (n + 1, ok + int(inside))

    print("\n── Calibration validation (realistic draws vs empirical p1–p99) ──")
    if not tally:
        print("  (no calibrated parameters were drawn in this bank)")
        return tally
    all_ok = True
    for (atype, param), (n, ok) in sorted(tally.items()):
        flag = "OK " if ok == n else "!! "
        all_ok &= (ok == n)
        print(f"  {flag}{atype:24}{param:22} {ok}/{n} in envelope")
    print(f"  {'PASS' if all_ok else 'FAIL'} — "
          f"{'all realistic draws within empirical envelope' if all_ok else 'some draws escaped the envelope; review the fit'}")
    return tally


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CORRELATION DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

class CorrelationDiscovery:
    def __init__(self, returns_wide: pd.DataFrame,
                 rolling_window: int = 120, seed: int = 42):
        self.returns = returns_wide   # (T x N) DataFrame
        self.window  = rolling_window
        self.seed    = seed

    def compute_full_sample(self, at_date=None) -> dict:
        R = self.returns
        if at_date is not None:
            idx = R.index[R.index <= at_date]
            if len(idx) > self.window:
                idx = idx[-self.window:]
            R = R.loc[idx]

        R = R.fillna(0.0)
        min_obs     = self.window // 2
        active_bool = (R.abs() > 1e-12).sum(axis=0) >= min_obs
        R           = R.loc[:, active_bool]

        all_cols    = self.returns.columns.tolist()
        active_cols = [all_cols.index(c) for c in R.columns if c in all_cols]

        T_w, N_act = R.shape

        if N_act < 3:
            return {"corr_matrix": np.eye(N_act),
                    "cluster_labels": np.ones(N_act, dtype=int),
                    "n_clusters": 1, "silhouette_score": 0.0,
                    "intra_cluster_pairs": [], "inter_cluster_pairs": [],
                    "active_cols": active_cols, "at_date": at_date}

        R_np = R.to_numpy(dtype=np.float64)
        q    = T_w / N_act
        R_c  = R_np - R_np.mean(axis=0)
        cov  = R_c.T @ R_c / max(T_w - 1, 1)
        cov_dn, _ = denoise_cov(cov, q)
        corr      = cov2corr(cov_dn)

        dist_mat  = np.sqrt(np.maximum(0.5 * (1 - corr), 0.0))
        np.fill_diagonal(dist_mat, 0.0)

        try:
            linkage = sch.ward(squareform(dist_mat, checks=False))
        except Exception:
            return {"corr_matrix": corr,
                    "cluster_labels": np.ones(N_act, dtype=int),
                    "n_clusters": 1, "silhouette_score": 0.0,
                    "intra_cluster_pairs": [], "inter_cluster_pairs": [],
                    "active_cols": active_cols, "at_date": at_date}

        k_range    = range(2, min(21, max(N_act // 5, 3)))
        best_k     = 2
        best_score = -np.inf

        for k in k_range:
            labels = sch.fcluster(linkage, k, criterion="maxclust")
            if len(np.unique(labels)) < 2:
                continue
            try:
                score = silhouette_score(dist_mat, labels, metric="precomputed")
                if score > best_score:
                    best_k, best_score = k, score
            except Exception:
                pass

        labels = sch.fcluster(linkage, best_k, criterion="maxclust")

        intra = [(i, j) for i in range(N_act) for j in range(i + 1, N_act)
                 if labels[i] == labels[j]]
        inter = [(i, j) for i in range(N_act) for j in range(i + 1, N_act)
                 if labels[i] != labels[j] and abs(corr[i, j]) < 0.2]

        return {"corr_matrix": corr, "cluster_labels": labels,
                "n_clusters": best_k, "silhouette_score": float(best_score),
                "intra_cluster_pairs": intra, "inter_cluster_pairs": inter,
                "active_cols": active_cols, "at_date": at_date}

    def compute_rolling(self, dates: list) -> dict:
        return {d: self.compute_full_sample(at_date=d) for d in dates}


# ─────────────────────────────────────────────────────────────────────────────
# 10.  SCENARIO COMPOSER + select_curves
# ─────────────────────────────────────────────────────────────────────────────

def select_curves(policy: dict, corr_info: dict, N: int,
                  rng: np.random.Generator, params: dict = None,
                  restrict_to: list = None) -> list:
    mode = policy["mode"]

    if mode == "random_subset":
        # Bucket-coherent (a1): when restrict_to is given, sample the subset
        # from within that single tf bucket so the drawn params (fit on that
        # bucket) match the curves receiving them.
        pool = restrict_to if restrict_to else list(range(N))
        if not pool:
            pool = list(range(N))
        n_min, n_max = policy["n_curves_range"]
        n = int(rng.integers(n_min, min(n_max, len(pool)) + 1))
        return list(rng.choice(pool, size=n, replace=False).tolist())

    if mode == "correlated_cluster":
        labels        = corr_info["cluster_labels"]
        unique_labels = np.unique(labels)
        sizes  = np.array([np.sum(labels == k) for k in unique_labels], float)
        probs  = sizes / sizes.sum()
        cid    = unique_labels[int(rng.choice(len(unique_labels), p=probs))]
        local  = list(np.where(labels == cid)[0])
        ac     = corr_info["active_cols"]

        # Cap to at most 10 members to keep the active_mask feasible.
        # Pick the members most correlated to the cluster centroid so the
        # decorrelation test targets the tightest sub-group in the cluster.
        max_curves = policy.get("max_curves", 10)
        if len(local) > max_curves:
            corr = corr_info["corr_matrix"]
            # Mean correlation of each member to all other cluster members
            mean_corr = np.array([
                corr[np.ix_([i], local)].mean() for i in local
            ])
            top_idx = np.argsort(mean_corr)[::-1][:max_curves]
            local   = [local[i] for i in top_idx]

        return [ac[i] for i in local if i < len(ac)]

    if mode == "inter_cluster_sample":
        n_groups = int((params or {}).get("n_groups", 2))
        labels   = corr_info["cluster_labels"]
        unique   = np.unique(labels)
        chosen   = rng.choice(unique, size=min(n_groups, len(unique)), replace=False)
        ac       = corr_info["active_cols"]
        curves   = []
        for cid in chosen:
            members = np.where(labels == cid)[0]
            li = int(rng.choice(members))
            if li < len(ac):
                curves.append(ac[li])
        return curves

    if mode == "all":
        return list(range(N))

    raise ValueError(f"Unknown targeting mode: {mode!r}")


class ScenarioComposer:
    def __init__(self, returns_wide: pd.DataFrame, zero_mask: pd.DataFrame,
                 corr_discovery: CorrelationDiscovery, seed: int,
                 calibration: "Calibration" = None):
        self.returns   = returns_wide.to_numpy().astype(np.float64)
        self.zero_mask = zero_mask.to_numpy().astype(bool)
        self.tickers   = returns_wide.columns.tolist()
        self.T, self.N = self.returns.shape
        self.corr_disc = corr_discovery
        self.base_seed = seed
        self.calibration = calibration
        print("  Computing correlation structure...", flush=True)
        self.corr_info = corr_discovery.compute_full_sample()
        print(f"  n_clusters={self.corr_info['n_clusters']}  "
              f"silhouette={self.corr_info['silhouette_score']:.3f}", flush=True)

    def _choose_bucket(self, rng) -> str:
        """Pick a tf bucket weighted by the number of calibratable curves in it."""
        items = [(tf, len(idx)) for tf, idx in self.calibration.buckets.items() if idx]
        if not items:
            return "_pooled"
        tfs, sizes = zip(*items)
        p = np.asarray(sizes, float)
        p = p / p.sum()
        return tfs[int(rng.choice(len(tfs), p=p))]

    def compose_scenario(self, anomaly_specs: list, scenario_seed: int,
                         density: str = "realistic") -> dict:
        perturbed = self.returns.copy()
        rng       = np.random.default_rng(scenario_seed)
        records   = []

        anomaly_specs = sorted(
            anomaly_specs,
            key=lambda s: (
                LAYER_ORDER.index(ANOMALY_REGISTRY[s["type"]].layer),
                ANOMALY_REGISTRY[s["type"]].layer_priority,
            )
        )

        for spec in anomaly_specs:
            atype      = spec["type"]
            injector   = ANOMALY_REGISTRY[atype]()
            mode       = injector.targeting_policy["mode"]
            used_specs = injector.param_specs

            # Resolve (curves, params) with params drawn from the bucket-
            # appropriate calibrated band. Ordering differs by targeting mode:
            #   random_subset -> bucket chosen first, curves sampled within it
            #   cluster/all   -> curves chosen on correlation, bucket = their
            #                    majority tf (a1); a provisional draw supplies
            #                    n_groups so targeting & injection stay consistent
            if spec["params"] is not None:                      # explicit override
                params = spec["params"]
                curves = select_curves(injector.targeting_policy, self.corr_info,
                                       self.N, rng, params=params)

            elif self.calibration is None:                      # uncalibrated path
                params = ParameterSampler.sample(injector.param_specs, rng)
                curves = select_curves(injector.targeting_policy, self.corr_info,
                                       self.N, rng, params=params)

            elif mode == "random_subset":
                tf         = self._choose_bucket(rng)
                pool       = self.calibration.buckets.get(tf, [])
                curves     = select_curves(injector.targeting_policy,
                                           self.corr_info, self.N, rng,
                                           restrict_to=pool)
                used_specs = self.calibration.spec_for(atype, tf, density)
                params     = ParameterSampler.sample(used_specs, rng)

            else:                                               # cluster / all
                prov   = ParameterSampler.sample(
                            self.calibration.spec_for(atype, "_pooled", density), rng)
                curves = select_curves(injector.targeting_policy, self.corr_info,
                                       self.N, rng, params=prov)
                tf         = self.calibration.majority_tf(curves)
                used_specs = self.calibration.spec_for(atype, tf, density)
                params     = ParameterSampler.sample(used_specs, rng)
                if "n_groups" in prov:      # keep selection & injection consistent
                    params["n_groups"] = prov["n_groups"]

            if not curves:
                continue

            # any: at least one selected curve is active on that day.
            # apply() handles per-curve activity via _active_days internally,
            # so the all() constraint here causes regime/cross-curve types
            # (large curve sets) to never find a valid window.
            active_mask = (~self.zero_mask[:, curves]).any(axis=1)
            req_len     = injector.required_length(params)

            try:
                window = injector.sample_window(self.T, active_mask, req_len, rng)
            except ValueError:
                continue

            perturbed = injector.apply(
                perturbed, self.zero_mask, curves, window, params, rng
            )

            records.append(AnomalyRecord(
                anomaly_id      = str(uuid4()),
                anomaly_type    = atype,
                layer           = injector.layer,
                affected_curves = curves,
                window          = window,
                params          = _to_serialisable(params),
                param_specs     = _serialise_param_specs(used_specs),
            ))

        return {
            "perturbed":  perturbed,
            "original":   self.returns,
            "delta":      perturbed - self.returns,
            "zero_mask":  self.zero_mask,
            "tickers":    self.tickers,
            "anomalies":  records,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 11.  SCENARIO SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

class ScenarioSampler:
    LAYER_WEIGHTS = {
        "realistic":   {"univariate": 0.50, "cross_curve": 0.25,
                        "regime":     0.15, "pattern":     0.10},
        "adversarial": {"univariate": 0.40, "cross_curve": 0.30,
                        "regime":     0.20, "pattern":     0.10},
    }
    LAYER_TYPES = {
        "univariate":  ["vol_spike", "vol_cluster_burst", "persistent_vol_shift",
                        "artificial_drawdown", "drift_injection", "trend_reversal",
                        "heavy_tail_sub", "ar1_injection", "merton_jump"],
        "cross_curve": ["decorrelation", "contagion", "sync_drawdown_recovery"],
        "regime":      ["vol_regime_swap", "drawdown_recovery_var"],
        "pattern":     ["regime_persistence", "oscillating_pattern"],
    }

    def __init__(self, density: str = "realistic", seed: int = 42,
                 enabled_types: list = None):
        self.density = density
        self.preset  = DENSITY_PRESETS[density]
        self.weights = self.LAYER_WEIGHTS[density]
        self.rng     = np.random.default_rng(seed)

        # Realistic bank: drop the estimator-probe types entirely. This is the
        # gate that makes "realistic" a property of the bank rather than a
        # label on it — previously every pattern type was drawn at the same
        # 10% layer weight in both banks.
        layer_types = {
            layer: [t for t in types
                    if density == "adversarial" or t not in ADVERSARIAL_ONLY_TYPES]
            for layer, types in self.LAYER_TYPES.items()
        }

        if enabled_types is not None:
            enabled = set(enabled_types)
            layer_types = {
                layer: [t for t in types if t in enabled]
                for layer, types in layer_types.items()
            }

        self._layer_types = {k: v for k, v in layer_types.items() if v}

    def sample_scenario_spec(self, n_anomalies: int = None) -> list:
        n = n_anomalies if n_anomalies is not None \
            else int(self.rng.integers(1, self.preset["max_concurrent"] + 1))

        available = list(self._layer_types.keys())
        if not available:
            return []

        # At least one univariate anomaly
        anchor = "univariate" if "univariate" in self._layer_types else available[0]
        sampled_layers = [anchor]

        if n > 1:
            total_w = sum(self.weights.get(k, 0.1) for k in available)
            probs   = [self.weights.get(k, 0.1) / total_w for k in available]
            extra   = list(self.rng.choice(available, size=n - 1,
                                            p=probs, replace=True))
            sampled_layers += extra

        types = [
            self._layer_types[layer][
                int(self.rng.integers(0, len(self._layer_types[layer])))
            ]
            for layer in sampled_layers
        ]
        return [{"type": t, "params": None} for t in types]

    def generate_bank(self, n_scenarios: int,
                      composer: ScenarioComposer) -> list:
        results = []
        for i in range(n_scenarios):
            spec = self.sample_scenario_spec()
            if not spec:
                continue
            seed   = int(self.rng.integers(0, 2**31))
            result = composer.compose_scenario(spec, scenario_seed=seed,
                                               density=self.density)
            result["_seed"]    = seed
            result["_density"] = self.density
            results.append(result)
            if (i + 1) % 50 == 0:
                print(f"    generated {i+1}/{n_scenarios}", flush=True)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 12.  SCENARIO BANK
# ─────────────────────────────────────────────────────────────────────────────

class ScenarioBank:
    def __init__(self, registry: RunRegistry, data_run_id: str):
        self.registry    = registry
        self.data_run_id = data_run_id
        self._con        = get_connection()
        self._ensure_tables()

    def _ensure_tables(self):
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS scenarios (
                scenario_id   VARCHAR PRIMARY KEY,
                engine_run_id VARCHAR,
                data_run_id   VARCHAR,
                seed          INTEGER,
                density_mode  VARCHAR,
                n_anomalies   INTEGER,
                created_at    TIMESTAMP
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_instances (
                anomaly_id     VARCHAR PRIMARY KEY,
                scenario_id    VARCHAR,
                anomaly_type   VARCHAR,
                layer          VARCHAR,
                layer_priority INTEGER,
                window_start   INTEGER,
                window_end     INTEGER,
                params         VARCHAR,
                param_specs    VARCHAR,
                targeting_mode VARCHAR
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_affected_curves (
                anomaly_id VARCHAR,
                curve_idx  INTEGER
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS scenario_deltas (
                scenario_id VARCHAR,
                curve_idx   INTEGER,
                ticker      VARCHAR,
                t_idx       INTEGER,
                delta_value DOUBLE
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS scenario_snapshots (
                scenario_id      VARCHAR,
                curve_idx        INTEGER,
                ticker           VARCHAR,
                original_vol     DOUBLE,
                perturbed_vol    DOUBLE,
                vol_ratio        DOUBLE,
                original_sharpe  DOUBLE,
                perturbed_sharpe DOUBLE,
                sharpe_delta     DOUBLE,
                max_dd_original  DOUBLE,
                max_dd_perturbed DOUBLE
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS model_stress_results (
                result_id     VARCHAR PRIMARY KEY,
                engine_run_id VARCHAR,
                scenario_id   VARCHAR,
                model_id      VARCHAR,
                model_run_id  VARCHAR,
                metric        VARCHAR,
                value         DOUBLE,
                evaluated_at  TIMESTAMP
            )
        """)

    def store_scenario(self, scenario_result: dict, engine_run_id: str,
                       seed: int, density_mode: str) -> str:
        sid       = str(uuid4())
        anomalies = scenario_result["anomalies"]
        delta     = scenario_result["delta"]
        original  = scenario_result["original"]
        perturbed = scenario_result["perturbed"]
        zero_mask = scenario_result["zero_mask"]
        tickers   = scenario_result.get("tickers",
                        [f"c{i}" for i in range(delta.shape[1])])

        ann = np.sqrt(252)

        # Build all rows before touching the DB so the transaction is short.
        ai_rows = []
        ac_rows = []
        for rec in anomalies:
            ai_rows.append([
                rec.anomaly_id, sid, rec.anomaly_type, rec.layer,
                ANOMALY_REGISTRY[rec.anomaly_type].layer_priority,
                rec.window.start_idx, rec.window.end_idx,
                json.dumps(rec.params), json.dumps(rec.param_specs),
                ANOMALY_REGISTRY[rec.anomaly_type].targeting_policy["mode"],
            ])
            ac_rows.extend([rec.anomaly_id, int(ci)] for ci in rec.affected_curves)

        # Vectorised delta + snapshot assembly (avoids per-element Python loop).
        delta_rows    = []
        snapshot_rows = []
        nz_cols       = np.where(np.any(delta != 0, axis=0))[0]
        for ci in nz_cols:
            col_d  = delta[:, ci]
            nz     = np.flatnonzero(col_d)
            ticker = tickers[ci] if ci < len(tickers) else f"c{ci}"
            ci_int = int(ci)

            vals   = col_d[nz]
            delta_rows.extend(
                zip([sid] * len(nz), [ci_int] * len(nz),
                    [ticker] * len(nz), nz.tolist(), vals.tolist())
            )

            orig_r = original[nz, ci]
            pert_r = perturbed[nz, ci]
            # A handful of perturbed days (vol_spike/merton_jump can touch as
            # few as 1-3) makes np.std(orig_r) a noisy, occasionally
            # near-zero baseline — the same pathology _local_vol already
            # guards against. Reuse it here instead of the raw few-point std.
            o_vol  = _local_vol(original, zero_mask, ci, int(nz[0])) * ann
            p_vol  = float(np.std(pert_r) * ann) if len(pert_r) > 1 else 0.0
            v_rat  = p_vol / o_vol if o_vol > 1e-6 else 1.0
            o_sr   = _annualised_sharpe(orig_r)
            p_sr   = _annualised_sharpe(pert_r)
            snapshot_rows.append((
                sid, ci_int, ticker,
                o_vol, p_vol, v_rat,
                o_sr, p_sr, p_sr - o_sr,
                float(_max_drawdown(orig_r)), float(_max_drawdown(pert_r)),
            ))

        # Single transaction: all inserts commit together or not at all.
        self._con.execute("BEGIN")
        try:
            self._con.execute(
                "INSERT INTO scenarios VALUES (?, ?, ?, ?, ?, ?, ?)",
                [sid, engine_run_id, self.data_run_id,
                 seed, density_mode, len(anomalies), datetime.now()]
            )
            if ai_rows:
                self._con.executemany(
                    "INSERT INTO anomaly_instances VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ai_rows
                )
            if ac_rows:
                self._con.executemany(
                    "INSERT INTO anomaly_affected_curves VALUES (?, ?)", ac_rows
                )
            if delta_rows:
                self._con.executemany(
                    "INSERT INTO scenario_deltas VALUES (?, ?, ?, ?, ?)",
                    delta_rows
                )
            if snapshot_rows:
                self._con.executemany(
                    "INSERT INTO scenario_snapshots VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", snapshot_rows
                )
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise

        return sid

    def quick_view(self, scenario_id: str) -> pd.DataFrame:
        return self._con.execute(
            "SELECT * FROM scenario_snapshots WHERE scenario_id = ?",
            [scenario_id]
        ).df()

    def list_scenarios(self, filters: dict = None) -> pd.DataFrame:
        # scenarios/anomaly_instances accumulate across every past run
        # (CREATE TABLE IF NOT EXISTS never clears old rows), so
        # engine_run_id is included here for callers who need to scope to
        # one run rather than the table's entire history.
        df = self._con.execute("""
            SELECT s.scenario_id, s.engine_run_id, s.density_mode,
                   s.n_anomalies, s.created_at,
                   ai.anomaly_type, ai.layer, ai.window_start, ai.window_end
            FROM   scenarios s
            LEFT JOIN anomaly_instances ai ON s.scenario_id = ai.scenario_id
        """).df()
        if filters and df is not None and not df.empty:
            for k, v in filters.items():
                if k in df.columns:
                    df = df[df[k] == v]
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 14.  STRESS TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class StressTestRunner:
    DEFAULT_METRICS = ["sharpe", "max_dd", "turnover", "calmar", "tracking_error"]

    def __init__(self, bank: ScenarioBank, registry: RunRegistry):
        self.bank     = bank
        self.registry = registry
        self._con     = get_connection()

    @staticmethod
    def weekly_rebalance_dates(T: int, zero_mask: np.ndarray,
                                dates_index=None) -> list:
        if dates_index is not None:
            rebalance = []
            for i, d in enumerate(dates_index):
                if d.weekday() == 4:
                    for j in range(i, max(i - 7, -1), -1):
                        if not zero_mask[j, :].all():
                            rebalance.append(j)
                            break
            return sorted(set(rebalance))
        return list(range(4, T, 5))

    def run(self, model_fn, model_id: str, model_run_id: str = None,
            scenario_ids: list = None, metrics: list = None,
            engine_run_id: str = None):
        if scenario_ids is None:
            df = self.bank.list_scenarios()
            if df is None or df.empty:
                print("No scenarios in bank.")
                return
            scenario_ids = df["scenario_id"].unique().tolist()

        metrics = metrics or self.DEFAULT_METRICS

        rows = []
        for sid in scenario_ids:
            # ScenarioBank.reconstruct_scenario is left to run_stress_test.py
            # because it needs model-side data loading. Instead, model_fn
            # receives the result dict directly if passed through generate_bank.
            pass

        if rows:
            self._con.executemany(
                "INSERT INTO model_stress_results VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            df = pd.DataFrame(rows,
                              columns=["result_id", "engine_run_id", "scenario_id",
                                       "model_id", "model_run_id", "metric",
                                       "value", "evaluated_at"])
            print(f"\n── Stress Test: {model_id} {'─'*40}")
            for m in metrics:
                vals = df[df["metric"] == m]["value"].values
                if len(vals):
                    print(f"  {m:<25} mean={np.mean(vals):>8.3f}  "
                          f"std={np.std(vals):>7.3f}  "
                          f"p5={np.percentile(vals,5):>8.3f}  "
                          f"p95={np.percentile(vals,95):>8.3f}  "
                          f"n={len(vals)}")

    def store_result(self, engine_run_id: str, scenario_id: str,
                     model_id: str, result: dict,
                     metrics: list = None, model_run_id: str = None):
        """Store a single model_fn result dict for one scenario."""
        metrics = metrics or self.DEFAULT_METRICS
        rows    = []
        for metric, value in result.items():
            if metric not in metrics:
                continue
            rows.append([str(uuid4()), engine_run_id, scenario_id,
                         model_id, model_run_id or "",
                         metric, float(value), datetime.now()])
        if rows:
            self._con.executemany(
                "INSERT INTO model_stress_results VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)", rows
            )


# ─────────────────────────────────────────────────────────────────────────────
# 15.  MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main(
    data_run_id:       str   = None,
    n_scenarios_core:  int   = 1200,
    n_scenarios_adv:   int   = 700,
    rolling_window:    int   = 24 * WEEK_DAYS,
    seed:              int   = 42,
    enabled_types:     list  = None,
    recalibrate:       bool  = False,
    validate:          bool  = False,
) -> tuple:
    """
    Build the scenario bank and return (bank, composer, engine_run_id).

    Parameters
    ----------
    enabled_types : list[str] | None
        Restrict sampling to this subset of anomaly type names.
        None → all 16 types.
        Pass MINIMAL_TEST_SET for a one-per-layer smoke test.

    Returns
    -------
    bank            : ScenarioBank
    composer        : ScenarioComposer  (holds tickers / T / N)
    engine_run_id   : str
    """
    registry = RunRegistry()
    con      = get_connection()

    # ── Resolve data_run_id ───────────────────────────────────────────────────
    if data_run_id is None:
        row = con.execute("""
            SELECT run_id FROM runs
            WHERE json_extract_string(labels, '$.type') = 'algo_data_pull'
            ORDER BY created_at DESC LIMIT 1
        """).fetchone()
        if row is None:
            raise RuntimeError("No algo_data_pull run found in registry.")
        data_run_id = row[0]

    print(f"\n── Scenario Engine ──────────────────────────────────────────────")
    print(f"  data_run_id     : {data_run_id}")
    print(f"  n_core          : {n_scenarios_core}")
    print(f"  n_adversarial   : {n_scenarios_adv}")
    print(f"  enabled_types   : {enabled_types or 'all'}")

    # ── Load returns ──────────────────────────────────────────────────────────
    # Algo-only universe: fixed-notional arithmetic returns (ret = pnl / E_0),
    # not log returns — see pull_algo_data.py for why log-return-on-equity
    # doesn't apply to these accounts. CorrelationDiscovery/ScenarioComposer
    # below operate on a generic (T x N) returns matrix and don't care which
    # return convention produced it.
    print("\n  Loading returns...", flush=True)
    algo = con.execute(
        'SELECT "Date" AS date, key AS ticker, ret FROM algo_returns_daily WHERE run_id = ?',
        [data_run_id]
    ).df()
    returns_all = (algo.pivot(index="date", columns="ticker", values="ret")
                       .sort_index())
    returns_all.index = pd.to_datetime(returns_all.index)

    zero_mask   = returns_all.isna() | (returns_all.abs() < 1e-12)
    returns_all = returns_all.fillna(0.0)

    print(f"  Shape           : {returns_all.shape}  "
          f"(T={returns_all.shape[0]}, N={returns_all.shape[1]})")

    # ── Timeframe buckets + empirical calibration ─────────────────────────────
    # tf_of[j] is the timeframe of the curve in column j, aligned to the panel
    # column order (== composer.tickers). Calibration bands are keyed by tf.
    meta = con.execute(
        'SELECT key, tf FROM algo_meta WHERE run_id = ?', [data_run_id]
    ).df()
    tf_lookup = (dict(zip(meta["key"], meta["tf"]))
                 if meta is not None and not meta.empty else {})
    tf_of = [str(tf_lookup.get(k, "UNK")) for k in returns_all.columns]

    calibration = resolve_or_build_calibration(
        con, registry, data_run_id,
        returns_all.to_numpy(dtype=float),
        zero_mask.to_numpy(dtype=bool),
        tf_of,
        recalibrate=recalibrate,
    )

    # ── Register engine run ───────────────────────────────────────────────────
    engine_run_id = registry.start_run(
        type           = "scenario_engine",
        parent_run_id  = data_run_id,
        n_core         = n_scenarios_core,
        n_adv          = n_scenarios_adv,
        seed           = seed,
        enabled_types  = json.dumps(enabled_types or []),
    )

    # ── Build composer ──────────────────────────────────────────────────────
    corr_disc = CorrelationDiscovery(returns_all, rolling_window, seed)
    composer  = ScenarioComposer(returns_all, zero_mask, corr_disc, seed,
                                 calibration=calibration)
    bank      = ScenarioBank(registry, data_run_id)

    # ── Core bank ────────────────────────────────────────────────────────────
    print(f"\n  Generating {n_scenarios_core} core (realistic) scenarios...",
          flush=True)
    sampler_core = ScenarioSampler(density="realistic", seed=seed,
                                   enabled_types=enabled_types)
    core = sampler_core.generate_bank(n_scenarios_core, composer)

    print(f"\n  Generating {n_scenarios_adv} adversarial scenarios...",
          flush=True)
    sampler_adv = ScenarioSampler(density="adversarial", seed=seed + 1,
                                   enabled_types=enabled_types)
    adv = sampler_adv.generate_bank(n_scenarios_adv, composer)

    # ── Validate ───────────────────────────────────────────────────────────────
    if validate and calibration is not None:
        validate_calibration(core, calibration)

    # ── Store ────────────────────────────────────────────────────────────────
    print("\n  Storing scenarios...", flush=True)
    for result in core:
        bank.store_scenario(result, engine_run_id,
                            result.get("_seed", 0), "realistic")
    for result in adv:
        bank.store_scenario(result, engine_run_id,
                            result.get("_seed", 0), "adversarial")

    # ── Summary ──────────────────────────────────────────────────────────────
    registry.log_metrics(engine_run_id,
                         n_core=n_scenarios_core,
                         n_adversarial=n_scenarios_adv)
    registry.end_run(engine_run_id)

    df_summary = bank.list_scenarios(filters={"engine_run_id": engine_run_id})
    if df_summary is not None and not df_summary.empty:
        print("\n  Scenarios by anomaly type:")
        print(df_summary.groupby(["layer", "anomaly_type"])
                        .size()
                        .rename("count")
                        .to_string())

    print(f"\n── Done — engine_run_id: {engine_run_id}")
    return bank, composer, engine_run_id


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scenario injection engine for portfolio stress testing"
    )
    parser.add_argument("--data-run-id",       type=str, default=None)
    parser.add_argument("--n-scenarios-core",  type=int, default=1200)
    parser.add_argument("--n-scenarios-adv",   type=int, default=700)
    parser.add_argument("--rolling-window",    type=int, default=24 * WEEK_DAYS)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument(
        "--recalibrate", action="store_true",
        help="Force a refit of the empirical calibration even if one is already "
             "persisted for this data run (otherwise the stored fit is reused).",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="After building the bank, check that every realistic draw lands "
             "inside the empirical p1–p99 envelope, and print a per-type report.",
    )
    parser.add_argument(
        "--enabled-types", type=str, nargs="*", default=None,
        metavar="TYPE",
        help=(
            "Restrict sampling to these anomaly types. "
            f"Defaults to all types. "
            f"Minimal test set: {MINIMAL_TEST_SET}"
        ),
    )
    args = parser.parse_args()

    main(
        data_run_id      = args.data_run_id,
        n_scenarios_core = args.n_scenarios_core,
        n_scenarios_adv  = args.n_scenarios_adv,
        rolling_window   = args.rolling_window,
        seed             = args.seed,
        enabled_types    = args.enabled_types,
        recalibrate      = args.recalibrate,
        validate         = args.validate,
    )
