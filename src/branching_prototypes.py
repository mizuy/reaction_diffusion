#!/usr/bin/env python3
"""Prototype mechanisms that give Gray-Scott pits an intrinsic finite length and
tip splitting, so that an isolated pit grows into a small branched *tree*
instead of an endless stripe.

Baseline (unchanged, same kernel / dt=1 / dA, dB as ``sweep_video``):

    A_t = dA Lap(A) - A B^2 + f (1 - A)
    B_t = dB Lap(B) + A B^2 - (k + f) B

Each mechanism adds at most one extra field and one or two coupling terms.
"How does the tip learn how long it already is?" is answered per mechanism:

``none``  baseline.

``H``     slow, far-ranging inhibitor secreted by the pit (mass / size limit):
              H_t = dH Lap(H) + delta (rho B - H),   k_eff = k + gamma H
          H relaxes to B smoothed over sqrt(dH/delta).  The tip sits inside the
          H cloud of its own trunk; the cloud grows with the trunk's mass and
          reaches the tip because sqrt(dH/delta) >= L*.

``N``     slowly recovering nutrient consumed by the pit (depletion version of H):
              N_t = dN Lap(N) + rho (1 - N) - kappa B N,   f_eff = f (1 - c (1 - N))
          The trunk has eaten the nutrient around it; the tip only sees fresh
          tissue within sqrt(dN/rho) of the trunk, so a long trunk starves it.

``M``     mean-field coverage cap (no extra field):
              k_eff = k + gamma max(0, <B> - phi*)
          The tip does not know its own length; it knows the total coverage,
          which grows as every pit elongates, and growth is frozen when the
          field-wide cap is hit (frozen transient).

``S``     root resource transported along the pit (needs a static root mask):
              root(x) = B(x, t_on) at the moment the mechanism is switched on
              S_t = div(dS B grad S) + rho root - delta S,   g = S/(S + S0)
              coupling (s_mode): reaction * g  and/or  k_eff = k + s_kill (1-g) - s_widen (1-g)
          S leaks along the B channel from the root and decays, so its value at
          the tip falls off as exp(-L/sqrt(dS/delta)): the tip reads its own
          distance from the root.

``--grow`` (G) growing domain: the grid is resized linearly between two steps
          (all fields re-interpolated), giving the pits room to spread out.

``--sweep`` embeds the mechanism in the v2 path sweep of ``sweep_video`` and
          reports the phase at the end of every dwell (spots / worm / labyrinth / holes).

Examples:
    # dense protocol: spots -> ramp to worms with the root resource S
    python3 src/branching_prototypes.py --mech S --fk 0.026,0.061 --fk1 0.036,0.059 \
        --t-on 3000 --ramp-steps 4000 --seeds dense --steps 30000 \
        --set s_mode=kill --set s_kill=0.004 --set s_delta=0.0005 --out /tmp/proto/S
    # same, on a domain growing from 64^2 to 128^2
    python3 src/branching_prototypes.py --mech S --size 64 --grow 128,3000,23000 ...
    # sweep embedding check
    python3 src/branching_prototypes.py --mech S --sweep 60000 --seeds dense \
        --set s_mode=kill --sweep-on 0.7 --sweep-off 0.9 --out /tmp/proto/sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_pit_steps import (  # noqa: E402
    KERNEL,
    make_initial_state,
    prepare_skeleton_topology,
    skeleton_endpoints,
    zhang_suen_thin,
)


@dataclass
class ProtoConfig:
    mech: str = "none"
    size: int = 128
    steps: int = 12000
    f: float = 0.036
    k: float = 0.059
    d_a: float = 1.0
    d_b: float = 0.5
    noise: float = 0.0
    seed: int = 7
    # seeds: isolated (1 blob), sparse (n blobs on a jittered grid), dense (repo IC), hex (relaxed spots)
    seeds: str = "isolated"
    n_seeds: int = 9
    seed_radius: int = 3
    # mechanism switch-on step (fields are initialised to their B=0 equilibrium before)
    t_on: int = 0
    # H
    d_h: float = 1.0
    h_rho: float = 1.0
    h_delta: float = 0.002
    h_gamma: float = 0.02
    # N
    d_n: float = 1.0
    n_rho: float = 0.001
    n_kappa: float = 0.02
    n_c: float = 0.3
    # M
    m_gamma: float = 0.05
    m_phi: float = 0.12
    # S
    d_s: float = 0.2  # channel conductance; explicit stability needs d_s <= 0.25
    s_rho: float = 0.05
    s_delta: float = 0.001  # screening length along the channel: sqrt(d_s/s_delta) ~ 14 px
    s_zero: float = 0.2  # half-saturation of g as a fraction of the root level s_rho/s_delta
    s_b_ref: float = 0.25
    s_chi0: float = 0.0  # background conductance of the S channel (halo); 0 = strictly inside B
    s_hill: int = 0  # 0: chi = min(1, B/b_ref) (leaky); n>0: chi = B^n/(B^n + b_ref^n) (selective channel)
    # root mask threshold on B at t_on.  0.3 = spot cores only (S then spreads over the whole
    # spot footprint and equilibrates at ~0.2 rho/delta, i.e. g ~ 0.5 inside the pit); 0.1 = whole footprint.
    s_root_thr: float = 0.3
    # Dirichlet root: S is clamped to rho/delta on the root every step (constant-supply pit mouth),
    # so S(L) = (rho/delta) exp(-L/l_S) along a channel with no dilution by the total B area.
    s_dirichlet: bool = False
    s_root_dilate: int = 0  # dilate the root mask by this many px (keeps a spot fed while it settles)
    s_root_keep: float = 1.0  # keep only this fraction of the root components (sparse-root demo device)
    # coupling of g = S/(S+S0): "react" -> reaction * (g_min + (1-g_min) g) ; "kill" -> k += s_kill (1-g) ;
    # "widen" -> k -= s_widen g (1-g)  (combine with '+', e.g. "kill+widen")
    s_mode: str = "kill"
    s_kill: float = 0.004
    s_widen: float = 0.002
    s_feed: float = 0.010
    s_gmin: float = 0.5
    # stage-2 couplings (S -> transport / activator side)
    s_beta_b: float = 1.0  # "dbeta":  d_B_eff = d_B (1 + beta_b (1-g)); need d_B (1+beta_b) <= 1
    s_beta_a: float = 0.0  # "dabeta": d_A_eff = d_A (1 + beta_a (1-g)); need d_A (1+beta_a) <= 1 -> use d_a<1
    s_boost: float = 0.5  # "boost":  A B^2 (1 + boost (1-g))
    s_cubic: float = 1.0  # "cubic":  A B^2 (1 + cubic (1-g) B)
    # optional linear ramp of (f,k) from (f,k) to (f1,k1), starting at t_on and
    # lasting ramp_steps (0 -> until the end of the run); afterwards hold (f1,k1)
    f1: float | None = None
    k1: float | None = None
    ramp_steps: int = 0
    # G: growing domain.  The grid is resized linearly from `size` to `g_size1`
    # between steps g_t0 and g_t1 (all fields are re-interpolated).  0 = off.
    g_size1: int = 0
    g_t0: int = 0
    g_t1: int = 0
    # BARW-like hybrid (stage 3): tips are particles that deposit B; the RD (GS + S)
    # provides the stripe width and the starvation stop.  0 = off.
    tips_per_root: int = 0
    tip_v: float = 0.03  # px / step
    tip_pb: float = 0.002  # branching probability per step, multiplied by g at the tip
    tip_dtheta: float = 0.6  # half opening angle of a branching event (rad)
    tip_sigma: float = 0.05  # angular noise per step (rad)
    tip_gstop: float = 0.3  # tip is removed when g at the tip falls below this
    tip_r: float = 2.0  # deposit radius (px)
    tip_b: float = 0.5  # deposited B level (max with the field)
    tip_refractory: int = 600  # steps after a branching event during which a tip does not branch
    tip_max: int = 600  # global cap on the number of live tips
    t_tips: int = 0  # step at which the tips are spawned (0 = t_on)
    tip_max_path: float = 0.0  # tip is removed after travelling this path length (px); 0 = S gate only
    snapshots: int = 6


@dataclass
class State:
    a: np.ndarray
    b: np.ndarray
    x: np.ndarray  # extra field (H / N / S), zeros otherwise
    root: np.ndarray | None = None


def lap(u: np.ndarray) -> np.ndarray:
    # periodic boundaries (the repo uses reflect); keeps the Laplacian consistent with the
    # np.roll based transport terms and with the growing-domain resize
    padded = np.pad(u, 1, mode="wrap")
    return cv2.filter2D(padded, -1, KERNEL, borderType=cv2.BORDER_REPLICATE)[1:-1, 1:-1]


def channel_chi(b: np.ndarray, b_ref: float, hill: int) -> np.ndarray:
    """Channel conductance chi(B) in [0,1].  hill=0: min(1, B/b_ref) (leaky: the GS background
    B ~ 0.02-0.08 between spots still conducts 10-30%).  hill>=1: B^n/(B^n + b_ref^n) (selective)."""
    if hill <= 0:
        return np.minimum(1.0, b / b_ref)
    bn = b ** hill
    return bn / (bn + b_ref**hill)


def channel_diffusion(s: np.ndarray, b: np.ndarray, d: float, b_ref: float = 0.25, chi0: float = 0.0,
                      hill: int = 0) -> np.ndarray:
    """Conservative div(d chi(B) grad s), chi = max(chi0, channel_chi(B)), face-averaged (4-neighbour).

    chi0 > 0 lets S leak a few px beyond the B envelope (halo), so that the pit's own
    boundary is not read as "starved".  Explicit stability requires 4 d <= 1 (d <= 0.25)."""
    mob = d * np.maximum(chi0, channel_chi(b, b_ref, hill))
    out = np.zeros_like(s)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        s_n = np.roll(s, shift, axis=axis)
        m_n = np.roll(mob, shift, axis=axis)
        out += 0.5 * (mob + m_n) * (s_n - s)
    return out


_STENCIL9 = [(0, 1, 0.2), (0, -1, 0.2), (1, 0, 0.2), (-1, 0, 0.2),
             (1, 1, 0.05), (1, -1, 0.05), (-1, 1, 0.05), (-1, -1, 0.05)]


def div_grad(u: np.ndarray, mob: np.ndarray) -> np.ndarray:
    """Conservative div(mob grad u) on the same 9-point stencil as KERNEL (mob=const -> mob*lap(u)).

    Explicit stability requires max(mob) <= 1."""
    out = np.zeros_like(u)
    for dy, dx, w in _STENCIL9:
        u_n = np.roll(np.roll(u, dy, axis=0), dx, axis=1)
        m_n = np.roll(np.roll(mob, dy, axis=0), dx, axis=1)
        out += w * 0.5 * (mob + m_n) * (u_n - u)
    return out


def make_seeds(cfg: ProtoConfig) -> tuple[np.ndarray, np.ndarray]:
    n = cfg.size
    rng = np.random.default_rng(cfg.seed)
    if cfg.seeds == "dense":
        return make_initial_state(n, cfg.seed, 0.035)
    a = np.ones((n, n), dtype=np.float32)
    b = np.zeros((n, n), dtype=np.float32)
    if cfg.seeds == "isolated":
        centres = [(n // 2, n // 2)]
    elif cfg.seeds == "sparse":
        g = int(np.ceil(np.sqrt(cfg.n_seeds)))
        pitch = n / g
        centres = []
        for i in range(g):
            for j in range(g):
                if len(centres) >= cfg.n_seeds:
                    break
                cy = int((i + 0.5) * pitch + rng.uniform(-0.15, 0.15) * pitch)
                cx = int((j + 0.5) * pitch + rng.uniform(-0.15, 0.15) * pitch)
                centres.append((cy, cx))
    else:
        raise ValueError(cfg.seeds)
    for cy, cx in centres:
        cv2.circle(b, (cx, cy), cfg.seed_radius, 0.85, -1)
        cv2.circle(a, (cx, cy), cfg.seed_radius, 0.25, -1)
    a += 0.01 * rng.normal(0, 1, (n, n)).astype(np.float32)
    b += 0.01 * rng.normal(0, 1, (n, n)).astype(np.float32)
    return np.clip(a, 0, 1), np.clip(b, 0, 1)


def init_state(cfg: ProtoConfig) -> State:
    a, b = make_seeds(cfg)
    if cfg.mech == "N":
        x = np.ones_like(b)
    else:
        x = np.zeros_like(b)
    return State(a=a, b=b, x=x)


def step(st: State, cfg: ProtoConfig, f: float, k: float, active: bool, rng, gain: float = 1.0) -> State:
    """One explicit Euler step.  ``gain`` scales the S coupling amplitudes (sweep embedding)."""
    a, b, x = st.a, st.b, st.x
    f_eff: float | np.ndarray = f
    k_eff: float | np.ndarray = k
    react_gain: float | np.ndarray = 1.0
    mob_a: np.ndarray | None = None
    mob_b: np.ndarray | None = None
    next_x = x

    if active and cfg.mech == "H":
        # H relaxes to B smoothed over sqrt(d_h/h_delta): a slow "local coverage" field.
        k_eff = k + cfg.h_gamma * x
        next_x = x + cfg.d_h * lap(x) + cfg.h_delta * (cfg.h_rho * b - x)
    elif active and cfg.mech == "N":
        # Bounded effect: f is reduced by at most n_c when the nutrient is exhausted.
        f_eff = f * (1.0 - cfg.n_c * (1.0 - x))
        next_x = x + cfg.d_n * lap(x) + cfg.n_rho * (1.0 - x) - cfg.n_kappa * b * x
        next_x = np.clip(next_x, 0.0, 1.0)
    elif active and cfg.mech == "M":
        k_eff = k + cfg.m_gamma * max(0.0, float(b.mean()) - cfg.m_phi)
    elif active and cfg.mech == "S":
        s_root = cfg.s_rho / cfg.s_delta
        g = x / (x + cfg.s_zero * s_root)
        if "react" in cfg.s_mode:
            react_gain = 1.0 - gain * (1.0 - cfg.s_gmin) * (1.0 - g)
        if "kill" in cfg.s_mode:
            k_eff = k_eff + gain * cfg.s_kill * (1.0 - g)
        if "widen" in cfg.s_mode:
            # strongest at intermediate supply (the "half-fed" zone just behind the tip)
            k_eff = k_eff - gain * cfg.s_widen * 4.0 * g * (1.0 - g)
        if "feed" in cfg.s_mode:
            # starved tissue gets a higher feed -> wider tip -> tip splitting (coral-like)
            f_eff = f_eff + gain * cfg.s_feed * (1.0 - g)
        if "fedwide" in cfg.s_mode:
            # well-fed tissue (near the root) gets a higher feed -> wide, fingering base
            f_eff = f_eff + gain * cfg.s_feed * g
        if "boost" in cfg.s_mode:
            # activator side: starved tissue autocatalyses harder (AB^2 gain > 1 at the tip)
            react_gain = react_gain * (1.0 + gain * cfg.s_boost * (1.0 - g))
        if "cubic" in cfg.s_mode:
            # activator side: starved tissue gets an extra A B^3 term (steeper autocatalysis)
            react_gain = react_gain + gain * cfg.s_cubic * (1.0 - g) * b
        # starvation weighted by tissue presence: outside the B envelope S is always ~0,
        # so the pit's own boundary must not be read as starved tissue.
        starved = (1.0 - g) * np.minimum(1.0, b / cfg.s_b_ref)
        if "dbeta" in cfg.s_mode:
            # tip flattening: B diffuses faster (beta>0) or slower (beta<0) where starved
            mob_b = cfg.d_b * (1.0 + gain * cfg.s_beta_b * starved)
        if "dabeta" in cfg.s_mode:
            # Mullins-Sekerka style: the substrate A diffuses faster/slower where the tissue is starved
            mob_a = cfg.d_a * (1.0 + gain * cfg.s_beta_a * starved)
        next_x = (
            x
            + channel_diffusion(x, b, cfg.d_s, cfg.s_b_ref, cfg.s_chi0, cfg.s_hill)
            + cfg.s_rho * st.root
            - cfg.s_delta * x
        )
        next_x = np.clip(next_x, 0.0, 2.0 * s_root)
        if cfg.s_dirichlet:
            next_x = np.where(st.root > 0, s_root, next_x)

    ab2 = a * b * b * react_gain
    diff_a = div_grad(a, mob_a) if mob_a is not None else cfg.d_a * lap(a)
    diff_b = div_grad(b, mob_b) if mob_b is not None else cfg.d_b * lap(b)
    next_a = a + diff_a - ab2 + f_eff * (1.0 - a)
    next_b = b + diff_b + ab2 - (k_eff + f_eff) * b
    if cfg.noise > 0:
        next_b = next_b + cfg.noise * rng.standard_normal(b.shape, dtype=np.float32)
    return State(
        a=np.clip(next_a, 0, 1).astype(np.float32),
        b=np.clip(next_b, 0, 1).astype(np.float32),
        x=next_x.astype(np.float32),
        root=st.root,
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def binarize(b: np.ndarray) -> np.ndarray:
    if float(b.std()) < 0.01:
        return np.zeros_like(b, dtype=bool)
    u8 = np.clip(b * 255, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(thr), 40.0)
    return u8 >= thr


def tree_metrics(b: np.ndarray) -> dict[str, float]:
    fg = binarize(b)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= 6
    comps = int(keep.sum())
    out: dict[str, float] = {
        "coverage": float(fg.mean()),
        "components": comps,
        "largest_frac": float(areas[keep].max() / areas[keep].sum()) if comps else 0.0,
    }
    if comps == 0:
        out.update(skeleton_px=0, endpoints=0, junc3=0, trees=0, max_junc_tree=0, mean_arm=0.0, longest_skel=0)
        return out
    sk_raw = zhang_suen_thin(fg)
    sk, branch = prepare_skeleton_topology(sk_raw, prune_spurs=4, min_arm_length=5)
    nj, jlab = cv2.connectedComponents(branch.astype(np.uint8), connectivity=8)
    ends = skeleton_endpoints(sk)
    ns, slab, sstats, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), 8)
    skel_areas = sstats[1:, cv2.CC_STAT_AREA] if ns > 1 else np.array([0])
    # trees = skeleton components containing >= 1 three-arm junction;
    # max_junc_tree = most junction clusters in one component (>=3 -> at least 2 generations)
    trees = 0
    max_junc_tree = 0
    for lab in range(1, ns):
        jl = np.unique(jlab[(slab == lab) & branch])
        nj_here = int((jl > 0).sum())
        if nj_here:
            trees += 1
            max_junc_tree = max(max_junc_tree, nj_here)
    segs = sk & ~branch
    nseg, _, segstats, _ = cv2.connectedComponentsWithStats(segs.astype(np.uint8), 8)
    mean_arm = float(segstats[1:, cv2.CC_STAT_AREA].mean()) if nseg > 1 else 0.0
    out.update(
        skeleton_px=int(sk.sum()),
        endpoints=int(ends.sum()),
        junc3=int(nj - 1),
        trees=trees,
        max_junc_tree=max_junc_tree,
        mean_arm=mean_arm,
        longest_skel=int(skel_areas.max()),
    )
    return out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def render(b: np.ndarray, label: str, scale: int = 2) -> np.ndarray:
    img = np.clip(b / max(float(b.max()), 0.3) * 255, 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(img, (0, 0), (img.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(img, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return img


@dataclass
class Tip:
    y: float
    x: float
    theta: float
    gen: int = 0
    last_branch: int = 0
    path: float = 0.0


def spawn_tips(root: np.ndarray, cfg: ProtoConfig, rng, t: int) -> list[Tip]:
    n, labels, stats, cents = cv2.connectedComponentsWithStats(root.astype(np.uint8), 8)
    tips: list[Tip] = []
    for i in range(1, n):
        cx, cy = cents[i]
        th0 = rng.uniform(0, 2 * np.pi)
        for j in range(cfg.tips_per_root):
            tips.append(Tip(float(cy), float(cx), th0 + 2 * np.pi * j / cfg.tips_per_root, 0, t))
    return tips


def update_tips(tips: list[Tip], st: State, cfg: ProtoConfig, rng, t: int, s_root: float) -> tuple[list[Tip], int]:
    """Move / branch / stop the tip particles and deposit B.  Returns (tips, n_branch_events)."""
    n = st.b.shape[0]
    g_field = st.x / (st.x + cfg.s_zero * s_root)
    alive: list[Tip] = []
    births = 0
    for tp in tips:
        iy, ix = int(round(tp.y)) % n, int(round(tp.x)) % n
        g = float(g_field[iy, ix])
        if g < cfg.tip_gstop or (cfg.tip_max_path > 0 and tp.path >= cfg.tip_max_path):
            continue  # starved / exhausted: the tip stops (the RD keeps the deposited stripe)
        tp.theta += cfg.tip_sigma * rng.standard_normal()
        tp.y = (tp.y + cfg.tip_v * np.sin(tp.theta)) % n
        tp.x = (tp.x + cfg.tip_v * np.cos(tp.theta)) % n
        tp.path += cfg.tip_v
        if (t - tp.last_branch > cfg.tip_refractory and len(tips) + births < cfg.tip_max
                and rng.random() < cfg.tip_pb * g):
            births += 1
            child = Tip(tp.y, tp.x, tp.theta - cfg.tip_dtheta, tp.gen + 1, t, tp.path)
            tp.theta += cfg.tip_dtheta
            tp.gen += 1
            tp.last_branch = t
            alive.append(child)
        alive.append(tp)
    if alive:
        r = int(cfg.tip_r)
        dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
        disk = (dy * dy + dx * dx) <= cfg.tip_r * cfg.tip_r
        oy, ox = dy[disk], dx[disk]
        ys = (np.array([round(tp.y) for tp in alive], int)[:, None] + oy[None, :]) % n
        xs = (np.array([round(tp.x) for tp in alive], int)[:, None] + ox[None, :]) % n
        st.b[ys, xs] = np.maximum(st.b[ys, xs], cfg.tip_b)
        st.a[ys, xs] = np.minimum(st.a[ys, xs], 0.3)
    return alive, births


def domain_size(cfg: ProtoConfig, t: int) -> int:
    if cfg.g_size1 <= 0 or t <= cfg.g_t0:
        return cfg.size
    if t >= cfg.g_t1:
        return cfg.g_size1
    u = (t - cfg.g_t0) / max(cfg.g_t1 - cfg.g_t0, 1)
    return int(round(cfg.size + u * (cfg.g_size1 - cfg.size)))


def resize_state(st: State, n: int) -> State:
    rs = lambda u, interp: cv2.resize(u, (n, n), interpolation=interp).astype(np.float32)  # noqa: E731
    root = None
    if st.root is not None:
        root = (rs(st.root, cv2.INTER_LINEAR) > 0.5).astype(np.float32)
    return State(a=rs(st.a, cv2.INTER_LINEAR), b=rs(st.b, cv2.INTER_LINEAR), x=rs(st.x, cv2.INTER_LINEAR), root=root)


def run(cfg: ProtoConfig, out: Path | None = None, verbose: bool = True) -> list[dict]:
    rng = np.random.default_rng(cfg.seed + 101)
    st = init_state(cfg)
    snap_steps = set(np.linspace(cfg.t_on, cfg.steps, cfg.snapshots, dtype=int).tolist()) | {cfg.steps}
    rows: list[dict] = []
    tiles: list[np.ndarray] = []
    tile_size = max(cfg.size, cfg.g_size1)
    tips: list[Tip] = []
    n_births = 0
    for t in range(1, cfg.steps + 1):
        active = t >= cfg.t_on
        n_now = domain_size(cfg, t)
        if n_now != st.b.shape[0]:
            scale = n_now / st.b.shape[0]
            st = resize_state(st, n_now)
            for tp in tips:
                tp.y *= scale
                tp.x *= scale
        if cfg.mech == "S" and t == max(cfg.t_on, 1):
            root = (st.b > cfg.s_root_thr).astype(np.uint8)
            if cfg.s_root_dilate > 0:
                d = 2 * cfg.s_root_dilate + 1
                root = cv2.dilate(root, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
            if cfg.s_root_keep < 1.0:
                n_lab, lab = cv2.connectedComponents(root, connectivity=8)
                keep = rng.random(n_lab) < cfg.s_root_keep
                keep[0] = False
                root = keep[lab].astype(np.uint8)
            st.root = root.astype(np.float32)
            st.x = (cfg.s_rho / cfg.s_delta) * st.root
        if cfg.tips_per_root > 0 and t == max(cfg.t_tips or cfg.t_on, 1) and st.root is not None:
            tips = spawn_tips(st.root, cfg, rng, t)
        if cfg.f1 is not None and active:
            span = cfg.ramp_steps if cfg.ramp_steps > 0 else (cfg.steps - cfg.t_on)
            u = min(1.0, (t - cfg.t_on) / max(span, 1))
            f = cfg.f + u * (cfg.f1 - cfg.f)
            k = cfg.k + u * ((cfg.k1 if cfg.k1 is not None else cfg.k) - cfg.k)
        else:
            f, k = cfg.f, cfg.k
        st = step(st, cfg, f, k, active, rng)
        if tips and active:
            tips, births = update_tips(tips, st, cfg, rng, t, cfg.s_rho / cfg.s_delta)
            n_births += births
        if t in snap_steps:
            m = tree_metrics(st.b)
            m.update(step=t, f=f, k=k, x_mean=float(st.x.mean()), x_max=float(st.x.max()), b_mean=float(st.b.mean()),
                     tips_alive=len(tips), tip_births=n_births,
                     tip_max_gen=max((tp.gen for tp in tips), default=0))
            rows.append(m)
            if verbose:
                extra = f" tips={len(tips)} births={n_births} gen={m['tip_max_gen']}" if cfg.tips_per_root else ""
                print(
                    f"t={t:6d} f={f:.4f} k={k:.4f} cov={m['coverage']:.3f} comp={m['components']:3d} "
                    f"trees={m['trees']:2d} junc3={m['junc3']:3d} mjt={m['max_junc_tree']:2d} ends={m['endpoints']:3d} "
                    f"longest={m['longest_skel']:4d} arm={m['mean_arm']:5.1f} x={m['x_mean']:.3f}/{m['x_max']:.3f}" + extra,
                    flush=True,
                )
            tile = render(st.b, f"{cfg.mech} t={t} n={st.b.shape[0]} j3={m['junc3']} tr={m['trees']} L={m['longest_skel']}")
            if tile.shape[0] != 2 * tile_size:
                pad = np.zeros((2 * tile_size, 2 * tile_size, 3), np.uint8)
                pad[: tile.shape[0], : tile.shape[1]] = tile
                tile = pad
            tiles.append(tile)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "timeline.png"), np.concatenate(tiles, axis=1))
        cv2.imwrite(str(out / "final_b.png"), render(st.b, f"{cfg.mech} final", scale=3))
        np.save(str(out / "final_b.npy"), st.b)
        if cfg.mech in ("H", "N", "S"):
            x = st.x
            xi = np.clip((x - x.min()) / max(float(x.max() - x.min()), 1e-6) * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out / "final_x.png"), cv2.applyColorMap(xi, cv2.COLORMAP_VIRIDIS))
        (out / "metrics.json").write_text(json.dumps({"config": asdict(cfg), "rows": rows}, indent=1))
    return rows


def run_sweep(cfg: ProtoConfig, total_steps: int, out: Path | None, s_off: float = 0.9, s_on: float = 0.0,
              use_noise_schedule: bool = True, verbose: bool = True) -> list[dict]:
    """Embed the mechanism in the v2 path sweep of ``sweep_video`` (spots -> worm -> labyrinth -> holes).

    The S coupling is scaled by w(s) = clip((s_off - s)/(s_off - s_on), 0, 1): full
    strength up to s_on, fading linearly, off before the holes dwell (s = 0.94).
    The root mask is taken at the end of the Type I dwell (t = 0.12 T).
    """
    from sweep_video import (  # noqa: E402
        NOISE_SCHEDULE_V2, SCHEDULE_V2, add_noise, classify_phase, path_fk, phase_metrics, schedule_s,
    )

    p0, p1 = (0.026, 0.061), (0.040, 0.058)
    rng = np.random.default_rng(cfg.seed + 101)
    cfg.t_on = int(round(0.12 * total_steps))
    st = init_state(cfg)
    checkpoints = {int(round(u * total_steps)): name for u, name in
                   ((0.12, "spots"), (0.54, "worm"), (0.76, "labyrinth"), (1.0, "holes"))}
    rows: list[dict] = []
    tiles: list[np.ndarray] = []
    for t in range(1, total_steps + 1):
        s = schedule_s(t, total_steps, SCHEDULE_V2)
        f, k = path_fk(s, p0, p1)
        sigma = schedule_s(t, total_steps, NOISE_SCHEDULE_V2) if use_noise_schedule else cfg.noise
        active = t >= cfg.t_on
        if cfg.mech == "S" and t == cfg.t_on:
            st.root = (st.b > cfg.s_root_thr).astype(np.float32)
            st.x = (cfg.s_rho / cfg.s_delta) * st.root
        gain = float(np.clip((s_off - s) / max(s_off - s_on, 1e-9), 0.0, 1.0))
        st = step(st, cfg, f, k, active, rng, gain=gain)
        st.b = add_noise(st.b, sigma, rng)
        if t in checkpoints:
            m = tree_metrics(st.b)
            pm = phase_metrics(st.b)
            m.update(step=t, s=s, f=f, k=k, gain=gain, expected=checkpoints[t], phase=classify_phase(pm),
                     euler_ratio=pm.get("euler_ratio", 0.0), n_bg=pm.get("n_bg", 0))
            rows.append(m)
            if verbose:
                print(f"t={t:6d} s={s:.2f} w={gain:.2f} expect={checkpoints[t]:9s} got={m['phase']:9s} "
                      f"cov={m['coverage']:.3f} comp={m['components']:3d} junc3={m['junc3']:3d} "
                      f"longest={m['longest_skel']:4d} euler={m['euler_ratio']:+.2f}", flush=True)
            tiles.append(render(st.b, f"{cfg.mech} s={s:.2f} {m['phase']} j3={m['junc3']} L={m['longest_skel']}"))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "sweep_timeline.png"), np.concatenate(tiles, axis=1))
        (out / "sweep_metrics.json").write_text(json.dumps({"config": asdict(cfg), "rows": rows}, indent=1))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mech", default="none", choices=["none", "H", "N", "M", "S"])
    p.add_argument("--fk", default="0.036,0.059")
    p.add_argument("--fk1", default=None, help="optional end point of a linear (f,k) ramp")
    p.add_argument("--ramp-steps", type=int, default=0, help="ramp duration after --t-on (0: to the end)")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--t-on", type=int, default=0)
    p.add_argument("--seeds", default="isolated", choices=["isolated", "sparse", "dense"])
    p.add_argument("--n-seeds", type=int, default=9)
    p.add_argument("--seed-radius", type=int, default=3)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--d-b", type=float, default=0.5)
    p.add_argument("--grow", default=None, help="growing domain: final_size,t0,t1 (linear resize between t0 and t1)")
    p.add_argument("--sweep", type=int, default=0, metavar="TOTAL_STEPS",
                   help="instead of fixed (f,k): embed the mechanism in the v2 path sweep of sweep_video")
    p.add_argument("--sweep-on", type=float, default=0.0, help="s up to which the S coupling gain is 1")
    p.add_argument("--sweep-off", type=float, default=0.9, help="s at which the S coupling gain reaches 0")
    p.add_argument("--sweep-const-noise", action="store_true", help="use --noise instead of NOISE_SCHEDULE_V2")
    p.add_argument("--set", action="append", default=[], help="override any ProtoConfig field: name=value")
    p.add_argument("--snapshots", type=int, default=6)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    f, k = (float(v) for v in args.fk.split(","))
    cfg = ProtoConfig(
        mech=args.mech, size=args.size, steps=args.steps, f=f, k=k, d_b=args.d_b,
        noise=args.noise, seed=args.seed, seeds=args.seeds, n_seeds=args.n_seeds,
        seed_radius=args.seed_radius,
        t_on=args.t_on, snapshots=args.snapshots, ramp_steps=args.ramp_steps,
    )
    if args.fk1:
        cfg.f1, cfg.k1 = (float(v) for v in args.fk1.split(","))
    if args.grow:
        cfg.g_size1, cfg.g_t0, cfg.g_t1 = (int(v) for v in args.grow.split(","))
    for item in args.set:
        name, value = item.split("=")
        cur = getattr(cfg, name)
        if isinstance(cur, bool):
            setattr(cfg, name, value.lower() in ("1", "true", "yes"))
        else:
            setattr(cfg, name, type(cur)(value) if cur is not None else float(value))
    if args.sweep > 0:
        run_sweep(cfg, args.sweep, args.out, s_off=args.sweep_off, s_on=args.sweep_on,
                  use_noise_schedule=not args.sweep_const_noise)
    else:
        run(cfg, args.out)


if __name__ == "__main__":
    main()
