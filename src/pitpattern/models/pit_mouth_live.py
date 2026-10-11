#!/usr/bin/env python3
"""Real-time form of the crypt-mouth model: one state that follows the progression p.

Same two layers and the same p-dependent coefficients as simulate_pit_mouth, but nothing is
restarted when p changes:

  surface   The window looks at a growing surface.  Area per crypt A(p) sets a linear scale
            s(p) = sqrt(A(p) / A(0)); when p moves, crypt centres, grown grooves and the field are
            dilated about the window centre, so the number of crypts is conserved and they
            drift out of (or back into) view.
  units     Up to Branch (p <= 2) every crypt is a rigid dot / tube / Y whose arm lengths follow
            p; positions and orientations keep relaxing under the gap-D repulsion.
  growth    Beyond Branch the arm tips advance (self-avoiding, gap = groove pitch, occasional
            forks) while the branch is shorter than its length target, and retract while it is
            longer; the target is a function of p.
  field     du/dt = r u - (1 + lap/k(p)^2)^2 u - u^3 + g(p) u^2 + kappa(p)(T - u)
                    + kappa_g(p) P (u_p - u) + noise,  stepped every frame; T is redrawn from
            the units.

p itself moves towards the requested value at a limited rate, so the state stays close to
equilibrium.  This is a continuation in p: unlike simulate_pit_mouth it depends on history.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from scipy import fft as sfft

from pitpattern.figures.make_pit_grade_sweep import NPTS, PER_ARM, local_points, rotate, smooth
from pitpattern.models.pit_mouth_kernels import relax_forces, thin
from pitpattern.models.simulate_pit_mouth import Config, colour, params


@dataclass(frozen=True)
class LiveConfig:
    size: int = 330               # side of the displayed window (px)
    margin: int = 51              # simulated border around the window (hidden); 330 + 2*51 = 432 = 2^4 3^3
    keep: float = 160.0           # crypts stay active this far outside the simulated box
    p_rate: float = 0.12          # how fast p follows the slider (per second of model time)
    fps: float = 30.0             # model seconds per frame = 1 / fps
    relax_steps: int = 4          # unit relaxation steps per frame (p <= 2)
    relax_noise: float = 0.15
    grow_ticks: int = 2           # tip growth / retraction ticks per frame
    field_steps: int = 5          # field steps per frame
    live_noise: float = 0.03      # field noise per step (lets islands and grooves nucleate)
    warmup: int = 500             # relaxation steps before the first frame
    fft_workers: int = 4          # threads of the field's FFT
    # Model coefficients.  Two differ from the validated independent-run values (extra 420,
    # kappa_gyrus 0.5): here the surface dilation stretches the arms and that length counts towards
    # the budget, so the same budget leaves a sparser Gyrus.
    base: Config = field(default_factory=lambda: replace(Config(), extra=650.0, kappa_gyrus=0.25))


@dataclass
class Branch:
    owner: int
    pts: list                      # grown points; pts[0] is the fork point / core tip
    heading: float
    arcs: list                     # arc position of every point from the crypt centre
    parent: int                    # branch id of the parent (-1: core arm)
    frac: float                    # share of the parent's remaining budget (1 for a core arm)
    anc: dict                      # branch id -> arc limit of the stretch to ignore near the tip
    fork_at: dict = field(default_factory=dict)     # child branch id -> index of the fork point
    wait: int = 0                  # ticks to skip after the tip was blocked

    @property
    def arc0(self) -> float:
        return self.arcs[0]

    def length(self) -> float:
        return self.arcs[-1] - self.arcs[0]


class Live:
    def __init__(self, cfg: LiveConfig = LiveConfig(), seed: int = 1, p: float = 0.2):
        self.cfg, self.m = cfg, cfg.base
        self.rng = np.random.default_rng(seed)
        self.N = cfg.size + 2 * cfg.margin
        self.centre = np.array([self.N / 2, self.N / 2])
        self.p = self.p_target = float(p)
        self.time = 0.0
        m = self.m
        # crypt population: fills the simulated box at p = 0 (scale 1)
        self.A0 = float(params(0.0, m)["area"])
        n = int(round(self.N * self.N / self.A0))
        xi = self.rng.uniform(0, self.N, (n, 2))
        self.c = self.centre + self.scale(self.p) * (xi - self.centre)
        self.th = self.rng.uniform(0, 2 * np.pi, n)
        self.branches: dict[int, Branch] = {}
        self.next_bid = 3 * n
        self._warp = 1.0
        self.K2 = ((2 * np.pi * np.fft.fftfreq(self.N))[:, None] ** 2
                   + (2 * np.pi * np.fft.rfftfreq(self.N))[None, :] ** 2).astype(np.float32)
        self._den = (None, None)
        self.u = (m.noise * self.rng.normal(size=(self.N, self.N))).astype(np.float32)      # single precision
        for _ in range(cfg.warmup):
            self.relax(noise=cfg.relax_noise)
        self.T = self.template()
        for _ in range(60):
            self.field_step()

    # ------------------------------------------------------------ surface
    def scale(self, p: float) -> float:
        return float(np.sqrt(params(p, self.m)["area"] / self.A0))

    def active(self):
        lo, hi = -self.cfg.keep, self.N + self.cfg.keep
        return np.flatnonzero((self.c[:, 0] > lo) & (self.c[:, 0] < hi) & (self.c[:, 1] > lo) & (self.c[:, 1] < hi))

    def dilate(self, rho: float):
        if rho == 1.0:
            return
        self.c = self.centre + rho * (self.c - self.centre)
        for b in self.branches.values():
            b.pts = [self.centre + rho * (q - self.centre) for q in b.pts]
            a0 = b.arcs[0] * (rho if b.parent >= 0 else 1.0)       # a core tip keeps its arc
            b.arcs = [a0 + rho * (a - b.arcs[0]) for a in b.arcs]
            b.anc = {k: (v * rho if np.isfinite(v) and k >= 3 * len(self.c) else v) for k, v in b.anc.items()}
        self._warp *= rho
        if abs(self._warp - 1) > 0.003:              # resample the field in a few larger steps
            M = np.array([[self._warp, 0, (1 - self._warp) * self.centre[0]],
                          [0, self._warp, (1 - self._warp) * self.centre[1]]])
            self.u = cv2.warpAffine(self.u, M, (self.N, self.N), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            self._warp = 1.0

    # ------------------------------------------------------------ units (p <= 2)
    def local(self):
        m = self.m
        return local_points(min(self.p, 2.0), m.tube_len, m.arm_len, m.arm_len)

    def relax(self, noise: float = 0.0, k: float = 0.25, max_step: float = 1.0):
        """One overdamped step of the gap-D repulsion between the active rigid units."""
        act = self.active()
        if len(act) < 2:
            return
        D = self.m.D
        local = self.local()
        reach = np.linalg.norm(local, axis=1).max()
        c, th = self.c[act], self.th[act]
        F, Tq = relax_forces(c, th, local, D, k, D + 2 * reach + 6)
        step = F / np.sqrt(NPTS)
        norm = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, max_step / np.maximum(norm, 1e-9))
        inert = max(float((local ** 2).sum()), 1.0)
        self.c[act] = c + step + self.rng.normal(0, noise, c.shape)
        self.th[act] = th + np.clip(2.0 * Tq / inert, -0.05, 0.05) + self.rng.normal(0, 0.05 * noise, len(act)) * (self.p > 0.15)

    # ------------------------------------------------------------ growth (p > 2)
    def core_points(self, act):
        """World points of the rigid cores: (i, k) -> point; tips at k = PER_ARM, 2 PER_ARM, 3 PER_ARM."""
        return rotate(np.broadcast_to(self.local(), (len(act), NPTS, 2)), self.th[act]) + self.c[act][:, None]

    def target(self, bid: int, root: float, memo: dict) -> float:
        """Length target of a branch: a function of p (root) and of where it forked."""
        if bid in memo:
            return memo[bid]
        b = self.branches[bid]
        if b.parent < 0:
            t = root
        else:
            par = self.branches.get(b.parent)
            t = 0.0 if par is None else b.frac * (self.target(b.parent, root, memo) - (b.arc0 - par.arc0))
        memo[bid] = t
        return t

    def groove_cloud(self, act, core):
        """All centreline points with owner, branch id and arc position."""
        l = self.m.arm_len
        P = [self.c[act]]
        own = [act]
        br = [np.full(len(act), -1)]
        arc = [np.zeros(len(act))]
        fr = np.linspace(0.25, 1, 4)
        for arm in range(3):
            tip = core[:, (arm + 1) * PER_ARM]
            v = tip - self.c[act]
            for f in fr:
                P.append(self.c[act] + f * v)
                own.append(act)
                br.append(3 * act + arm)
                arc.append(np.full(len(act), f * l))
        grown = [(q, b.owner, bid, a) for bid, b in self.branches.items() for q, a in zip(b.pts[1:], b.arcs[1:])]
        if grown:
            P.append(np.array([g[0] for g in grown]))
            own.append(np.array([g[1] for g in grown]))
            br.append(np.array([g[2] for g in grown]))
            arc.append(np.array([g[3] for g in grown]))
        return np.concatenate(P), np.concatenate(own), np.concatenate(br), np.concatenate(arc)

    def grow(self):
        """One tick: tips of too-short branches advance, too-long branches retract."""
        m, rng = self.m, self.rng
        act = self.active()
        if self.p <= 2.0:
            self.branches.clear()
            return
        core = self.core_points(act)
        idx = {int(i): k for k, i in enumerate(act)}
        # core arms carry one root branch each
        for i in act:
            for arm in range(3):
                bid = 3 * int(i) + arm
                tip = core[idx[int(i)], (arm + 1) * PER_ARM]
                if bid not in self.branches:
                    v = tip - self.c[i]
                    sib = {3 * int(i) + k: m.arm_len + m.D / np.sqrt(3) for k in range(3)}
                    sib[bid] = np.inf
                    self.branches[bid] = Branch(int(i), [tip.copy()], float(np.arctan2(v[1], v[0])), [m.arm_len], -1, 1.0, sib)
                else:
                    self.branches[bid].pts[0] = tip.copy()
        for bid in [b for b, br_ in self.branches.items() if br_.owner not in idx]:
            del self.branches[bid]                       # crypt left the active region
        pr = params(self.p, m)
        gap = float(pr["lam"])
        root = float(pr["grow"]) * m.extra / 3
        memo: dict = {}
        fork_p = m.fork * float(smooth(self.p - 2))
        step = 2.0
        cloud = None
        newP, newown, newbr, newarc = [], [], [], []
        for bid in rng.permutation(list(self.branches)):
            b = self.branches.get(int(bid))
            if b is None:
                continue
            tgt, ln = self.target(int(bid), root, memo), b.length()
            if ln > tgt + step or tgt <= 0:              # retract
                self.retract(int(bid), tgt)
                continue
            if ln > tgt - step:
                continue
            if b.wait > 0:
                b.wait -= 1
                continue
            tip = b.pts[-1]
            if not (-20 < tip[0] < self.N + 20 and -20 < tip[1] < self.N + 20):
                continue
            if cloud is None:
                cloud = self.groove_cloud(act, core)
            P, own, br, arc = cloud
            a_tip = b.arcs[-1]
            core_lim = m.arm_len + 1 if a_tip < m.arm_len + gap else 0.0
            near0 = ((P - tip) ** 2).sum(1) < (gap + step + 1) ** 2
            Pn, on, bn, an = P[near0], own[near0], br[near0], arc[near0]
            if newP:
                Pn, on = np.concatenate([Pn, np.array(newP)]), np.concatenate([on, np.array(newown)])
                bn, an = np.concatenate([bn, np.array(newbr)]), np.concatenate([an, np.array(newarc)])
            b.wait = 8                                   # cleared below if the tip moves
            for dh in sorted([0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.25, -1.25], key=lambda v: abs(v) + 0.2 * rng.random()):
                hh = b.heading + dh + rng.normal(0, 0.08)
                q = tip + step * np.array([np.cos(hh), np.sin(hh)])
                close = ((Pn - q) ** 2).sum(1) < gap * gap
                if close.any():
                    mine = on[close] == b.owner
                    skip = an[close] < core_lim
                    for bb, lim in b.anc.items():
                        skip |= (bn[close] == bb) & (an[close] <= lim) & (np.abs(an[close] - (a_tip + step)) < 1.05 * gap)
                    if (~(mine & skip)).any():
                        continue
                b.pts.append(q)
                b.arcs.append(a_tip + step)
                b.heading = hh
                b.wait = 0
                newP.append(q)
                newown.append(b.owner)
                newbr.append(int(bid))
                newarc.append(a_tip + step)
                if rng.random() < fork_p + 0.10 * (abs(dh) > 0.8) * float(smooth(self.p - 2)):
                    near = a_tip + step + 1.3 * gap
                    cid = self.next_bid
                    self.next_bid += 1
                    canc = dict(b.anc)
                    canc[int(bid)] = near
                    canc[cid] = np.inf
                    b.anc[cid] = near
                    b.fork_at[cid] = len(b.pts) - 1
                    self.branches[cid] = Branch(b.owner, [q.copy()], hh + rng.choice([-1, 1]) * np.pi / 3,
                                                [a_tip + step], int(bid), 0.6, canc)
                break

    def retract(self, bid: int, tgt: float):
        b = self.branches[bid]
        if len(b.pts) > 1:
            last = len(b.pts) - 1
            for cid, at in list(b.fork_at.items()):
                if at >= last:
                    self.drop(cid)
                    b.fork_at.pop(cid, None)
                    b.anc.pop(cid, None)
            b.pts.pop()
            b.arcs.pop()
            if len(b.pts) > 1:
                d = b.pts[-1] - b.pts[-2]
                b.heading = float(np.arctan2(d[1], d[0]))
        elif b.parent >= 0 and tgt <= 0:
            par = self.branches.get(b.parent)
            if par is not None:
                par.fork_at.pop(bid, None)
                par.anc.pop(bid, None)
            self.drop(bid)

    def drop(self, bid: int):
        b = self.branches.pop(bid, None)
        if b is not None:
            for cid in list(b.fork_at):
                self.drop(cid)

    # ------------------------------------------------------------ field
    def template(self, ss: int = 2):
        m, N = self.m, self.N
        act = self.active()
        img = np.zeros((N * ss, N * ss), np.uint8)
        core = self.core_points(act)
        Pt = lambda v: (int(round(v[0] * ss)), int(round(v[1] * ss)))
        r, w = int(m.dot_d / 2 * ss), int(m.line_w * ss)
        for k in range(len(act)):
            cv2.circle(img, Pt(core[k, 0]), r, 1, -1, cv2.LINE_AA)
            for arm in range(3):
                tip = core[k, (arm + 1) * PER_ARM]
                if np.abs(tip - core[k, 0]).max() > 0.5:
                    cv2.line(img, Pt(core[k, 0]), Pt(tip), 1, w, cv2.LINE_AA)
        for b in self.branches.values():
            if len(b.pts) > 1:
                poly = np.round(np.asarray(b.pts) * ss).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [poly], False, 1, w, cv2.LINE_AA)
        f = cv2.resize(img.astype(np.float32), (N, N), interpolation=cv2.INTER_AREA)
        T = 2 * cv2.GaussianBlur(f, (0, 0), 1.5) - 1
        self.P = np.clip((T + 1) / 2, 0, 1)             # 1 on the grooves
        return T

    def field_step(self):
        m = self.m
        pr = params(self.p, m)
        kappa, kappa_g, g, lam = (float(pr[k]) for k in ("kappa", "kappa_g", "g", "lam"))
        if self._den[0] != lam:
            self._den = (lam, (1 - m.dt * (m.r - (1 - self.K2 / (2 * np.pi / lam) ** 2) ** 2 - 1)).astype(np.float32))
        u, T = self.u, self.T
        f32 = np.float32
        nl = u - u * u * u + f32(g) * u * u
        if kappa:
            nl += f32(kappa) * (T - u)
        if kappa_g:
            nl += f32(kappa_g) * self.P * (f32(m.pin_level) - u)
        if self.cfg.live_noise:
            nl += f32(self.cfg.live_noise / np.sqrt(m.dt)) * self.rng.standard_normal(u.shape, dtype=f32)
        w = self.cfg.fft_workers
        self.u = sfft.irfft2(sfft.rfft2(u + f32(m.dt) * nl, workers=w) / self._den[1], s=u.shape, workers=w)

    def paint(self, x: float, y: float, groove: bool, radius: float = 14.0):
        """Perturbation: force a disc of the field (window coordinates) to groove or to mucosa."""
        mg = self.cfg.margin
        yy, xx = np.ogrid[:self.N, :self.N]
        w = np.clip(radius + 0.5 - np.hypot(xx - (x + mg), yy - (y + mg)), 0, 1)      # soft 1 px edge
        self.u = (self.u * (1 - w) + (1.0 if groove else -1.0) * w).astype(np.float32)

    # ------------------------------------------------------------ one frame
    def step(self):
        cfg = self.cfg
        dt = 1.0 / cfg.fps
        self.time += dt
        dp = np.clip(self.p_target - self.p, -cfg.p_rate * dt, cfg.p_rate * dt)
        if dp:
            s0 = self.scale(self.p)
            self.p = float(np.clip(self.p + dp, 0.0, 4.0))
            self.dilate(self.scale(self.p) / s0)
        if self.p <= 2.0:
            for _ in range(cfg.relax_steps):
                self.relax(noise=cfg.relax_noise * (0.3 if dp == 0 else 1.0))
        for _ in range(cfg.grow_ticks):
            self.grow()
        self.T = self.template()
        for _ in range(cfg.field_steps):
            self.field_step()

    # ------------------------------------------------------------ output
    def window(self, a):
        mg = self.cfg.margin
        return a[mg:-mg, mg:-mg]

    def binary(self, fixed_width: bool = True):
        """Pattern in the window: the field's centreline at the fixed width, or u > 0 itself."""
        mb = self.u > 0
        if not fixed_width:
            return self.window(mb)
        sk = thin(mb)
        d = int(round(self.m.render_w))
        return self.window(cv2.dilate(sk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))).astype(bool))

    def rgb(self, fixed_width: bool = True):
        return colour(self.binary(fixed_width).astype(np.float32), replace(self.m, width=self.cfg.size, height=self.cfg.size))

    def n_visible(self) -> int:
        mg, s = self.cfg.margin, self.cfg.size
        return int(((self.c > mg) & (self.c < mg + s)).all(1).sum())


def main() -> None:
    """Ramp p up and back in one continuous run and tile the snapshots (docs figure)."""
    import argparse
    from pathlib import Path

    from PIL import Image, ImageDraw

    from pitpattern.evaluation.evaluate_pit_mouth import label, measure
    from pitpattern.figures.make_pit_ideal_sweep import font

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/images/pit_mouth_live_ramp.png"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--up", type=float, nargs="+", default=[0.2, 0.6, 1.0, 1.5, 2.0, 2.4, 2.7, 3.0, 3.2, 3.5, 3.8, 4.0])
    ap.add_argument("--down", type=float, nargs="+", default=[3.2, 2.4, 2.0, 1.0, 0.2])
    ap.add_argument("--settle", type=float, default=2.0, help="seconds of model time held at every stop")
    a = ap.parse_args()
    cfg = LiveConfig()
    L = Live(cfg, a.seed, a.up[0])
    mcfg = replace(cfg.base, width=cfg.size, height=cfg.size)
    tiles = []
    for k, p in enumerate(a.up + a.down):
        L.p_target, held = p, 0
        while abs(L.p - p) > 1e-9 or held < a.settle * cfg.fps:
            L.step()
            held += abs(L.p - p) < 1e-9
        lab = label(measure(L.binary(), mcfg, periodic=False))
        print(f"{'up' if k < len(a.up) else 'down':4s} p={p:.1f} {lab:10s} crypts in view {L.n_visible()}", flush=True)
        tiles.append((("↑" if k < len(a.up) else "↓") + f" p = {p:.1f}   {lab}", L.rgb()))
    t, cols, pad = 220, 6, 14
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (2 * pad + cols * (t + 6), 44 + rows * (t + 24) + pad), "#fcfcfb")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 12), "リアルタイム版: 1 つの状態で p を上げて戻した連続計算", fill="black", font=font(18, True))
    for k, (name, rgb) in enumerate(tiles):
        x, y = pad + (k % cols) * (t + 6), 44 + (k // cols) * (t + 24)
        sheet.paste(Image.fromarray(rgb).resize((t, t), Image.LANCZOS), (x, y + 18))
        d.text((x + 2, y), name, fill="#52514e", font=font(13))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
