"""Compiled inner loops for the real-time crypt-mouth model (numba).

Both reproduce the numpy routines they replace: `relax_forces` the pair forces of
make_pit_grade_sweep.relax, `thin` the Zhang-Suen thinning of evaluate_pit_steps.zhang_suen_thin.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def relax_forces(c, th, local, D, k, cutoff):
    """Force and torque on every rigid unit from the gap-D repulsion between unit points."""
    n, npts = c.shape[0], local.shape[0]
    W = np.empty((n, npts, 2))
    for i in range(n):
        co, si = np.cos(th[i]), np.sin(th[i])
        for a in range(npts):
            W[i, a, 0] = local[a, 0] * co - local[a, 1] * si
            W[i, a, 1] = local[a, 0] * si + local[a, 1] * co
    F = np.zeros((n, 2))
    Tq = np.zeros(n)
    c2 = cutoff * cutoff
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = c[i, 0] - c[j, 0], c[i, 1] - c[j, 1]
            if dx * dx + dy * dy >= c2:
                continue
            for a in range(npts):
                ax, ay = W[i, a, 0] + dx, W[i, a, 1] + dy
                for b in range(npts):
                    vx, vy = ax - W[j, b, 0], ay - W[j, b, 1]
                    dist = np.sqrt(vx * vx + vy * vy) + 1e-9
                    if dist < D:
                        f = k * (D - dist) / dist
                        fx, fy = f * vx, f * vy
                        F[i, 0] += fx
                        F[i, 1] += fy
                        F[j, 0] -= fx
                        F[j, 1] -= fy
                        Tq[i] += W[i, a, 0] * fy - W[i, a, 1] * fx
                        Tq[j] -= W[j, b, 0] * fy - W[j, b, 1] * fx
    return F, Tq


@njit(cache=True)
def thin(binary):
    """Zhang-Suen thinning; only the remaining foreground pixels are revisited."""
    h, w = binary.shape
    img = np.zeros((h, w), np.uint8)
    n = 0
    ys = np.empty(h * w, np.int32)
    xs = np.empty(h * w, np.int32)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if binary[y, x]:
                img[y, x] = 1
                ys[n] = y
                xs[n] = x
                n += 1
    kill = np.empty(n, np.bool_)
    for _ in range(120):
        changed = False
        for sub in range(2):
            any_kill = False
            for t in range(n):
                y, x = ys[t], xs[t]
                p2, p3, p4, p5 = img[y - 1, x], img[y - 1, x + 1], img[y, x + 1], img[y + 1, x + 1]
                p6, p7, p8, p9 = img[y + 1, x], img[y + 1, x - 1], img[y, x - 1], img[y - 1, x - 1]
                nb = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                kill[t] = False
                if nb < 2 or nb > 6:
                    continue
                tr = ((p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) + (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1)
                      + (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) + (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1))
                if tr != 1:
                    continue
                if sub == 0:
                    ok = p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
                else:
                    ok = p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
                if ok:
                    kill[t] = True
                    any_kill = True
            if any_kill:
                changed = True
                m = 0
                for t in range(n):
                    if kill[t]:
                        img[ys[t], xs[t]] = 0
                    else:
                        ys[m] = ys[t]
                        xs[m] = xs[t]
                        m += 1
                n = m
        if not changed:
            break
    return img
