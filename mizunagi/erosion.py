"""Peyzaj evrimi: implicit stream-power + hillslope difüzyonu (Braun & Willett 2013, Perron 2008).

dz/dt = U - K A^m S^n + D lap(z),  m = 0.5, n = 1
* Akış yönü D8 en dik iniş; taban seviyesi (vadi tabanı) sabit
* Stream-power terimi implicit (koşulsuz kararlı), difüzyon explicit
* Dere aralığı ~ (D/K)^(1/(2m+1)) ölçeğinde belirlenir -> gerçek ölçek için D/K seçilir
"""
import math
import numpy as np
from numba import njit

DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
DL = np.array([math.sqrt(2), 1, math.sqrt(2), 1, 1, math.sqrt(2), 1, math.sqrt(2)])


@njit(cache=True)
def _receivers(z, base, dx):
    h, w = z.shape
    rec = np.empty(h * w, np.int64)
    rdist = np.ones(h * w)
    for i in range(h):
        for j in range(w):
            k = i * w + j
            rec[k] = k
            if base[i, j]:
                continue
            best = 0.0
            for d in range(8):
                ii, jj = i + DY[d], j + DX[d]
                if ii < 0 or ii >= h or jj < 0 or jj >= w:
                    continue
                s = (z[i, j] - z[ii, jj]) / (DL[d] * dx)
                if s > best:
                    best = s
                    rec[k] = ii * w + jj
                    rdist[k] = DL[d] * dx
    return rec, rdist


@njit(cache=True)
def _stack(rec, n):
    # alıcı -> verici listeleri, sonra taban düğümlerinden DFS ile yukarı akış sırası
    ndon = np.zeros(n, np.int64)
    for k in range(n):
        if rec[k] != k:
            ndon[rec[k]] += 1
    start = np.zeros(n + 1, np.int64)
    for k in range(n):
        start[k + 1] = start[k] + ndon[k]
    fill = start[:-1].copy()
    don = np.empty(max(1, start[n]), np.int64)
    for k in range(n):
        r = rec[k]
        if r != k:
            don[fill[r]] = k
            fill[r] += 1
    stack = np.empty(n, np.int64)
    ns = 0
    tmp = np.empty(n, np.int64)
    for k in range(n):
        if rec[k] == k:
            top = 0
            tmp[top] = k
            top += 1
            while top > 0:
                top -= 1
                c = tmp[top]
                stack[ns] = c
                ns += 1
                for q in range(start[c], start[c + 1]):
                    tmp[top] = don[q]
                    top += 1
    return stack


@njit(cache=True)
def _step(z, base, U, K, D, dt, dx, m):
    h, w = z.shape
    n = h * w
    zf = z.ravel()
    Uf = U.ravel()
    bf = base.ravel()
    for k in range(n):
        if not bf[k]:
            zf[k] += Uf[k] * dt
    rec, rdist = _receivers(z, base, dx)
    stack = _stack(rec, n)
    A = np.full(n, dx * dx)
    for q in range(n - 1, -1, -1):
        k = stack[q]
        r = rec[k]
        if r != k:
            A[r] += A[k]
    Kf = K.ravel()
    for q in range(n):
        k = stack[q]
        r = rec[k]
        if r != k and not bf[k]:
            F = Kf[k] * dt * A[k] ** m / rdist[k]
            zf[k] = (zf[k] + F * zf[r]) / (1.0 + F)
    # difüzyon (explicit, 5 nokta)
    out = z.copy()
    c = D * dt / (dx * dx)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if not base[i, j]:
                out[i, j] = z[i, j] + c[i, j] * (z[i - 1, j] + z[i + 1, j] + z[i, j - 1] + z[i, j + 1] - 4 * z[i, j])
    return out, A.reshape(h, w)


def evolve(z0, base, U, K, D, dt, steps, dx=1.0, m=0.5, max_slope=None, log=None):
    z = z0.astype(np.float64).copy()
    U = np.broadcast_to(U, z.shape).astype(np.float64).copy()
    K = np.broadcast_to(K, z.shape).astype(np.float64).copy()
    D = np.broadcast_to(D, z.shape).astype(np.float64).copy()
    A = None
    for t in range(steps):
        z, A = _step(z, base, U, K, D, dt, dx, m)
        if max_slope is not None:                 # eşik yamaç (kaya düşmesi): tan(phi) üstü kırpılır
            z = _clip_slope(z, base, max_slope * dx)
        if log and t % max(1, steps // 5) == 0:
            log(f"  erozyon adımı {t}/{steps}, z {z.min():.1f}..{z.max():.1f}")
    return z, A


@njit(cache=True)
def _clip_slope(z, base, smax):
    h, w = z.shape
    for _ in range(2):
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if base[i, j]:
                    continue
                lo = min(z[i - 1, j], z[i + 1, j], z[i, j - 1], z[i, j + 1])
                if z[i, j] - lo > smax:
                    z[i, j] = lo + smax
    return z


# ------------------------------------------------------------------ parçacık (damla) tabanlı hidrolik erozyon
@njit(cache=True)
def _grad_h(z, x, y):
    h, w = z.shape
    i, j = int(y), int(x)
    if i < 0 or j < 0 or i >= h - 1 or j >= w - 1:
        return 0.0, 0.0, 1e9
    u, v = x - j, y - i
    a, b, c, d = z[i, j], z[i, j + 1], z[i + 1, j], z[i + 1, j + 1]
    gx = (b - a) * (1 - v) + (d - c) * v
    gy = (c - a) * (1 - u) + (d - b) * u
    hh = a * (1 - u) * (1 - v) + b * u * (1 - v) + c * (1 - u) * v + d * u * v
    return gx, gy, hh


@njit(cache=True)
def droplets(z, spawn, nonerode, n, seed, radius=3, inertia=0.05, capacity=4.0, min_slope=0.01, erode=0.3,
             deposit=0.3, evaporate=0.012, gravity=4.0, max_steps=80):
    """Beyer (2015) tarzı damla erozyonu. z normalize yükseklik (eğim ~0.01/hücre mertebesi)."""
    np.random.seed(seed)
    h, w = z.shape
    # fırça ağırlıkları
    R = radius
    offs_i = []
    offs_j = []
    wts = []
    tot = 0.0
    for di in range(-R, R + 1):
        for dj in range(-R, R + 1):
            dd = math.sqrt(di * di + dj * dj)
            if dd <= R:
                ww = R - dd
                offs_i.append(di)
                offs_j.append(dj)
                wts.append(ww)
                tot += ww
    nb = len(wts)
    cand = np.argwhere(spawn)
    for _ in range(n):
        k = np.random.randint(0, len(cand))
        y = cand[k, 0] + np.random.random()
        x = cand[k, 1] + np.random.random()
        dx_, dy_ = 0.0, 0.0
        speed, water, sed = 1.0, 1.0, 0.0
        for _s in range(max_steps):
            i, j = int(y), int(x)
            if i < 1 or j < 1 or i >= h - 2 or j >= w - 2:
                break
            u, v = x - j, y - i
            gx, gy, hh = _grad_h(z, x, y)
            dx_ = dx_ * inertia - gx * (1 - inertia)
            dy_ = dy_ * inertia - gy * (1 - inertia)
            L = math.sqrt(dx_ * dx_ + dy_ * dy_)
            if L < 1e-12:
                break
            dx_ /= L
            dy_ /= L
            x += dx_
            y += dy_
            if y < 1 or x < 1 or y >= h - 2 or x >= w - 2:
                break
            _, _, hn = _grad_h(z, x, y)
            dh = hn - hh
            cap = max(-dh, min_slope) * speed * water * capacity
            if sed > cap or dh > 0:
                amt = min(dh, sed) if dh > 0 else (sed - cap) * deposit
                sed -= amt
                z[i, j] += amt * (1 - u) * (1 - v)
                z[i, j + 1] += amt * u * (1 - v)
                z[i + 1, j] += amt * (1 - u) * v
                z[i + 1, j + 1] += amt * u * v
            else:
                amt = min((cap - sed) * erode, -dh)
                if not nonerode[i, j]:
                    got = 0.0
                    for q in range(nb):
                        ii, jj = i + offs_i[q], j + offs_j[q]
                        if 0 <= ii < h and 0 <= jj < w and not nonerode[ii, jj]:
                            e = amt * wts[q] / tot
                            e = min(e, max(0.0, z[ii, jj] - hn))       # damlanın indiği kotun altına oyma yok (kararlılık)
                            z[ii, jj] -= e
                            got += e
                    sed += got
            speed = min(6.0, math.sqrt(max(0.0, speed * speed - dh * gravity)))
            water *= (1 - evaporate)
    return z
