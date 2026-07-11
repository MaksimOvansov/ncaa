
import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def arccot(z):
    """
    Вещественная версия arccot, согласованная по ветви с кодом пользователя:
    arccot(z) in (0, pi) for real z.
    """
    z = float(z)
    if z == 0.0:
        return 0.5 * np.pi
    res = np.arctan(1.0 / z)
    if z < 0.0:
        res += np.pi
    return float(res)


def self_cons_equations_single(N, M, x, y, z, theta):
    """
    Правая часть самосогласованных уравнений одиночной примеси.

    Возвращает (n, m), где решение ищется из
        N = n(N, M; x, y, z, theta),
        M = m(N, M; x, y, z, theta).

    Реализована численно устойчивая обработка случая chi -> 0.
    """
    N = float(N)
    M = float(M)
    x = float(x)
    y = float(y)
    z = -1 * float(z)
    theta = float(theta)

    c = np.cos(theta)
    s = np.sin(theta)
    chi_sq = z ** 2 + (y * M) ** 2 - 2.0 * z * y * M * c
    if chi_sq < 0.0 and abs(chi_sq) < 1e-15:
        chi_sq = 0.0
    chi = np.sqrt(chi_sq)

    a = x + y * N
    n = (1.0 / np.pi) * (arccot(a + chi) + arccot(a - chi))

    pref_num = z * c - y * M
    if chi < 1e-12:
        # Лимит [arccot(a+chi)-arccot(a-chi)] / chi -> -2/(1+a^2)
        m = (-2.0 / np.pi) * pref_num / (1.0 + a * a)
    else:
        m = (1.0 / np.pi) * (arccot(a + chi) - arccot(a - chi)) * (pref_num / chi)

    return float(n), float(m)


def residual_single_nm(nm, x, y, z, theta):
    N, M = np.asarray(nm, dtype=float)
    n, m = self_cons_equations_single(N, M, x, y, z, theta)
    return np.array([N - n, M - m], dtype=float)


def residual_single_xm(xm, N_fixed, y, z, theta):
    x, M = np.asarray(xm, dtype=float)
    n, m = self_cons_equations_single(N_fixed, M, x, y, z, theta)
    return np.array([N_fixed - n, M - m], dtype=float)


def _clip_nm(nm):
    N, M = np.asarray(nm, dtype=float)
    return np.array([np.clip(N, 0.0, 2.0), np.clip(M, -1.0, 1.0)], dtype=float)


def _clip_xm(xm, x_bounds):
    x, M = np.asarray(xm, dtype=float)
    xl, xr = x_bounds
    return np.array([np.clip(x, xl, xr), np.clip(M, -1.0, 1.0)], dtype=float)


def projected_picard_nm(nm0, x, y, z, theta, damping=0.65, n_steps=3):
    nm = _clip_nm(nm0)
    for _ in range(n_steps):
        rhs = np.array(self_cons_equations_single(nm[0], nm[1], x, y, z, theta), dtype=float)
        nm = _clip_nm((1.0 - damping) * nm + damping * rhs)
    return nm


def _halton_points(n, dim=2):
    primes = [2, 3, 5, 7][:dim]

    def van_der_corput(index, base):
        f = 1.0 / base
        r = 0.0
        i = index
        while i > 0:
            r += f * (i % base)
            i //= base
            f /= base
        return r

    pts = np.empty((n, dim), dtype=float)
    for i in range(n):
        idx = i + 1
        for j, b in enumerate(primes):
            pts[i, j] = van_der_corput(idx, b)
    return pts


def _dedupe_rows(points, atol=1e-8):
    unique = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if not any(np.allclose(p, q, atol=atol, rtol=0.0) for q in unique):
            unique.append(p)
    if not unique:
        return np.empty((0, 2), dtype=float)
    return np.vstack(unique)


def _coarse_grid(bounds1, bounds2, n1, n2):
    v1 = np.linspace(bounds1[0], bounds1[1], int(n1))
    v2 = np.linspace(bounds2[0], bounds2[1], int(n2))
    return np.array(list(itertools.product(v1, v2)), dtype=float)


def generate_nm_seeds(x, y, z, theta,
                      N_grid=7, M_grid=7,
                      n_halton=24,
                      picard_steps=3,
                      damping=0.65,
                      max_start_points=24):
    grid_pts = _coarse_grid((0.0, 2.0), (-1.0, 1.0), N_grid, M_grid)
    halton = _halton_points(n_halton, dim=2) if n_halton > 0 else np.empty((0, 2), dtype=float)
    if len(halton):
        halton[:, 0] = 2.0 * halton[:, 0]
        halton[:, 1] = -1.0 + 2.0 * halton[:, 1]

    special = np.array([
        [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
        [1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
        [2.0, -1.0], [2.0, 0.0], [2.0, 1.0],
    ], dtype=float)

    seeds = np.vstack([special, grid_pts, halton])
    scored = []
    for s in seeds:
        sr = projected_picard_nm(s, x, y, z, theta, damping=damping, n_steps=picard_steps)
        score = float(np.linalg.norm(residual_single_nm(sr, x, y, z, theta), ord=2))
        scored.append((score, sr))
    scored.sort(key=lambda t: t[0])
    selected = _dedupe_rows([p for _, p in scored], atol=1e-6)
    return selected[:max_start_points]


def generate_xm_seeds(N_fixed, y, z, theta,
                      x_bounds=(-20.0, 20.0),
                      x_grid=9, M_grid=7,
                      n_halton=32,
                      max_start_points=32):
    xl, xr = map(float, x_bounds)
    grid_pts = _coarse_grid((xl, xr), (-1.0, 1.0), x_grid, M_grid)
    halton = _halton_points(n_halton, dim=2) if n_halton > 0 else np.empty((0, 2), dtype=float)
    if len(halton):
        halton[:, 0] = xl + (xr - xl) * halton[:, 0]
        halton[:, 1] = -1.0 + 2.0 * halton[:, 1]

    special = np.array([
        [xl, -1.0], [xl, 0.0], [xl, 1.0],
        [0.5 * (xl + xr), -1.0], [0.5 * (xl + xr), 0.0], [0.5 * (xl + xr), 1.0],
        [xr, -1.0], [xr, 0.0], [xr, 1.0],
    ], dtype=float)

    seeds = np.vstack([special, grid_pts, halton])
    scored = []
    for s in seeds:
        s = _clip_xm(s, x_bounds)
        score = float(np.linalg.norm(residual_single_xm(s, N_fixed, y, z, theta), ord=2))
        scored.append((score, s))
    scored.sort(key=lambda t: t[0])
    selected = _dedupe_rows([p for _, p in scored], atol=1e-6)
    return selected[:max_start_points]


@dataclass
class SolverDiagnostics:
    solutions: list
    seeds: np.ndarray
    raw: list


def validate_nm_solution(sol, x, y, z, theta, residual_tol=1e-9, bound_tol=1e-10):
    sol = np.asarray(sol, dtype=float)
    N, M = sol
    if not (0.0 - bound_tol <= N <= 2.0 + bound_tol and -1.0 - bound_tol <= M <= 1.0 + bound_tol):
        return False, np.inf
    res = residual_single_nm(sol, x, y, z, theta)
    res_inf = float(np.max(np.abs(res)))
    return bool(np.isfinite(res_inf) and res_inf <= residual_tol), res_inf


def validate_xm_solution(sol, N_fixed, y, z, theta, x_bounds=(-20.0, 20.0), residual_tol=1e-9, bound_tol=1e-10):
    sol = np.asarray(sol, dtype=float)
    x, M = sol
    xl, xr = x_bounds
    if not (xl - bound_tol <= x <= xr + bound_tol and -1.0 - bound_tol <= M <= 1.0 + bound_tol and 0.0 - bound_tol <= N_fixed <= 2.0 + bound_tol):
        return False, np.inf
    res = residual_single_xm(sol, N_fixed, y, z, theta)
    res_inf = float(np.max(np.abs(res)))
    return bool(np.isfinite(res_inf) and res_inf <= residual_tol), res_inf


def _cluster_solutions(candidates, atol=1e-7):
    unique = []
    for sol in candidates:
        sol = np.asarray(sol, dtype=float)
        if not any(np.allclose(sol, u, atol=atol, rtol=0.0) for u in unique):
            unique.append(sol)
    return unique


def self_cons_single(x, y, z, theta,
                     N_grid=7,
                     M_grid=7,
                     n_halton=24,
                     picard_steps=3,
                     max_start_points=24,
                     max_nfev=80,
                     residual_tol=1e-9,
                     uniqueness_tol=1e-7,
                     damping=0.65,
                     return_diagnostics=False):
    """
    Решатель для фиксированных (x, y, z, theta) по неизвестным (N, M).

    Возвращает список решений:
        [np.array([N, M]), ...]
    с ограничениями 0 <= N <= 2, -1 <= M <= 1.
    """
    seeds = generate_nm_seeds(x, y, z, theta,
                              N_grid=N_grid, M_grid=M_grid,
                              n_halton=n_halton,
                              picard_steps=picard_steps,
                              damping=damping,
                              max_start_points=max_start_points)
    raw = []
    accepted = []
    for seed in seeds:
        ans = least_squares(
            lambda nm: residual_single_nm(nm, x, y, z, theta),
            seed,
            bounds=([0.0, -1.0], [2.0, 1.0]),
            method='trf',
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=max_nfev,
            x_scale='jac',
            loss='linear',
        )
        sol = _clip_nm(ans.x)
        ok, res_inf = validate_nm_solution(sol, x, y, z, theta, residual_tol=residual_tol)
        item = {
            'success': bool(ans.success),
            'status': int(ans.status),
            'message': ans.message,
            'sol': sol,
            'residual_inf': res_inf,
            'cost': float(ans.cost),
            'nfev': int(ans.nfev),
        }
        raw.append(item)
        if ok:
            accepted.append(np.round(sol, 12))

    unique = _cluster_solutions(accepted, atol=uniqueness_tol)
    unique.sort(key=lambda s: (float(s[0]), float(s[1])))

    if return_diagnostics:
        return SolverDiagnostics(solutions=unique, seeds=seeds, raw=raw)
    return unique


def self_cons_single_fixed_N(N, y, z, theta,
                             x_bounds=(-20.0, 20.0),
                             x_grid=9,
                             M_grid=7,
                             n_halton=32,
                             max_start_points=32,
                             max_nfev=80,
                             residual_tol=1e-9,
                             uniqueness_tol=1e-7,
                             return_diagnostics=False):
    """
    Решатель для фиксированных (N, y, z, theta) по неизвестным (x, M).

    ВАЖНО:
    x не ограничивается самой системой, поэтому для полноты поиска
    нужно явно задавать диапазон x_bounds.

    Возвращает список решений:
        [np.array([x, M]), ...]
    с ограничениями x in x_bounds, -1 <= M <= 1, 0 <= N <= 2.
    """
    N = float(N)
    if not (0.0 <= N <= 2.0):
        raise ValueError("Для физически допустимого решения фиксированное N должно лежать в [0, 2].")

    seeds = generate_xm_seeds(N, y, z, theta,
                              x_bounds=x_bounds,
                              x_grid=x_grid, M_grid=M_grid,
                              n_halton=n_halton,
                              max_start_points=max_start_points)
    xl, xr = map(float, x_bounds)

    raw = []
    accepted = []
    for seed in seeds:
        ans = least_squares(
            lambda xm: residual_single_xm(xm, N, y, z, theta),
            seed,
            bounds=([xl, -1.0], [xr, 1.0]),
            method='trf',
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=max_nfev,
            x_scale='jac',
            loss='linear',
        )
        sol = _clip_xm(ans.x, x_bounds)
        ok, res_inf = validate_xm_solution(sol, N, y, z, theta, x_bounds=x_bounds, residual_tol=residual_tol)
        item = {
            'success': bool(ans.success),
            'status': int(ans.status),
            'message': ans.message,
            'sol': sol,
            'residual_inf': res_inf,
            'cost': float(ans.cost),
            'nfev': int(ans.nfev),
        }
        raw.append(item)
        if ok:
            accepted.append(np.round(sol, 12))

    unique = _cluster_solutions(accepted, atol=uniqueness_tol)
    unique.sort(key=lambda s: (float(s[0]), float(s[1])))

    if return_diagnostics:
        return SolverDiagnostics(solutions=unique, seeds=seeds, raw=raw)
    return unique
