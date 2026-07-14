import itertools
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import least_squares

from main_functions.dimer_funcs import self_cons_equations_numba, up_limit_of_energy_of_state

def nm_to_occ(sol_nm):
    """[N1,M1,N2,M2] -> [n1_up,n1_dn,n2_up,n2_dn], each ideally in [0,1]."""
    N1, M1, N2, M2 = np.asarray(sol_nm, dtype=float)
    return np.array([
        0.5 * (N1 + M1),
        0.5 * (N1 - M1),
        0.5 * (N2 + M2),
        0.5 * (N2 - M2),
    ], dtype=float)


def occ_to_nm(occ):
    """[n1_up,n1_dn,n2_up,n2_dn] -> [N1,M1,N2,M2]."""
    n1u, n1d, n2u, n2d = np.asarray(occ, dtype=float)
    return np.array([
        n1u + n1d,
        n1u - n1d,
        n2u + n2d,
        n2u - n2d,
    ], dtype=float)


def self_cons_map_occ(occ, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    """Самосогласованное отображение в переменных заполнений [0,1]^4."""
    N1, M1, N2, M2 = occ_to_nm(occ)
    n1, m1, n2, m2 = self_cons_equations_numba(
        float(N1), float(M1), float(N2), float(M2),
        x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
    )
    return np.array([
        0.5 * (n1 + m1),
        0.5 * (n1 - m1),
        0.5 * (n2 + m2),
        0.5 * (n2 - m2),
    ], dtype=float)


def residual_occ(occ, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    return occ - self_cons_map_occ(occ, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)


def projected_picard(occ0, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
                     damping=0.65, n_steps=4):
    """Несколько дешёвых проекционных итераций для притяжения к физической области."""
    occ = np.clip(np.asarray(occ0, dtype=float), 0.0, 1.0)
    for _ in range(n_steps):
        fmap = self_cons_map_occ(occ, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
        occ = np.clip((1.0 - damping) * occ + damping * fmap, 0.0, 1.0)
    return occ


def _coarse_grid_points(grid_per_dim):
    vals = np.linspace(0.0, 1.0, grid_per_dim)
    return np.array(list(itertools.product(vals, repeat=4)), dtype=float)


def _halton_points(n, dim=4):
    # Детерминированная малодисперсная последовательность без дополнительных зависимостей.
    primes = [2, 3, 5, 7, 11, 13, 17, 19][:dim]

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


def _dedupe_rows(points, atol=1e-6):
    unique = []
    for p in points:
        if not any(np.allclose(p, q, atol=atol, rtol=0.0) for q in unique):
            unique.append(np.asarray(p, dtype=float))
    if not unique:
        return np.empty((0, 4), dtype=float)
    return np.vstack(unique)


def generate_seed_points(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
                         grid_per_dim=3, n_halton=64, picard_steps=3,
                         damping=0.65, max_start_points=64):
    grid_pts = _coarse_grid_points(grid_per_dim)
    halton_pts = _halton_points(n_halton, dim=4) if n_halton > 0 else np.empty((0, 4), dtype=float)

    # Полезно явно добавить физически симметричные углы гиперкуба и центр.
    special = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ], dtype=float)

    seeds = np.vstack([special, grid_pts, halton_pts])
    relaxed = []
    scored = []
    for s in seeds:
        sr = projected_picard(s, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
                              damping=damping, n_steps=picard_steps)
        r = residual_occ(sr, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
        score = float(np.linalg.norm(r, ord=2))
        relaxed.append(sr)
        scored.append((score, sr))

    scored.sort(key=lambda t: t[0])
    selected = _dedupe_rows([p for _, p in scored], atol=1e-4)
    return selected[:max_start_points]


class SolverResult(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def validate_solution_nm(sol_nm, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
                         residual_tol=1e-8, bound_tol=1e-8):
    sol_nm = np.asarray(sol_nm, dtype=float)
    occ = nm_to_occ(sol_nm)
    if np.any(occ < -bound_tol) or np.any(occ > 1.0 + bound_tol):
        return False, np.inf, occ

    rhs_occ = self_cons_map_occ(occ, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
    res = occ - rhs_occ
    res_norm = float(np.max(np.abs(res)))
    is_ok = np.isfinite(res_norm) and res_norm <= residual_tol and np.all(rhs_occ >= -bound_tol) and np.all(rhs_occ <= 1.0 + bound_tol)
    return bool(is_ok), res_norm, occ


def _solve_one_seed(payload):
    seed, params, lsq_kwargs = payload
    x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2 = params

    def fun(occ):
        return residual_occ(occ, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)

    seed = np.clip(np.asarray(seed, dtype=float), 0.0, 1.0)
    try:
        ans = least_squares(
            fun,
            seed,
            bounds=(np.zeros(4), np.ones(4)),
            method=lsq_kwargs.get('method', 'trf'),
            ftol=lsq_kwargs.get('ftol', 1e-10),
            xtol=lsq_kwargs.get('xtol', 1e-10),
            gtol=lsq_kwargs.get('gtol', 1e-10),
            max_nfev=lsq_kwargs.get('max_nfev', 80),
            x_scale=lsq_kwargs.get('x_scale', 'jac'),
            loss=lsq_kwargs.get('loss', 'linear'),
        )
        occ_sol = np.clip(ans.x, 0.0, 1.0)
        nm_sol = occ_to_nm(occ_sol)
        res_inf = float(np.max(np.abs(fun(occ_sol))))
        return {
            'success': bool(ans.success),
            'status': int(ans.status),
            'message': ans.message,
            'occ': occ_sol,
            'nm': nm_sol,
            'residual_inf': res_inf,
            'cost': float(ans.cost),
            'nfev': int(ans.nfev),
        }
    except Exception as exc:
        return {
            'success': False,
            'status': -999,
            'message': repr(exc),
            'occ': None,
            'nm': None,
            'residual_inf': np.inf,
            'cost': np.inf,
            'nfev': 0,
        }


def _cluster_solutions(candidates_nm, atol_nm=1e-6):
    unique = []
    for sol in candidates_nm:
        sol = np.asarray(sol, dtype=float)
        if not any(np.allclose(sol, u, atol=atol_nm, rtol=0.0) for u in unique):
            unique.append(sol)
    return unique


def self_cons_all_solutions_bounded(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
                                    grid_per_dim=3,
                                    n_halton=64,
                                    picard_steps=3,
                                    max_start_points=64,
                                    max_nfev=80,
                                    residual_tol=1e-8,
                                    uniqueness_tol=1e-6,
                                    damping=0.65,
                                    sort_by='energy',
                                    return_diagnostics=False):
    """
    Находит все физические решения x=f(x) в переменных
    u = [(N1+M1)/2, (N1-M1)/2, (N2+M2)/2, (N2-M2)/2] ∈ [0,1]^4.

    Идея:
    1) строим детерминированный набор стартов в [0,1]^4;
    2) делаем несколько проекционных Picard-итераций;
    3) полируем кандидаты bounded least_squares;
    4) оставляем только точки с малой невязкой.
    """
    params = (x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
    seeds = generate_seed_points(*params,
                                 grid_per_dim=grid_per_dim,
                                 n_halton=n_halton,
                                 picard_steps=picard_steps,
                                 damping=damping,
                                 max_start_points=max_start_points)

    lsq_kwargs = dict(max_nfev=max_nfev)
    raw = [_solve_one_seed((seed, params, lsq_kwargs)) for seed in seeds]

    accepted = []
    diagnostics = []
    for item in raw:
        diagnostics.append(item)
        if item['nm'] is None:
            continue
        ok, res_norm, occ = validate_solution_nm(item['nm'], *params, residual_tol=residual_tol)
        if ok:
            accepted.append(np.round(item['nm'], 12))

    unique = _cluster_solutions(accepted, atol_nm=uniqueness_tol)

    if sort_by == 'energy':
        unique.sort(key=lambda sol: float(up_limit_of_energy_of_state(sol, *params)))
    elif sort_by == 'residual':
        unique.sort(key=lambda sol: validate_solution_nm(sol, *params, residual_tol=np.inf)[1])

    if return_diagnostics:
        return SolverResult(solutions=unique, seeds=seeds, raw=raw)
    return unique


def self_cons_all_solutions_bounded_mp(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
                                       n_jobs=None,
                                       **kwargs):
    params = (x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
    seeds = generate_seed_points(*params,
                                 grid_per_dim=kwargs.get('grid_per_dim', 3),
                                 n_halton=kwargs.get('n_halton', 64),
                                 picard_steps=kwargs.get('picard_steps', 3),
                                 damping=kwargs.get('damping', 0.65),
                                 max_start_points=kwargs.get('max_start_points', 64))
    lsq_kwargs = dict(max_nfev=kwargs.get('max_nfev', 80))
    payloads = [(seed, params, lsq_kwargs) for seed in seeds]
    if n_jobs is None:
        n_jobs = cpu_count()
    with Pool(n_jobs) as pool:
        raw = pool.map(_solve_one_seed, payloads)

    residual_tol = kwargs.get('residual_tol', 1e-8)
    uniqueness_tol = kwargs.get('uniqueness_tol', 1e-6)
    accepted = []
    for item in raw:
        if item['nm'] is None:
            continue
        ok, _, _ = validate_solution_nm(item['nm'], *params, residual_tol=residual_tol)
        if ok:
            accepted.append(np.round(item['nm'], 12))

    unique = _cluster_solutions(accepted, atol_nm=uniqueness_tol)
    sort_by = kwargs.get('sort_by', 'energy')
    if sort_by == 'energy':
        unique.sort(key=lambda sol: float(up_limit_of_energy_of_state(sol, *params)))
    elif sort_by == 'residual':
        unique.sort(key=lambda sol: validate_solution_nm(sol, *params, residual_tol=np.inf)[1])
    return unique
