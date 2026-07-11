import itertools
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import least_squares

from dimer_funcs import self_cons_equations_numba


# === преобразования переменных ===

def nm_to_occ(sol_nm):
    """[N1,M1,N2,M2] -> [n1_up,n1_dn,n2_up,n2_dn]."""
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


def posm_vars_to_occ(v):
    """
    v = [a1, s1, a2, s2],  a_i,s_i in [0,1]
    n_up = a, n_dn = a*s  => 0 <= n_dn <= n_up <= 1, следовательно M >= 0.
    """
    a1, s1, a2, s2 = np.asarray(v, dtype=float)
    return np.array([
        a1,
        a1 * s1,
        a2,
        a2 * s2,
    ], dtype=float)


def occ_to_posm_vars(occ):
    """Обратное преобразование для начальных точек."""
    n1u, n1d, n2u, n2d = np.asarray(occ, dtype=float)
    s1 = 0.0 if n1u <= 1e-15 else np.clip(n1d / n1u, 0.0, 1.0)
    s2 = 0.0 if n2u <= 1e-15 else np.clip(n2d / n2u, 0.0, 1.0)
    return np.array([
        np.clip(n1u, 0.0, 1.0),
        s1,
        np.clip(n2u, 0.0, 1.0),
        s2,
    ], dtype=float)


# === самосогласованное отображение ===

def self_cons_map_occ(occ, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
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


def residual_posm(v, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    occ = posm_vars_to_occ(v)
    return residual_occ(occ, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)


def projected_picard_posm(v0, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
                          damping=0.65, n_steps=4):
    """Проекционные Picard-итерации в occ-пространстве с возвратом в posM-параметризацию."""
    v = np.clip(np.asarray(v0, dtype=float), 0.0, 1.0)
    occ = posm_vars_to_occ(v)
    for _ in range(n_steps):
        fmap = self_cons_map_occ(occ, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
        # enforce M1 >= 0, M2 >= 0 by sorting each spin pair so that n_up >= n_dn
        fmap = np.array([
            max(fmap[0], fmap[1]), min(fmap[0], fmap[1]),
            max(fmap[2], fmap[3]), min(fmap[2], fmap[3]),
        ], dtype=float)
        occ = np.clip((1.0 - damping) * occ + damping * fmap, 0.0, 1.0)
        occ = np.array([
            max(occ[0], occ[1]), min(occ[0], occ[1]),
            max(occ[2], occ[3]), min(occ[2], occ[3]),
        ], dtype=float)
    return occ_to_posm_vars(occ)


# === генерация стартов ===

def _coarse_grid_points_posm(grid_per_dim):
    vals = np.linspace(0.0, 1.0, grid_per_dim)
    return np.array(list(itertools.product(vals, repeat=4)), dtype=float)


def _halton_points(n, dim=4):
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


def generate_seed_points_posm(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
                              grid_per_dim=3, n_halton=64, picard_steps=3,
                              damping=0.65, max_start_points=64):
    grid_pts = _coarse_grid_points_posm(grid_per_dim)
    halton_pts = _halton_points(n_halton, dim=4) if n_halton > 0 else np.empty((0, 4), dtype=float)

    special = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 0.5, 1.0, 0.5],
        [0.75, 0.25, 0.75, 0.25],
    ], dtype=float)

    seeds = np.vstack([special, grid_pts, halton_pts])
    scored = []
    for s in seeds:
        sr = projected_picard_posm(s, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
                                   damping=damping, n_steps=picard_steps)
        occ = posm_vars_to_occ(sr)
        r = residual_occ(occ, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
        score = float(np.linalg.norm(r, ord=2))
        scored.append((score, sr))

    scored.sort(key=lambda t: t[0])
    selected = _dedupe_rows([p for _, p in scored], atol=1e-4)
    return selected[:max_start_points]


class SolverResult(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


# === проверка решений ===

def validate_solution_nm_posm(sol_nm, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
                              residual_tol=1e-8, bound_tol=1e-8):
    sol_nm = np.asarray(sol_nm, dtype=float)
    occ = nm_to_occ(sol_nm)
    M1 = float(sol_nm[1])
    M2 = float(sol_nm[3])

    if np.any(occ < -bound_tol) or np.any(occ > 1.0 + bound_tol):
        return False, np.inf, occ
    if M1 < -bound_tol or M2 < -bound_tol:
        return False, np.inf, occ

    rhs_occ = self_cons_map_occ(occ, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
    res = occ - rhs_occ
    res_norm = float(np.max(np.abs(res)))
    is_ok = (
        np.isfinite(res_norm)
        and res_norm <= residual_tol
        and np.all(rhs_occ >= -bound_tol)
        and np.all(rhs_occ <= 1.0 + bound_tol)
        and M1 >= -bound_tol
        and M2 >= -bound_tol
    )
    return bool(is_ok), res_norm, occ


def _solve_one_seed_posm(payload):
    seed, params, lsq_kwargs = payload
    x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2 = params

    def fun(v):
        return residual_posm(v, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)

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
        v_sol = np.clip(ans.x, 0.0, 1.0)
        occ_sol = posm_vars_to_occ(v_sol)
        nm_sol = occ_to_nm(occ_sol)
        res_inf = float(np.max(np.abs(residual_occ(occ_sol, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2))))
        return {
            'success': bool(ans.success),
            'status': int(ans.status),
            'message': ans.message,
            'vars': v_sol,
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
            'vars': None,
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


# === публичные функции ===

def self_cons_all_solutions_bounded_posM(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
                                         grid_per_dim=3,
                                         n_halton=64,
                                         picard_steps=3,
                                         max_start_points=64,
                                         max_nfev=80,
                                         residual_tol=1e-8,
                                         uniqueness_tol=1e-6,
                                         damping=0.65,
                                         return_diagnostics=False):
    """
    Находит все физические решения x=f(x) в переменных [N1,M1,N2,M2]
    при дополнительных ограничениях M1 >= 0 и M2 >= 0.

    Параметризация на каждом атоме:
        n_up = a, n_dn = a*s,  a,s in [0,1]
    поэтому автоматически
        0 <= n_dn <= n_up <= 1,
    а значит
        0 <= (N_i ± M_i)/2 <= 1  и  M_i >= 0.
    """
    params = (x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
    seeds = generate_seed_points_posm(*params,
                                      grid_per_dim=grid_per_dim,
                                      n_halton=n_halton,
                                      picard_steps=picard_steps,
                                      damping=damping,
                                      max_start_points=max_start_points)

    lsq_kwargs = dict(max_nfev=max_nfev)
    raw = [_solve_one_seed_posm((seed, params, lsq_kwargs)) for seed in seeds]

    accepted = []
    for item in raw:
        if item['nm'] is None:
            continue
        ok, _, _ = validate_solution_nm_posm(item['nm'], *params, residual_tol=residual_tol)
        if ok:
            accepted.append(np.round(item['nm'], 12))

    unique = _cluster_solutions(accepted, atol_nm=uniqueness_tol)
    # детерминированный порядок без сортировки по энергии
    unique.sort(key=lambda sol: (sol[0], sol[2], sol[1], sol[3]))

    if return_diagnostics:
        return SolverResult(solutions=unique, seeds=seeds, raw=raw)
    return unique


def self_cons_all_solutions_bounded_posM_mp(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
                                            n_jobs=None,
                                            **kwargs):
    params = (x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
    seeds = generate_seed_points_posm(*params,
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
        raw = pool.map(_solve_one_seed_posm, payloads)

    residual_tol = kwargs.get('residual_tol', 1e-8)
    uniqueness_tol = kwargs.get('uniqueness_tol', 1e-6)
    accepted = []
    for item in raw:
        if item['nm'] is None:
            continue
        ok, _, _ = validate_solution_nm_posm(item['nm'], *params, residual_tol=residual_tol)
        if ok:
            accepted.append(np.round(item['nm'], 12))

    unique = _cluster_solutions(accepted, atol_nm=uniqueness_tol)
    unique.sort(key=lambda sol: (sol[0], sol[2], sol[1], sol[3]))
    return unique
