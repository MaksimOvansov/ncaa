import itertools
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import least_squares

from dimer_funcs import self_cons_equations_numba


class SolverResult(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def feasible_u_interval_nonnegative_M(N, tol=1e-12):
    """
    Для фиксированного N и дополнительного условия M >= 0
    используем параметризацию
        u = (N + M)/2,
        M = 2u - N.

    Физические ограничения
        0 <= (N + M)/2 <= 1,
        0 <= (N - M)/2 <= 1
    дают
        u in [max(0, N-1), min(1, N)],
    а условие
        M >= 0
    эквивалентно
        u >= N/2.

    Итого
        u in [max(0, N-1, N/2), min(1, N)].

    Для N in [0, 2] это упрощается до
        u in [N/2, min(1, N)].
    """
    N = float(N)
    lo = max(0.0, N - 1.0, 0.5 * N)
    hi = min(1.0, N)
    if hi < lo - tol:
        raise ValueError(
            f"Для фиксированного N={N} не существует физически допустимого M. "
            f"Нужно N in [0, 2]."
        )
    return lo, hi


def u_to_M(u, N):
    return 2.0 * float(u) - float(N)


def M_to_u(M, N):
    return 0.5 * (float(N) + float(M))


def vars_to_xf_M1_M2(vars_xu, N_1, N_2):
    x_f, u_1, u_2 = np.asarray(vars_xu, dtype=float)
    M_1 = u_to_M(u_1, N_1)
    M_2 = u_to_M(u_2, N_2)
    return float(x_f), float(M_1), float(M_2)


def xf_M1_M2_to_vars(x_f, M_1, M_2, N_1, N_2):
    return np.array([
        float(x_f),
        M_to_u(M_1, N_1),
        M_to_u(M_2, N_2),
    ], dtype=float)


def residual_fixed_N_variable_xf_nonnegative_M(vars_xu, N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2):
    """
    Переопределённая система из 4 уравнений на 3 неизвестных:
        N_1 = n_1(x_f, M_1, M_2),
        M_1 = m_1(x_f, M_1, M_2),
        N_2 = n_2(x_f, M_1, M_2),
        M_2 = m_2(x_f, M_1, M_2).

    Неизвестные параметризованы как [x_f, u_1, u_2],
    где u_i = (N_i + M_i)/2 и физические ограничения превращаются в box bounds.
    """
    x_f, M_1, M_2 = vars_to_xf_M1_M2(vars_xu, N_1, N_2)

    n1, m1, n2, m2 = self_cons_equations_numba(
        float(N_1), float(M_1), float(N_2), float(M_2),
        x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
    )

    return np.array([
        float(N_1 - n1),
        float(M_1 - m1),
        float(N_2 - n2),
        float(M_2 - m2),
    ], dtype=float)


def validate_solution_fixed_N_variable_xf_nonnegative_M(sol_xfM, N_1, N_2, x_0, y, z, w,
                                          theta_1, theta_2, phi_1, phi_2,
                                          xf_bounds=(-20.0, 20.0),
                                          residual_tol=1e-8,
                                          bound_tol=1e-8):
    x_f, M_1, M_2 = np.asarray(sol_xfM, dtype=float)

    xf_min, xf_max = map(float, xf_bounds)
    if x_f < xf_min - bound_tol or x_f > xf_max + bound_tol:
        return False, np.inf

    u_1 = 0.5 * (float(N_1) + M_1)
    d_1 = 0.5 * (float(N_1) - M_1)
    u_2 = 0.5 * (float(N_2) + M_2)
    d_2 = 0.5 * (float(N_2) - M_2)

    occ = np.array([u_1, d_1, u_2, d_2], dtype=float)
    if np.any(occ < -bound_tol) or np.any(occ > 1.0 + bound_tol):
        return False, np.inf
    if M_1 < -bound_tol or M_2 < -bound_tol:
        return False, np.inf

    vars_xu = xf_M1_M2_to_vars(x_f, M_1, M_2, N_1, N_2)
    res = residual_fixed_N_variable_xf_nonnegative_M(
        vars_xu, N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2
    )
    res_norm = float(np.max(np.abs(res)))
    return bool(np.isfinite(res_norm) and res_norm <= residual_tol), res_norm


def _halton_points(n, dim=3):
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


def _scale_box_points(unit_pts, bounds):
    bounds = np.asarray(bounds, dtype=float)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return lo + unit_pts * (hi - lo)


def _dedupe_rows(points, atol=1e-8):
    unique = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if not any(np.allclose(p, q, atol=atol, rtol=0.0) for q in unique):
            unique.append(p)
    if not unique:
        return np.empty((0, 3), dtype=float)
    return np.vstack(unique)


def _special_seed_points(xf_bounds, u1_bounds, u2_bounds):
    xf_min, xf_max = xf_bounds
    u1_min, u1_max = u1_bounds
    u2_min, u2_max = u2_bounds
    xf_mid = 0.5 * (xf_min + xf_max)
    u1_mid = 0.5 * (u1_min + u1_max)
    u2_mid = 0.5 * (u2_min + u2_max)

    return np.array([
        [xf_min, u1_min, u2_min],
        [xf_min, u1_max, u2_max],
        [xf_max, u1_min, u2_min],
        [xf_max, u1_max, u2_max],
        [xf_mid, u1_mid, u2_mid],
        [xf_mid, u1_min, u2_mid],
        [xf_mid, u1_max, u2_mid],
        [xf_mid, u1_mid, u2_min],
        [xf_mid, u1_mid, u2_max],
    ], dtype=float)


def _score_seed(seed, params):
    return float(np.linalg.norm(residual_fixed_N_variable_xf_nonnegative_M(seed, *params), ord=2))


def generate_seed_points_fixed_N_variable_xf_nonnegative_M(N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2,
                                             xf_bounds=(-20.0, 20.0),
                                             xf_grid=9,
                                             u_grid=3,
                                             n_halton=16,
                                             per_xf_keep=2,
                                             max_start_points=24):
    """
    Генерация стартов:
    1) грубая детерминированная сетка по (x_f, u1, u2),
    2) несколько специальных точек,
    3) малодисперсные Halton-старты.

    Затем старты ранжируются по norm(residual) и прореживаются.
    Отдельно сохраняем лучшие точки для каждого среза x_f, чтобы не потерять решения
    в разных диапазонах x_f.
    """
    u1_bounds = feasible_u_interval_nonnegative_M(N_1)
    u2_bounds = feasible_u_interval_nonnegative_M(N_2)
    xf_bounds = tuple(map(float, xf_bounds))

    xf_vals = np.linspace(xf_bounds[0], xf_bounds[1], int(xf_grid))
    u1_vals = np.linspace(u1_bounds[0], u1_bounds[1], int(u_grid))
    u2_vals = np.linspace(u2_bounds[0], u2_bounds[1], int(u_grid))

    params = (N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2)

    grid_selected = []
    for xf in xf_vals:
        slice_pts = np.array(list(itertools.product([xf], u1_vals, u2_vals)), dtype=float)
        scored = [(_score_seed(pt, params), pt) for pt in slice_pts]
        scored.sort(key=lambda t: t[0])
        for _, pt in scored[:max(1, int(per_xf_keep))]:
            grid_selected.append(pt)

    special = _special_seed_points(xf_bounds, u1_bounds, u2_bounds)

    if n_halton > 0:
        unit_halton = _halton_points(int(n_halton), dim=3)
        halton = _scale_box_points(unit_halton, np.array([
            [xf_bounds[0], xf_bounds[1]],
            [u1_bounds[0], u1_bounds[1]],
            [u2_bounds[0], u2_bounds[1]],
        ], dtype=float))
        halton_scored = [(_score_seed(pt, params), pt) for pt in halton]
        halton_scored.sort(key=lambda t: t[0])
        n_halton_keep = max(4, int(np.ceil(max_start_points / 3)))
        halton_selected = [pt for _, pt in halton_scored[:n_halton_keep]]
    else:
        halton_selected = []

    seeds = _dedupe_rows(np.vstack([
        np.asarray(grid_selected, dtype=float),
        special,
        np.asarray(halton_selected, dtype=float) if len(halton_selected) else np.empty((0, 3), dtype=float),
    ]), atol=1e-6)

    scored = [(_score_seed(pt, params), pt) for pt in seeds]
    scored.sort(key=lambda t: t[0])
    selected = _dedupe_rows([pt for _, pt in scored], atol=1e-5)
    return selected[:int(max_start_points)]


def _solve_one_seed(payload):
    seed, params, bounds, lsq_kwargs = payload

    def fun(vars_xu):
        return residual_fixed_N_variable_xf_nonnegative_M(vars_xu, *params)

    lower, upper = bounds
    seed = np.clip(np.asarray(seed, dtype=float), lower, upper)

    try:
        ans = least_squares(
            fun,
            seed,
            bounds=(lower, upper),
            method=lsq_kwargs.get('method', 'trf'),
            ftol=lsq_kwargs.get('ftol', 1e-10),
            xtol=lsq_kwargs.get('xtol', 1e-10),
            gtol=lsq_kwargs.get('gtol', 1e-10),
            max_nfev=lsq_kwargs.get('max_nfev', 80),
            x_scale=lsq_kwargs.get('x_scale', 'jac'),
            loss=lsq_kwargs.get('loss', 'linear'),
        )
        vars_sol = np.clip(ans.x, lower, upper)
        x_f, M_1, M_2 = vars_to_xf_M1_M2(vars_sol, params[0], params[1])
        res_inf = float(np.max(np.abs(fun(vars_sol))))
        return {
            'success': bool(ans.success),
            'status': int(ans.status),
            'message': ans.message,
            'vars': vars_sol,
            'solution': np.array([x_f, M_1, M_2], dtype=float),
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
            'solution': None,
            'residual_inf': np.inf,
            'cost': np.inf,
            'nfev': 0,
        }


def _cluster_solutions(candidates, atol=1e-6):
    unique = []
    for sol in candidates:
        sol = np.asarray(sol, dtype=float)
        if not any(np.allclose(sol, u, atol=atol, rtol=0.0) for u in unique):
            unique.append(sol)
    return unique


def self_cons_all_solutions_fixed_N_variable_xf_nonnegative_M(N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2,
                                                xf_bounds=(-20.0, 20.0),
                                                xf_grid=9,
                                                u_grid=3,
                                                n_halton=16,
                                                per_xf_keep=2,
                                                max_start_points=24,
                                                max_nfev=80,
                                                residual_tol=1e-8,
                                                uniqueness_tol=1e-6,
                                                return_diagnostics=False):
    """
    Находит все физические решения при фиксированных N_1 и N_2,
    когда x_f является неизвестной переменной.

    Неизвестные:
        [x_f, u_1, u_2],
    где
        u_i = (N_i + M_i)/2.

    На выходе возвращаются точки
        [x_f, M_1, M_2].

    Важно:
    - решается переопределённая система из 4 невязок на 3 неизвестных;
    - решение принимается только если все 4 невязки одновременно малы.
    """
    u1_bounds = feasible_u_interval_nonnegative_M(N_1)
    u2_bounds = feasible_u_interval_nonnegative_M(N_2)
    xf_bounds = tuple(map(float, xf_bounds))

    params = (N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2)
    seeds = generate_seed_points_fixed_N_variable_xf_nonnegative_M(
        N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2,
        xf_bounds=xf_bounds,
        xf_grid=xf_grid,
        u_grid=u_grid,
        n_halton=n_halton,
        per_xf_keep=per_xf_keep,
        max_start_points=max_start_points,
    )

    lower = np.array([xf_bounds[0], u1_bounds[0], u2_bounds[0]], dtype=float)
    upper = np.array([xf_bounds[1], u1_bounds[1], u2_bounds[1]], dtype=float)
    lsq_kwargs = dict(max_nfev=max_nfev)

    raw = [_solve_one_seed((seed, params, (lower, upper), lsq_kwargs)) for seed in seeds]

    accepted = []
    for item in raw:
        if item['solution'] is None:
            continue
        ok, _ = validate_solution_fixed_N_variable_xf_nonnegative_M(
            item['solution'],
            N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2,
            xf_bounds=xf_bounds,
            residual_tol=residual_tol,
        )
        if ok:
            accepted.append(np.round(item['solution'], 12))

    unique = _cluster_solutions(accepted, atol=uniqueness_tol)
    unique.sort(key=lambda sol: (float(sol[0]), float(sol[1]), float(sol[2])))

    if return_diagnostics:
        return SolverResult(solutions=unique, seeds=seeds, raw=raw)
    return unique


def self_cons_all_solutions_fixed_N_variable_xf_nonnegative_M_mp(N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2,
                                                   n_jobs=None,
                                                   **kwargs):
    u1_bounds = feasible_u_interval_nonnegative_M(N_1)
    u2_bounds = feasible_u_interval_nonnegative_M(N_2)
    xf_bounds = tuple(map(float, kwargs.get('xf_bounds', (-20.0, 20.0))))

    params = (N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2)
    seeds = generate_seed_points_fixed_N_variable_xf_nonnegative_M(
        N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2,
        xf_bounds=xf_bounds,
        xf_grid=kwargs.get('xf_grid', 9),
        u_grid=kwargs.get('u_grid', 3),
        n_halton=kwargs.get('n_halton', 16),
        per_xf_keep=kwargs.get('per_xf_keep', 2),
        max_start_points=kwargs.get('max_start_points', 24),
    )

    lower = np.array([xf_bounds[0], u1_bounds[0], u2_bounds[0]], dtype=float)
    upper = np.array([xf_bounds[1], u1_bounds[1], u2_bounds[1]], dtype=float)
    lsq_kwargs = dict(max_nfev=kwargs.get('max_nfev', 80))

    payloads = [(seed, params, (lower, upper), lsq_kwargs) for seed in seeds]
    if n_jobs is None:
        n_jobs = cpu_count()

    with Pool(n_jobs) as pool:
        raw = pool.map(_solve_one_seed, payloads)

    residual_tol = kwargs.get('residual_tol', 1e-8)
    uniqueness_tol = kwargs.get('uniqueness_tol', 1e-6)

    accepted = []
    for item in raw:
        if item['solution'] is None:
            continue
        ok, _ = validate_solution_fixed_N_variable_xf_nonnegative_M(
            item['solution'],
            N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2,
            xf_bounds=xf_bounds,
            residual_tol=residual_tol,
        )
        if ok:
            accepted.append(np.round(item['solution'], 12))

    unique = _cluster_solutions(accepted, atol=uniqueness_tol)
    unique.sort(key=lambda sol: (float(sol[0]), float(sol[1]), float(sol[2])))
    return unique
