from gpt_solvers.dimer_solver_fixedN_xf_posM import (
    feasible_u_interval_nonnegative_M,
    residual_fixed_N_variable_xf_nonnegative_M,
    validate_solution_fixed_N_variable_xf_nonnegative_M,
    self_cons_all_solutions_fixed_N_variable_xf_nonnegative_M,
    self_cons_all_solutions_fixed_N_variable_xf_nonnegative_M_mp,
)


# def self_cons_fixed_N_xf_pM(N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs):
#     return self_cons_all_solutions_fixed_N_variable_xf_nonnegative_M(
#         N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs
#     )


def self_cons_fixed_N_xf_pM(N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2, xf_bounds=(-100.0, 100.0), xf_grid=41, u_grid=4, n_halton=24, per_xf_keep=2, max_start_points=96, max_nfev=100, residual_tol=1e-8, uniqueness_tol=1e-6, **kwargs):
    return self_cons_all_solutions_fixed_N_variable_xf_nonnegative_M(
        N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2, xf_bounds=(-100.0, 100.0), xf_grid=41, u_grid=4, n_halton=24, per_xf_keep=2, max_start_points=96, max_nfev=100, residual_tol=1e-8, uniqueness_tol=1e-6, **kwargs
    )


def self_cons_fixed_N_variable_xf_nonnegative_M_multiprocessing(N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs):
    return self_cons_all_solutions_fixed_N_variable_xf_nonnegative_M_mp(
        N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs
    )
