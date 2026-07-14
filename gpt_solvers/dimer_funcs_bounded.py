from gpt_solvers.dimer_solver_alt import (
    nm_to_occ,
    occ_to_nm,
    self_cons_map_occ,
    residual_occ,
    projected_picard,
    validate_solution_nm,
    self_cons_all_solutions_bounded,
    self_cons_all_solutions_bounded_mp,
)


# drop-in friendly aliases

# def self_cons_bounded(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs):
#     return self_cons_all_solutions_bounded(
#         x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs
#     )

def self_cons_bounded(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, grid_per_dim=3, n_halton=64, picard_steps=3, max_start_points=64, max_nfev=80, residual_tol=1e-8, uniqueness_tol=1e-6, **kwargs):
    return self_cons_all_solutions_bounded(
        x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, grid_per_dim=3, n_halton=64, picard_steps=3, max_start_points=64, max_nfev=80, residual_tol=1e-8, uniqueness_tol=1e-6, **kwargs
    )


def self_cons_bounded_multiprocessing(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs):
    return self_cons_all_solutions_bounded_mp(
        x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs
    )
