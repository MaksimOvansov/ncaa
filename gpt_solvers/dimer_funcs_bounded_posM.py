from gpt_solvers.dimer_solver_alt_posM import (
    nm_to_occ,
    occ_to_nm,
    posm_vars_to_occ,
    occ_to_posm_vars,
    self_cons_map_occ,
    residual_occ,
    residual_posm,
    projected_picard_posm,
    validate_solution_nm_posm,
    self_cons_all_solutions_bounded_posM,
    self_cons_all_solutions_bounded_posM_mp,
)


def self_cons_bounded_posM(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs):
    return self_cons_all_solutions_bounded_posM(
        x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs
    )


def self_cons_bounded_posM_multiprocessing(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs):
    return self_cons_all_solutions_bounded_posM_mp(
        x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs
    )
