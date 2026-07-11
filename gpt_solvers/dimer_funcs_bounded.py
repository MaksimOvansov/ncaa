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

def self_cons_bounded(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs):
    return self_cons_all_solutions_bounded(
        x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs
    )


def self_cons_bounded_multiprocessing(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs):
    return self_cons_all_solutions_bounded_mp(
        x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2, **kwargs
    )
