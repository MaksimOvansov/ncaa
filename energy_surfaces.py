from main_functions.dimer_funcs import *
from gpt_solvers.dimer_funcs_bounded import self_cons_bounded
from gpt_solvers.dimer_funcs_bounded_posM import self_cons_bounded_posM
from gpt_solvers.dimer_funcs_fixedN_xf_posM import self_cons_fixed_N_xf_pM

# === Параметры системы ===
x = -12                 # (E_0-E_f)/Г
x_0 = -6                # E_0/Г
x_f = x_0 - x           # E_f/Г
y = 6.5                 # U/2Г
z = 0.0                 # (mu B)/Г
w = 3                   # V/Г
theta_1 = 0.0
theta_2 = - theta_1
phi_1 = 0
phi_2 = 0
N_1 = 1.2
N_2 = 1.2

# === Решения, соответствующие M_i>=0, M_1==M_2 ===
def dimer_canonical_energy_surface_posM_sym(N, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2):

    N_1, N_2 = N, N

    # Расчет немагнитного решения, соответствующего началу отсчета энергии (немагнитное решение при z=0)
    sols_m0 = self_cons_fixed_N_xf_pM(
        N_1, N_2, x_0, y, 0, w, theta_1, theta_2, phi_1, phi_2
    )
    sol_m0 = np.array([0.0, 0.0, 0.0])
    for arr in sols_m0:
        if np.isclose(arr[1], 0.0, atol=1e-4) and np.isclose(arr[2], 0.0, atol=1e-4):
            sol_m0 = arr
            break
    
    # Расчет начала отсчета энергии
    Energy_m0 = up_limit_of_energy_E_of_state(sol_m0, N_1, N_2, x_0, y, 0, w, theta_1, theta_2, phi_1, phi_2)

    # Расчет самосогласованных решений при z!=0 (для отрисовки самосогласованных точек на поверхности)
    sols = self_cons_fixed_N_xf_pM(
        N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2
    )

    # Сетка для построения поверхности
    x_values = np.linspace(-10.0, -7.5, 60)
    M_values = np.linspace(0.0, 0.8, 60)

    # Расчет энергетической поверхности
    E_list = np.zeros((len(x_values), len(M_values)))
    for x in range(len(x_values)):
        for m in range(len(M_values)):
            x_f_ast = x_0-x_values[x]
            M_1_ast = M_values[m]
            M_2_ast = M_values[m]
            state = np.array([x_f_ast, M_1_ast, M_2_ast])
            E_list[x, m] = up_limit_of_energy_E_of_state(state, N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2) - Energy_m0

    # Построение поверхности
    x_grid, M_grid = np.meshgrid(x_values, M_values, indexing="ij")

    plt.figure(figsize=(6,5))
    plt.contourf(x_grid, M_grid, E_list,
                 levels=60,
                 cmap="viridis")
    plt.colorbar()

    points = np.array([list(p) for p in sols])  # точки самосогласованных решений
    plt.scatter(x_0-points[:, 0], points[:, 1],
                c='black', marker='o', s=50,
                edgecolors='white', linewidth=1.5,
                label='Solutions')

    plt.xlabel("$x$")
    plt.ylabel("$M$")
    plt.title(rf"$\Delta E(x,M)/\Gamma,\ \theta={theta_1:.3f}$")
    # plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()


def dimer_grand_canonical_energy_surface_posM_sym(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    
    # Расчет немагнитного решения, соответствующего началу отсчета энергии (немагнитное решение при z=0)
    sols_m0 = self_cons_bounded_posM(
        x_0, x_f, y, 0, w, theta_1, theta_2, phi_1, phi_2
    )
    sol_m0 = np.array([0.0, 0.0, 0.0, 0.0])
    for arr in sols_m0:
        if (np.isclose(arr[0], arr[2], atol=1e-4) and np.isclose(arr[1], 0.0, atol=1e-4) and np.isclose(arr[3], 0.0, atol=1e-4)):
            sol_m0 = arr
            break
    
    # Расчет начала отсчета энергии
    Energy_m0 = up_limit_of_energy_of_state(sol_m0, x_0, x_f, y, 0, w, theta_1, theta_2, phi_1, phi_2)

    # Расчет самосогласованных решений при z!=0 (для отрисовки самосогласованных точек на поверхности)
    sols = self_cons_bounded_posM(
        x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2
    )

    # Сетка для построения поверхности
    N_values = np.linspace(0.0, 2.0, 60)
    M_values = np.linspace(-1.0, 1.0, 60)

    # Расчет энергетической поверхности
    E_list = np.zeros((len(N_values), len(M_values)))
    for n in range(len(N_values)):
        for m in range(len(M_values)):
            N_1_ast = N_values[n]
            M_1_ast = M_values[m]
            N_2_ast = N_values[n]
            M_2_ast = M_values[m]
            state = np.array([N_1_ast, M_1_ast, N_2_ast, M_2_ast])
            E_list[n, m] = up_limit_of_energy_of_state(state, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2) - Energy_m0

    # Построение поверхности
    N_grid, M_grid = np.meshgrid(N_values, M_values, indexing="ij")
    plt.figure(figsize=(6,5))
    plt.contourf(N_grid, M_grid, E_list,
                 levels=60,
                 cmap="viridis")
    plt.colorbar()

    points = np.array([list(p) for p in sols])  # точки самосогласованных решений
    plt.scatter(points[:, 0], points[:, 1],
                c='black', marker='o', s=50,
                edgecolors='white', linewidth=1.5,
                label='Solutions')

    plt.xlabel("$N$")
    plt.ylabel("$M$")
    plt.title(rf"$\Delta\Omega(N,M)/\Gamma,\ \theta={theta_1:.3f}$")
    # plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()


# === Решения, соответствующие M_1==M_2 ===
def dimer_grand_canonical_energy_surface_sym(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    
    # Расчет немагнитного решения, соответствующего началу отсчета энергии (немагнитное решение при z=0)
    sols_m0 = self_cons_bounded(
        x_0, x_f, y, 0, w, theta_1, theta_2, phi_1, phi_2
    )
    sol_m0 = np.array([0.0, 0.0, 0.0, 0.0])
    for arr in sols_m0:
        if (np.isclose(arr[0], arr[2], atol=1e-4) and np.isclose(arr[1], 0.0, atol=1e-4) and np.isclose(arr[3], 0.0, atol=1e-4)):
            sol_m0 = arr
            break
    
    # Расчет начала отсчета энергии
    Energy_m0 = up_limit_of_energy_of_state(sol_m0, x_0, x_f, y, 0, w, theta_1, theta_2, phi_1, phi_2)

    # Расчет самосогласованных симметричных решений при z!=0 (для отрисовки самосогласованных точек на поверхности)
    sols_all = self_cons_bounded(
        x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2
    )
    sols = []
    for arr in sols_all:
        if (np.isclose(arr[0], arr[2], atol=1e-4) and np.isclose(arr[1], arr[3], atol=1e-4)):
            sols.append(arr)

    # Сетка для построения поверхности
    N_values = np.linspace(0.0, 2.0, 60)
    M_values = np.linspace(-1.0, 1.0, 60)

    # Расчет энергетической поверхности
    E_list = np.zeros((len(N_values), len(M_values)))
    for n in range(len(N_values)):
        for m in range(len(M_values)):
            N_1_ast = N_values[n]
            M_1_ast = M_values[m]
            N_2_ast = N_values[n]
            M_2_ast = M_values[m]
            state = np.array([N_1_ast, M_1_ast, N_2_ast, M_2_ast])
            E_list[n, m] = up_limit_of_energy_of_state(state, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2) - Energy_m0

    # Построение поверхности
    N_grid, M_grid = np.meshgrid(N_values, M_values, indexing="ij")
    plt.figure(figsize=(6,5))
    plt.contourf(N_grid, M_grid, E_list,
                 levels=60,
                 cmap="viridis")
    plt.colorbar()

    points = np.array([list(p) for p in sols])  # точки самосогласованных решений
    plt.scatter(points[:, 0], points[:, 1],
                c='black', marker='o', s=50,
                edgecolors='white', linewidth=1.5,
                label='Solutions')

    plt.xlabel("$N$")
    plt.ylabel("$M$")
    plt.title(rf"$\Delta\Omega(N,M)/\Gamma,\ \theta={theta_1:.3f}$")
    # plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()



# dimer_canonical_energy_surface_posM_sym(N_1, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2)

# dimer_canonical_energy_surface_grand_posM_sym(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)

dimer_grand_canonical_energy_surface_sym(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)