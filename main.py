from main_functions.dimer_funcs import *
from gpt_solvers.dimer_funcs_bounded import self_cons_bounded
from gpt_solvers.dimer_funcs_bounded_posM import self_cons_bounded_posM
from gpt_solvers.dimer_funcs_fixedN_xf_posM import self_cons_fixed_N_xf_pM

# В начале кода (глобальные настройки)
plt.rcParams.update({
    'figure.figsize': (5, 3.5),
    'figure.dpi': 100,

    # Сохранение
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',

    # Шрифт
    'font.family': 'serif',
    'mathtext.fontset': 'cm',  # Computer Modern math font
    'mathtext.rm': 'serif',
    'mathtext.it': 'serif:italic',
    'mathtext.bf': 'serif:bold',
    'font.size': 14,

    # Оси, шкалы, легенда
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 6,    # длина основной риски
    'ytick.major.size': 6,
})

# === Параметры системы ===
x = -12                 # (E_0-E_f)/Г
x_0 = -6                # E_0/Г
x_f = x_0 - x           # E_f/Г     # !!!!!!!!!!!!!!!
y = 6.5                 # U/2Г
z = 0.0                 # (g mu B)/2Г # должно быть отрицательным (g фактор)
w = 3                   # V/Г
# theta_1 = np.pi/2-0.05
# theta_2 = - theta_1
theta_1 = 0.0
theta_2 = - theta_1
# theta_1 = np.pi/4 - 0.1
# theta_2 = - theta_1
# theta_1 = 0
# theta_2 = 0
phi_1 = 0
phi_2 = 0

# sols = self_cons_bounded_posM(
#     x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2,
#     grid_per_dim=3,      # 3 -> 81 узел грубой сетки
#     n_halton=64,         # ещё 64 квазислучайных старта
#     picard_steps=3,
#     max_start_points=64, # после предварительного отбора
#     max_nfev=80,
#     residual_tol=1e-8,
#     uniqueness_tol=1e-6,
#     # sort_by='energy',    # сортировка по вашей энергии
# )
# print(sols)

N_1 = 1.0
N_2 = 1.0

# sols = self_cons_fixed_N_xf_pM(
#     N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2
# )
# print(sols)
