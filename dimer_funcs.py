import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy import integrate
import time
from multiprocessing import Pool, cpu_count
from numba import njit
from njit_funcs import *


# === Вспомогательные функции ===

def func_to_coeffs(func, E, V, omega_symbol='omega'):
    """
    Преобразует функцию в ее коэффициенты.
    E и V — реальные (числовые или символьные) массивы.
    """
    omega = sp.Symbol(omega_symbol)

    # Получаем исходный код
    import inspect
    source = inspect.getsource(func)

    # Извлекаем выражение
    return_line = source.split('return')[1].strip()

    # Создаем временное пространство имен с символьными переменными
    sym_globals = {
        'E': E,  # Заглушка
        'V': V,  # Заглушка
        'omega': omega,
        '1j': sp.I
    }

    # Выполняем выражение в символьном контексте
    expr = eval(return_line, sym_globals)

    # Раскрываем и получаем полином
    poly = sp.Poly(sp.expand(expr), omega)

    # Получаем коэффициенты от старшей степени к младшей
    coeffs = poly.all_coeffs()

    # Преобразуем коэффициенты к float (если они числа)
    coeffs = [float(c) if c.is_real else complex(c) for c in coeffs]

    return np.array(coeffs)


def find_roots_numpy(denominator_coeffs):
    """
    Находит корни полинома с помощью numpy.roots

    Parameters:
    denominator_coeffs: список коэффициентов от старшей степени к младшей
                       [a_n, a_{n-1}, ..., a_1, a_0]

    Returns:
    roots: массив корней (комплексных)
    multiplicities: кратности корней
    """
    # Находим корни
    roots = np.roots(denominator_coeffs)

    # Определяем кратности (приближенно)
    roots_rounded = np.round(roots, 10)  # округляем для группировки
    unique_roots, multiplicities = np.unique(roots_rounded, return_counts=True)

    # return unique_roots, multiplicities
    return unique_roots


def partial_fractions(num_func, roots):
    """
    num_func : функция числителя (например num_G_pp_11)
    roots    : ndarray комплексных простых корней знаменателя

    Возвращает массив коэффициентов A_i
    в разложении Σ A_i / (ω - roots[i])
    """

    A = np.zeros(len(roots), dtype=complex)

    for i, r in enumerate(roots):
        # 1. Числитель в точке r_i
        numerator = num_func(r)

        # 2. Производная знаменателя через произведение (r_i - r_j)
        denom = np.prod([r - roots[j] for j in range(len(roots)) if j != i])

        # 3. Формула коэффициента
        A[i] = numerator / denom

        # 4. Убираем численный шум
        if abs(A[i].imag) < 1e-12:
            A[i] = A[i].real

    return A


def E_and_V(N_1, M_1, N_2, M_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2):
    # === Заполнение матрицы E (индексы: спин1, спин2, номер атома) ===
    E = np.zeros((2, 2, 2), dtype=complex)  # первый индекс матрицы E^{ab}_i - строка, второй - столбец
    E[0, 0, 0] = x_0 - z + y * (N_1 - M_1 * np.cos(theta_1))            # E^{++}_1  везде поменял на -z (от g фактора)
    E[0, 1, 0] = - np.exp(1j * phi_1) * y * M_1 * np.sin(theta_1)       # E^{+-}_1
    E[1, 0, 0] = - np.exp(- 1j * phi_1) * y * M_1 * np.sin(theta_1)     # E^{-+}_1
    E[1, 1, 0] = x_0 + z + y * (N_1 + M_1 * np.cos(theta_1))            # E^{--}_1

    E[0, 0, 1] = x_0 - z + y * (N_2 - M_2 * np.cos(theta_2))            # E^{++}_2
    E[0, 1, 1] = - np.exp(1j * phi_2) * y * M_2 * np.sin(theta_2)       # E^{+-}_2
    E[1, 0, 1] = - np.exp(- 1j * phi_2) * y * M_2 * np.sin(theta_2)     # E^{-+}_2
    E[1, 1, 1] = x_0 + z + y * (N_2 + M_2 * np.cos(theta_2))            # E^{--}_2

    # === Заполнение матрицы V (индексы: спин1, спин2, атом1, атом2) ===
    V = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    V[0, 0, 0, 1] = w  # V^{++}_{12}
    V[0, 0, 1, 0] = w  # V^{++}_{21}
    V[1, 1, 0, 1] = w  # V^{--}_{12}
    V[1, 1, 1, 0] = w  # V^{--}_{21}

    V[0, 1, 0, 0] = - np.exp(1j * phi_1) * y * M_1 * np.sin(theta_1)    # V^{+-}_{11}
    V[1, 0, 0, 0] = - np.exp(- 1j * phi_1) * y * M_1 * np.sin(theta_1)  # V^{-+}_{11}
    V[0, 1, 1, 1] = - np.exp(1j * phi_2) * y * M_2 * np.sin(theta_2)    # V^{+-}_{22}
    V[1, 0, 1, 1] = - np.exp(- 1j * phi_2) * y * M_2 * np.sin(theta_2)  # V^{-+}_{22}
    return E, V

# === Уравнения самосогласования ===


def self_cons_equations_numba(N_1, M_1, N_2, M_2, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    """
    Правая часть уравнений самосогласования (считаем корни знаменателя простыми).
    Время одного вызова около 0.08 сек
    """
    # === Заполнение матрицы E (индексы: спин1, спин2, номер атома) ===
    E = np.zeros((2, 2, 2), dtype=complex)  # первый индекс матрицы E^{ab}_i - строка, второй - столбец
    E[0, 0, 0] = x_0 - z + y * (N_1 - M_1 * np.cos(theta_1))  # E^{++}_1
    E[0, 1, 0] = - np.exp(1j * phi_1) * y * M_1 * np.sin(theta_1)  # E^{+-}_1
    E[1, 0, 0] = - np.exp(- 1j * phi_1) * y * M_1 * np.sin(theta_1)  # E^{-+}_1
    E[1, 1, 0] = x_0 + z + y * (N_1 + M_1 * np.cos(theta_1))  # E^{--}_1

    E[0, 0, 1] = x_0 - z + y * (N_2 - M_2 * np.cos(theta_2))  # E^{++}_2
    E[0, 1, 1] = - np.exp(1j * phi_2) * y * M_2 * np.sin(theta_2)  # E^{+-}_2
    E[1, 0, 1] = - np.exp(- 1j * phi_2) * y * M_2 * np.sin(theta_2)  # E^{-+}_2
    E[1, 1, 1] = x_0 + z + y * (N_2 + M_2 * np.cos(theta_2))  # E^{--}_2

    # === Заполнение матрицы V (индексы: спин1, спин2, атом1, атом2) ===
    V = np.zeros((2, 2, 2, 2), dtype=complex)
    V[0, 0, 0, 1] = w  # V^{++}_{12}
    V[0, 0, 1, 0] = w  # V^{++}_{21}
    V[1, 1, 0, 1] = w  # V^{--}_{12}
    V[1, 1, 1, 0] = w  # V^{--}_{21}

    V[0, 1, 0, 0] = E[0, 1, 0]  # V^{+-}_{11}
    V[1, 0, 0, 0] = E[1, 0, 0]  # V^{-+}_{11}
    V[0, 1, 1, 1] = E[0, 1, 1]  # V^{+-}_{22}
    V[1, 0, 1, 1] = E[1, 0, 1]  # V^{-+}_{22}

    den_roots = find_roots_numpy(denominator_coeffs(E, V))

    # === Коэффициенты разложения на простейшие дроби ===

    A_pp_11 = partial_fractions_numba(num_G_pp_11, den_roots, E, V)
    A_mp_11 = partial_fractions_numba(num_G_mp_11, den_roots, E, V)
    A_pm_11 = partial_fractions_numba(num_G_pm_11, den_roots, E, V)
    A_mm_11 = partial_fractions_numba(num_G_mm_11, den_roots, E, V)
    A_pp_22 = partial_fractions_numba(num_G_pp_22, den_roots, E, V)
    A_mp_22 = partial_fractions_numba(num_G_mp_22, den_roots, E, V)
    A_pm_22 = partial_fractions_numba(num_G_pm_22, den_roots, E, V)
    A_mm_22 = partial_fractions_numba(num_G_mm_22, den_roots, E, V)

    # === Правые части уравнений самосогласования ===

    n1 = int_G(A_pp_11, den_roots, x_f) + int_G(A_mm_11, den_roots, x_f)
    m1 = (int_G(A_pp_11, den_roots, x_f) - int_G(A_mm_11, den_roots, x_f)) * np.cos(theta_1) + (
            np.exp(- 1j * phi_1) * int_G(A_pm_11, den_roots, x_f) + np.exp(1j * phi_1) * int_G(A_mp_11, den_roots, x_f)) * np.sin(theta_1)
    n2 = int_G(A_pp_22, den_roots, x_f) + int_G(A_mm_22, den_roots, x_f)
    m2 = (int_G(A_pp_22, den_roots, x_f) - int_G(A_mm_22, den_roots, x_f)) * np.cos(theta_2) + (
            np.exp(- 1j * phi_2) * int_G(A_pm_22, den_roots, x_f) + np.exp(1j * phi_2) * int_G(A_mp_22, den_roots, x_f)) * np.sin(theta_2)
    n1 = float(n1.real)
    m1 = float(m1.real)
    n2 = float(n2.real)
    m2 = float(m2.real)

    return n1, m1, n2, m2

def self_cons_equations_numba_x(N_1, M_1, N_2, M_2, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    """
    Правая часть уравнений самосогласования (считаем корни знаменателя простыми).
    Время одного вызова около 0.08 сек
    """
    # === Заполнение матрицы E (индексы: спин1, спин2, номер атома) ===
    E = np.zeros((2, 2, 2), dtype=complex)  # первый индекс матрицы E^{ab}_i - строка, второй - столбец
    E[0, 0, 0] = x_0 - z + y * (N_1 - M_1 * np.cos(theta_1))  # E^{++}_1
    E[0, 1, 0] = - np.exp(1j * phi_1) * y * M_1 * np.sin(theta_1)  # E^{+-}_1
    E[1, 0, 0] = - np.exp(- 1j * phi_1) * y * M_1 * np.sin(theta_1)  # E^{-+}_1
    E[1, 1, 0] = x_0 + z + y * (N_1 + M_1 * np.cos(theta_1))  # E^{--}_1

    E[0, 0, 1] = x_0 - z + y * (N_2 - M_2 * np.cos(theta_2))  # E^{++}_2
    E[0, 1, 1] = - np.exp(1j * phi_2) * y * M_2 * np.sin(theta_2)  # E^{+-}_2
    E[1, 0, 1] = - np.exp(- 1j * phi_2) * y * M_2 * np.sin(theta_2)  # E^{-+}_2
    E[1, 1, 1] = x_0 + z + y * (N_2 + M_2 * np.cos(theta_2))  # E^{--}_2

    # === Заполнение матрицы V (индексы: спин1, спин2, атом1, атом2) ===
    V = np.zeros((2, 2, 2, 2), dtype=complex)
    V[0, 0, 0, 1] = w  # V^{++}_{12}
    V[0, 0, 1, 0] = w  # V^{++}_{21}
    V[1, 1, 0, 1] = w  # V^{--}_{12}
    V[1, 1, 1, 0] = w  # V^{--}_{21}

    V[0, 1, 0, 0] = E[0, 1, 0]  # V^{+-}_{11}
    V[1, 0, 0, 0] = E[1, 0, 0]  # V^{-+}_{11}
    V[0, 1, 1, 1] = E[0, 1, 1]  # V^{+-}_{22}
    V[1, 0, 1, 1] = E[1, 0, 1]  # V^{-+}_{22}

    den_roots = find_roots_numpy(denominator_coeffs(E, V))

    # === Коэффициенты разложения на простейшие дроби ===

    A_pp_11 = partial_fractions_numba(num_G_pp_11, den_roots, E, V)
    A_mp_11 = partial_fractions_numba(num_G_mp_11, den_roots, E, V)
    A_pm_11 = partial_fractions_numba(num_G_pm_11, den_roots, E, V)
    A_mm_11 = partial_fractions_numba(num_G_mm_11, den_roots, E, V)
    A_pp_22 = partial_fractions_numba(num_G_pp_22, den_roots, E, V)
    A_mp_22 = partial_fractions_numba(num_G_mp_22, den_roots, E, V)
    A_pm_22 = partial_fractions_numba(num_G_pm_22, den_roots, E, V)
    A_mm_22 = partial_fractions_numba(num_G_mm_22, den_roots, E, V)

    # === Правые части уравнений самосогласования ===

    n1 = int_G(A_pp_11, den_roots, x_f) + int_G(A_mm_11, den_roots, x_f)
    m1 = (int_G(A_pp_11, den_roots, x_f) - int_G(A_mm_11, den_roots, x_f)) * np.cos(theta_1) + (
            np.exp(- 1j * phi_1) * int_G(A_pm_11, den_roots, x_f) + np.exp(1j * phi_1) * int_G(A_mp_11, den_roots, x_f)) * np.sin(theta_1)
    n2 = int_G(A_pp_22, den_roots, x_f) + int_G(A_mm_22, den_roots, x_f)
    m2 = (int_G(A_pp_22, den_roots, x_f) - int_G(A_mm_22, den_roots, x_f)) * np.cos(theta_2) + (
            np.exp(- 1j * phi_2) * int_G(A_pm_22, den_roots, x_f) + np.exp(1j * phi_2) * int_G(A_mp_22, den_roots, x_f)) * np.sin(theta_2)
    n1 = float(n1.real)
    m1 = float(m1.real)
    n2 = float(n2.real)
    m2 = float(m2.real)

    return n1, m1, n2, m2


def self_cons_equations_symmetric(N, M, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    """
    Правая часть уравнений самосогласования (считаем корни знаменателя простыми).
    Время одного вызова около 0.08 сек
    """
    N_1 = N
    N_2 = N
    M_1 = M
    M_2 = M
    # === Заполнение матрицы E (индексы: спин1, спин2, номер атома) ===
    E = np.zeros((2, 2, 2), dtype=complex)  # первый индекс матрицы E^{ab}_i - строка, второй - столбец
    E[0, 0, 0] = x_0 - z + y * (N_1 - M_1 * np.cos(theta_1))  # E^{++}_1
    E[0, 1, 0] = - np.exp(1j * phi_1) * y * M_1 * np.sin(theta_1)  # E^{+-}_1
    E[1, 0, 0] = - np.exp(- 1j * phi_1) * y * M_1 * np.sin(theta_1)  # E^{-+}_1
    E[1, 1, 0] = x_0 + z + y * (N_1 + M_1 * np.cos(theta_1))  # E^{--}_1

    E[0, 0, 1] = x_0 - z + y * (N_2 - M_2 * np.cos(theta_2))  # E^{++}_2
    E[0, 1, 1] = - np.exp(1j * phi_2) * y * M_2 * np.sin(theta_2)  # E^{+-}_2
    E[1, 0, 1] = - np.exp(- 1j * phi_2) * y * M_2 * np.sin(theta_2)  # E^{-+}_2
    E[1, 1, 1] = x_0 + z + y * (N_2 + M_2 * np.cos(theta_2))  # E^{--}_2

    # === Заполнение матрицы V (индексы: спин1, спин2, атом1, атом2) ===
    V = np.zeros((2, 2, 2, 2), dtype=complex)
    V[0, 0, 0, 1] = w  # V^{++}_{12}
    V[0, 0, 1, 0] = w  # V^{++}_{21}
    V[1, 1, 0, 1] = w  # V^{--}_{12}
    V[1, 1, 1, 0] = w  # V^{--}_{21}

    V[0, 1, 0, 0] = E[0, 1, 0]  # V^{+-}_{11}
    V[1, 0, 0, 0] = E[1, 0, 0]  # V^{-+}_{11}
    V[0, 1, 1, 1] = E[0, 1, 1]  # V^{+-}_{22}
    V[1, 0, 1, 1] = E[1, 0, 1]  # V^{-+}_{22}

    den_roots = find_roots_numpy(denominator_coeffs(E, V))

    # === Коэффициенты разложения на простейшие дроби ===

    A_pp_11 = partial_fractions_numba(num_G_pp_11, den_roots, E, V)
    A_mp_11 = partial_fractions_numba(num_G_mp_11, den_roots, E, V)
    A_pm_11 = partial_fractions_numba(num_G_pm_11, den_roots, E, V)
    A_mm_11 = partial_fractions_numba(num_G_mm_11, den_roots, E, V)
    # A_pp_22 = partial_fractions_numba(num_G_pp_22, den_roots, E, V)
    # A_mp_22 = partial_fractions_numba(num_G_mp_22, den_roots, E, V)
    # A_pm_22 = partial_fractions_numba(num_G_pm_22, den_roots, E, V)
    # A_mm_22 = partial_fractions_numba(num_G_mm_22, den_roots, E, V)

    # === Правые части уравнений самосогласования ===

    n = int_G(A_pp_11, den_roots, x_f) + int_G(A_mm_11, den_roots, x_f)
    m = (int_G(A_pp_11, den_roots, x_f) - int_G(A_mm_11, den_roots, x_f)) * np.cos(theta_1) + (
            np.exp(- 1j * phi_1) * int_G(A_pm_11, den_roots, x_f) + np.exp(1j * phi_1) * int_G(A_mp_11, den_roots, x_f)) * np.sin(theta_1)
    # n2 = int_G(A_pp_22, den_roots, x_f) + int_G(A_mm_22, den_roots, x_f)
    # m2 = (int_G(A_pp_22, den_roots, x_f) - int_G(A_mm_22, den_roots, x_f)) * np.cos(theta_2) + (
    #         np.exp(- 1j * phi_2) * int_G(A_pm_22, den_roots, x_f) + np.exp(1j * phi_2) * int_G(A_mp_22, den_roots, x_f)) * np.sin(theta_2)
    n = float(n.real)
    m = float(m.real)
    # n2 = float(n2.real)
    # m2 = float(m2.real)

    return n, m


# === Решатели ===

def solve_from_initial_guess(args):
    """
    Решает систему из одной стартовой точки.
    Запускается в отдельном процессе.
    """
    initial_guess, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2 = args
    def equations(vars):
        N_1, M_1, N_2, M_2 = vars
        n1, m1, n2, m2 = self_cons_equations_numba(
            N_1, M_1, N_2, M_2,
            x_0, x_f, y, z, w,
            theta_1, theta_2, phi_1, phi_2
        )
        return [
            N_1 - n1,
            M_1 - m1,
            N_2 - n2,
            M_2 - m2
        ]

    sol = fsolve(equations, initial_guess)
    return np.round(sol, 6)


def self_cons_print(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    # === Скан по сетке начальных условий ===
    N_1_range = np.linspace(0, 2, 10)
    M_1_range = np.linspace(-1, 1, 10)
    N_2_range = np.linspace(0, 2, 10)
    M_2_range = np.linspace(-1, 1, 10)

    solutions = []

    def equations(vars):
        N_1, M_1, N_2, M_2 = vars
        n1, m1, n2, m2 = self_cons_equations_numba(N_1, M_1, N_2, M_2, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
        eq1 = N_1 - n1
        eq2 = M_1 - m1
        eq3 = N_2 - n2
        eq4 = M_2 - m2
        return [eq1, eq2, eq3, eq4]

    for N_1 in N_1_range:
        for M_1 in M_1_range:
            for N_2 in N_2_range:
                for M_2 in M_2_range:
                    sol = fsolve(equations, (N_1, M_1, N_2, M_2))
                    sol = np.round(sol, 6)
                    N_1_sol, M_1_sol, N_2_sol, M_2_sol = sol

                    # Проверяем, что решение в нужных границах
                    # if 0 <= N_1_sol <= 2 and -1 <= M_1_sol <= 1 and 0 <= N_2_sol <= 2 and -1 <= M_2_sol <= 1:
                    if 0 <= 0.5*(N_1_sol+M_1_sol) <= 1 and 0 <= 0.5*(N_1_sol-M_1_sol) <= 1 and 0 <= 0.5*(N_2_sol+M_2_sol) <= 1 and 0 <= 0.5*(N_2_sol-M_2_sol) <= 1:
                        # Проверяем, что это новое уникальное решение
                        if not any(np.allclose(sol, s, atol=1e-2) for s in solutions):
                            solutions.append(sol)

        # === Вывод ===
    print(f"Найдено {len(solutions)} уникальных самосогласованных решений:")
    for s in solutions:
        print(f"N_1 = {s[0]:.6f},  M_1 = {s[1]:.6f}, N_2 = {s[2]:.6f},  M_2 = {s[3]:.6f}")

    return solutions


def self_cons(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    # === Скан по сетке начальных условий ===
    N_1_range = np.linspace(0, 2, 10)
    M_1_range = np.linspace(-1, 1, 10)
    N_2_range = np.linspace(0, 2, 10)
    M_2_range = np.linspace(-1, 1, 10)

    solutions = []

    def equations(vars):
        N_1, M_1, N_2, M_2 = vars
        n1, m1, n2, m2 = self_cons_equations_numba(N_1, M_1, N_2, M_2, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
        eq1 = N_1 - n1
        eq2 = M_1 - m1
        eq3 = N_2 - n2
        eq4 = M_2 - m2
        return [eq1, eq2, eq3, eq4]

    for N_1 in N_1_range:
        for M_1 in M_1_range:
            for N_2 in N_2_range:
                for M_2 in M_2_range:
                    sol = fsolve(equations, (N_1, M_1, N_2, M_2))
                    sol = np.round(sol, 6)
                    N_1_sol, M_1_sol, N_2_sol, M_2_sol = sol

                    # Проверяем, что решение в нужных границах
                    # if 0 <= N_1_sol <= 2 and -1 <= M_1_sol <= 1 and 0 <= N_2_sol <= 2 and -1 <= M_2_sol <= 1:
                    if 0 <= 0.5*(N_1_sol+M_1_sol) <= 1 and 0 <= 0.5*(N_1_sol-M_1_sol) <= 1 and 0 <= 0.5*(N_2_sol+M_2_sol) <= 1 and 0 <= 0.5*(N_2_sol-M_2_sol) <= 1:
                        # Проверяем, что это новое уникальное решение
                        if not any(np.allclose(sol, s, atol=1e-2) for s in solutions):
                            solutions.append(sol)

    #     # === Вывод ===
    # print(f"Найдено {len(solutions)} уникальных самосогласованных решений:")
    # for s in solutions:
    #     print(f"N_1 = {s[0]:.6f},  M_1 = {s[1]:.6f}, N_2 = {s[2]:.6f},  M_2 = {s[3]:.6f}")

    return solutions


def self_cons_multiprocessing_print(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    N_1_range = np.linspace(0, 2, 6)
    M_1_range = np.linspace(-1, 1, 6)
    N_2_range = np.linspace(0, 2, 6)
    M_2_range = np.linspace(-1, 1, 6)

    # Все стартовые точки
    initial_points = [
        ((N_1, M_1, N_2, M_2), x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
        for N_1 in N_1_range
        for M_1 in M_1_range
        for N_2 in N_2_range
        for M_2 in M_2_range
    ]

    # 🔥 Параллельный запуск
    with Pool(cpu_count()) as pool:
        results = pool.map(solve_from_initial_guess, initial_points)

    # Фильтрация решений
    solutions = []

    for sol in results:
        N_1_sol, M_1_sol, N_2_sol, M_2_sol = sol

        # if 0 <= N_1_sol <= 2 and -1 <= M_1_sol <= 1 \
        #    and 0 <= N_2_sol <= 2 and -1 <= M_2_sol <= 1:
        if 0 <= 0.5 * (N_1_sol + M_1_sol) <= 1 and 0 <= 0.5 * (N_1_sol - M_1_sol) <= 1 \
                and 0 <= 0.5 * (N_2_sol + M_2_sol) <= 1 and 0 <= 0.5 * (N_2_sol - M_2_sol) <= 1:

            if not any(np.allclose(sol, s, atol=1e-2) for s in solutions):
                solutions.append(sol)

    print(f"Найдено {len(solutions)} уникальных решений:")
    for s in solutions:
        print(f"N_1 = {s[0]:.6f}, M_1 = {s[1]:.6f}, "
              f"N_2 = {s[2]:.6f}, M_2 = {s[3]:.6f}")

    return solutions


def self_cons_multiprocessing(x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    N_1_range = np.linspace(0, 2, 6)
    M_1_range = np.linspace(-1, 1, 6)
    N_2_range = np.linspace(0, 2, 6)
    M_2_range = np.linspace(-1, 1, 6)

    # Все стартовые точки
    initial_points = [
        ((N_1, M_1, N_2, M_2), x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
        for N_1 in N_1_range
        for M_1 in M_1_range
        for N_2 in N_2_range
        for M_2 in M_2_range
    ]

    # 🔥 Параллельный запуск
    with Pool(cpu_count()) as pool:
        results = pool.map(solve_from_initial_guess, initial_points)

    # Фильтрация решений
    solutions = []

    for sol in results:
        N_1_sol, M_1_sol, N_2_sol, M_2_sol = sol

        # if 0 <= N_1_sol <= 2 and -1 <= M_1_sol <= 1 \
        #    and 0 <= N_2_sol <= 2 and -1 <= M_2_sol <= 1:
        if 0 <= 0.5 * (N_1_sol + M_1_sol) <= 1 and 0 <= 0.5 * (N_1_sol - M_1_sol) <= 1 \
                and 0 <= 0.5 * (N_2_sol + M_2_sol) <= 1 and 0 <= 0.5 * (N_2_sol - M_2_sol) <= 1:

            if not any(np.allclose(sol, s, atol=1e-2) for s in solutions):
                solutions.append(sol)

    # print(f"Найдено {len(solutions)} уникальных решений:")
    # for s in solutions:
    #     print(f"N_1 = {s[0]:.6f}, M_1 = {s[1]:.6f}, "
    #           f"N_2 = {s[2]:.6f}, M_2 = {s[3]:.6f}")

    return solutions

# === Энергии ===


def up_limit_of_energy_of_state(solution, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2):
    N_1_ast = solution[0]
    M_1_ast = solution[1]
    N_2_ast = solution[2]
    M_2_ast = solution[3]

    E, V = E_and_V(N_1_ast, M_1_ast, N_2_ast, M_2_ast, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2)

    den_roots = find_roots_numpy(denominator_coeffs(E, V))

    A_pp_11 = partial_fractions_numba(num_G_pp_11, den_roots, E, V)
    A_mm_11 = partial_fractions_numba(num_G_mm_11, den_roots, E, V)
    A_pp_22 = partial_fractions_numba(num_G_pp_22, den_roots, E, V)
    A_mm_22 = partial_fractions_numba(num_G_mm_22, den_roots, E, V)

    Energy = (
            int_rho_i_cot(A_pp_11, A_mm_11, den_roots, x_f)
            + int_rho_i_ln(A_pp_11, A_mm_11, den_roots, x_f)
            + int_rho_i_cot(A_pp_22, A_mm_22, den_roots, x_f)
            + int_rho_i_ln(A_pp_22, A_mm_22, den_roots, x_f)
            - 0.5 * y * ((N_1_ast ** 2 - M_1_ast ** 2) + (N_2_ast ** 2 - M_2_ast ** 2))
    )

    return Energy.real


def up_limit_of_energy_E_of_state(solution, N_1, N_2, x_0, y, z, w, theta_1, theta_2, phi_1, phi_2):
    state = np.array([N_1, solution[1], N_2, solution[2]])
    Omega = up_limit_of_energy_of_state(state, x_0, solution[0], y, z, w, theta_1, theta_2, phi_1, phi_2)
    Energy = Omega + solution[0] * (N_1 + N_2)
    return Energy.real








# ================================================== Черновики ==================================================

# ===================== взято из dimer_numba =====================

# def solve_from_initial_guess(initial_guess):
#     """
#     Решает систему из одной стартовой точки.
#     Запускается в отдельном процессе.
#     """
#     def equations(vars):
#         N_1, M_1, N_2, M_2 = vars
#         n1, m1, n2, m2 = self_cons_equations_numba(
#             N_1, M_1, N_2, M_2,
#             x_0, x_f, y, z, w,
#             theta_1, theta_2, phi_1, phi_2
#         )
#         return [
#             N_1 - n1,
#             M_1 - m1,
#             N_2 - n2,
#             M_2 - m2
#         ]
#
#     sol = fsolve(equations, initial_guess)
#     return np.round(sol, 6)
#
# def self_cons_multiprocessing():
#     N_1_range = np.linspace(0, 2, 6)
#     M_1_range = np.linspace(-1, 1, 6)
#     N_2_range = np.linspace(0, 2, 6)
#     M_2_range = np.linspace(-1, 1, 6)
#
#     # Все стартовые точки
#     initial_points = [
#         (N_1, M_1, N_2, M_2)
#         for N_1 in N_1_range
#         for M_1 in M_1_range
#         for N_2 in N_2_range
#         for M_2 in M_2_range
#     ]
#
#     # 🔥 Параллельный запуск
#     with Pool(cpu_count()) as pool:
#         results = pool.map(solve_from_initial_guess, initial_points)
#
#     # Фильтрация решений
#     solutions = []
#
#     for sol in results:
#         N_1_sol, M_1_sol, N_2_sol, M_2_sol = sol
#
#         if 0 <= N_1_sol <= 2 and -1 <= M_1_sol <= 1 \
#            and 0 <= N_2_sol <= 2 and -1 <= M_2_sol <= 1:
#
#             if not any(np.allclose(sol, s, atol=1e-2) for s in solutions):
#                 solutions.append(sol)
#
#     print(f"Найдено {len(solutions)} уникальных решений:")
#     for s in solutions:
#         print(f"N_1 = {s[0]:.6f}, M_1 = {s[1]:.6f}, "
#               f"N_2 = {s[2]:.6f}, M_2 = {s[3]:.6f}")
#
#     return solutions
#
# def solve_symmetric(self_cons_equations_symmetric, epsilon=0.0001):
#     """
#     Решатель системы уравнений с возвратом всех вычисленных значений
#     """
#     m_values = np.arange(-1.0, 1.0 + epsilon, epsilon)
#     n_values = np.zeros_like(m_values)
#     m_star_values = np.zeros_like(m_values)
#     is_solution = np.zeros_like(m_values, dtype=np.bool_)
#
#     for i, m_k in enumerate(m_values):
#         # Бинарный поиск для n
#         n_left, n_right = 0, 2
#
#         for _ in range(100):
#             n_k = (n_left + n_right) / 2
#             f_n_val, _ = self_cons_equations_symmetric(n_k, m_k, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
#
#             if abs(n_k - f_n_val) < 1e-6:
#                 break
#             elif n_k > f_n_val:
#                 n_right = n_k
#             else:
#                 n_left = n_k
#
#         n_values[i] = n_k
#
#         # Вычисляем m*_k
#         _, f_m_val = self_cons_equations_symmetric(n_k, m_k, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
#         m_star_values[i] = f_m_val
#
#     # Находим решения как пересечения (где m_star - m меняет знак)
#     difference = m_star_values - m_values
#
#     for i in range(len(difference) - 1):
#         if difference[i] * difference[i + 1] <= 0:  # есть пересечение или касание
#             # Более точное решение - выбираем точку с меньшей разностью
#             if abs(difference[i]) < abs(difference[i + 1]):
#                 is_solution[i] = True
#             else:
#                 is_solution[i + 1] = True
#
#     return m_values, n_values, m_star_values, is_solution
#
# # === Решение системы (скан по углам) ===
#
# def self_cons_theta(theta_1, theta_2):
#     # === Скан по сетке начальных условий ===
#     # N_1_range = np.linspace(0, 2, 41)  # шаг 0.05
#     # M_1_range = np.linspace(-1, 1, 41)
#     # N_2_range = np.linspace(0, 2, 41)  # шаг 0.05
#     # M_2_range = np.linspace(-1, 1, 41)
#
#     # N_1_range = np.linspace(0, 2, 2)  # шаг 0.05
#     # M_1_range = np.linspace(-1, 1, 2)
#     # N_2_range = np.linspace(0, 2, 2)  # шаг 0.05
#     # M_2_range = np.linspace(-1, 1, 2)
#
#     N_1_range = np.linspace(0, 2, 6)
#     M_1_range = np.linspace(-1, 1, 6)
#     N_2_range = np.linspace(0, 2, 6)
#     M_2_range = np.linspace(-1, 1, 6)
#
#     solutions = []
#
#     def equations(vars):
#         N_1, M_1, N_2, M_2 = vars
#         n1, m1, n2, m2 = self_cons_equations_numba(N_1, M_1, N_2, M_2, x_0, x_f, y, z, w, theta_1, theta_2, phi_1, phi_2)
#         eq1 = N_1 - n1
#         eq2 = M_1 - m1
#         eq3 = N_2 - n2
#         eq4 = M_2 - m2
#         return [eq1, eq2, eq3, eq4]
#
#     for N_1 in N_1_range:
#         for M_1 in M_1_range:
#             for N_2 in N_2_range:
#                 for M_2 in M_2_range:
#                     sol = fsolve(equations, (N_1, M_1, N_2, M_2))
#                     sol = np.round(sol, 6)
#                     N_1_sol, M_1_sol, N_2_sol, M_2_sol = sol
#
#                     # Проверяем, что решение в нужных границах
#                     if 0 <= N_1_sol <= 2 and -1 <= M_1_sol <= 1 and 0 <= N_2_sol <= 2 and -1 <= M_2_sol <= 1:
#                         # Проверяем, что это новое уникальное решение
#                         if not any(np.allclose(sol, s, atol=1e-2) for s in solutions):
#                             solutions.append(sol)
#
#     return solutions
#
# def solve_from_initial_guess_theta(args):
#     """
#     Решает систему из одной стартовой точки.
#     Запускается в отдельном процессе.
#     """
#     initial_guess, theta_1, theta_2 = args
#     def equations(vars):
#         N_1, M_1, N_2, M_2 = vars
#         n1, m1, n2, m2 = self_cons_equations_numba(
#             N_1, M_1, N_2, M_2,
#             x_0, x_f, y, z, w,
#             theta_1, theta_2, phi_1, phi_2
#         )
#         return [
#             N_1 - n1,
#             M_1 - m1,
#             N_2 - n2,
#             M_2 - m2
#         ]
#
#     sol = fsolve(equations, initial_guess)
#     return np.round(sol, 6)
#
# def self_cons_multiprocessing_theta(theta_1, theta_2):
#
#     # N_1_range = np.linspace(0, 2, 2)
#     # M_1_range = np.linspace(-1, 1, 2)
#     # N_2_range = np.linspace(0, 2, 2)
#     # M_2_range = np.linspace(-1, 1, 2)
#
#     # N_1_range = np.linspace(0, 2, 4)
#     # M_1_range = np.linspace(-1, 1, 4)
#     # N_2_range = np.linspace(0, 2, 4)
#     # M_2_range = np.linspace(-1, 1, 4)
#
#     N_1_range = np.linspace(0, 2, 6)
#     M_1_range = np.linspace(-1, 1, 6)
#     N_2_range = np.linspace(0, 2, 6)
#     M_2_range = np.linspace(-1, 1, 6)
#
#     # N_1_range = np.linspace(0, 2, 10)
#     # M_1_range = np.linspace(-1, 1, 10)
#     # N_2_range = np.linspace(0, 2, 10)
#     # M_2_range = np.linspace(-1, 1, 10)
#
#     # Все стартовые точки
#     initial_points = [
#         ((N_1, M_1, N_2, M_2), theta_1, theta_2)
#         for N_1 in N_1_range
#         for M_1 in M_1_range
#         for N_2 in N_2_range
#         for M_2 in M_2_range
#     ]
#
#     # 🔥 Параллельный запуск
#     with Pool(cpu_count()) as pool:
#         results = pool.map(solve_from_initial_guess_theta, initial_points)
#
#     # Фильтрация решений
#     solutions = []
#
#     for sol in results:
#         N_1_sol, M_1_sol, N_2_sol, M_2_sol = sol
#
#         if 0 <= N_1_sol <= 2 and -1 <= M_1_sol <= 1 \
#            and 0 <= N_2_sol <= 2 and -1 <= M_2_sol <= 1:
#
#             if not any(np.allclose(sol, s, atol=1e-2) for s in solutions):
#                 solutions.append(sol)
#
#     # print(f"Найдено {len(solutions)} уникальных решений:")
#     # for s in solutions:
#     #     print(f"N_1 = {s[0]:.6f}, M_1 = {s[1]:.6f}, "
#     #           f"N_2 = {s[2]:.6f}, M_2 = {s[3]:.6f}")
#
#     return solutions

