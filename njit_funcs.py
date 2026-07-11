from numba import njit
import numpy as np

@njit
def arccot(z):
    if z == 0:
        return 0.5 * np.pi

    res = np.arctan(1.0 / z)

    if z.real < 0:
        res = res + np.pi

    return res

@njit
def partial_fractions_numba(num_func, roots, E, V):
    """
    num_func : функция числителя
    roots    : ndarray комплексных простых корней

    Возвращает коэффициенты A_i
    """

    n = len(roots)
    A = np.zeros(n, dtype=np.complex128)

    for i in range(n):

        r = roots[i]

        # 1. Числитель в точке r_i
        numerator = num_func(r, E, V)

        # 2. Произведение (r_i - r_j), j ≠ i
        denom = 1.0 + 0.0j
        for j in range(n):
            if j != i:
                denom *= (r - roots[j])

        A[i] = numerator / denom

        # 3. Убираем малую мнимую часть
        if abs(A[i].imag) < 1e-12:
            A[i] = A[i].real + 0.0j

    return A

# === Функции для метода цепной дроби ===

@njit
def diag_numpy_coeffs(H, y1):
    # Нормализуем начальный вектор
    norm_y1 = np.sqrt(np.vdot(y1, y1).real)
    y1 = y1 / norm_y1

    # Шаг 1: a1
    H_y1 = H @ y1
    a1 = np.vdot(y1, H_y1).real

    # Шаг 2: b1 и y2
    b1_vec = H_y1 - a1 * y1
    b1 = np.sqrt(np.vdot(b1_vec, b1_vec).real)
    y2 = b1_vec / b1

    # Шаг 3: a2
    H_y2 = H @ y2
    a2 = np.vdot(y2, H_y2).real

    # Шаг 4: b2 и y3
    b2_vec = H_y2 - b1 * y1 - a2 * y2
    b2 = np.sqrt(np.vdot(b2_vec, b2_vec).real)
    y3 = b2_vec / b2

    # Шаг 5: a3
    H_y3 = H @ y3
    a3 = np.vdot(y3, H_y3).real

    # Шаг 6: b3 и y4
    b3_vec = H_y3 - b2 * y2 - a3 * y3
    b3 = np.sqrt(np.vdot(b3_vec, b3_vec).real)
    y4 = b3_vec / b3

    # Шаг 7: a4
    H_y4 = H @ y4
    a4 = np.vdot(y4, H_y4).real

    return a1, b1, a2, b2, a3, b3, a4

@njit
def lanczos_to_partial_fractions(cfs):
    """
    Преобразование коэффициентов Ланцоша [a1,b1,a2,b2,a3,b3,a4]
    в сумму простейших дробей G = Σ p_i/(w - q_i)
    """
    a1, b1, a2, b2, a3, b3, a4 = cfs
    eps = 1e-12

    # === Уровень 1 ===
    d_sq = (a3 + a4) ** 2 - 4 * (a3 * a4 - b3 ** 2)
    if d_sq < 0:
        d_sq = 0.0
    d = np.sqrt(d_sq)

    q1 = ((a3 + a4) - d) / 2
    q2 = ((a3 + a4) + d) / 2

    # Защита от совпадающих корней
    if abs(q1 - q2) < eps:
        q2 = q1 + eps

    # Вычеты уровня 1
    p1_1 = b2 ** 2 * (q1 - a4) / (q1 - q2)
    p2_1 = b2 ** 2 * (q2 - a4) / (q2 - q1)

    # === Динамический диапазон ===
    all_vals = np.array([a1, a2, a3, a4, b1, b2, b3, q1, q2])
    max_abs = np.abs(all_vals).max()
    margin = max(10.0, max_abs * 2)
    margin = min(margin, 1e6)  # ограничение для защиты от переполнения

    # === Уровень 2 ===
    q_prev = np.array([q1, q2])
    p_prev1 = np.array([p1_1, p2_1])

    # Сортируем полюса для интервалов
    poles = np.sort(q_prev)
    intervals = np.array([-margin, poles[0], poles[1], margin])

    roots2 = np.zeros(3)

    for i in range(3):
        # Динамические границы с отступом от полюсов
        left = intervals[i]
        right = intervals[i + 1]

        # Отступ от границ
        if i == 0:
            left = intervals[i] + margin / 100
            right = intervals[i + 1] - 0.1
        elif i == 2:
            left = intervals[i] + 0.1
            right = intervals[i + 1] - margin / 100
        else:
            left = intervals[i] + 0.1
            right = intervals[i + 1] - 0.1

        # Метод бисекции
        for _ in range(100):
            mid = (left + right) / 2

            # Защита от совпадения с полюсами
            if abs(mid - poles[0]) < eps or abs(mid - poles[1]) < eps:
                mid += eps

            # Вычисление функции
            f_left = left - a2
            f_mid = mid - a2
            for j in range(2):
                if abs(left - q_prev[j]) > eps:
                    f_left -= p_prev1[j] / (left - q_prev[j])
                if abs(mid - q_prev[j]) > eps:
                    f_mid -= p_prev1[j] / (mid - q_prev[j])

            if abs(f_mid) < 1e-10:
                break
            if f_left * f_mid < 0:
                right = mid
            else:
                left = mid
        roots2[i] = mid

    # Сортируем корни
    roots2 = np.sort(roots2)

    # Вычеты уровня 2
    p1_2 = b1 ** 2 * (roots2[0] - q1) * (roots2[0] - q2)
    p2_2 = b1 ** 2 * (roots2[1] - q1) * (roots2[1] - q2)
    p3_2 = b1 ** 2 * (roots2[2] - q1) * (roots2[2] - q2)

    # Знаменатели
    denom1 = (roots2[0] - roots2[1]) * (roots2[0] - roots2[2])
    denom2 = (roots2[1] - roots2[0]) * (roots2[1] - roots2[2])
    denom3 = (roots2[2] - roots2[0]) * (roots2[2] - roots2[1])

    if abs(denom1) < eps:
        denom1 = eps if denom1 >= 0 else -eps
    if abs(denom2) < eps:
        denom2 = eps if denom2 >= 0 else -eps
    if abs(denom3) < eps:
        denom3 = eps if denom3 >= 0 else -eps

    p1_2 /= denom1
    p2_2 /= denom2
    p3_2 /= denom3

    p_prev2 = np.array([p1_2, p2_2, p3_2])

    # === Уровень 3 (финальный) ===
    poles2 = np.sort(roots2)
    intervals = np.zeros(5)
    intervals[0] = -margin
    intervals[1:4] = poles2
    intervals[4] = margin

    roots_final = np.zeros(4)

    for i in range(4):
        left = intervals[i]
        right = intervals[i + 1]

        # Отступ от границ
        if i == 0:
            left = intervals[i] + margin / 100
            right = intervals[i + 1] - 0.1
        elif i == 3:
            left = intervals[i] + 0.1
            right = intervals[i + 1] - margin / 100
        else:
            left = intervals[i] + 0.1
            right = intervals[i + 1] - 0.1

        # Метод бисекции
        for _ in range(100):
            mid = (left + right) / 2

            # Защита от совпадения с полюсами
            pole_close = False
            for j in range(3):
                if abs(mid - roots2[j]) < eps:
                    mid += eps
                    pole_close = True
                    break
            if pole_close:
                continue

            # Вычисление функции
            f_left = left - a1
            f_mid = mid - a1
            for j in range(3):
                if abs(left - roots2[j]) > eps:
                    f_left -= p_prev2[j] / (left - roots2[j])
                if abs(mid - roots2[j]) > eps:
                    f_mid -= p_prev2[j] / (mid - roots2[j])

            if abs(f_mid) < 1e-10:
                break
            if f_left * f_mid < 0:
                right = mid
            else:
                left = mid
        roots_final[i] = mid

    # Сортируем финальные корни
    roots_final = np.sort(roots_final)

    # Финальные вычеты
    p_final = np.zeros(4)
    for i in range(4):
        prod_num = 1.0
        prod_den = 1.0

        # Числитель: произведение по корням предыдущего уровня
        for j in range(3):
            prod_num *= (roots_final[i] - roots2[j])

        # Знаменатель: произведение по остальным финальным корням
        for j in range(4):
            if j != i:
                prod_den *= (roots_final[i] - roots_final[j])

        if abs(prod_den) < eps:
            prod_den = eps if prod_den >= 0 else -eps

        p_final[i] = prod_num / prod_den

    return p_final + 0.0j, roots_final + 0.0j

@njit
def nondiag_part_fractions(H, y1_initial, y2_initial):
    y1_a = 1 / np.sqrt(2) * (y1_initial + y2_initial)
    y1_b = 1 / np.sqrt(2) * (y1_initial - y2_initial)
    y1_c = 1 / np.sqrt(2) * (y1_initial + 1j * y2_initial)
    y1_d = 1 / np.sqrt(2) * (y1_initial - 1j * y2_initial)

    G_a_cfs = diag_numpy_coeffs(H, y1_a)
    G_b_cfs = diag_numpy_coeffs(H, y1_b)
    G_c_cfs = diag_numpy_coeffs(H, y1_c)
    G_d_cfs = diag_numpy_coeffs(H, y1_d)

    p_a, q_a = lanczos_to_partial_fractions(G_a_cfs)
    p_b, q_b = lanczos_to_partial_fractions(G_b_cfs)
    p_c, q_c = lanczos_to_partial_fractions(G_c_cfs)
    p_d, q_d = lanczos_to_partial_fractions(G_d_cfs)

    n = len(p_a)
    p = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        p[i] = 1 / 2 * (p_a[i] - p_b[i] + 1j * (p_d[i] - p_c[i]))

    return p

# === Основные njit функции ===

@njit
def num_G_pp_11(omega, E, V):
    return (omega - E[1, 1, 0]) * (omega - E[1, 1, 1]) * (omega - E[0, 0, 1]) - (omega - E[0, 0, 1]) * V[
        1, 1, 0, 1] * V[1, 1, 1, 0] - (omega - E[1, 1, 0]) * V[1, 0, 1, 1] * V[0, 1, 1, 1]

@njit
def num_G_mp_11(omega, E, V):
    return (omega - E[0, 0, 1]) * (omega - E[1, 1, 1]) * V[1, 0, 0, 0] - V[1, 0, 0, 0] * V[0, 1, 1, 1] * V[
        1, 0, 1, 1] + V[1, 1, 0, 1] * V[1, 0, 1, 1] * V[0, 0, 1, 0]

@njit
def num_G_pm_11(omega, E, V):
    return (omega - E[1, 1, 1]) * (omega - E[0, 0, 1]) * V[0, 1, 0, 0] - V[0, 1, 0, 0] * V[1, 0, 1, 1] * V[
        0, 1, 1, 1] + V[0, 0, 0, 1] * V[0, 1, 1, 1] * V[1, 1, 1, 0]

@njit
def num_G_mm_11(omega, E, V):
    return (omega - E[0, 0, 0]) * (omega - E[0, 0, 1]) * (omega - E[1, 1, 1]) - (omega - E[1, 1, 1]) * V[
        0, 0, 0, 1] * V[0, 0, 1, 0] - (omega - E[0, 0, 0]) * V[0, 1, 1, 1] * V[1, 0, 1, 1]

@njit
def num_G_pp_22(omega, E, V):
    return (omega - E[1, 1, 1]) * (omega - E[1, 1, 0]) * (omega - E[0, 0, 0]) - (omega - E[0, 0, 0]) * V[
        1, 1, 1, 0] * V[1, 1, 0, 1] - (omega - E[1, 1, 1]) * V[1, 0, 0, 0] * V[0, 1, 0, 0]

@njit
def num_G_mp_22(omega, E, V):
    return (omega - E[0, 0, 0]) * (omega - E[1, 1, 0]) * V[1, 0, 1, 1] - V[1, 0, 1, 1] * V[0, 1, 0, 0] * V[
        1, 0, 0, 0] + V[1, 1, 1, 0] * V[1, 0, 0, 0] * V[0, 0, 0, 1]

@njit
def num_G_pm_22(omega, E, V):
    return (omega - E[1, 1, 0]) * (omega - E[0, 0, 0]) * V[0, 1, 1, 1] - V[0, 1, 1, 1] * V[1, 0, 0, 0] * V[
        0, 1, 0, 0] + V[0, 0, 1, 0] * V[0, 1, 0, 0] * V[1, 1, 0, 1]

@njit
def num_G_mm_22(omega, E, V):
    return (omega - E[0, 0, 1]) * (omega - E[0, 0, 0]) * (omega - E[1, 1, 0]) - (omega - E[1, 1, 0]) * V[
        0, 0, 1, 0] * V[0, 0, 0, 1] - (omega - E[0, 0, 1]) * V[0, 1, 0, 0] * V[1, 0, 0, 0]

@njit
def int_G(coeffs, roots, x_f):
    s = 0.0 + 0.0j
    for i in range(len(roots)):
        s += (1 / np.pi) * coeffs[i] * arccot(roots[i] - x_f)
    return s

@njit
def int_rho_i_cot(coeffs_plus, coeffs_minus, roots, x_f):
    s = 0.0 + 0.0j
    for i in range(len(roots)):
        s += (1 / np.pi) * (coeffs_plus[i] + coeffs_minus[i]) * (roots[i] - x_f) * arccot(roots[i] - x_f)
    return s

@njit
def int_rho_i_ln(coeffs_plus, coeffs_minus, roots, x_f):
    s = 0.0 + 0.0j
    for i in range(len(roots)):
        s += (1 / np.pi) * (coeffs_plus[i] + coeffs_minus[i]) * (1 / 2) * np.log((roots[i] - x_f)**2 + 1)
    return s

@njit
def denominator_coeffs(E, V):
    den_coeffs = np.zeros(5, dtype=np.complex128)
    den_coeffs[0] = 1
    den_coeffs[1] = - (E[0, 0, 0] + E[1, 1, 0] + E[0, 0, 1] + E[1, 1, 1])
    den_coeffs[2] = E[0, 0, 1] * E[1, 1, 1] + E[0, 0, 0] * E[1, 1, 0] + (E[0, 0, 0] + E[1, 1, 0]) * (
                E[0, 0, 1] + E[1, 1, 1]) - V[0, 1, 1, 1] * V[1, 0, 1, 1] - V[0, 0, 0, 1] * V[0, 0, 1, 0] - V[
              1, 0, 0, 0] * V[0, 1, 0, 0] - V[1, 1, 0, 1] * V[1, 1, 1, 0]
    den_coeffs[3] = - ((E[0, 0, 1] + E[1, 1, 1]) * E[0, 0, 0] * E[1, 1, 0] + (E[0, 0, 0] + E[1, 1, 0]) * E[0, 0, 1] * E[
        1, 1, 1]) + (E[0, 0, 0] + E[1, 1, 0]) * V[0, 1, 1, 1] * V[1, 0, 1, 1] + (E[1, 1, 0] + E[1, 1, 1]) * V[
              0, 0, 0, 1] * V[0, 0, 1, 0] + (E[0, 0, 1] + E[1, 1, 1]) * V[1, 0, 0, 0] * V[0, 1, 0, 0] + (
                      E[0, 0, 0] + E[0, 0, 1]) * V[1, 1, 0, 1] * V[1, 1, 1, 0]
    den_coeffs[4] = E[0, 0, 0] * E[1, 1, 0] * E[0, 0, 1] * E[1, 1, 1] - E[0, 0, 0] * E[1, 1, 0] * V[0, 1, 1, 1] * V[
        1, 0, 1, 1] - E[1, 1, 0] * E[1, 1, 1] * V[0, 0, 0, 1] * V[0, 0, 1, 0] - E[0, 0, 1] * E[1, 1, 1] * V[
              1, 0, 0, 0] * V[0, 1, 0, 0] - E[0, 0, 0] * E[0, 0, 1] * V[1, 1, 0, 1] * V[1, 1, 1, 0] - V[
              1, 1, 0, 1] * V[0, 1, 0, 0] * V[1, 0, 1, 1] * V[0, 0, 1, 0] + V[1, 0, 0, 0] * V[0, 1, 1, 1] * V[
              0, 1, 0, 0] * V[1, 0, 1, 1] + V[1, 1, 0, 1] * V[1, 1, 1, 0] * V[0, 0, 0, 1] * V[0, 0, 1, 0] - V[
              1, 0, 0, 0] * V[0, 1, 1, 1] * V[1, 1, 1, 0] * V[0, 0, 0, 1]
    # den_coeffs = np.where(np.abs(den_coeffs.imag) < 1e-12, den_coeffs.real, den_coeffs) # не работает
    return den_coeffs