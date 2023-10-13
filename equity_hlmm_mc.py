from math import (
    sqrt,
)
import numpy as np

kappa = 1.2
xi_bar = 0.1
gamma = 0.5
s_0 = 1
xi_0 = 0.1
beta = 0.5
sigma = 0.25
_lambda = 1
v_0 = 1
eta = 0.1
tau = 1

terminal_n = 10
maturity = 2
strike = 0.4

bond_list = [1, 0.9512, 0.9048, 0.8607, 0.8187,
             0.7788, 0.7408, 0.7047, 0.6703, 0.6376, 0.6065]


delta_h = 1/20

rho_l_l = 0.98


def generate_rho(n_number):
    rho_xi_x = np.array([[1, -0.3],
                         [-0.3, 1]])

    rho_libor_libor = np.zeros((n_number, n_number))
    for i in range(n_number):
        for j in range(n_number):
            rho_libor_libor[i, j] = 1 if i == j else rho_l_l

    rho_libor_x = np.array([0.5] * n_number)

    rho_v = np.array([0]*(n_number+2))
    rho_xi = np.array([0]*n_number)

    c_1 = np.vstack((rho_libor_x, rho_xi))
    c_2 = np.hstack((rho_xi_x, c_1))
    c_3 = np.hstack((c_1.T, rho_libor_libor))
    c_4 = np.vstack((c_2, c_3))
    c_5 = np.vstack((c_4, rho_v))
    c_6 = np.hstack((rho_v, [1]))
    rho = np.vstack((c_5.T, c_6)).T

    return rho


def libor(k,  v_ini, libor_list, w_libor):
    tmp1 = libor_list[k-1] + sigma * \
        phi(k,  libor_list[k-1]) * sqrt(abs(v_ini)) * sqrt(delta_h)*w_libor
    if k == terminal_n:
        return tmp1

    tmp2 = -phi(k,  libor_list[k-1]) * sigma * v_ini
    sum = 0
    for j in range(k+1, terminal_n+1):
        tmp3 = tau * phi(j, libor_list=libor_list) * \
            sigma / (1 + tau * libor_list[j-1]) * rho_l_l * delta_h
        sum += tmp2 * tmp3

    return sum + tmp1


def v_libor(v_ini, w_v):
    return v_ini + _lambda * (v_0-v_ini)*delta_h + eta * sqrt(abs(v_ini))*sqrt(delta_h)*w_v


def phi(k, libor_k=None, libor_list=None):
    return beta * libor_k + (1 - beta) * libor_zero(k) if libor_k is not None else beta * libor_list[k-1] + \
        (1 - beta) * libor_zero(k)


def libor_zero(k):
    return 1 / tau * (bond_list[k-1] / bond_list[k]-1)


def forward_equity(f_e_ini, xi_ini, v_ini, libor_list, w_x, w_libor_list, t):
    tmp1 = sqrt(abs(xi_ini)) * sqrt(delta_h) * w_x
    sum = 0
    for j in range(m_func(t)+1, terminal_n+1):
        tmp2 = tau * sigma * phi(j, libor_list=libor_list) * sqrt(abs(v_ini))
        tmp3 = 1 + tau * libor_list[j-1]
        tmp4 = sqrt(delta_h) * w_libor_list[j-1]
        sum += tmp2 / tmp3 * tmp4

    f_e = f_e_ini + f_e_ini * (tmp1 + sum)

    return f_e


def m_func(t):
    import math
    return math.ceil(t)


def xi_forward_equity(xi_ini, w_xi):
    return xi_ini + kappa*(xi_bar-xi_ini)*delta_h + gamma * sqrt(abs(xi_ini))*sqrt(delta_h)*w_xi


def generate_rv(rho):
    rho_tri = np.linalg.cholesky(rho)
    w_list = rho_tri@np.random.normal(0, 1, terminal_n + 3)
    w_x = w_list[0]
    w_xi = w_list[1]
    w_libor_list = w_list[2:terminal_n+2]
    w_v = w_list[terminal_n+2]

    return w_x, w_xi, w_libor_list, w_v


def get_strike(k, libor_list, maturity, terminal):
    tmp = 1
    for i in range(maturity+1, terminal+1):
        tmp = tmp * (1 + tau * libor_list[i-1])
    return k * tmp


def calculate_equity(maturity, strike):
    libor_list = [libor_zero(i) for i in range(1, terminal_n+1)]
    rho = generate_rho(terminal_n)

    grid_list = [grid*delta_h for grid in range(1, int(maturity/delta_h+1))]

    _v = v_0
    _xi = xi_0
    _f = s_0 / bond_list[terminal_n]
    for t_grid in grid_list:
        w_x, w_xi, w_libor_list, w_v = generate_rv(rho)

        tmp_libor_list = []
        for k in range(1, terminal_n+1):
            tmp_libor_list.append(libor(k, _v, libor_list, w_libor_list[k-1]))

        _f = forward_equity(_f, _xi, _v, libor_list, w_x, w_libor_list, t_grid)
        v_calc = v_libor(_v, w_v)
        xi_calc = xi_forward_equity(_xi, w_xi)

        libor_list = tmp_libor_list
        _v = v_calc
        _xi = xi_calc

    strike = get_strike(strike, libor_list, maturity, terminal_n)
    return max(_f-strike, 0)


def calculate_equity_loop(maturity, strike):
    _N = 20000
    sum = 0
    for i in range(0, _N):
        sum += calculate_equity(maturity, strike)
    return bond_list[terminal_n] * sum / _N


def beep(freq, dur=100):

    import winsound
    winsound.Beep(freq, dur)


if __name__ == "__main__":
    eq = 0
    for i in range(0, 10):
        tmp = calculate_equity_loop(maturity, strike)
        eq += tmp
        print(i)
        print(tmp)
    print('finish!!!!!')
    print(eq/10)
    beep(500, 2000)
