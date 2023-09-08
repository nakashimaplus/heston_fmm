from math import (
    exp,
    sqrt,
)
import numpy as np

kappa = 1.2
xi_bar = 0.01
gamma = 0.5
s_0 = 1
xi_0 = 0.01
beta = 0.5
sigma = 0.25
_lambda = 1
v_0 = 1
eta = 0.1
tau = 1

bond_list = [1, 0.9512, 0.9048, 0.8607, 0.8187,
             0.7788, 0.7408, 0.7047, 0.6703, 0.6376, 0.6065]
delta_h = 1/20

libor_0 = 1/0.9512-1

N = 1

rho = np.array([[1, -0.3,   0.5,    0],
                [-0.3, 1,    0,  0],
                [0.5,   0,  1,   0],
                [0,  0,    0,   1]
                ])


def generate_rho(n_number):
    rho_xi_x = np.array([[1, -0.3],
                         [-0.3, 1]])


def f_libor(libor_ini, v_ini, w_libor):
    return libor_ini + sigma * (beta*libor_ini + (1-beta)*libor_0) * sqrt(abs(v_ini)) * sqrt(delta_h)*w_libor


def f_v(v_ini, w_v):
    return v_ini + _lambda * (v_0-v_ini)*delta_h + eta * sqrt(abs(v_ini))*sqrt(delta_h)*w_v


def f_f(f_ini, xi_ini, v_ini, libor_ini, w_x, w_libor):
    tmp1 = sqrt(abs(xi_ini))*sqrt(delta_h)*w_x
    tmp2 = tau*sigma*(beta*libor_ini + (1-beta)*libor_0)
    tmp3 = sqrt(abs(v_ini)) * sqrt(delta_h)*w_libor
    tmp4 = 1 + tau * libor_ini
    f = f_ini + f_ini * (tmp1 + tmp2 * tmp3 / tmp4)
    return f


def f_xi(xi_ini, w_xi):
    return xi_ini + kappa*(xi_bar-xi_ini)*delta_h + gamma * sqrt(abs(xi_ini))*sqrt(delta_h)*w_xi


def generate_rv():
    rho_tri = np.linalg.cholesky(rho)
    w_x, w_xi, w_libor, w_v = rho_tri@np.random.normal(0, 1, N + 3)

    return w_x, w_xi, w_libor, w_v


def calculate_equity(T, K):

    grid_list = [grid*delta_h for grid in range(1, int(T/delta_h+1))]

    _v = v_0
    _libor = 1/0.9512-1
    _xi = xi_0
    _f = s_0 / bond_list[1]
    for grid in grid_list:
        w_x, w_xi, w_libor, w_v = generate_rv()
        _libor = f_libor(_libor, _v, w_libor)
        _v = f_v(_v, w_v)
        _f = f_f(_f, _xi, _v, _libor, w_x, w_libor)
        _xi = f_xi(_xi, w_xi)

    return max(_f-K, 0)


def calculate_equity_loop(T, K):
    _N = 50000
    sum = 0
    for i in range(0, _N):
        sum += calculate_equity(T, K)
    return bond_list[1]*sum/_N


def beep(freq, dur=100):

    import winsound
    winsound.Beep(freq, dur)


if __name__ == "__main__":

    # print(a)

    fx = calculate_equity_loop(1, 0.4)

    print(fx)
    beep(500, 2000)
