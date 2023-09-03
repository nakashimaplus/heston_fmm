from math import (
    exp,
    sqrt,
)
import numpy as np
from scipy.linalg import ldl

delta_list = [-1.5, -1.0, -0.5, 0, 0.5, 1, 1.5]
maturity_list = [1/2, 1, 3, 5, 7, 10, 15, 20, 25, 30]

# モデルパラメータの設定

kappa = 0.5
gamma = 0.3
sigma_bar = 0.1
sigma_zero = 0.1
xi_zero = 1.35

delta_h = 1/20

beta_d_k = 0.95
sigma_d_K = 0.15
lambda_d = 1
eta_d = 0.1

beta_f_k = 0.5
sigma_f_K = 0.25
lambda_f = 0.7
eta_f = 0.2

v_d_zero = 1
v_f_zero = 1

N = 2  # N = 2とする
libor_d = [0] * N
libor_f = [0] * N
tau = 1


c_d = np.array([[1, 0.9, 0.9],
                [0.9, 1, 0.9],
                [0.9, 0.9, 1]])

c_f = np.array([[1, 0.7, 0.7],
                [0.7, 1, 0.7],
                [0.7, 0.7, 1]])

c_d_f = np.array([[1, 0.25, 0.25],
                  [0.25, 1, 0.25],
                  [0.25, 0.25, 1]
                  ])

c_xi_d = np.array([[-0.15],
                  [-0.15],
                  [-0.15]])

c_xi_f = np.array([[-0.15],
                   [-0.15],
                   [-0.15]])

c_1 = np.array([[1]])

c_row_1 = np.concatenate((c_d, c_d_f), axis=1)
c_row_1 = np.concatenate((c_row_1, c_xi_d), axis=1)
c_row_2 = np.concatenate((c_d_f.T, c_f), axis=1)
c_row_2 = np.concatenate((c_row_2, c_xi_f), axis=1)
c_row_3 = np.concatenate((c_xi_d.T, c_xi_f.T), axis=1)
c_row_3 = np.concatenate((c_row_3, c_1), axis=1)


c = np.concatenate((c_row_1, c_row_2), axis=0)
c = np.concatenate((c, c_row_3), axis=0)


# def d_fx(t,T,fx_ini,sigma_ini,w_xi,w_d,w_f):
#     fx = sqrt(abs(sigma_ini))*sqrt(delta_h)*w_xi-eta_d*b_d(t,T)*sqrt(delta_h)*w_d+eta_f*b_f(t,T)*sqrt(delta_h)*w_f
#     fx = fx_ini+fx_ini*fx
#     return fx

libor_zero = 111


def L_d(k, libor_ini, v_d_ini, w_d_T_k):
    return libor_ini + sigma_d_K * phi_d_k(k, libor_zero) * sqrt(v_d_ini) * \
        (mu_d(k, libor_zero) * sqrt(v_d_ini) * delta_h + w_d_T_k)


def v_d(v_d_ini, w_d_T_v):
    return v_d_ini * lambda_d * (v_d_zero-v_d_ini) * delta_h + eta_d * sqrt(v_d_ini) * w_d_T_v


def mu_d(k, libor_zero):
    if k+1 > N:
        return 0
    sum = 0
    for j in range(k+1, N+1):
        sum += - tau * phi_d_k(j, libor_zero) * sigma_d_K / \
            (1 + tau * libor_d[j-1]) * c_d[k-1, j-1]
    return sum


def mu_f(k, libor_zero):
    if k+1 > N:
        return 0
    sum = 0
    for j in range(k+1, N+1):
        sum += - tau * phi_f_k(j, libor_zero) * sigma_f_K / \
            (1 + tau * libor_f[j-1]) * c_f[k-1, j-1]
    return sum


def phi_d_k(k, libor_zero):
    return beta_d_k * libor_d[k-1] + (1-beta_d_k) * libor_zero


def phi_f_k(k, libor_zero):
    return beta_f_k * libor_f[k-1] + (1-beta_f_k) * libor_zero


def generate_rv():
    # rho_tri = np.linalg.cholesky(c)
    L, D, Perm = ldl(c)

    w_d_1, w_d_2, w_f_1, w_f_2, w_xi = L[Perm,
                                         :]@np.random.normal(0, 1, 2 * N + 1)

    return w_d_1, w_d_2, w_f_1, w_f_2, w_xi


if __name__ == "__main__":
    # print(generate_rv())
    # e = np.array([[1, 0.9,   1,    0.4, -0.15],
    #               [0.9, 1,    0.4,  1,   -0.15],
    #               [1,   0.5,  1,    0.45,  -0.15],
    #               [0.5,  1,    0.45,   1,   -0.15],
    #               [-0.15, -0.15, -0.15, -0.15,  1],
    #               ])
    # print(np.linalg.eigvals(e))
    #rho_tri = np.linalg.cholesky(e)

    print(c)
    print(np.linalg.eigvals(c))
# print(L@D@L.T)
