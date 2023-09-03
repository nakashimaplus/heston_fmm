from math import (
    exp,
    sqrt,
)
import numpy as np

delta_list = [-1.5, -1.0, -0.5, 0, 0.5, 1, 1.5]
maturity_list = [1/2, 1, 3, 5, 7, 10, 15, 20, 25, 30]

# モデルパラメータの設定
eta_d = 0.007
eta_f = 0.012
lambda_d = 0.01
lambda_f = 0.05

kappa = 0.5
gamma = 0.3
sigma_bar = 0.1
sigma_zero = 0.1
xi_zero = 1.35

delta_h = 1/20


def strike(T, delta):
    fx_0 = exp(-0.05*T)/exp(-0.02*T)*1.35
    strike = fx_0 * exp(0.1*delta*sqrt(T))
    return strike


def make_strike_table(delta_list, maturity_list):
    strike_list = []
    strike_table = []
    for delta in delta_list:
        for maturity in maturity_list:
            strike_list.append(strike(maturity, delta))
        strike_table.append(strike_list)
        strike_list = []
    return strike_table


def b_f(t, T):
    return 1/lambda_f*(exp(-lambda_f*(T-t))-1)


def b_d(t, T):
    return 1/lambda_d*(exp(-lambda_d*(T-t))-1)


def d_fx(t, T, fx_ini, sigma_ini, w_xi, w_d, w_f):
    fx = sqrt(abs(sigma_ini))*sqrt(delta_h)*w_xi-eta_d*b_d(t, T) * \
        sqrt(delta_h)*w_d+eta_f*b_f(t, T)*sqrt(delta_h)*w_f
    fx = fx_ini+fx_ini*fx
    return fx


def d_sigma(t, T, sigma_ini, w_sigma):
    rho_sigma_d = 0.3
    sigma = sigma_ini+(kappa*(sigma_bar-sigma_ini)+gamma*rho_sigma_d*eta_d*b_d(t, T)
                       * sqrt(abs(sigma_ini)))*delta_h + gamma*sqrt(abs(sigma_ini))*sqrt(delta_h)*w_sigma
    return sigma


def generate_rv():
    rho = np.array([[1, -0.4, -0.15, -0.15],
                    [-0.4, 1, 0.3, 0.3],
                    [-0.15, 0.3, 1, 0.25],
                    [-0.15, 0.3, 0.25, 1]])
    rho_tri = np.linalg.cholesky(rho)
    nrand = rho_tri@np.random.normal(0, 1, 4)

    return nrand


def calculate_fx(T, K):

    grid_list = [grid*delta_h for grid in range(1, int(T/delta_h+1))]
    # grid_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    sigma = 0.1
    fx = (exp(-0.05*T))/(exp(-0.02*T))*xi_zero

    for grid in grid_list:
        w_xi, w_sigma, w_d, w_f = generate_rv()
        fx = d_fx(grid, T, fx, sigma, w_xi, w_d, w_f)
        sigma = d_sigma(grid, T, sigma, w_sigma)

    return max(fx-K, 0)


def calculate_fx_loop(T, K):
    N = 50000
    sum = 0
    for i in range(0, N):
        sum += calculate_fx(T, K)
    return exp(-0.02*T)*sum/N


def beep(freq, dur=100):

    import winsound
    winsound.Beep(freq, dur)


if __name__ == "__main__":
    a = make_strike_table(delta_list, maturity_list)
    # print(a)

    sum = 0
    for i in range(0, 20):
        fx = calculate_fx_loop(0.5, 1.1960668818837148)
        sum += fx
        print(fx)
    b = sum/20
    print(b)
    beep(500, 2000)
