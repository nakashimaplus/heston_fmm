import numpy as np
import scipy.integrate as integrate
import enum

# This class defines puts and calls


class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0


def CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, tau, K, N, L, P0T):

    # cf   - Characteristic function as a functon, in the book denoted by \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # tau  - Time to maturity
    # K    - List of strikes
    # N    - Number of expansion terms
    # L    - Size of truncation domain (typ.:L=8 or L=10)
    # P0T  - Zero-coupon bond for maturity T.

    # Reshape K to become a column vector

    if K is not np.array:
        K = np.array(K).reshape([len(K), 1])

    # Assigning i=sqrt(-1)

    i = np.complex(0.0, 1.0)
    x0 = np.log(S0 / K)

    # Truncation domain

    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)

    # Summation from k = 0 to k=N-1

    k = np.linspace(0, N-1, N).reshape([N, 1])
    u = k * np.pi / (b - a)

    # Determine coefficients for put prices

    H_k = CallPutCoefficients(OptionType.PUT, a, b, k)
    mat = np.exp(i * np.outer((x0 - a), u))
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]
    value = K * np.real(mat.dot(temp))

    # We use the put-call parity for call options

    if CP == OptionType.CALL:
        value = value + S0 - K * P0T

    return value

# Determine coefficients for put prices


def CallPutCoefficients(CP, a, b, k):
    if CP == OptionType.CALL:
        c = 0.0
        d = b
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k), 1])
        else:
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k)
    elif CP == OptionType.PUT:
        c = a
        d = 0.0
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k = 2.0 / (b - a) * (- Chi_k + Psi_k)

    return H_k


def Chi_Psi(a, b, c, d, k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - \
        np.sin(k * np.pi * (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c

    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)), 2.0))
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d) - np.cos(k * np.pi
                                                                     * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi *
                                         (d - a) / (b - a)) - k * np.pi / (b - a) * np.sin(k
                                                                                           * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)

    value = {"chi": chi, "psi": psi}
    return value


def CfFH1LMM_EQ(u, tau, T):
    i = np.complex(0.0, 1.0)
    M = 500
    delta_s = tau / M
    delta_tau = 1

    kappa = 1.2
    xi_bar = 0.1
    gamma = 0.5
    xi_0 = 0.1

    # LMM model parameter settings

    beta = 0.5
    sigma = 0.25
    _lambda = 1
    v_0 = 1
    eta = 0.1

    # Correlations

    rho_x_xi = -0.3
    rho_x_l = 0.5
    rho_x_v = 0
    rho_l_l = 0.98

    def libor_0(j):
        return (bond_list[j-1]/bond_list[j]-1)/delta_tau

    def psi(j, t):
        _gamma = 0
        if t < tennor_list[j-1]:
            _gamma = 1
        elif t > tennor_list[j-1] and t <= tennor_list[j]:
            _gamma = (tennor_list[j]-t)/(tennor_list[j]-tennor_list[j-1])
        elif t > tennor_list[j]:
            _gamma = 0

        return delta_tau * sigma * _gamma * libor_0(j) / (1+delta_tau*libor_0(j))

    def m_func(t):
        import math
        return math.ceil(t)

    def a_1(t):
        if m_func(t) > T:
            return 0
        a_1 = 0
        for i in range(m_func(t), T+1):
            a_1 += psi(i, t) ** 2
        for i in range(m_func(t), T+1):
            for j in range(m_func(t), T+1):
                if not i == j:
                    a_1 += psi(i, t)*psi(j, t)*rho_l_l
        return a_1

    def a_2(t):
        if m_func(t) > T:
            return 0
        a_2 = 0
        for i in range(m_func(t), T+1):
            a_2 += psi(i, t) * rho_x_l
        return a_2

    def d_1_j(u):
        tmp1 = (rho_x_xi*gamma*i*u-kappa)**2
        tmp2 = gamma**2 * (i * u + u**2)
        return np.sqrt(tmp1 + tmp2)

    # meanSqrtV_3の代わり
    def meanSqrt(kappa, v0, vbar, gamma):
        a = np.sqrt(vbar-gamma**2/(8*kappa))
        b = np.sqrt(v0)-a
        c1 = 1/(4*kappa)*gamma**2*(1-np.exp(-kappa))
        delta = 4 * kappa * vbar / gamma/gamma
        kappa1 = 4 * kappa * v0 * \
            np.exp(-kappa) / (gamma**2*(1-np.exp(-kappa)))
        epsilon1 = c1 * (kappa1-1) + c1 * delta + \
            c1 * delta / (2*(delta+kappa1))
        epsilon1 = np.sqrt(epsilon1)
        c = - np.log((epsilon1 - a) / b)

        def temp1(t_list):

            return [a_1(t)+2*(a + b * np.exp(-c * t))*a_2(t) for t in t_list]
            # return (a + b * np.exp(-c * t_list))
        return temp1

    # def d_2_j(u, t):
    #     tmp = _lambda ** 2 + eta ** 2 * a_1(t) * (u**2 + i * u)
    #     return np.sqrt(tmp)

    def g_1_j(u, b_xi_ini):
        tmp1 = (kappa - rho_x_xi * gamma * i * u) - \
            d_1_j(u)-gamma**2 * b_xi_ini
        tmp2 = (kappa - rho_x_xi * gamma * i * u) + \
            d_1_j(u)-gamma**2 * b_xi_ini
        return tmp1 / tmp2

    # def g_2_j(u, b_v_ini, t):
    #     tmp1 = _lambda - d_2_j(u, t) - eta**2 * b_v_ini
    #     tmp2 = _lambda + d_2_j(u, t) - eta**2 * b_v_ini
    #     return tmp1 / tmp2

    def b_xi(b_xi_ini, u):
        tmp1 = (kappa-rho_x_xi * gamma * i * u - d_1_j(u)-gamma**2 *
                b_xi_ini) * (1-np.exp(-d_1_j(u)*delta_s))
        tmp2 = gamma ** 2 * (1 - g_1_j(u, b_xi_ini)*np.exp(-d_1_j(u)*delta_s))
        return tmp1 / tmp2

    # def b_v(b_v_ini, u, t):
    #     tmp1 = (_lambda - d_2_j(u, t)-eta**2*b_v_ini) * \
    #         (1-np.exp(-d_2_j(u, t)*delta_s))
    #     tmp2 = eta ** 2 * (1-g_2_j(u, b_v_ini, t)
    #                        * np.exp(-d_2_j(u, t)*delta_s))
    #     return tmp1 / tmp2

    def a_func(b_xi_ini, u, theta_integral, t):
        tmp_log_1 = (1-g_1_j(u, b_xi_ini)*np.exp(-d_1_j(u)
                     * delta_s))/(1-g_1_j(u, b_xi_ini))
        tmp1 = kappa * xi_bar / (gamma**2) * ((kappa-rho_x_xi*gamma *
                                               i*u-d_1_j(u))*delta_s - 2*np.log(tmp_log_1))
        # tmp_log_2 = (1-g_2_j(u, b_v_ini, t)*np.exp(-d_1_j(u)
        #              * delta_s))/(1-g_2_j(u, b_v_ini, t))

        # tmp2 = _lambda * v_0 / \
        #     (eta**2) * ((_lambda - d_2_j(u, t))*delta_s - 2*np.log(tmp_log_2))

        # TODO　↓が変わる
        # tmp3 = a_2(t) * (u**2 + i * u) * theta_integral
        tmp3 = (u**2 + i * u)/2 * theta_integral

        # return tmp1 + tmp2 - tmp3
        return tmp1 - tmp3

    # Integration within the function theta(u,tau)
    # v_sqrt = meanSqrt(_lambda, v_0, v_0, eta)
    xi_sqrt = meanSqrt(kappa, xi_0, xi_bar, gamma)

    def temp1(z1, tau): return xi_sqrt(tau-z1)

    # recursive calculation
    tau_list = [tau / M * i for i in range(0, M)]
    temp_b_xi = 0
    # temp_b_v = 0
    temp_a = 0

    for _tau in tau_list:
        tau_l = _tau
        tau_u = _tau + tau / M
        _N = 100
        z_u = np.linspace(0+1e-10, tau_u-1e-10, _N)
        z_l = np.linspace(0+1e-10, tau_l-1e-10, _N)

        theta_integral = integrate.trapz(np.real(
            temp1(z_u, tau_u)), z_u) - integrate.trapz(np.real(temp1(z_l, tau_l)), z_l)
        # temp_a += a_func(temp_b_xi, temp_b_v, u, theta_integral, T-_tau)
        temp_a += a_func(temp_b_xi,  u, theta_integral, T-_tau)
        temp_b_xi += b_xi(temp_b_xi, u)
        # temp_b_v += b_v(temp_b_v, u, T-_tau)

    # cf = np.exp(temp_a + temp_b_xi * xi_0 + temp_b_v * v_0)
    cf = np.exp(temp_a + temp_b_xi * xi_0)
    return cf


bond_list = [1, 0.9512, 0.9048, 0.8607, 0.8187,
             0.7788, 0.7408, 0.7047, 0.6703, 0.6376, 0.6065]

tennor_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def mainCalculation():
    CP = OptionType.CALL
    T = 10
    t = 0
    tau = T-t

    # Settings for the COS method

    N = 500
    L = 8

    # Market settings

    # Strike prices

    K = [2.4]

    # Value from the COS method

    def cf(u): return CfFH1LMM_EQ(u, tau, T)
    valCOS_H1HW = bond_list[T]*CallPutOptionPriceCOSMthd_StochIR(cf,
                                                                 CP, 1.0/bond_list[T], T, K, N, L, 1.0)
    print(valCOS_H1HW)


mainCalculation()
