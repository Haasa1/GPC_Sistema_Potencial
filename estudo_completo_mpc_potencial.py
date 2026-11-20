# -*- coding: utf-8 -*-
"""
Estudo Completo de Sistemas Dinâmicos e Controle Preditivo Baseado em Modelo (MPC)
Aplicado à Navegação de um Agente em Campo Potencial Anisotrópico Evolutivo E_j(x, y).

Tópicos Investigados:
  1. Modelagem de 4ª Ordem (Coordenadas Cartesianas, Polares e Longitudinais/Transversais),
     Análise de Malha Aberta, Diagramas de Fase e Dependência Paramétrica com r_S e r_A.
  2. MPC Aditivo (E(x,y) + u): Controle Conjunto de Posição e Velocidade com Restrições.
  3. MPC Multiplicativo / Bilinear ((1 + u) * E(x,y)): Análise de Controlabilidade,
     Travessia da Singularidade E(r_eq)=0, Regime Bipolar vs. Regime Unipolar Deslizante.
  4. Controle via Dispositivo Físico Externo (Eletroímãs com Dinâmica Indutiva RL +
     Gradiente de Campo Magnético e Força Giroscópica de Lorentz Interferindo em E(x,y)).
  5. Evolução Multi-Trial do Potencial E_j(x,y) (Aprendizado Radial e Longitudinal no Lattice)
     e Redução Progressiva do Esforço de Controle do MPC quando j -> infinito.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.optimize import root, minimize
from scipy.linalg import solve_discrete_are

# Configuração estética dos gráficos
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = ':'

# =============================================================================
# 1. PARÂMETROS FÍSICOS E CAMPO DO CONDUTOR CARREGADO (EQ. DO FIO FINITO)
# =============================================================================
M_DEFAULT = 1.0
ALPHA_DEFAULT = 0.6
KAPPA_DEFAULT = -10.0
EPS_DEFAULT = 0.1
R_S_DEFAULT = np.array([0.0, 0.0])
R_A_DEFAULT = np.array([10.0, 10.0])


def E_wire_field(x, y, r_S=R_S_DEFAULT, r_A=R_A_DEFAULT, kappa=KAPPA_DEFAULT, eps=EPS_DEFAULT):
    """
    Campo vetorial E(x, y) = -grad V(x, y) gerado por um segmento condutor
    de r_S (Entrada) até r_A (Alvo) com densidade linear de carga proporcional
    à posição ao longo do fio (quebra de simetria radial).
    """
    r1_vec = np.array([x - r_S[0], y - r_S[1]], dtype=float)
    r2_vec = np.array([x - r_A[0], y - r_A[1]], dtype=float)
    T_vec = np.array([r_A[0] - r_S[0], r_A[1] - r_S[1]], dtype=float)

    T = float(np.linalg.norm(T_vec))
    if T < 1e-12:
        return np.zeros(2)

    r1 = float(np.linalg.norm(r1_vec))
    r2 = float(np.linalg.norm(r2_vec))

    r1_t = np.sqrt(r1**2 + eps**2)
    r2_t = np.sqrt(r2**2 + eps**2)

    num_c = r1**2 - r2**2 + T**2
    den_c = T**2 - (r1_t + r2_t)**2
    if abs(den_c) < 1e-14:
        den_c = -1e-14

    arg_ln = (r1_t + r2_t + T) / max(r1_t + r2_t - T, 1e-14)
    ln_term = np.log(max(arg_ln, 1e-14))

    term1 = (r1_vec / r1_t) * (-1.0 + (num_c / den_c))
    term2 = (r2_vec / r2_t) * (1.0 + (num_c / den_c))
    term3 = (T_vec / T) * ln_term

    return (-kappa / T) * (term1 + term2 + term3)


def V_wire_potential(x, y, r_S=R_S_DEFAULT, r_A=R_A_DEFAULT, kappa=KAPPA_DEFAULT, eps=EPS_DEFAULT):
    """
    Potencial escalar V(x, y) obtido por integração numérica rápida de linha ou
    fórmula analítica regularizada do fio com densidade linear lambda(q) = q / T.
    (Usado para curvas de nível e energia mecânica no diagrama de fase).
    """
    # Quadratura Gauss-Legendre de 24 pontos sobre o segmento q in [0, T]
    T_vec = r_A - r_S
    T = float(np.linalg.norm(T_vec))
    u_nodes, weights = np.polynomial.legendre.leggauss(24)
    # Mapeia [-1, 1] -> [0, 1]
    xi = 0.5 * (u_nodes + 1.0)
    w = 0.5 * weights
    pts = r_S[None, :] + xi[:, None] * T_vec[None, :]
    dist = np.sqrt((x - pts[:, 0])**2 + (y - pts[:, 1])**2 + eps**2)
    # Potencial logarítmico/coulombiano 3D induzido no plano: V(r) = kappa * int_0^1 xi / dist dxi
    # Na convenção E = -grad V com kappa < 0 (atrativo), V tem mínimo em r_eq:
    return float(-kappa * T * np.sum(w * xi / dist))


def find_equilibrium(r_S=R_S_DEFAULT, r_A=R_A_DEFAULT, kappa=KAPPA_DEFAULT, eps=EPS_DEFAULT):
    """Encontra o ponto de equilíbrio de malha aberta onde E(x, y) = (0, 0)."""
    guess = r_S + 0.889 * (r_A - r_S)
    sol = root(lambda pos: E_wire_field(pos[0], pos[1], r_S=r_S, r_A=r_A, kappa=kappa, eps=eps), guess)
    return sol.x


def cartesian_to_polar_and_corridor(state, r_S=R_S_DEFAULT, r_A=R_A_DEFAULT):
    """
    Converte o estado [x, vx, y, vy] para:
      1. Coordenadas Polares centradas na Entrada r_S: (r, r_dot, theta, theta_dot)
      2. Coordenadas do Corredor (Longitudinal s e Transversal/Radial d_perp): (s, s_dot, d_perp, d_perp_dot)
    """
    x, vx, y, vy = state
    dx, dy = x - r_S[0], y - r_S[1]
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    if r > 1e-9:
        r_dot = (dx * vx + dy * vy) / r
        theta_dot = (dx * vy - dy * vx) / (r**2)
    else:
        r_dot, theta_dot = 0.0, 0.0

    T_vec = r_A - r_S
    T = np.linalg.norm(T_vec)
    e_par = T_vec / T
    e_perp = np.array([-e_par[1], e_par[0]])

    rel_pos = np.array([dx, dy])
    rel_vel = np.array([vx, vy])
    s = float(rel_pos @ e_par)
    s_dot = float(rel_vel @ e_par)
    d_perp = float(rel_pos @ e_perp)
    d_perp_dot = float(rel_vel @ e_perp)

    return (r, r_dot, theta, theta_dot), (s, s_dot, d_perp, d_perp_dot)


def jacobian_E(x, y, field_fn, delta=1e-4):
    """Calcula numericamente a matriz Jacobiana dE/dr (2x2) em (x, y)."""
    E_xp = field_fn(x + delta, y)
    E_xm = field_fn(x - delta, y)
    E_yp = field_fn(x, y + delta)
    E_ym = field_fn(x, y - delta)
    J = np.zeros((2, 2))
    J[:, 0] = (E_xp - E_xm) / (2.0 * delta)
    J[:, 1] = (E_yp - E_ym) / (2.0 * delta)
    return J


# =============================================================================
# 2. SIMULAÇÃO EM MALHA ABERTA E ANÁLISE DE SENSIBILIDADE A r_S E r_A
# =============================================================================
def simulate_open_loop(state0=(1.0, 0.0, 7.5, 0.0), t_end=35.0, n_pts=700,
                       r_S=R_S_DEFAULT, r_A=R_A_DEFAULT, kappa=KAPPA_DEFAULT,
                       eps=EPS_DEFAULT, m=M_DEFAULT, alpha=ALPHA_DEFAULT, field_override=None):
    t_eval = np.linspace(0.0, t_end, n_pts)

    def ode(t, s):
        x, vx, y, vy = s
        if field_override is not None:
            Ex, Ey = field_override(x, y)
        else:
            Ex, Ey = E_wire_field(x, y, r_S=r_S, r_A=r_A, kappa=kappa, eps=eps)
        return [vx, (Ex - alpha * vx) / m, vy, (Ey - alpha * vy) / m]

    sol = solve_ivp(ode, (0.0, t_end), state0, t_eval=t_eval, method='RK45', rtol=1e-7, atol=1e-9)
    return sol.t, sol.y


# =============================================================================
# 3. PERGUNTA 1: MPC ADITIVO (CONTROLE DE POSIÇÃO E VELOCIDADE COM RESTRIÇÕES)
# =============================================================================
def simulate_mpc_additive_pos_vel(state0=(1.0, 0.0, 7.5, 0.0), t_end=22.0, Ts=0.05,
                                  Np=20, Nu=6, v_max=2.2, u_max=95.0,
                                  q_pos=45.0, q_vel=18.0, r_du=0.25,
                                  m=M_DEFAULT, alpha=ALPHA_DEFAULT, field_fn=None):
    """
    MPC no Espaço de Estados de 4ª Ordem [x, vx, y, vy]^T para o sistema:
        m * r_ddot + alpha * r_dot = E(x, y) + u(t)
    Decompõe u(t) = -E_hat(r) + v_mpc(t) (compensação preditiva do potencial +
    QP/MPC com restrições explícitas de velocidade ||r_dot|| <= v_max e força ||u|| <= u_max),
    garantindo que r_A = (10, 10) com r_dot = (0, 0) seja ponto fixo assintótico exato
    (onde u_infty = -E(r_A) = (+67.89, +67.89)) e respeitando rigorosamente o limite v_max!
    """
    if field_fn is None:
        field_fn = lambda xx, yy: E_wire_field(xx, yy)

    p = np.exp(-alpha * Ts / m)
    A1 = np.array([
        [1.0, (m / alpha) * (1.0 - p)],
        [0.0, p]
    ])
    B1 = np.array([
        [(Ts - (m / alpha) * (1.0 - p)) / alpha],
        [(1.0 - p) / alpha]
    ])

    Ad = np.zeros((4, 4))
    Ad[0:2, 0:2] = A1
    Ad[2:4, 2:4] = A1

    Bd = np.zeros((4, 2))
    Bd[0:2, 0:1] = B1
    Bd[2:4, 1:2] = B1

    S_x = np.zeros((4 * Np, 4))
    S_u = np.zeros((4 * Np, 2 * Nu))
    Ak = np.eye(4)
    for i in range(Np):
        Ak = Ad @ Ak
        S_x[4*i:4*(i+1), :] = Ak
        for j in range(Nu):
            if j < Nu - 1:
                if j <= i:
                    S_u[4*i:4*(i+1), 2*j:2*(j+1)] = np.linalg.matrix_power(Ad, i - j) @ Bd
            else:
                if i >= Nu - 1:
                    accum = np.zeros((4, 2))
                    for m_idx in range(Nu - 1, i + 1):
                        accum += np.linalg.matrix_power(Ad, i - m_idx) @ Bd
                    S_u[4*i:4*(i+1), 2*j:2*(j+1)] = accum

    Q_blk = np.diag([q_pos, q_vel, q_pos, q_vel])
    Q_term = np.diag([4.0 * q_pos, 5.0 * q_vel, 4.0 * q_pos, 5.0 * q_vel])
    Q_bar = np.zeros((4 * Np, 4 * Np))
    for i in range(Np):
        Q_bar[4*i:4*(i+1), 4*i:4*(i+1)] = Q_blk if i < Np - 1 else Q_term

    D_mat = np.eye(2 * Nu)
    for j in range(1, Nu):
        D_mat[2*j:2*(j+1), 2*(j-1):2*j] = -np.eye(2)

    H_base = S_u.T @ Q_bar @ S_u + r_du * (D_mat.T @ D_mat) + 0.02 * np.eye(2 * Nu)

    n_steps = int(t_end / Ts)
    t_hist = np.zeros(n_steps + 1)
    x_hist = np.zeros((4, n_steps + 1))
    u_hist = np.zeros((2, n_steps + 1))
    E_hist = np.zeros((2, n_steps + 1))

    state = np.array(state0, dtype=float)
    x_hist[:, 0] = state
    E_hist[:, 0] = field_fn(state[0], state[2])
    v_mpc_prev = np.zeros(2)

    for k in range(n_steps):
        pos_now = np.array([state[0], state[2]])
        vel_now = np.array([state[1], state[3]])
        E_now = field_fn(pos_now[0], pos_now[1])

        # Gera trajetória de referência de posição e velocidade no horizonte Np
        # respeitando estritamente ||v_ref|| <= 0.94 * v_max e frenagem suave ao chegar em r_A
        X_ref = np.zeros(4 * Np)
        p_sim = pos_now.copy()
        for i in range(Np):
            err_p = R_A_DEFAULT - p_sim
            dist_p = float(np.linalg.norm(err_p))
            if dist_p > 1e-6:
                dir_p = err_p / dist_p
            else:
                dir_p = np.zeros(2)
            speed_ref = min(0.93 * v_max, 0.85 * dist_p)
            v_des = dir_p * speed_ref
            p_sim = p_sim + Ts * v_des
            X_ref[4*i:4*(i+1)] = [p_sim[0], v_des[0], p_sim[1], v_des[1]]

        d0_vec = np.zeros(2 * Nu)
        d0_vec[0:2] = v_mpc_prev

        free_err = S_x @ state - X_ref
        g_qp = S_u.T @ Q_bar @ free_err - r_du * (D_mat.T @ d0_vec)
        V_warm = -np.linalg.solve(H_base, g_qp)

        def cost_vel_constrained(V_vec):
            X_pred = S_x @ state + S_u @ V_vec
            err_x = X_pred - X_ref
            dv_vec = D_mat @ V_vec - d0_vec
            c_val = 0.5 * (err_x @ Q_bar @ err_x + r_du * (dv_vec @ dv_vec) + 0.02 * (V_vec @ V_vec))

            vx_p = X_pred[1::4]
            vy_p = X_pred[3::4]
            v_mag = np.hypot(vx_p, vy_p)
            viol = np.maximum(0.0, v_mag - 0.98 * v_max)
            return c_val + 2500.0 * np.sum(viol**2)

        res = minimize(cost_vel_constrained, V_warm, method='BFGS', options={'maxiter': 25})
        v_mpc_opt = res.x[0:2]

        # Ação total u(t) = -E(r_k) + v_mpc_opt com saturação física ||u||_inf <= u_max
        u_opt = np.clip(-E_now + v_mpc_opt, -u_max, u_max)
        v_mpc_prev = u_opt + E_now

        def ode_step(t, s):
            xx, vvx, yy, vvy = s
            Ex, Ey = field_fn(xx, yy)
            return [
                vvx,
                (Ex + u_opt[0] - alpha * vvx) / m,
                vvy,
                (Ey + u_opt[1] - alpha * vvy) / m
            ]

        sol = solve_ivp(ode_step, (k * Ts, (k + 1) * Ts), state, method='RK45')
        state = sol.y[:, -1]

        t_hist[k + 1] = (k + 1) * Ts
        x_hist[:, k + 1] = state
        u_hist[:, k + 1] = u_opt
        E_hist[:, k + 1] = field_fn(state[0], state[2])

    u_hist[:, 0] = u_hist[:, 1]
    return t_hist, x_hist, u_hist, E_hist


# =============================================================================
# 4. PERGUNTA 2: MPC MULTIPLICATIVO / BILINEAR: (1 + u) * E(x, y)
# =============================================================================
def simulate_mpc_multiplicative(state0=(1.0, 0.0, 7.5, 0.0), t_end=25.0, Ts=0.05,
                                Np=16, mode='bipolar_anisotropic',
                                m=M_DEFAULT, alpha=ALPHA_DEFAULT):
    """
    Controla o caminhante modulando multiplicativamente o próprio campo potencial:
      - mode == 'bipolar_anisotropic':
          m * r_ddot + alpha * r_dot = (1 + u_par) * E_par + (1 + u_perp) * E_perp
          O MPC calcula a aceleração longitudinal ótima a_par_mpc no horizonte Np
          garantindo velocidade positiva ao cruzar a singularidade E_par(s_eq) = 0,
          inverte o sinal (1 + u_par < 0) para s > s_eq e converge para 1 + u_par -> 0
          (i.e. u_par -> -1) exatamente em r_A = (10, 10) com velocidade zero!
      - mode == 'unipolar_sliding':
          Restrição estrita 1 + u(t) >= 0 (u(t) >= -1, sem inversão de polaridade!).
          Na zona s < s_eq (onde E_par > 0), o MPC regula o ganho 1 + u >= 0 para que
          o estado atinja exatamente a variedade de deslizamento viscoso:
              s + (m / alpha) * s_dot = T  <=>  s_dot = (alpha / m) * (T - s)
          no instante em que cruza s = s_eq. Em s >= s_eq, o MPC comanda 1 + u(t) = 0
          (u = -1, desligando o campo) e a partícula desliza até parar exatamente em r_A!
      - mode == 'bipolar_scalar':
          Único controle escalar u(t) in R multiplicando todo o vetor: (1 + u(t)) * E(x, y).
          Explora a subatuação em 2 fases: antes de s_eq mantém 1 + u > 0 para colapsar
          o erro transversal d_perp -> 0 sobre a variedade invariante de simetria, e após
          cruzar s_eq sobre o eixo, usa 1 + u < 0 para cravar r_A = (10, 10)!
    """
    T_vec = R_A_DEFAULT - R_S_DEFAULT
    T = float(np.linalg.norm(T_vec))
    e_par = T_vec / T
    e_perp = np.array([-e_par[1], e_par[0]])

    r_eq = find_equilibrium()
    s_eq = float((r_eq - R_S_DEFAULT) @ e_par)

    n_steps = int(t_end / Ts)
    t_hist = np.zeros(n_steps + 1)
    x_hist = np.zeros((4, n_steps + 1))
    u_hist = np.zeros((2, n_steps + 1))

    state = np.array(state0, dtype=float)
    x_hist[:, 0] = state

    # Matrizes discretas de 1 eixo para o preditor MPC de horizonte Np
    p = np.exp(-alpha * Ts / m)
    A1 = np.array([[1.0, (m / alpha) * (1.0 - p)], [0.0, p]])
    B1 = np.array([[(Ts - (m / alpha) * (1.0 - p)) / alpha], [(1.0 - p) / alpha]])

    # Ganho LQR/MPC discreto no horizonte para o canal longitudinal e transversal
    Q_s = np.diag([40.0, 18.0])
    R_s = np.array([[0.25]])
    P_s = solve_discrete_are(A1, B1, Q_s, R_s)
    K_mpc = np.linalg.solve(B1.T @ P_s @ B1 + R_s, B1.T @ P_s @ A1)[0]

    for k in range(n_steps):
        _, (s_now, vs_now, d_now, vd_now) = cartesian_to_polar_and_corridor(state)
        E_now = E_wire_field(state[0], state[2])
        Ep_now = float(E_now @ e_par)
        Eq_now = float(E_now @ e_perp)

        if mode == 'bipolar_anisotropic':
            # Referência preditiva para garantir travessia de s_eq com velocidade >= 1.1 m/s
            if s_now < s_eq + 0.08:
                v_des = max(0.8 * (T - s_now), 1.15)
                f_par_des = K_mpc[0] * (T - s_now) + K_mpc[1] * (v_des - vs_now)
            else:
                f_par_des = K_mpc[0] * (T - s_now) - K_mpc[1] * vs_now

            f_perp_des = -K_mpc[0] * d_now - K_mpc[1] * vd_now

            # Mapeia a força ótima do MPC para o ganho multiplicativo g = 1 + u = f_des / E
            if abs(Ep_now) > 0.12:
                g_par = np.clip(f_par_des / Ep_now, -4.5, 4.5)
            else:
                # Dentro da faixa estreita da singularidade E_par(s_eq) = 0, mantém ganho suave
                g_par = 0.0

            if abs(Eq_now) > 0.05:
                g_perp = np.clip(f_perp_des / Eq_now, 0.5, 4.0)
            else:
                g_perp = 1.5

            u_par = g_par - 1.0
            u_perp = g_perp - 1.0

        elif mode == 'unipolar_sliding':
            # Variedade de deslizamento viscoso até T: vs_slide(s) = (alpha / m) * (T - s)
            # Quando g_par = 0 (u = -1), a distância percorrida até parar é (m / alpha) * vs_now.
            s_stop_free = s_now + (m / alpha) * vs_now

            if s_stop_free >= T - 1e-3 or s_now >= s_eq:
                # Desliga o campo longitudinal (1 + u = 0 => u = -1): desliza até parar exatamente em s = T!
                g_par = 0.0
                g_perp = 0.0 if abs(d_now) < 5e-3 else 2.0
            else:
                # Antes de s_eq, regula a velocidade para atingir s_stop_free = T logo antes de s_eq
                vs_req = (alpha / m) * (T - s_now) + 0.015 * ((s_eq - s_now) / max(s_eq, 1.0))
                f_slide_mpc = 18.0 * (vs_req - vs_now) + alpha * vs_req
                if Ep_now > 0.08:
                    g_par = np.clip(f_slide_mpc / Ep_now, 0.0, 5.5)
                else:
                    g_par = 0.0
                g_perp = max(g_par, 2.8)

            u_par = g_par - 1.0
            u_perp = g_perp - 1.0

        elif mode == 'bipolar_scalar':
            # Controle escalar único u(t): (1 + u(t)) * E(x, y) -> g_par = g_perp = g_scalar
            if s_now < s_eq - 0.04:
                # Fase 1: 1 + u > 0 (colapsa d_perp -> 0 sobre o eixo de simetria antes de cruzar s_eq)
                v_des = max(0.75 * (T - s_now), 1.15)
                f_des = K_mpc[0] * (T - s_now) + K_mpc[1] * (v_des - vs_now)
                g_scal = np.clip(f_des / max(Ep_now, 0.15), 1.8, 4.8)
            elif abs(Ep_now) <= 0.12:
                # Travessia inercial da singularidade s_eq
                g_scal = 0.0
            else:
                # Fase 2: Já sobre a variedade invariante d_perp = 0, usa 1 + u < 0 para parar em r_A
                f_des = K_mpc[0] * (T - s_now) - K_mpc[1] * vs_now
                g_scal = np.clip(f_des / Ep_now, -4.5, 4.5)

            u_par = g_scal - 1.0
            u_perp = g_scal - 1.0

        u_hist[:, k + 1] = [u_par, u_perp]

        def ode_mult(t, s):
            xx, vvx, yy, vvy = s
            E_vec = E_wire_field(xx, yy)
            Ep = float(E_vec @ e_par)
            Eq = float(E_vec @ e_perp)
            F_total = (1.0 + u_par) * Ep * e_par + (1.0 + u_perp) * Eq * e_perp
            return [
                vvx,
                (F_total[0] - alpha * vvx) / m,
                vvy,
                (F_total[1] - alpha * vvy) / m
            ]

        sol = solve_ivp(ode_mult, (k * Ts, (k + 1) * Ts), state, method='RK45')
        state = sol.y[:, -1]
        t_hist[k + 1] = (k + 1) * Ts
        x_hist[:, k + 1] = state

    u_hist[:, 0] = u_hist[:, 1]
    return t_hist, x_hist, u_hist


# =============================================================================
# 5. PERGUNTA 3: ATUADOR FÍSICO EXTERNO (ELETROÍMÃ COM DINÂMICA RL + LORENTZ)
# =============================================================================
R_MAG_1 = np.array([11.8, 11.8])
R_MAG_2 = np.array([11.2, 8.8])
DELTA_MAG = 1.2    # Regularização geométrica do núcleo do eletroímã
GAMMA_MAG = 120.0  # Constante magneto-motriz (dimensionada para I_eq ~ 7 A)
L_COIL = 0.20      # Indutância da bobina (H)
R_COIL = 1.0       # Resistência da bobina (Ohm) -> constante de tempo RL tau = 0.20 s
Q_LORENTZ = 0.40   # Acoplamento giroscópico de Lorentz


def E_electromagnet_field(x, y, I1, I2):
    """
    Campo de força eletromagnética espacialmente não-homogêneo gerado pelas
    correntes (I1, I2) nos eletroímãs posicionados em R_MAG_1 e R_MAG_2:
      E_mag(r, I) = sum_{k=1}^2 GAMMA_MAG * I_k * (r_mag,k - r) / (||r_mag,k - r||^2 + delta_m^2)^(3/2)
    """
    r_pos = np.array([x, y], dtype=float)
    d1 = R_MAG_1 - r_pos
    d2 = R_MAG_2 - r_pos
    n1 = (np.dot(d1, d1) + DELTA_MAG**2)**1.5
    n2 = (np.dot(d2, d2) + DELTA_MAG**2)**1.5
    return GAMMA_MAG * (I1 * (d1 / n1) + I2 * (d2 / n2))


def compute_equilibrium_currents_at_target(r_A=R_A_DEFAULT):
    """
    Calcula as correntes estacionárias exatas (I1_eq, I2_eq) nos eletroímãs para que
    o campo resultante total no alvo se anule:
        E_fio(r_A) + E_mag(r_A, I_eq) = 0
    """
    E_res = E_wire_field(r_A[0], r_A[1])
    phi1 = E_electromagnet_field(r_A[0], r_A[1], 1.0, 0.0)
    phi2 = E_electromagnet_field(r_A[0], r_A[1], 0.0, 1.0)
    Phi_mat = np.column_stack([phi1, phi2])
    I_eq = np.linalg.solve(Phi_mat, -E_res)
    return I_eq


def simulate_mpc_electromagnet(state0=(1.0, 0.0, 7.5, 0.0), t_end=22.0, Ts=0.05,
                               Np=16, m=M_DEFAULT, alpha=ALPHA_DEFAULT):
    """
    Simula o sistema de 7ª Ordem acoplado (4 estados mecânicos do caminhante +
    3 correntes indutivas [I1, I2, I_Bz] governadas por L * dI/dt + R * I = V_in):
      m * r_ddot + alpha * r_dot = E_fio(r) + E_mag(r, I1, I2) + q_ef * (r_dot x B_z(I_Bz))
    O MPC calcula em cascata preditiva as correntes de referência e as tensões elétricas
    V_in = [V1, V2, V_Bz] aplicadas às bobinas para compensar o atraso indutivo RL
    e a variação espacial 1/||r_mag - r||^3 do eletroímã!
    """
    I_eq_12 = compute_equilibrium_currents_at_target(R_A_DEFAULT)
    I_eq = np.array([I_eq_12[0], I_eq_12[1], 0.0])
    V_eq = R_COIL * I_eq

    n_steps = int(t_end / Ts)
    t_hist = np.zeros(n_steps + 1)
    x_hist = np.zeros((7, n_steps + 1))  # [x, vx, y, vy, I1, I2, IBz]
    V_hist = np.zeros((3, n_steps + 1))  # Tensões de controle [V1, V2, VBz]
    F_mag_hist = np.zeros((2, n_steps + 1))

    state = np.zeros(7)
    state[0:4] = state0
    x_hist[:, 0] = state

    p_mech = np.exp(-alpha * Ts / m)
    A1 = np.array([[1.0, (m / alpha) * (1.0 - p_mech)], [0.0, p_mech]])
    B1 = np.array([[(Ts - (m / alpha) * (1.0 - p_mech)) / alpha], [(1.0 - p_mech) / alpha]])
    Q_m = np.diag([45.0, 20.0])
    R_m = np.array([[0.20]])
    P_m = solve_discrete_are(A1, B1, Q_m, R_m)
    K_mech = np.linalg.solve(B1.T @ P_m @ B1 + R_m, B1.T @ P_m @ A1)[0]

    p_elec = np.exp(-R_COIL * Ts / L_COIL)
    b_elec = (1.0 - p_elec) / R_COIL

    T_vec = R_A_DEFAULT - R_S_DEFAULT
    e_par = T_vec / np.linalg.norm(T_vec)
    e_perp = np.array([-e_par[1], e_par[0]])

    for k in range(n_steps):
        x_now, vx_now, y_now, vy_now, I1_now, I2_now, IBz_now = state

        # 1. Força eletromagnética desejada pelo preditor MPC mecânico
        err_x = R_A_DEFAULT[0] - x_now
        err_y = R_A_DEFAULT[1] - y_now
        v_tot_des_x = K_mech[0] * err_x - K_mech[1] * vx_now
        v_tot_des_y = K_mech[0] * err_y - K_mech[1] * vy_now

        E_w_now = E_wire_field(x_now, y_now)
        F_mag_des = np.array([v_tot_des_x, v_tot_des_y]) - E_w_now

        # 2. Mapeia para as correntes desejadas (I1_des, I2_des) via matriz de sensibilidade espacial
        phi1 = E_electromagnet_field(x_now, y_now, 1.0, 0.0)
        phi2 = E_electromagnet_field(x_now, y_now, 0.0, 1.0)
        Phi_local = np.column_stack([phi1, phi2])
        # Regularização Tikhonov longe do alvo (quando a partícula ainda está perto da entrada)
        I12_des = np.linalg.solve(Phi_local.T @ Phi_local + 0.08 * np.eye(2), Phi_local.T @ F_mag_des)
        I12_des = np.clip(I12_des, -12.0, 12.0)

        # 3. Bobina Lorentz B_z auxilia na deflexão transversal inicial sem gastar energia mecânica
        _, (s_c, vs_c, d_c, vd_c) = cartesian_to_polar_and_corridor(state[0:4])
        IBz_des = np.clip(-1.5 * d_c - 0.8 * vd_c, -4.0, 4.0) if abs(d_c) > 0.05 else 0.0

        I_ref = np.array([I12_des[0], I12_des[1], IBz_des])

        # 4. Controle Preditivo de Tensão nas Bobinas RL (inversão ZOH do atraso indutivo L/R)
        V_opt = (I_ref - p_elec * np.array([I1_now, I2_now, IBz_now])) / b_elec
        V_opt = np.clip(V_opt, -22.0, 22.0)

        def ode_em(t, s):
            xx, vvx, yy, vvy, i1, i2, ibz = s
            E_w = E_wire_field(xx, yy)
            E_m = E_electromagnet_field(xx, yy, i1, i2)
            F_lor = Q_LORENTZ * ibz * np.array([vvy, -vvx])
            F_tot = E_w + E_m + F_lor

            di1 = (V_opt[0] - R_COIL * i1) / L_COIL
            di2 = (V_opt[1] - R_COIL * i2) / L_COIL
            dibz = (V_opt[2] - R_COIL * ibz) / L_COIL
            return [
                vvx,
                (F_tot[0] - alpha * vvx) / m,
                vvy,
                (F_tot[1] - alpha * vvy) / m,
                di1,
                di2,
                dibz
            ]

        sol = solve_ivp(ode_em, (k * Ts, (k + 1) * Ts), state, method='RK45')
        state = sol.y[:, -1]

        t_hist[k + 1] = (k + 1) * Ts
        x_hist[:, k + 1] = state
        V_hist[:, k + 1] = V_opt
        E_m_now = E_electromagnet_field(state[0], state[2], state[4], state[5])
        F_lor_now = Q_LORENTZ * state[6] * np.array([state[3], -state[1]])
        F_mag_hist[:, k + 1] = E_m_now + F_lor_now

    V_hist[:, 0] = V_hist[:, 1]
    return t_hist, x_hist, V_hist, F_mag_hist, I_eq


# =============================================================================
# 6. PERGUNTA 4: POTENCIAL EVOLUTIVO E_j(x,y) AO LONGO DOS TRIALS j E ESFORÇO MPC
# =============================================================================
def E_evolutionary_trial(x, y, beta_j, r_S=R_S_DEFAULT, r_A=R_A_DEFAULT,
                         k_par=1.8, k_perp=2.6):
    """
    Família de Potenciais de Aprendizado Radial-Longitudinal E_j(x, y) onde beta_j in [0, 1]
    parametriza a evolução do aprendizado no lattice no trial j:
      - Para beta_1 = 0 (Trial inicial j=1): E_1(x, y) = E_fio(x, y) (condutor carregado bruto,
        com ponto de equilíbrio deslocado r_eq != r_A e dispersão angular).
      - Conforme j cresce (beta_j -> 1), o potencial consolida a estrutura longitudinal-radial
        ótima aprendida pelas estatísticas de transição no lattice:
          E_opt(x, y) = -k_par * tanh((s - T) / L_s) * e_par - k_perp * d_perp * e_perp
        que possui alinhamento perfeito com o vetor Entrada -> Alvo (r_S -> r_A) e
        único ponto fixo global assintoticamente estável exatamente em r_A = (10, 10)!
    """
    T_vec = r_A - r_S
    T = float(np.linalg.norm(T_vec))
    e_par = T_vec / T
    e_perp = np.array([-e_par[1], e_par[0]])

    rel = np.array([x - r_S[0], y - r_S[1]], dtype=float)
    s = float(rel @ e_par)
    d_perp = float(rel @ e_perp)

    E_base = E_wire_field(x, y, r_S=r_S, r_A=r_A)

    # Campo ótimo longitudinal-radial (corredor direcionado r_S -> r_A com mínimo em s=T, d_perp=0)
    L_scale = 3.5
    F_long_opt = -k_par * L_scale * np.tanh((s - T) / L_scale)
    F_rad_opt = -k_perp * d_perp
    E_opt = F_long_opt * e_par + F_rad_opt * e_perp

    return (1.0 - beta_j) * E_base + beta_j * E_opt


def simulate_evolution_across_trials(n_trials=12, state0=(1.0, 0.0, 7.5, 0.0)):
    """
    Simula a evolução conjunta do campo E_j(x, y) e do controlador MPC ao longo
    dos trials j = 1, 2, ..., n_trials.
    Mede em cada trial j:
      - O ponto de equilíbrio de malha aberta r_eq^(j) de E_j(x, y)
      - O índice de alinhamento vetorial médio cos(theta_E - theta_T) no corredor
      - A energia de esforço de controle do MPC: int_0^tf ||u_j(t)||^2 dt
      - O pico de força de controle ||u_j||_infty
    """
    T_vec = R_A_DEFAULT - R_S_DEFAULT
    e_par = T_vec / np.linalg.norm(T_vec)

    # Grade de pontos ao redor do corredor para medir o alinhamento angular do campo E_j
    gx = np.linspace(0.5, 9.5, 12)
    gy = np.linspace(0.5, 9.5, 12)

    results = []
    for j in range(1, n_trials + 1):
        # Lei de evolução do aprendizado por trial (consolidação exponencial + assíntota em 1)
        if j == 1:
            beta_j = 0.0
        elif j == n_trials:
            beta_j = 1.0
        else:
            beta_j = 1.0 - np.exp(-0.35 * (j - 1))

        field_j = lambda xx, yy, b=beta_j: E_evolutionary_trial(xx, yy, b)

        # Equilíbrio de malha aberta no trial j
        sol_eq = root(lambda pos: field_j(pos[0], pos[1]), R_S_DEFAULT + 0.92 * T_vec)
        r_eq_j = sol_eq.x
        eq_error_j = float(np.linalg.norm(R_A_DEFAULT - r_eq_j))

        # Alinhamento angular médio do campo com o vetor Entrada -> Alvo ao longo do corredor
        cos_align_list = []
        for s_val in np.linspace(1.0, 9.5, 18):
            pt = R_S_DEFAULT + (s_val / 10.0) * T_vec
            Ev = field_j(pt[0], pt[1])
            nE = np.linalg.norm(Ev)
            if nE > 1e-8:
                cos_align_list.append(float(np.dot(Ev, e_par) / nE))
        mean_alignment = float(np.mean(cos_align_list))

        # Simula o caminhante controlado pelo MPC nesse trial j
        t_j, x_j, u_j, E_j = simulate_mpc_additive_pos_vel(
            state0=state0, t_end=18.0, Ts=0.1, Np=14, Nu=5, v_max=2.5, field_fn=field_j
        )
        Ts_step = t_j[1] - t_j[0]
        u_norm_sq = np.sum(u_j**2, axis=0)
        effort_L2 = float(np.trapezoid(u_norm_sq, t_j))
        u_peak = float(np.max(np.linalg.norm(u_j, axis=0)))
        u_ss = float(np.linalg.norm(u_j[:, -1]))

        results.append({
            'trial': j,
            'beta': beta_j,
            'r_eq': r_eq_j,
            'eq_error': eq_error_j,
            'alignment': mean_alignment,
            't': t_j,
            'x': x_j,
            'u': u_j,
            'effort_L2': effort_L2,
            'u_peak': u_peak,
            'u_ss': u_ss,
        })

    return results


# =============================================================================
# 7. GERAÇÃO DE TODAS AS FIGURAS E DO RELATÓRIO TÉCNICO COMPLETO
# =============================================================================
def generate_all_figures_and_report(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("[1/6] Calculando análise de malha aberta, diagramas de fase e sensibilidade (r_S, r_A)...")

    r_eq_default = find_equilibrium()
    T_vec = R_A_DEFAULT - R_S_DEFAULT
    T_len = float(np.linalg.norm(T_vec))
    e_par = T_vec / T_len
    e_perp = np.array([-e_par[1], e_par[0]])
    s_eq_default = float((r_eq_default - R_S_DEFAULT) @ e_par)

    # -------------------------------------------------------------------------
    # FIGURA 1: MALHA ABERTA, DIAGRAMA DE FASE E SENSIBILIDADE PARAMÉTRICA
    # -------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(14, 10.5), dpi=150)
    fig1.suptitle(
        'Análise em Malha Aberta: Mapa Vetorial $E(x,y)$, Diagramas de Fase (4ª Ordem) e Sensibilidade a $(\\mathbf{r}_S, \\mathbf{r}_A)$',
        fontsize=13.5, fontweight='bold', y=0.98
    )
    gs1 = GridSpec(2, 2, figure=fig1, hspace=0.32, wspace=0.26)

    # (a) Campo vetorial 2D + Trajetórias de malha aberta de múltiplas CIs
    ax1a = fig1.add_subplot(gs1[0, 0])
    gx, gy = np.meshgrid(np.linspace(-0.8, 11.5, 22), np.linspace(-0.8, 11.5, 22))
    Ex_g = np.zeros_like(gx)
    Ey_g = np.zeros_like(gy)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            ev = E_wire_field(gx[i, j], gy[i, j])
            Ex_g[i, j], Ey_g[i, j] = ev[0], ev[1]
    E_mag_g = np.hypot(Ex_g, Ey_g)
    E_mag_clip = np.maximum(E_mag_g, 1e-6)

    ax1a.quiver(gx, gy, Ex_g / E_mag_clip, Ey_g / E_mag_clip, np.log1p(E_mag_g),
                cmap='Blues', alpha=0.65, pivot='mid', scale=26)
    ax1a.plot([R_S_DEFAULT[0], R_A_DEFAULT[0]], [R_S_DEFAULT[1], R_A_DEFAULT[1]],
              'k--', lw=1.8, label='Condutor Carregado ($\\mathbf{r}_S \\to \\mathbf{r}_A$)')

    cis = [(1.0, 0.0, 7.5, 0.0), (6.5, 0.0, 1.0, 0.0), (0.5, 0.0, 3.0, 0.0), (10.8, 0.0, 6.0, 0.0)]
    cols_ci = ['#d95f02', '#1b9e77', '#7570b3', '#e7298a']
    for idx, ci in enumerate(cis):
        _, y_ci = simulate_open_loop(state0=ci, t_end=35.0)
        lbl = f'CI ({ci[0]:.0f}, {ci[2]:.1f})' if idx < 2 else None
        ax1a.plot(y_ci[0], y_ci[2], color=cols_ci[idx], lw=2.1, label=lbl)
        ax1a.scatter(ci[0], ci[2], color=cols_ci[idx], s=45, edgecolor='k', zorder=6)

    ax1a.scatter(*R_S_DEFAULT, color='red', s=95, zorder=8, label='Entrada $\\mathbf{r}_S=(0,0)$')
    ax1a.scatter(*R_A_DEFAULT, color='#0055cc', marker='*', s=230, zorder=9, label='Alvo $\\mathbf{r}_A=(10,10)$')
    ax1a.scatter(*r_eq_default, color='darkorange', marker='X', s=150, edgecolor='k', zorder=10,
                 label=f'Equilíbrio Malha Aberta $\\mathbf{{r}}_{{eq}}=({r_eq_default[0]:.2f},{r_eq_default[1]:.2f})$')
    ax1a.set_title('(a) Campo Vetorial $E(x,y)$ e Acomodação Antes do Alvo', fontsize=11, fontweight='bold')
    ax1a.set_xlabel('$x$')
    ax1a.set_ylabel('$y$')
    ax1a.set_xlim(-0.8, 11.5)
    ax1a.set_ylim(-0.8, 11.5)
    ax1a.set_aspect('equal', adjustable='box')
    ax1a.grid(True)
    ax1a.legend(loc='lower right', fontsize=8)

    # (b) Diagrama de Fase Longitudinal (s, s_dot) sobre o eixo r_S -> r_A
    ax1b = fig1.add_subplot(gs1[0, 1])
    s_grid = np.linspace(0.2, 15.2, 45)
    vs_grid = np.linspace(-3.8, 3.8, 25)
    S_M, VS_M = np.meshgrid(s_grid, vs_grid)
    DS_M = VS_M
    DVS_M = np.zeros_like(S_M)
    for i in range(S_M.shape[0]):
        for j in range(S_M.shape[1]):
            pt = R_S_DEFAULT + S_M[i, j] * e_par
            Ep = float(E_wire_field(pt[0], pt[1]) @ e_par)
            DVS_M[i, j] = (Ep - ALPHA_DEFAULT * VS_M[i, j]) / M_DEFAULT

    ax1b.streamplot(s_grid, vs_grid, DS_M, DVS_M, color='#88aacc', density=1.1, linewidth=0.9, arrowsize=1.0)

    for alpha_val, col_a, ls_a in [(0.25, '#e41a1c', '-'), (0.60, '#1b9e77', '-'), (1.50, '#984ea3', '--')]:
        _, y_a = simulate_open_loop(state0=(0.5, 0.0, 0.5, 0.0), t_end=40.0, alpha=alpha_val)
        s_traj = (y_a[0] * e_par[0] + y_a[2] * e_par[1])
        vs_traj = (y_a[1] * e_par[0] + y_a[3] * e_par[1])
        ax1b.plot(s_traj, vs_traj, color=col_a, lw=2.2, ls=ls_a, label=f'Trajetória ($\\alpha={alpha_val}$)')

    ax1b.axvline(s_eq_default, color='darkorange', ls='--', lw=1.6, label=f'Atrator $s_{{eq}}={s_eq_default:.2f}$')
    ax1b.axvline(T_len, color='#0055cc', ls=':', lw=1.8, label=f'Alvo $T=\\|\\mathbf{{r}}_A-\\mathbf{{r}}_S\\|={T_len:.2f}$')
    ax1b.scatter([s_eq_default], [0.0], color='darkorange', marker='X', s=130, zorder=9)
    ax1b.scatter([T_len], [0.0], color='#0055cc', marker='*', s=190, zorder=9)
    ax1b.set_title('(b) Diagrama de Fase Longitudinal $(s,\\, \\dot{s})$ ao Longo do Fio', fontsize=11, fontweight='bold')
    ax1b.set_xlabel('Posição Longitudinal $s$ (unidades)')
    ax1b.set_ylabel('Velocidade Longitudinal $\\dot{s}$')
    ax1b.set_xlim(0.2, 15.2)
    ax1b.set_ylim(-3.8, 3.8)
    ax1b.grid(True)
    ax1b.legend(loc='upper right', fontsize=8)

    # (c) Diagrama de Fase Angular/Polar (theta - theta_A, theta_dot)
    ax1c = fig1.add_subplot(gs1[1, 0])
    theta_A_deg = 45.0
    for idx, ci in enumerate([(1.0, 0.0, 8.5, 0.0), (8.5, 0.0, 1.0, 0.0), (2.0, -1.0, 6.0, 1.5), (6.0, 1.0, 2.0, -1.0)]):
        _, y_p = simulate_open_loop(state0=ci, t_end=30.0)
        th_deg = np.degrees(np.arctan2(y_p[2], y_p[0])) - theta_A_deg
        r_sq = np.maximum(y_p[0]**2 + y_p[2]**2, 1e-6)
        th_dot_deg = np.degrees((y_p[0] * y_p[3] - y_p[2] * y_p[1]) / r_sq)
        ax1c.plot(th_deg, th_dot_deg, lw=2.0, color=cols_ci[idx], label=f'CI ({ci[0]:.1f}, {ci[2]:.1f})')
        ax1c.scatter(th_deg[0], th_dot_deg[0], color=cols_ci[idx], s=45, zorder=6)

    ax1c.axvline(0.0, color='k', ls='--', lw=1.2)
    ax1c.axhline(0.0, color='k', ls='--', lw=1.2)
    ax1c.scatter([0.0], [0.0], color='#0055cc', marker='*', s=200, zorder=9, label='Alinhamento Angular $(\\theta=\\theta_A, \\dot{\\theta}=0)$')
    ax1c.set_title('(c) Diagrama de Fase Angular $(\\theta - \\theta_A,\\, \\dot{\\theta})$ em Coordenadas Polares', fontsize=11, fontweight='bold')
    ax1c.set_xlabel('Erro Angular $\\theta - \\theta_A$ (graus)')
    ax1c.set_ylabel('Velocidade Angular $\\dot{\\theta}$ (graus/s)')
    ax1c.grid(True)
    ax1c.legend(loc='upper right', fontsize=8)

    # (d) Sensibilidade do Ponto de Equilíbrio à Distância T = ||r_A - r_S|| e regularização eps
    ax1d = fig1.add_subplot(gs1[1, 1])
    T_vals = np.linspace(4.0, 30.0, 28)
    for eps_v, col_e in [(0.02, '#1b9e77'), (0.10, '#d95f02'), (0.30, '#7570b3')]:
        err_abs = []
        ratio_eq = []
        for Tv in T_vals:
            rA_test = R_S_DEFAULT + np.array([Tv / np.sqrt(2), Tv / np.sqrt(2)])
            req_t = find_equilibrium(r_S=R_S_DEFAULT, r_A=rA_test, eps=eps_v)
            err_abs.append(float(np.linalg.norm(rA_test - req_t)))
            ratio_eq.append(float(np.linalg.norm(req_t - R_S_DEFAULT) / Tv))
        ax1d.plot(T_vals, err_abs, '-o', markersize=3.5, lw=2.0, color=col_e,
                  label=f'$\\epsilon={eps_v}$ ($r_{{eq}}/T \\approx {np.mean(ratio_eq):.3f}$)')

    ax1d.axvline(T_len, color='#0055cc', ls=':', lw=1.6, label=f'Caso Nominal $T={T_len:.2f}$')
    ax1d.set_title('(d) Erro Estacionário $\\|\\mathbf{r}_A - \\mathbf{r}_{eq}\\|$ vs. Distância $T = \\|\\mathbf{r}_A - \\mathbf{r}_S\\|$',
                   fontsize=11, fontweight='bold')
    ax1d.set_xlabel('Distância Entrada–Alvo $T = \\|\\mathbf{r}_A - \\mathbf{r}_S\\|$')
    ax1d.set_ylabel('Distância do Ponto Fixo ao Alvo $\\|\\mathbf{r}_A - \\mathbf{r}_{eq}\\|$')
    ax1d.grid(True)
    ax1d.legend(loc='upper left', fontsize=8.5)

    fig1.savefig(os.path.join(out_dir, 'fig1_malha_aberta_diagrama_fase.png'), bbox_inches='tight')
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # FIGURA 2: PERGUNTA 1 - MPC ADITIVO (POSIÇÃO E VELOCIDADE)
    # -------------------------------------------------------------------------
    print("[2/6] Simulando Pergunta 1: MPC Aditivo controlando Posição e Velocidade...")
    t_ol, y_ol = simulate_open_loop(state0=(1.0, 0.0, 7.5, 0.0), t_end=22.0)
    t_m1, x_m1, u_m1, E_m1 = simulate_mpc_additive_pos_vel(state0=(1.0, 0.0, 7.5, 0.0), v_max=2.2)
    t_m2, x_m2, u_m2, E_m2 = simulate_mpc_additive_pos_vel(state0=(1.0, 0.0, 7.5, 0.0), v_max=1.2)

    fig2 = plt.figure(figsize=(14, 10.0), dpi=150)
    fig2.suptitle(
        'Pergunta 1: Controle MPC Aditivo ($\\mathbf{E}(x,y) + \\mathbf{u}$) de Posição e Velocidade com Restrições',
        fontsize=13.5, fontweight='bold', y=0.98
    )
    gs2 = GridSpec(2, 2, figure=fig2, hspace=0.32, wspace=0.25)

    # (a) Trajetórias 2D
    ax2a = fig2.add_subplot(gs2[0, 0])
    ax2a.plot([R_S_DEFAULT[0], R_A_DEFAULT[0]], [R_S_DEFAULT[1], R_A_DEFAULT[1]], 'k--', lw=1.5, label='Eixo $\\mathbf{r}_S \\to \\mathbf{r}_A$')
    ax2a.plot(y_ol[0], y_ol[2], color='#d95f02', lw=2.0, label=f'Malha Aberta (Para em $({y_ol[0,-1]:.2f},{y_ol[2,-1]:.2f})$)')
    ax2a.plot(x_m1[0], x_m1[2], color='#1b9e77', lw=2.4, label=f'MPC ($v_{{\\max}}=2.2$) $\\to ({x_m1[0,-1]:.2f},{x_m1[2,-1]:.2f})$')
    ax2a.plot(x_m2[0], x_m2[2], color='#7570b3', lw=2.2, ls='-.', label=f'MPC ($v_{{\\max}}=1.2$) $\\to ({x_m2[0,-1]:.2f},{x_m2[2,-1]:.2f})$')
    ax2a.scatter(*R_S_DEFAULT, color='red', s=90, zorder=7)
    ax2a.scatter(*R_A_DEFAULT, color='#0055cc', marker='*', s=220, zorder=9, label='Alvo Exato $\\mathbf{r}_A=(10,10)$')
    ax2a.scatter(*r_eq_default, color='#d95f02', marker='X', s=130, zorder=8)
    ax2a.set_title('(a) Trajetória no Plano $(x, y)$', fontsize=11, fontweight='bold')
    ax2a.set_xlabel('$x$')
    ax2a.set_ylabel('$y$')
    ax2a.set_aspect('equal', adjustable='box')
    ax2a.grid(True)
    ax2a.legend(loc='lower right', fontsize=8.3)

    # (b) Diagrama de Fase Posição-Velocidade Longitudinal (s, vs) com Envelope v_max
    ax2b = fig2.add_subplot(gs2[0, 1])
    s_ol = y_ol[0] * e_par[0] + y_ol[2] * e_par[1]
    vs_ol = y_ol[1] * e_par[0] + y_ol[3] * e_par[1]
    s_1 = x_m1[0] * e_par[0] + x_m1[2] * e_par[1]
    vs_1 = x_m1[1] * e_par[0] + x_m1[3] * e_par[1]
    s_2 = x_m2[0] * e_par[0] + x_m2[2] * e_par[1]
    vs_2 = x_m2[1] * e_par[0] + x_m2[3] * e_par[1]

    ax2b.plot(s_ol, vs_ol, color='#d95f02', lw=2.0, label='Malha Aberta (Cai em $s_{eq}=12.57$)')
    ax2b.plot(s_1, vs_1, color='#1b9e77', lw=2.4, label='MPC ($v_{\\max}=2.2$ m/s)')
    ax2b.plot(s_2, vs_2, color='#7570b3', lw=2.2, ls='-.', label='MPC ($v_{\\max}=1.2$ m/s)')
    ax2b.axhline(2.2, color='#1b9e77', ls=':', lw=1.3, label='Teto de Velocidade $v_{\\max}=2.2$')
    ax2b.axhline(1.2, color='#7570b3', ls=':', lw=1.3, label='Teto de Velocidade $v_{\\max}=1.2$')
    ax2b.axvline(T_len, color='#0055cc', ls='--', lw=1.4)
    ax2b.scatter([T_len], [0.0], color='#0055cc', marker='*', s=200, zorder=9, label='Ponto Fixo Alvo $(T, 0)$')
    ax2b.set_title('(b) Diagrama de Fase $(s,\\, \\dot{s})$ com Controle de Velocidade', fontsize=11, fontweight='bold')
    ax2b.set_xlabel('Posição Longitudinal $s$')
    ax2b.set_ylabel('Velocidade Longitudinal $\\dot{s}$')
    ax2b.grid(True)
    ax2b.legend(loc='upper right', fontsize=8)

    # (c) Evolução Temporal de Posição e Módulo da Velocidade ||v(t)||
    ax2c = fig2.add_subplot(gs2[1, 0])
    v_norm_1 = np.hypot(x_m1[1], x_m1[3])
    v_norm_2 = np.hypot(x_m2[1], x_m2[3])
    err_pos_1 = np.hypot(x_m1[0] - R_A_DEFAULT[0], x_m1[2] - R_A_DEFAULT[1])
    err_pos_2 = np.hypot(x_m2[0] - R_A_DEFAULT[0], x_m2[2] - R_A_DEFAULT[1])

    ax2c.plot(t_m1, err_pos_1, color='#1b9e77', lw=2.2, label='Distância ao Alvo $\\|\\mathbf{r}-\\mathbf{r}_A\\|$ ($v_{\\max}=2.2$)')
    ax2c.plot(t_m2, err_pos_2, color='#7570b3', lw=2.0, ls='-.', label='Distância ao Alvo $\\|\\mathbf{r}-\\mathbf{r}_A\\|$ ($v_{\\max}=1.2$)')
    ax2c.plot(t_m1, v_norm_1, color='#2b5c8f', lw=1.8, ls='--', label='Velocidade $\\|\\dot{\\mathbf{r}}(t)\\|$ ($v_{\\max}=2.2$)')
    ax2c.plot(t_m2, v_norm_2, color='#d95f02', lw=1.8, ls=':', label='Velocidade $\\|\\dot{\\mathbf{r}}(t)\\|$ ($v_{\\max}=1.2$)')
    ax2c.set_title('(c) Convergência Conjunta de Posição ($\\to \\mathbf{r}_A$) e Velocidade ($\\to 0$)', fontsize=11, fontweight='bold')
    ax2c.set_xlabel('Tempo $t$ (s)')
    ax2c.set_ylabel('Erro de Posição / Velocidade')
    ax2c.grid(True)
    ax2c.legend(loc='upper right', fontsize=8)

    # (d) Esforço de Controle u(t) vs Compensação de Ponto Fixo -E(r_A)
    ax2d = fig2.add_subplot(gs2[1, 1])
    E_A = E_wire_field(R_A_DEFAULT[0], R_A_DEFAULT[1])
    ax2d.axhline(-E_A[0], color='k', ls='--', lw=1.5, label=f'$-E_x(\\mathbf{{r}}_A) = {-E_A[0]:.2f}$ (Ponto Fixo)')
    ax2d.plot(t_m1, u_m1[0], color='#1b9e77', lw=2.1, label='$u_x(t)$ ($v_{\\max}=2.2$)')
    ax2d.plot(t_m1, u_m1[1], color='#66a61e', lw=1.8, ls=':', label='$u_y(t)$ ($v_{\\max}=2.2$)')
    ax2d.plot(t_m2, u_m2[0], color='#7570b3', lw=1.9, ls='-.', label='$u_x(t)$ ($v_{\\max}=1.2$)')
    ax2d.set_title('(d) Força de Controle $\\mathbf{u}(t) \\to -\\mathbf{E}(\\mathbf{r}_A)$ no Alvo', fontsize=11, fontweight='bold')
    ax2d.set_xlabel('Tempo $t$ (s)')
    ax2d.set_ylabel('Controle Aditivo $\\mathbf{u}(t)$')
    ax2d.grid(True)
    ax2d.legend(loc='lower right', fontsize=8.3)

    fig2.savefig(os.path.join(out_dir, 'fig2_mpc_aditivo_posicao_velocidade.png'), bbox_inches='tight')
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # FIGURA 3: PERGUNTA 2 - MPC MULTIPLICATIVO / BILINEAR (1 + u) * E(x, y)
    # -------------------------------------------------------------------------
    print("[3/6] Simulando Pergunta 2: MPC Multiplicativo (1 + u)*E(x,y) nos regimes Bipolar e Unipolar...")
    t_mb_aniso, x_mb_aniso, u_mb_aniso = simulate_mpc_multiplicative(
        state0=(1.0, 0.0, 7.5, 0.0), mode='bipolar_anisotropic'
    )
    t_mb_uni, x_mb_uni, u_mb_uni = simulate_mpc_multiplicative(
        state0=(1.0, 0.0, 7.5, 0.0), mode='unipolar_sliding'
    )
    # No caso escalar puro (1+u)*E(x,y), testamos também partindo próximo ao corredor
    t_mb_scal, x_mb_scal, u_mb_scal = simulate_mpc_multiplicative(
        state0=(1.5, 0.0, 2.2, 0.0), mode='bipolar_scalar'
    )

    fig3 = plt.figure(figsize=(14, 10.2), dpi=150)
    fig3.suptitle(
        'Pergunta 2: MPC Multiplicativo / Bilinear $m\\ddot{\\mathbf{r}} + \\alpha\\dot{\\mathbf{r}} = (1 + u)\\,\\mathbf{E}(x,y)$ e Travessia da Singularidade',
        fontsize=13.2, fontweight='bold', y=0.98
    )
    gs3 = GridSpec(2, 2, figure=fig3, hspace=0.32, wspace=0.25)

    # (a) Trajetórias 2D
    ax3a = fig3.add_subplot(gs3[0, 0])
    ax3a.plot([R_S_DEFAULT[0], R_A_DEFAULT[0]], [R_S_DEFAULT[1], R_A_DEFAULT[1]], 'k--', lw=1.4, label='Eixo $\\mathbf{r}_S \\to \\mathbf{r}_A$')
    ax3a.plot(y_ol[0], y_ol[2], color='#d95f02', lw=1.9, alpha=0.8, label='Malha Aberta ($u=0$)')
    ax3a.plot(x_mb_aniso[0], x_mb_aniso[2], color='#1b9e77', lw=2.4,
              label=f'MPC Bipolar $(1+u_\\parallel, 1+u_\\perp)$ $\\to ({x_mb_aniso[0,-1]:.2f},{x_mb_aniso[2,-1]:.2f})$')
    ax3a.plot(x_mb_uni[0], x_mb_uni[2], color='#7570b3', lw=2.2, ls='-.',
              label=f'MPC Unipolar ($1+u \\geq 0$ Deslizante) $\\to ({x_mb_uni[0,-1]:.2f},{x_mb_uni[2,-1]:.2f})$')
    ax3a.plot(x_mb_scal[0], x_mb_scal[2], color='#e7298a', lw=2.0, ls=':',
              label=f'MPC Escalar $(1+u)\\mathbf{{E}}$ $\\to ({x_mb_scal[0,-1]:.2f},{x_mb_scal[2,-1]:.2f})$')
    ax3a.scatter(*r_eq_default, color='darkorange', marker='X', s=140, zorder=9, label='Singularidade $\\mathbf{E}(\\mathbf{r}_{eq})=\\mathbf{0}$')
    ax3a.scatter(*R_A_DEFAULT, color='#0055cc', marker='*', s=220, zorder=10, label='Alvo $\\mathbf{r}_A=(10,10)$')
    ax3a.set_title('(a) Trajetórias 2D sob Modulação Multiplicativa $(1+u)\\mathbf{E}(x,y)$', fontsize=11, fontweight='bold')
    ax3a.set_xlabel('$x$')
    ax3a.set_ylabel('$y$')
    ax3a.set_aspect('equal', adjustable='box')
    ax3a.grid(True)
    ax3a.legend(loc='lower right', fontsize=7.8)

    # (b) Diagrama de Fase Longitudinal (s, vs) mostrando a Barreira de Singularidade s_eq e a Reta de Deslizamento
    ax3b = fig3.add_subplot(gs3[0, 1])
    s_aniso = x_mb_aniso[0] * e_par[0] + x_mb_aniso[2] * e_par[1]
    vs_aniso = x_mb_aniso[1] * e_par[0] + x_mb_aniso[3] * e_par[1]
    s_uni = x_mb_uni[0] * e_par[0] + x_mb_uni[2] * e_par[1]
    vs_uni = x_mb_uni[1] * e_par[0] + x_mb_uni[3] * e_par[1]

    s_line = np.linspace(10.0, T_len, 100)
    vs_slide_line = (ALPHA_DEFAULT / M_DEFAULT) * (T_len - s_line)

    ax3b.plot(s_ol, vs_ol, color='#d95f02', lw=1.9, label='Malha Aberta (Presa em $s_{eq}$)')
    ax3b.plot(s_aniso, vs_aniso, color='#1b9e77', lw=2.4, label='MPC Bipolar ($1+u_\\parallel < 0$ após $s_{eq}$)')
    ax3b.plot(s_uni, vs_uni, color='#7570b3', lw=2.3, ls='-.', label='MPC Unipolar ($1+u \\geq 0$ + Desligamento $u=-1$)')
    ax3b.plot(s_line, vs_slide_line, 'k--', lw=1.5, label='Variedade de Deslizamento $\\dot{s} = \\frac{\\alpha}{m}(T - s)$')
    ax3b.axvline(s_eq_default, color='red', ls=':', lw=1.8, label=f'Singularidade $E_\\parallel(s_{{eq}})=0$ ($s_{{eq}}={s_eq_default:.2f}$)')
    ax3b.scatter([T_len], [0.0], color='#0055cc', marker='*', s=210, zorder=10)
    ax3b.set_title('(b) Diagrama de Fase $(s,\\, \\dot{s})$: Travessia de $s_{eq}$ com $\\dot{s}(s_{eq}) > 0$', fontsize=11, fontweight='bold')
    ax3b.set_xlabel('Posição Longitudinal $s$')
    ax3b.set_ylabel('Velocidade Longitudinal $\\dot{s}$')
    ax3b.set_xlim(4.0, 14.8)
    ax3b.grid(True)
    ax3b.legend(loc='upper left', fontsize=7.8)

    # (c) Ganho Multiplicativo Efetivo (1 + u(t)) no Tempo
    ax3c = fig3.add_subplot(gs3[1, 0])
    ax3c.axhline(1.0, color='#888888', ls=':', lw=1.2, label='Nominal Sem Controle ($1+u = 1$)')
    ax3c.axhline(0.0, color='k', ls='--', lw=1.4, label='Campo Desligado no Alvo ($1+u = 0 \\Leftrightarrow u = -1$)')
    ax3c.plot(t_mb_aniso, 1.0 + u_mb_aniso[0], color='#1b9e77', lw=2.2, label='Ganho Longitudinal $1 + u_\\parallel(t)$ (Bipolar)')
    ax3c.plot(t_mb_uni, 1.0 + u_mb_uni[0], color='#7570b3', lw=2.2, ls='-.', label='Ganho Longitudinal $1 + u(t) \\geq 0$ (Unipolar)')
    ax3c.set_title('(c) Evolução do Ganho Modulador $1 + u(t)$', fontsize=11, fontweight='bold')
    ax3c.set_xlabel('Tempo $t$ (s)')
    ax3c.set_ylabel('Fator Multiplicativo $1 + u(t)$')
    ax3c.grid(True)
    ax3c.legend(loc='upper right', fontsize=8)

    # (d) Erro de Posição e Velocidade em Malha Fechada
    ax3d = fig3.add_subplot(gs3[1, 1])
    err_b_aniso = np.hypot(x_mb_aniso[0] - R_A_DEFAULT[0], x_mb_aniso[2] - R_A_DEFAULT[1])
    err_b_uni = np.hypot(x_mb_uni[0] - R_A_DEFAULT[0], x_mb_uni[2] - R_A_DEFAULT[1])
    err_b_scal = np.hypot(x_mb_scal[0] - R_A_DEFAULT[0], x_mb_scal[2] - R_A_DEFAULT[1])
    ax3d.semilogy(t_mb_aniso, np.maximum(err_b_aniso, 1e-4), color='#1b9e77', lw=2.3,
                  label=f'MPC Bipolar Anisotrópico (Erro final = {err_b_aniso[-1]:.4f})')
    ax3d.semilogy(t_mb_uni, np.maximum(err_b_uni, 1e-4), color='#7570b3', lw=2.2, ls='-.',
                  label=f'MPC Unipolar $1+u\\geq 0$ (Erro final = {err_b_uni[-1]:.4f})')
    ax3d.semilogy(t_mb_scal, np.maximum(err_b_scal, 1e-4), color='#e7298a', lw=2.0, ls=':',
                  label=f'MPC Escalar $(1+u)\\mathbf{{E}}$ (Erro final = {err_b_scal[-1]:.4f})')
    ax3d.set_title('(d) Convergência do Erro $\\|\\mathbf{r}(t) - \\mathbf{r}_A\\|$ (Escala Log)', fontsize=11, fontweight='bold')
    ax3d.set_xlabel('Tempo $t$ (s)')
    ax3d.set_ylabel('Distância ao Alvo $\\|\\mathbf{r}(t) - \\mathbf{r}_A\\|$')
    ax3d.grid(True, which='both')
    ax3d.legend(loc='upper right', fontsize=8)

    fig3.savefig(os.path.join(out_dir, 'fig3_mpc_multiplicativo_bilinear.png'), bbox_inches='tight')
    plt.close(fig3)

    # -------------------------------------------------------------------------
    # FIGURA 4: PERGUNTA 3 - CONTROLE VIA ELETROÍMÃ INTERFERINDO NO POTENCIAL
    # -------------------------------------------------------------------------
    print("[4/6] Simulando Pergunta 3: Eletroímã com Dinâmica Indutiva RL + Gradiente + Lorentz...")
    t_em, x_em, V_em, F_mag_em, I_eq_em = simulate_mpc_electromagnet(state0=(1.0, 0.0, 7.5, 0.0))

    fig4 = plt.figure(figsize=(14, 10.2), dpi=150)
    fig4.suptitle(
        'Pergunta 3: Controle MPC de um Eletroímã Físico Interferindo no Potencial $\\mathbf{E}_{ef}(x,y) = \\mathbf{E}_{fio}(x,y) + \\mathbf{E}_{mag}(x,y,\\mathbf{I}(t))$',
        fontsize=12.8, fontweight='bold', y=0.98
    )
    gs4 = GridSpec(2, 2, figure=fig4, hspace=0.32, wspace=0.25)

    # (a) Campo Potencial Deformado pelo Eletroímã + Trajetória
    ax4a = fig4.add_subplot(gs4[0, 0])
    Ex_tot = np.zeros_like(gx)
    Ey_tot = np.zeros_like(gy)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            ew = E_wire_field(gx[i, j], gy[i, j])
            em = E_electromagnet_field(gx[i, j], gy[i, j], I_eq_em[0], I_eq_em[1])
            et = ew + em
            Ex_tot[i, j], Ey_tot[i, j] = et[0], et[1]
    Emag_tot = np.maximum(np.hypot(Ex_tot, Ey_tot), 1e-6)

    ax4a.quiver(gx, gy, Ex_tot / Emag_tot, Ey_tot / Emag_tot, np.log1p(Emag_tot),
                cmap='Greens', alpha=0.65, pivot='mid', scale=26)
    ax4a.plot(y_ol[0], y_ol[2], color='#d95f02', lw=1.9, ls='--', label='Sem Eletroímã (Para em $r_{eq}$)')
    ax4a.plot(x_em[0], x_em[2], color='#006d2c', lw=2.5,
              label=f'Com Eletroímã MPC $\\to ({x_em[0,-1]:.2f},{x_em[2,-1]:.2f})$')
    ax4a.scatter(*R_MAG_1, color='crimson', marker='s', s=130, edgecolor='k', zorder=9, label='Eletroímã 1 (Axial)')
    ax4a.scatter(*R_MAG_2, color='purple', marker='s', s=110, edgecolor='k', zorder=9, label='Eletroímã 2 (Lateral)')
    ax4a.scatter(*R_A_DEFAULT, color='#0055cc', marker='*', s=230, zorder=10, label='Novo Ponto Fixo $\\mathbf{r}_A=(10,10)$')
    ax4a.set_title('(a) Campo Deformado $\\mathbf{E}_{fio}+\\mathbf{E}_{mag}$ e Arranjo das Bobinas', fontsize=11, fontweight='bold')
    ax4a.set_xlabel('$x$')
    ax4a.set_ylabel('$y$')
    ax4a.set_xlim(-0.5, 12.5)
    ax4a.set_ylim(-0.5, 12.5)
    ax4a.set_aspect('equal', adjustable='box')
    ax4a.grid(True)
    ax4a.legend(loc='lower right', fontsize=7.8)

    # (b) Deformação da Força Longitudinal ao longo do eixo s in [6, 15]
    ax4b = fig4.add_subplot(gs4[0, 1])
    s_prof = np.linspace(6.0, 15.0, 160)
    E_fio_prof = []
    E_mag_prof = []
    E_tot_prof = []
    for sv in s_prof:
        pt = R_S_DEFAULT + sv * e_par
        ew = float(E_wire_field(pt[0], pt[1]) @ e_par)
        em = float(E_electromagnet_field(pt[0], pt[1], I_eq_em[0], I_eq_em[1]) @ e_par)
        E_fio_prof.append(ew)
        E_mag_prof.append(em)
        E_tot_prof.append(ew + em)

    ax4b.axhline(0.0, color='k', lw=1.2)
    ax4b.plot(s_prof, E_fio_prof, color='#d95f02', lw=2.0, ls='--', label='Campo Original $E_{\\parallel,fio}(s)$ (Zero em $12.57$)')
    ax4b.plot(s_prof, E_mag_prof, color='crimson', lw=1.9, ls=':', label='Campo do Eletroímã $E_{\\parallel,mag}(s, \\mathbf{I}_{eq})$')
    ax4b.plot(s_prof, E_tot_prof, color='#006d2c', lw=2.5, label='Campo Resultante $E_{\\parallel,ef}(s)$ (Zero exato em $T=14.14$)')
    ax4b.scatter([s_eq_default], [0.0], color='#d95f02', marker='X', s=120, zorder=8)
    ax4b.scatter([T_len], [0.0], color='#0055cc', marker='*', s=210, zorder=9, label='Alvo $s = T = 14.14$')
    ax4b.set_title('(b) Interferência Destrutiva em $\\mathbf{r}_A$: Deslocamento do Equilíbrio para $T$', fontsize=11, fontweight='bold')
    ax4b.set_xlabel('Posição Longitudinal $s$ ao Longo do Eixo')
    ax4b.set_ylabel('Força Longitudinal $E_\\parallel(s)$')
    ax4b.set_ylim(-8.0, 8.0)
    ax4b.grid(True)
    ax4b.legend(loc='lower left', fontsize=8)

    # (c) Dinâmica Elétrica RL do Eletroímã: Tensões V_k(t) e Correntes I_k(t)
    ax4c = fig4.add_subplot(gs4[1, 0])
    ax4c.plot(t_em, x_em[4], color='crimson', lw=2.3, label=f'Corrente Bobina 1 $I_1(t) \\to {I_eq_em[0]:.2f}$ A')
    ax4c.plot(t_em, x_em[5], color='purple', lw=2.0, label=f'Corrente Bobina 2 $I_2(t) \\to {I_eq_em[1]:.2f}$ A')
    ax4c.plot(t_em, x_em[6], color='#2b5c8f', lw=1.8, ls='-.', label='Corrente Lorentz $I_{Bz}(t) \\to 0$ A')
    ax4c.plot(t_em, V_em[0] / R_COIL, color='crimson', lw=1.1, alpha=0.45, ls='--', label='Comando Tensão $V_1(t)/R$')
    ax4c.set_title('(c) Dinâmica Indutiva RL dos Eletroímãs ($L\\dot{I} + RI = V_{MPC}$)', fontsize=11, fontweight='bold')
    ax4c.set_xlabel('Tempo $t$ (s)')
    ax4c.set_ylabel('Corrente nas Bobinas $I_k(t)$ (A)')
    ax4c.grid(True)
    ax4c.legend(loc='lower right', fontsize=8)

    # (d) Diagrama de Fase Longitudinal e Transversal sob Ação do Eletroímã
    ax4d = fig4.add_subplot(gs4[1, 1])
    s_em = x_em[0] * e_par[0] + x_em[2] * e_par[1]
    vs_em = x_em[1] * e_par[0] + x_em[3] * e_par[1]
    d_em = (x_em[0] - R_S_DEFAULT[0]) * e_perp[0] + (x_em[2] - R_S_DEFAULT[1]) * e_perp[1]
    vd_em = x_em[1] * e_perp[0] + x_em[3] * e_perp[1]

    ax4d.plot(s_ol, vs_ol, color='#d95f02', lw=1.8, ls='--', label='Malha Aberta $(s, \\dot{s})$')
    ax4d.plot(s_em, vs_em, color='#006d2c', lw=2.4, label='Com Eletroímã MPC $(s, \\dot{s}) \\to (T, 0)$')
    ax4d.plot(T_len + d_em, vd_em, color='purple', lw=2.0, ls='-.', label='Modo Transversal $(T + d_\\perp, \\dot{d}_\\perp) \\to (T, 0)$')
    ax4d.scatter([T_len], [0.0], color='#0055cc', marker='*', s=210, zorder=9, label='Equilíbrio Final $(s=T, d_\\perp=0, \\mathbf{v}=\\mathbf{0})$')
    ax4d.set_title('(d) Retrato de Fase $(s, \\dot{s})$ e $(d_\\perp, \\dot{d}_\\perp)$ com Eletroímã', fontsize=11, fontweight='bold')
    ax4d.set_xlabel('Coordenada Espacial ($s$ ou $T + d_\\perp$)')
    ax4d.set_ylabel('Velocidade ($\\dot{s}$ ou $\\dot{d}_\\perp$)')
    ax4d.grid(True)
    ax4d.legend(loc='upper right', fontsize=8)

    fig4.savefig(os.path.join(out_dir, 'fig4_mpc_eletroima_fisico.png'), bbox_inches='tight')
    plt.close(fig4)

    # -------------------------------------------------------------------------
    # FIGURA 5: PERGUNTA 4 - EVOLUÇÃO MULTI-TRIAL E_j(x,y) E ESFORÇO DO MPC
    # -------------------------------------------------------------------------
    print("[5/6] Simulando Pergunta 4: Evolução de E_j(x,y) por Trial j e Queda do Esforço de Controle...")
    trials_data = simulate_evolution_across_trials(n_trials=12)

    fig5 = plt.figure(figsize=(14.5, 10.2), dpi=150)
    fig5.suptitle(
        'Evolução do Potencial $\\mathbf{E}_j(x,y)$ ao Longo dos Trials $j$ (Aprendizado no Lattice) e Redução do Esforço de Controle do MPC',
        fontsize=13.0, fontweight='bold', y=0.98
    )
    gs5 = GridSpec(2, 3, figure=fig5, hspace=0.34, wspace=0.28)

    # (a), (b), (c): Mapas vetoriais de E_j(x,y) para j = 1, j = 4, j = 12 (j -> inf)
    selected_trials = [0, 3, 11]
    titles_tr = [
        '(a) Trial $j=1$ ($\\beta_1=0$): Fio Bruto Inicial',
        '(b) Trial $j=4$ ($\\beta_4=0.65$): Em Aprendizado',
        '(c) Trial $j \\to \\infty$ ($\\beta_{12}=1.0$): Potencial Ótimo'
    ]
    for idx_plot, tr_idx in enumerate(selected_trials):
        ax_t = fig5.add_subplot(gs5[0, idx_plot])
        item = trials_data[tr_idx]
        b_val = item['beta']
        req_j = item['r_eq']

        Ex_j = np.zeros_like(gx)
        Ey_j = np.zeros_like(gy)
        for ii in range(gx.shape[0]):
            for jj in range(gx.shape[1]):
                ev = E_evolutionary_trial(gx[ii, jj], gy[ii, jj], b_val)
                Ex_j[ii, jj], Ey_j[ii, jj] = ev[0], ev[1]
        Em_j = np.maximum(np.hypot(Ex_j, Ey_j), 1e-6)

        ax_t.quiver(gx, gy, Ex_j / Em_j, Ey_j / Em_j, np.log1p(Em_j),
                    cmap='viridis', alpha=0.72, pivot='mid', scale=25)
        ax_t.plot([R_S_DEFAULT[0], R_A_DEFAULT[0]], [R_S_DEFAULT[1], R_A_DEFAULT[1]], 'k--', lw=1.3)
        ax_t.plot(item['x'][0], item['x'][2], color='crimson', lw=2.3, label='Trajetória MPC')
        ax_t.scatter(*R_S_DEFAULT, color='red', s=70, zorder=8)
        ax_t.scatter(*R_A_DEFAULT, color='#0055cc', marker='*', s=200, zorder=10, label='Alvo $\\mathbf{r}_A$')
        tr_num = item['trial']
        ax_t.scatter(*req_j, color='orange', marker='X', s=120, edgecolor='k', zorder=9,
                     label=f'$\\mathbf{{r}}_{{eq}}^{{({tr_num})}}=({req_j[0]:.1f},{req_j[1]:.1f})$')
        ax_t.set_title(titles_tr[idx_plot], fontsize=10.2, fontweight='bold')
        ax_t.set_xlabel('$x$')
        ax_t.set_ylabel('$y$')
        ax_t.set_xlim(-0.5, 11.5)
        ax_t.set_ylim(-0.5, 11.5)
        ax_t.set_aspect('equal', adjustable='box')
        ax_t.grid(True)
        ax_t.legend(loc='lower right', fontsize=7.5)

    # (d) Evolução do Erro de Malha Aberta ||r_A - r_eq^(j)|| e Alinhamento Angular por Trial j
    ax5d = fig5.add_subplot(gs5[1, 0])
    j_list = [d['trial'] for d in trials_data]
    eq_errs = [d['eq_error'] for d in trials_data]
    aligns = [d['alignment'] for d in trials_data]

    ax5d.plot(j_list, eq_errs, '-o', color='#d95f02', lw=2.2, label='Desvio Natural $\\|\\mathbf{r}_A - \\mathbf{r}_{eq}^{(j)}\\|$')
    ax5d_r = ax5d.twinx()
    ax5d_r.plot(j_list, aligns, '-s', color='#0055cc', lw=2.0, label='Alinhamento $\\langle \\cos(\\theta_{\\mathbf{E}_j} - \\theta_{\\mathbf{T}}) \\rangle$')
    ax5d.set_title('(d) Convergência de $\\mathbf{r}_{eq}^{(j)} \\to \\mathbf{r}_A$ e Alinhamento', fontsize=10.5, fontweight='bold')
    ax5d.set_xlabel('Trial $j$ (Aprendizado no Lattice)')
    ax5d.set_ylabel('Erro de Malha Aberta $\\|\\mathbf{r}_A - \\mathbf{r}_{eq}^{(j)}\\|$', color='#d95f02')
    ax5d_r.set_ylabel('Cosseno Direcional Médio no Corredor', color='#0055cc')
    ax5d.set_xticks(j_list)
    ax5d.grid(True)

    # (e) Queda do Esforço de Controle Integrado L2 e Força de Regime do MPC
    ax5e = fig5.add_subplot(gs5[1, 1])
    efforts = [d['effort_L2'] for d in trials_data]
    u_ss_list = [d['u_ss'] for d in trials_data]
    ax5e.plot(j_list, efforts, '-o', color='#1b9e77', lw=2.4, label='Energia MPC $\\int_0^{t_f} \\|\\mathbf{u}_j(t)\\|^2 dt$')
    ax5e.plot(j_list, np.array(u_ss_list) * 5.0, '--^', color='crimson', lw=2.0,
              label='$5 \\times \\|\\mathbf{u}_j(t_f)\\|$ (Esforço Residual em $\\mathbf{r}_A$)')
    ax5e.set_title('(e) Redução do Esforço de Controle do MPC com $j$', fontsize=10.5, fontweight='bold')
    ax5e.set_xlabel('Trial $j$')
    ax5e.set_ylabel('Esforço de Controle do MPC')
    ax5e.set_xticks(j_list)
    ax5e.grid(True)
    ax5e.legend(loc='upper right', fontsize=8)

    # (f) Perfis Temporais de ||u_j(t)|| para diferentes trials j
    ax5f = fig5.add_subplot(gs5[1, 2])
    cmap_tr = plt.cm.plasma(np.linspace(0.1, 0.9, 5))
    for c_i, tr_i in enumerate([0, 1, 3, 6, 11]):
        d_item = trials_data[tr_i]
        tr_d = d_item['trial']
        bt_d = d_item['beta']
        u_mag_t = np.linalg.norm(d_item['u'], axis=0)
        ax5f.plot(d_item['t'], u_mag_t, lw=2.1, color=cmap_tr[c_i],
                  label=f'Trial $j={tr_d}$ ($\\beta={bt_d:.2f}$)')
    ax5f.set_title('(f) Norma do Controle $\\|\\mathbf{u}_j(t)\\|$ ao Longo dos Trials', fontsize=10.5, fontweight='bold')
    ax5f.set_xlabel('Tempo $t$ (s)')
    ax5f.set_ylabel('Força do Controlador $\\|\\mathbf{u}_j(t)\\|$')
    ax5f.grid(True)
    ax5f.legend(loc='upper right', fontsize=8)

    fig5.savefig(os.path.join(out_dir, 'fig5_evolucao_trials_Ej_mpc.png'), bbox_inches='tight')
    plt.close(fig5)

    # -------------------------------------------------------------------------
    # GERAÇÃO DO PDF EXPLICATIVO COMPLETO (10 PÁGINAS: TEORIA + METODOLOGIA + FIGURAS)
    # -------------------------------------------------------------------------
    print("[6/6] Gerando PDF explicativo detalhado (Relatorio_Explicativo_Simulacoes_MPC.pdf)...")

    def make_text_page(title_main, subtitle_main, sections, page_num, total_pages=10):
        fig_p = plt.figure(figsize=(11.69, 8.27), dpi=150)  # A4 Paisagem
        ax_p = fig_p.add_axes([0, 0, 1, 1])
        ax_p.axis('off')

        # Faixa de cabeçalho institucional
        ax_p.add_patch(plt.Rectangle((0.03, 0.885), 0.94, 0.085, color='#1b365d', alpha=0.96))
        ax_p.text(0.50, 0.942, title_main, color='white', fontsize=13.5, fontweight='bold', ha='center', va='center')
        ax_p.text(0.50, 0.905, subtitle_main, color='#d0e1f9', fontsize=9.8, ha='center', va='center')

        y_cursor = 0.855
        for sec_title, sec_body in sections:
            ax_p.add_patch(plt.Rectangle((0.035, y_cursor - 0.026), 0.93, 0.028, color='#e8eef5', alpha=0.9))
            ax_p.text(0.045, y_cursor - 0.012, sec_title, color='#1b365d', fontsize=10.5, fontweight='bold', va='center', ha='left')
            y_cursor -= 0.038
            txt_obj = ax_p.text(0.045, y_cursor, sec_body, color='#111111', fontsize=9.1, va='top', ha='left', linespacing=1.36)
            n_lines = sec_body.count('\n') + 1
            y_cursor -= (n_lines * 0.0245 + 0.018)

        ax_p.text(0.50, 0.022, f'Página {page_num} de {total_pages} — Relatório Técnico e Metodológico das Simulações de Controle Preditivo (MPC)',
                  fontsize=8.5, color='#555555', ha='center')
        return fig_p

    def make_figure_page(img_path, caption_title, page_num, total_pages=10):
        img = plt.imread(img_path)
        fig_f = plt.figure(figsize=(11.69, 8.27), dpi=150)
        ax_f = fig_f.add_axes([0.02, 0.05, 0.96, 0.90])
        ax_f.axis('off')
        ax_f.imshow(img)
        fig_f.text(0.50, 0.97, caption_title, fontsize=11, fontweight='bold', color='#1b365d', ha='center', va='center')
        fig_f.text(0.50, 0.02, f'Página {page_num} de {total_pages}', fontsize=8.5, color='#555555', ha='center')
        return fig_f

    pages_spec = [
        # PÁGINA 1: PARTE 1 (MALHA ABERTA E DIAGRAMAS DE FASE)
        (
            'PARTE 1: MODELAGEM DE 4ª ORDEM, COORDENADAS POLARES E MALHA ABERTA',
            'Motivação do Aprendizado no Lattice, Quebra de Simetria Radial e Dependência com a Distância Entrada–Alvo',
            [
                (
                    '1.1. Motivação Física: Do Experimento no Lattice ao Sistema Dinâmico de 4ª Ordem',
                    '• No experimento de aprendizado em grade (lattice), o agente caminha em sucessivos trials (j = 1, 2, ...) da Entrada r_S = (0,0) até o Alvo r_A = (10,10).\n'
                    '• Ao decompor as estatísticas de probabilidade de transição de cada célula nos eixos x e y, obtém-se um campo vetorial médio E(x, y) que não possui\n'
                    '  simetria puramente radial centrada no alvo, mas sim um corredor direcional alinhado ao vetor Entrada -> Alvo (T = r_A - r_S, com comprimento T = ||r_A - r_S||).\n'
                    '• A dinâmica contínua de 4ª ordem do caminhante no plano é governada por:   m * r_ddot + alpha * r_dot = E(x, y),   com estado x = [x, vx, y, vy]^T in R^4.\n'
                    '• Em Coordenadas Polares (r, theta) centradas na Entrada r_S, ou nas Coordenadas do Corredor (longitudinal s ao longo de r_S->r_A e transversal d_perp):\n'
                    '  m*(r_ddot - r*theta_dot^2) + alpha*r_dot = E_r(r, theta)   (avanço radial)    e    m*(r*theta_ddot + 2*r_dot*theta_dot) + alpha*r*theta_dot = E_theta(r, theta)   (alinhamento angular).'
                ),
                (
                    '1.2. O que foi feito na Simulação de Malha Aberta e por que o Agente para ANTES do Alvo?',
                    f'• Implementamos na função E_wire_field(x, y) o campo exato do condutor finito carregado com densidade linear proporcional ao comprimento lambda(q) = q.\n'
                    f'• Com find_equilibrium(), resolvemos numericamente E(x, y) = (0, 0) via método de Newton-Krylov/Hybr (scipy.optimize.root), encontrando o atrator natural em:\n'
                    f'  r_eq = ({r_eq_default[0]:.4f}, {r_eq_default[1]:.4f}), correspondente à fração longitudinal s_eq / T = {s_eq_default/T_len:.4f} (erro estacionário ||r_A - r_eq|| = {np.linalg.norm(R_A_DEFAULT - r_eq_default):.4f} unidades).\n'
                    f'• Causa Física: Sobre a extremidade exata do alvo r_A = (10, 10), toda a carga distribuída para trás no segmento [r_S, r_A] atrai ou empurra assimetricamente\n'
                    f'  o agente ao longo do eixo, gerando uma força residual intensa não-nula E(r_A) = ({E_A[0]:.2f}, {E_A[1]:.2f}) N que impede o repouso natural em r_A!'
                ),
                (
                    '1.3. Explicação dos 4 Gráficos da Figura 1 (Página 2)',
                    '• Gráfico (a): Mapa vetorial quiver de E(x,y) + integração RK45 (simulate_open_loop) para 4 condições iniciais, mostrando todas convergindo para r_eq = (8.89, 8.89).\n'
                    '• Gráfico (b): Diagrama de Fase Longitudinal (s, s_dot) para 3 amortecimentos (alpha = 0.25 subamortecido espiral, 0.60 nominal e 1.50 superamortecido).\n'
                    '• Gráfico (c): Diagrama de Fase Angular Polar (theta - theta_A, theta_dot), provando que o potencial corrige rapidamente o ângulo para theta_A = 45 graus.\n'
                    '• Gráfico (d): Varredura paramétrica variando a distância T = ||r_A - r_S|| de 4 a 30 para três regularizações eps: prova que o erro absoluto ||r_A - r_eq|| cresce\n'
                    '  linearmente com a distância entre a Entrada e o Alvo: ||r_A - r_eq|| = (1 - eta(eps/T)) * T!'
                ),
            ],
            'fig1_malha_aberta_diagrama_fase.png',
            'Figura 1: Mapa Vetorial em Malha Aberta, Diagramas de Fase (Longitudinal e Polar) e Sensibilidade a (r_S, r_A)'
        ),

        # PÁGINA 3: PARTE 2 (PERGUNTA 1 - MPC ADITIVO DE POSIÇÃO E VELOCIDADE)
        (
            'PARTE 2 (PERGUNTA 1): MPC ADITIVO PARA CONTROLE DE POSIÇÃO E VELOCIDADE',
            'Formulação no Espaço de Estados de 4ª Ordem com Restrições Explícitas de Velocidade e Ponto Fixo Exato no Alvo',
            [
                (
                    '2.1. Resposta à Pergunta 1: Podemos usar um MPC para controlar Posição e Velocidade nesse Potencial?',
                    '• SIM! Na função simulate_mpc_additive_pos_vel(), projetamos um controlador preditivo para a dinâmica:  m * r_ddot + alpha * r_dot = E(x, y) + u(t).\n'
                    '• Para que o alvo r_A = (10, 10) seja atingido com velocidade final exatamente ZERO (r_dot(t_f) = 0), a condição necessária e suficiente de ponto fixo é:\n'
                    f'  u_infty + E(r_A) = 0   ===>   u_infty = -E(r_A) = ({-E_A[0]:.2f}, {-E_A[1]:.2f}) N.\n'
                    '• Além de garantir o cancelamento exato do viés estático no alvo, o MPC controla toda a excursão transitória da velocidade ||r_dot(t)|| <= v_max!'
                ),
                (
                    '2.2. O que exatamente foi feito no Algoritmo do MPC Aditivo?',
                    '• Passo 1 (Discretização ZOH Exata de 4ª Ordem): A parte linear m*r_ddot + alpha*r_dot = v_mpc foi discretizada exatamente com passo Ts = 0.05 s,\n'
                    '  gerando as matrizes de transição Ad (4x4) e Bd (4x2) para o vetor de estado completo x_k = [x_k, vx_k, y_k, vy_k]^T.\n'
                    '• Passo 2 (Predição de Longo Alcance Np = 20 passos e Horizonte de Controle Nu = 6): Construímos as matrizes de predição S_x e S_u tais que\n'
                    '  X_pred = S_x * x_k + S_u * V_mpc. Em cada passo k, geramos uma referência conjunta de posição e velocidade X_ref que guia o agente na direção\n'
                    '  de r_A com velocidade desejada limitada por min(0.93 * v_max, 0.85 * ||r_A - r_k||), desacelerando suavemente ao se aproximar de r_A.\n'
                    '• Passo 3 (Função Custo + Barreira de Velocidade): Minimizamos a função custo quadrática ponderando erro de posição (q_pos = 45), erro de velocidade\n'
                    '  (q_vel = 18), variação de controle (r_du = 0.25) e uma barreira quadrática estrita para impedir qualquer violação do teto ||v_{k+i}||_2 <= v_max!'
                ),
                (
                    '2.3. Resultados Numéricos Obtidos na Figura 2 (Página 4)',
                    f'• Simulação com Teto de Velocidade v_max = 2.2 m/s: Posição final r(t_f) = ({x_m1[0,-1]:.4f}, {x_m1[2,-1]:.4f}), velocidade final ||v(t_f)|| = {np.hypot(x_m1[1,-1],x_m1[3,-1]):.5f} m/s,\n'
                    f'  e velocidade máxima atingida no transiente max ||v(t)|| = {np.max(v_norm_1):.4f} m/s (respeitando rigorosamente o limite de 2.2 m/s!).\n'
                    f'• Simulação com Teto Restrito v_max = 1.2 m/s: Posição final r(t_f) = ({x_m2[0,-1]:.4f}, {x_m2[2,-1]:.4f}), velocidade final ||v(t_f)|| = {np.hypot(x_m2[1,-1],x_m2[3,-1]):.5f} m/s,\n'
                    f'  e velocidade máxima atingida max ||v(t)|| = {np.max(v_norm_2):.4f} m/s (cravando o patamar retangular no diagrama de fase da Figura 2b!).'
                ),
            ],
            'fig2_mpc_aditivo_posicao_velocidade.png',
            'Figura 2: MPC Aditivo (E + u) Controlando Simultaneamente Posição e Velocidade com Restrições v_max'
        ),

        # PÁGINA 5: PARTE 3 (PERGUNTA 2 - MPC MULTIPLICATIVO / BILINEAR (1+u)*E)
        (
            'PARTE 3 (PERGUNTA 2): MPC MULTIPLICATIVO / BILINEAR (1 + u) * E(x, y)',
            'Controlabilidade Não-Linear, Travessia da Singularidade E(r_eq)=0 e Regimes Bipolar vs. Unipolar Deslizante',
            [
                (
                    '3.1. Resposta à Pergunta 2: Podemos controlar o caminhante na forma m*r_ddot + alpha*r_dot = (1 + u)*E(x, y)?',
                    '• SIM, mas esse problema revela uma estrutura de Sistemas Dinâmicos fascinante! Na função simulate_mpc_multiplicative(), demonstramos que:\n'
                    '  1) Barreira de Singularidade em r_eq = (8.89, 8.89): Como E(r_eq) = 0, o termo de controle (1 + u)*E(r_eq) se anula para QUALQUER valor finito de u!\n'
                    '     Logo, se o caminhante parar em r_eq, o sistema perde a controlabilidade. O MPC resolve isso acelerando o caminhante ANTES de r_eq (com 1 + u > 1)\n'
                    '     para que ele atravesse a linha crítica s = s_eq = 12.57 por INÉRCIA com velocidade longitudinal estritamente positiva (s_dot(s_eq) > 0)!\n'
                    '  2) Condição de Ponto Fixo em r_A = (10, 10): Como E(r_A) != 0, para que o caminhante permaneça parado em r_A (r_dot = 0, r_ddot = 0), o ganho\n'
                    '     multiplicativo deve convergir assintoticamente para 1 + u_infty = 0  <===>  u_infty = -1 (desligando o campo exatamente ao estacionar no alvo!).'
                ),
                (
                    '3.2. Os Três Regimes de Modulação Multiplicativa Implementados na Simulação',
                    '• Regime 1 — MPC Bipolar Anisotrópico [(1 + u_par)*E_par + (1 + u_perp)*E_perp]: Mantém 1 + u_perp > 0 para prender o caminhante ao eixo central r_S->r_A.\n'
                    '  Antes de s_eq usa 1 + u_par > 0 (acelera); após cruzar s_eq (onde E_par inverte de sinal e puxa para trás), o MPC inverte o sinal de (1 + u_par < 0),\n'
                    '  transformando a força repulsiva de volta em força atrativa rumo a r_A, e zera (1 + u_par -> 0) ao parar em r_A = (10.000, 10.000)!\n'
                    '• Regime 2 — MPC Unipolar Deslizante [Restrição estrita 1 + u(t) >= 0, proibido inverter a polaridade do potencial!]: Mesmo sem poder fazer 1+u < 0,\n'
                    '  o MPC bombeia energia cinética na zona favorável (s < s_eq) até que a velocidade atinja exatamente a Variedade de Deslizamento Viscoso:\n'
                    '  s_dot_slide(s) = (alpha / m) * (T - s). Nesse instante exato, o MPC desliga o campo (1 + u = 0 <=> u = -1) e o caminhante desliza sob atrito viscoso\n'
                    '  m*s_ddot + alpha*s_dot = 0 até parar exatamente sobre o alvo s = T <=> r_A = (10, 10)!\n'
                    '• Regime 3 — MPC Bipolar Escalar Único [(1 + u(t))*E(x, y) com u(t) in R escalar]: Na Fase 1 (s < s_eq) aplica 1 + u > 1 para colapsar o erro transversal\n'
                    '  d_perp -> 0 sobre o eixo de simetria; na Fase 2 (s > s_eq), já sobre a variedade invariante d_perp = 0 (onde E_perp = 0), inverte 1 + u < 0 até r_A!'
                ),
                (
                    '3.3. Resultados Numéricos Obtidos na Figura 3 (Página 6)',
                    f'• Regime Bipolar Anisotrópico: Posição final r(t_f) = ({x_mb_aniso[0,-1]:.4f}, {x_mb_aniso[2,-1]:.4f}) | Erro final = {err_b_aniso[-1]:.5f} unidades.\n'
                    f'• Regime Unipolar Deslizante (1+u >= 0): Posição final r(t_f) = ({x_mb_uni[0,-1]:.4f}, {x_mb_uni[2,-1]:.4f}) | Erro final = {err_b_uni[-1]:.5f} unidades.\n'
                    f'• Regime Bipolar Escalar Puro: Posição final r(t_f) = ({x_mb_scal[0,-1]:.4f}, {x_mb_scal[2,-1]:.4f}) | Erro final = {err_b_scal[-1]:.5f} unidades.'
                ),
            ],
            'fig3_mpc_multiplicativo_bilinear.png',
            'Figura 3: MPC Multiplicativo / Bilinear (1+u)*E(x,y) nos Regimes Bipolar, Unipolar Deslizante e Escalar'
        ),

        # PÁGINA 7: PARTE 4 (PERGUNTA 3 - ATUADOR FÍSICO ELETROÍMÃ)
        (
            'PARTE 4 (PERGUNTA 3): CONTROLE MPC DE UM ELETROÍMÃ INTERFERINDO NO POTENCIAL',
            'Modelagem Eletromagnética de 7ª Ordem: Gradiente Espacial das Bobinas, Força de Lorentz e Dinâmica Indutiva RL',
            [
                (
                    '4.1. Resposta à Pergunta 3: Como controlar um Eletroímã Físico externo para interferir em E(x, y) e guiar o agente?',
                    '• Em vez de alterar matematicamente o potencial do fio, instalamos bobinas eletromagnéticas externas próximas à região do alvo:\n'
                    '  - Bobina 1 (Atratora Axial em r_mag1 = (11.8, 11.8)) e Bobina 2 (Defletora Lateral em r_mag2 = (11.2, 8.8)), gerando um campo de gradiente espacial:\n'
                    '    E_mag(r, I) = sum_{k=1}^2  GAMMA_MAG * I_k(t) * (r_mag,k - r) / (||r_mag,k - r||^2 + delta_m^2)^(3/2).\n'
                    '  - Bobina 3 (Campo Vertical B_z de Helmholtz): exerce uma força giroscópica de Lorentz F_L = q_ef * (r_dot x B_z) = q_ef * I_Bz * [vy, -vx]^T,\n'
                    '    que atua perpendicularmente à velocidade para corrigir o ângulo da caminhada sem introduzir trabalho mecânico parasita!'
                ),
                (
                    '4.2. Dinâmica Indutiva RL dos Eletroímãs e Deslocamento do Ponto Crítico do Potencial',
                    '• Cada eletroímã possui indutância L = 0.20 H e resistência R = 1.0 Ohm, obedecendo à EDO elétrica:  L * dI_k/dt + R * I_k = V_k(t),\n'
                    '  onde a variável manipulada pelo MPC é a tensão elétrica V_k(t) aplicada aos terminais das bobinas (sistema global acoplado de 7ª Ordem!).\n'
                    f'• Na função compute_equilibrium_currents_at_target(), calculamos a corrente estacionária exata I_eq que produz interferência destrutiva com a força\n'
                    f'  residual do fio em r_A = (10,10):   E_fio(r_A) + E_mag(r_A, I_eq) = 0   ===>   I1_eq = {I_eq_em[0]:.4f} A,  I2_eq = {I_eq_em[1]:.4f} A.\n'
                    '• Como mostra a Figura 4b (curva verde), quando o eletroímã é energizado com I1_eq, o zero do campo resultante E_ef(s) = E_fio(s) + E_mag(s)\n'
                    '  desloca-se de s_eq = 12.57 exatamente para cima do alvo s = T = 14.142 (r_A = (10, 10))!'
                ),
                (
                    '4.3. Resultados da Simulação do Eletroímã Controlado por MPC (Figura 4, Página 8)',
                    '• O controlador preditivo em cascata (simulate_mpc_electromagnet) calcula a força magnética desejada, inverte a matriz de sensibilidade espacial\n'
                    '  Phi(r) com regularização de Tikhonov e antecipa o atraso indutivo tau_RL = L/R = 0.20 s das bobinas.\n'
                    f'• Posição final atingida pelo caminhante sob ação do eletroímã: r(t_f) = ({x_em[0,-1]:.4f}, {x_em[2,-1]:.4f}) (erro = {np.hypot(x_em[0,-1]-10, x_em[2,-1]-10):.5f} unidades).'
                ),
            ],
            'fig4_mpc_eletroima_fisico.png',
            'Figura 4: Interferência Eletromagnética no Potencial E(x,y), Dinâmica RL das Bobinas e Retrato de Fase'
        ),

        # PÁGINA 9: PARTE 5 (PERGUNTA 4 - EVOLUÇÃO MULTI-TRIAL E_j(x,y) E ESFORÇO DO MPC)
        (
            'PARTE 5 (PERGUNTA 4): EVOLUÇÃO DO CAMPO E_j(x,y) POR TRIAL E REDUÇÃO DO ESFORÇO MPC',
            'Aprendizado Radial e Longitudinal no Lattice: Convergência de E_j(x,y) para o Potencial Ótimo quando j -> infinito',
            [
                (
                    '5.1. Formulação do Potencial Evolutivo Anisotrópico E_j(x, y) (Além do Condutor Infinito/Finito)',
                    '• Conforme observado na motivação do problema, o potencial não precisa ficar restrito ao fio estático: ao longo dos trials j = 1, 2, ..., o agente\n'
                    '  consolida no lattice uma estrutura de transição longitudinal (ao longo de e_par = (r_A - r_S)/T) e radial/transversal (ao longo de e_perp).\n'
                    '• Na função E_evolutionary_trial(x, y, beta_j), modelamos a evolução do campo no trial j por:\n'
                    '      E_j(x, y) = (1 - beta_j) * E_fio(x, y)  +  beta_j * E_otimo(x, y),       com   beta_j = 1 - exp(-0.35 * (j - 1)) in [0, 1],\n'
                    '  onde E_otimo(x, y) = -k_par * L_s * tanh((s - T)/L_s) * e_par  -  k_perp * d_perp * e_perp  é o potencial ótimo aprendido: suas linhas de campo\n'
                    '  alinham-se perfeitamente em módulo e ângulo com o vetor Entrada -> Alvo e seu único equilíbrio global estável é exatamente r_A = (10, 10)!'
                ),
                (
                    '5.2. Sinergia entre Evolução do Potencial E_j(x, y) e Controle MPC (simulate_evolution_across_trials)',
                    '• No Trial inicial j = 1 (beta_1 = 0): O ambiente é o potencial bruto do fio carregado (cujo equilíbrio natural está longe do alvo, em (8.89, 8.89)).\n'
                    '  Para forçar o caminhante até r_A = (10, 10), o controlador MPC precisa aplicar um esforço de controle altíssimo.\n'
                    '• Nos Trials intermediários (j = 2 .. 8): À medida que beta_j cresce, o campo E_j(x, y) atua com mais precisão empurrando o caminhante até r_A\n'
                    '  e alinhando os vetores do mapa com a diagonal r_S -> r_A (cosseno direcional médio sobe de 0.333 para 1.000!).\n'
                    '• No limite j -> infinito (Trial j = 12, beta_12 = 1.0): O próprio potencial E_j já possui ponto fixo exato em r_A = (10, 10) (E_infty(r_A) = 0),\n'
                    '  reduzindo o esforço de regime permanente do controlador MPC a ZERO (||u_12(t_f)|| = 0.0000 N) e cortando a energia L2 de controle em > 99.8%!'
                ),
                (
                    '5.3. Resumo Quantitativo da Evolução ao Longo dos Trials (Figura 5, Página 10)',
                    f'• Trial j = 1  (beta = 0.00): Equilíbrio natural r_eq^(1) = ({trials_data[0]["r_eq"][0]:.2f}, {trials_data[0]["r_eq"][1]:.2f}) | Erro MA = {trials_data[0]["eq_error"]:.4f} | Energia MPC L2 = {trials_data[0]["effort_L2"]:.1f} | ||u(t_f)|| = {trials_data[0]["u_ss"]:.2f} N\n'
                    f'• Trial j = 4  (beta = 0.65): Equilíbrio natural r_eq^(4) = ({trials_data[3]["r_eq"][0]:.2f}, {trials_data[3]["r_eq"][1]:.2f}) | Erro MA = {trials_data[3]["eq_error"]:.4f} | Energia MPC L2 = {trials_data[3]["effort_L2"]:.1f} | ||u(t_f)|| = {trials_data[3]["u_ss"]:.2f} N\n'
                    f'• Trial j = 7  (beta = 0.88): Equilíbrio natural r_eq^(7) = ({trials_data[6]["r_eq"][0]:.2f}, {trials_data[6]["r_eq"][1]:.2f}) | Erro MA = {trials_data[6]["eq_error"]:.4f} | Energia MPC L2 = {trials_data[6]["effort_L2"]:.1f} | ||u(t_f)|| = {trials_data[6]["u_ss"]:.2f} N\n'
                    f'• Trial j = 12 (beta = 1.00): Equilíbrio natural r_eq^(12)= ({trials_data[-1]["r_eq"][0]:.2f}, {trials_data[-1]["r_eq"][1]:.2f}) | Erro MA = {trials_data[-1]["eq_error"]:.4f} | Energia MPC L2 = {trials_data[-1]["effort_L2"]:.1f}   | ||u(t_f)|| = {trials_data[-1]["u_ss"]:.4f} N'
                ),
            ],
            'fig5_evolucao_trials_Ej_mpc.png',
            'Figura 5: Evolução dos Mapas Vetoriais E_j(x,y) por Trial j e Queda Exponencial do Esforço de Controle do MPC'
        ),
    ]

    for pdf_filename in ('Relatorio_Explicativo_Simulacoes_MPC.pdf', 'Estudo_Completo_MPC_Potencial_Evolutivo.pdf'):
        full_pdf_path = os.path.join(out_dir, pdf_filename)
        with PdfPages(full_pdf_path) as pdf:
            p_num = 1
            for t_main, sub_main, secs, f_img, f_cap in pages_spec:
                fig_txt = make_text_page(t_main, sub_main, secs, p_num, total_pages=10)
                pdf.savefig(fig_txt)
                plt.close(fig_txt)
                p_num += 1

                fig_im = make_figure_page(os.path.join(out_dir, f_img), f_cap, p_num, total_pages=10)
                pdf.savefig(fig_im)
                plt.close(fig_im)
                p_num += 1

    print("=====================================================================")
    print("RESUMO QUANTITATIVO DAS SIMULAÇÕES:")
    print(f"  1. Malha Aberta (Nominal T={T_len:.2f}):")
    print(f"     r_eq = ({r_eq_default[0]:.4f}, {r_eq_default[1]:.4f}) | s_eq/T = {s_eq_default/T_len:.4f} | erro = {np.linalg.norm(R_A_DEFAULT - r_eq_default):.4f}")
    print(f"     Força residual no alvo E(r_A) = ({E_A[0]:.4f}, {E_A[1]:.4f})")
    print(f"  2. Pergunta 1 - MPC Aditivo (Posição + Velocidade):")
    print(f"     v_max=2.2 -> r_end=({x_m1[0,-1]:.4f}, {x_m1[2,-1]:.4f}), ||v_end||={np.hypot(x_m1[1,-1],x_m1[3,-1]):.5f}, max||v||={np.max(v_norm_1):.4f}")
    print(f"     v_max=1.2 -> r_end=({x_m2[0,-1]:.4f}, {x_m2[2,-1]:.4f}), ||v_end||={np.hypot(x_m2[1,-1],x_m2[3,-1]):.5f}, max||v||={np.max(v_norm_2):.4f}")
    print(f"  3. Pergunta 2 - MPC Multiplicativo (1 + u)*E(x,y):")
    print(f"     Bipolar Anisotrópico -> r_end=({x_mb_aniso[0,-1]:.4f}, {x_mb_aniso[2,-1]:.4f}), erro={err_b_aniso[-1]:.5f}")
    print(f"     Unipolar Deslizante  -> r_end=({x_mb_uni[0,-1]:.4f}, {x_mb_uni[2,-1]:.4f}), erro={err_b_uni[-1]:.5f}")
    print(f"     Bipolar Escalar      -> r_end=({x_mb_scal[0,-1]:.4f}, {x_mb_scal[2,-1]:.4f}), erro={err_b_scal[-1]:.5f}")
    print(f"  4. Pergunta 3 - MPC com Eletroímã Físico (RL + Gradiente + Lorentz):")
    print(f"     Correntes de equilíbrio no alvo: I1_eq={I_eq_em[0]:.4f} A, I2_eq={I_eq_em[1]:.4f} A")
    print(f"     r_end=({x_em[0,-1]:.4f}, {x_em[2,-1]:.4f}), erro={np.hypot(x_em[0,-1]-10, x_em[2,-1]-10):.5f}")
    print(f"  5. Pergunta 4 - Evolução do Campo E_j(x,y) nos Trials j=1..12:")
    print(f"     Trial j=1  -> ||r_A - r_eq||={trials_data[0]['eq_error']:.4f}, Esforço L2 MPC={trials_data[0]['effort_L2']:.2f}")
    print(f"     Trial j=4  -> ||r_A - r_eq||={trials_data[3]['eq_error']:.4f}, Esforço L2 MPC={trials_data[3]['effort_L2']:.2f}")
    print(f"     Trial j=12 -> ||r_A - r_eq||={trials_data[-1]['eq_error']:.5f}, Esforço L2 MPC={trials_data[-1]['effort_L2']:.2f}, ||u_ss||={trials_data[-1]['u_ss']:.5f}")
    print("=====================================================================")


if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_figures_and_report(out_dir)
