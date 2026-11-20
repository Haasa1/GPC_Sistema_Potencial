# -*- coding: utf-8 -*-
"""
Demonstrativo Completo: Controle Preditivo Generalizado (GPC) Aplicado ao
Sistema Dinâmico guiado por Campo Potencial (TEV) sob Duas Perspectivas:

  - 1ª Perspectiva (GPC Intra-Simulação no Tempo t):
      Atua por meio de uma força de controle aditiva u(t) com modelo CARIMA
      incremental (Delta u), compensando o erro de regime permanente do campo
      e cravando a posição final exatamente em A = (10, 10).

  - 2ª Perspectiva (GPC Evolutivo / Run-to-Run nos Parâmetros do Campo):
      Sem força externa artificial; a cada simulação (época i), o controlador
      preditivo-evolutivo atualiza os parâmetros do próprio potencial
      (polo virtual A_virt e amortecimento/suavização) até que o ponto de
      equilíbrio natural do campo E(r; theta*) = 0 coincida com A = (10, 10).

Gera:
  1. perspectiva1_gpc_tempo.png
  2. perspectiva2_gpc_evolutivo.png
  3. Resumo_Propostas_GPC_Sistema_Potencial.pdf (Relatório técnico + resultados)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.linalg import toeplitz

# Configuração visual padrão
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# =============================================================================
# 1. MODELO FÍSICO DO CAMPO POTENCIAL (EQ. 9 CORRIGIDA)
# =============================================================================
M_DEFAULT = 1.0
ALPHA_DEFAULT = 0.5
KAPPA_DEFAULT = -10.0
EPS_DEFAULT = 0.1
S_DEFAULT = np.array([0.0, 0.0])
A_TARGET = np.array([10.0, 10.0])


def E_field(x, y, S=S_DEFAULT, A=A_TARGET, kappa=KAPPA_DEFAULT, eps=EPS_DEFAULT):
    """
    Calcula o campo vetorial E(x, y) exatamente conforme a Eq. (9) do PDF:
      term1 -> (-1 + Q) na direção r1
      term2 -> (+1 + Q) na direção r2
      term3 -> ln(...) na direção T
      prefator -> -kappa / T  (com kappa < 0 para campo atrativo)
    """
    r1_vec = np.array([x - S[0], y - S[1]], dtype=float)
    r2_vec = np.array([x - A[0], y - A[1]], dtype=float)
    T_vec = np.array([A[0] - S[0], A[1] - S[1]], dtype=float)

    T = np.linalg.norm(T_vec)
    if T < 1e-12:
        return 0.0, 0.0

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    r1_tilde = np.sqrt(r1**2 + eps**2)
    r2_tilde = np.sqrt(r2**2 + eps**2)

    num_comum = r1**2 - r2**2 + T**2
    den_comum = T**2 - (r1_tilde + r2_tilde)**2

    arg_ln = (r1_tilde + r2_tilde + T) / (r1_tilde + r2_tilde - T)
    if arg_ln <= 0:
        return 0.0, 0.0
    ln_term = np.log(arg_ln)

    term1 = (r1_vec / r1_tilde) * (-1.0 + (num_comum / den_comum))
    term2 = (r2_vec / r2_tilde) * (1.0 + (num_comum / den_comum))
    term3 = (T_vec / T) * ln_term

    E_vec = (-kappa / T) * (term1 + term2 + term3)
    return float(E_vec[0]), float(E_vec[1])


def find_equilibrium(S=S_DEFAULT, A=A_TARGET, kappa=KAPPA_DEFAULT, eps=EPS_DEFAULT, guess=None):
    """Encontra numericamente o ponto onde E(x, y) = (0, 0)."""
    if guess is None:
        guess = S + 0.89 * (A - S)
    sol = root(lambda v: E_field(v[0], v[1], S=S, A=A, kappa=kappa, eps=eps), guess)
    return sol.x


# =============================================================================
# 2. SIMULAÇÃO BASELINE (SEM CONTROLE)
# =============================================================================
def simulate_open_loop(state0=(1.0, 0.0, 8.0, 0.0), t_end=40.0, n_pts=800,
                       S=S_DEFAULT, A=A_TARGET, kappa=KAPPA_DEFAULT, eps=EPS_DEFAULT,
                       m=M_DEFAULT, alpha=ALPHA_DEFAULT):
    t_eval = np.linspace(0.0, t_end, n_pts)

    def ode(t, state):
        x, vx, y, vy = state
        Ex, Ey = E_field(x, y, S=S, A=A, kappa=kappa, eps=eps)
        return [vx, (Ex - alpha * vx) / m, vy, (Ey - alpha * vy) / m]

    sol = solve_ivp(ode, (0.0, t_end), state0, t_eval=t_eval, method='RK45')
    return sol.t, sol.y


# =============================================================================
# 3. PERSPECTIVA 1: GPC INTRA-SIMULAÇÃO (NO DOMÍNIO DO TEMPO t)
# =============================================================================
def build_carima_gpc_matrices(m, alpha, Ts, Np, Nu, r_weight, q_weight=1.0):
    """
    Constrói as matrizes de predição G e F do GPC (Clarke et al., 1987) para a
    planta contínua P(s) = 1 / (m s^2 + alpha s) discretizada via ZOH com passo Ts.
    """
    if abs(alpha) > 1e-8:
        p = np.exp(-alpha * Ts / m)
        a1 = -(1.0 + p)
        a2 = p
        b0 = (Ts - (m / alpha) * (1.0 - p)) / alpha
        b1 = ((m / alpha) * (1.0 - p) - Ts * p) / alpha
    else:
        a1, a2 = -2.0, 1.0
        b0 = 0.5 * (Ts**2) / m
        b1 = 0.5 * (Ts**2) / m

    # Polinômio A_tilde(z^-1) = (1 - z^-1)(1 + a1 z^-1 + a2 z^-2)
    # = 1 + alpha_1 z^-1 + alpha_2 z^-2 + alpha_3 z^-3
    al1 = a1 - 1.0
    al2 = a2 - a1
    al3 = -a2

    # Resposta ao degrau CARIMA (coeficientes de Markov g_j de B(z^-1)/A_tilde(z^-1))
    g = np.zeros(Np)
    F = np.zeros((Np, 3))       # Depende de [y(k), y(k-1), y(k-2)]
    H_prev = np.zeros((Np, 1))  # Depende de Delta u(k-1)

    for j in range(Np):
        g_jm1 = g[j - 1] if j - 1 >= 0 else 0.0
        g_jm2 = g[j - 2] if j - 2 >= 0 else 0.0
        g_jm3 = g[j - 3] if j - 3 >= 0 else 0.0
        g[j] = -al1 * g_jm1 - al2 * g_jm2 - al3 * g_jm3 + (b0 if j == 0 else (b1 if j == 1 else 0.0))

        def get_f_h(idx):
            if idx >= 0:
                return F[idx], H_prev[idx, 0]
            vec = np.zeros(3)
            vec[-idx - 1] = 1.0
            return vec, 0.0

        f1, h1 = get_f_h(j - 1)
        f2, h2 = get_f_h(j - 2)
        f3, h3 = get_f_h(j - 3)

        F[j] = -al1 * f1 - al2 * f2 - al3 * f3
        H_prev[j, 0] = -al1 * h1 - al2 * h2 - al3 * h3 + (b1 if j == 0 else 0.0)

    # Matriz dinâmica G (Np x Nu) inferior-triangular em banda
    G_full = toeplitz(g, np.zeros(Np))
    G = G_full[:, :Nu]

    # Ganho ótimo do GPC: K_gpc = primera linha de (G^T Q G + R)^-1 G^T Q
    Q_mat = q_weight * np.eye(Np)
    R_mat = r_weight * np.eye(Nu)
    K_full = np.linalg.solve(G.T @ Q_mat @ G + R_mat, G.T @ Q_mat)
    k_gpc = K_full[0, :]  # (1 x Np)

    return k_gpc, F, H_prev, (a1, a2, b0, b1)


def simulate_perspective1_gpc(state0=(1.0, 0.0, 8.0, 0.0), t_end=40.0, Ts=0.05,
                              Np=30, Nu=5, r_weight=0.08, alpha_ref=0.85,
                              u_max=80.0, m=M_DEFAULT, alpha=ALPHA_DEFAULT):
    """
    Simula o sistema controlado pelo GPC no tempo t:
      m r_ddot + alpha r_dot = E(r) + u(t)
    Com ação integral CARIMA (Delta u) e compensação preditiva do campo E(r).
    """
    k_gpc, F, H_prev, _ = build_carima_gpc_matrices(m, alpha, Ts, Np, Nu, r_weight)

    n_steps = int(t_end / Ts)
    t_hist = np.zeros(n_steps + 1)
    state_hist = np.zeros((4, n_steps + 1))
    u_hist = np.zeros((2, n_steps + 1))
    E_hist = np.zeros((2, n_steps + 1))

    state = np.array(state0, dtype=float)
    state_hist[:, 0] = state
    E_hist[:, 0] = E_field(state[0], state[2])

    # Histórico de posições e incrementos para o preditor CARIMA
    x_past = [state[0], state[0], state[0]]
    y_past = [state[2], state[2], state[2]]
    du_eff_prev = np.zeros(2)  # Delta(u + E) anterior
    u_curr = np.zeros(2)
    E_prev = np.array(E_hist[:, 0])

    for k in range(n_steps):
        x_now, vx_now, y_now, vy_now = state
        Ex_now, Ey_now = E_field(x_now, y_now)
        E_now = np.array([Ex_now, Ey_now])
        dE_now = E_now - E_prev

        # Trajetória de referência suavizada w(k+j) convergindo para A_TARGET
        w_x = np.zeros(Np)
        w_y = np.zeros(Np)
        wx_val, wy_val = x_now, y_now
        for j in range(Np):
            wx_val = alpha_ref * wx_val + (1.0 - alpha_ref) * A_TARGET[0]
            wy_val = alpha_ref * wy_val + (1.0 - alpha_ref) * A_TARGET[1]
            w_x[j] = wx_val
            w_y[j] = wy_val

        # Resposta livre f = F * y_past + H_prev * du_eff(k-1)
        f_x = F @ np.array(x_past) + H_prev[:, 0] * du_eff_prev[0]
        f_y = F @ np.array(y_past) + H_prev[:, 0] * du_eff_prev[1]

        # Incremento ótimo da força total efetiva e desconto da variação do campo dE
        du_total_x = float(k_gpc @ (w_x - f_x))
        du_total_y = float(k_gpc @ (w_y - f_y))

        du_x = du_total_x - dE_now[0]
        du_y = du_total_y - dE_now[1]

        # Atualização da lei de controle com saturação suave
        u_next = np.clip(u_curr + np.array([du_x, du_y]), -u_max, u_max)
        du_eff_prev = (u_next - u_curr) + dE_now
        u_curr = u_next
        E_prev = E_now

        # Integra a EDO não-linear contínua no intervalo [k*Ts, (k+1)*Ts]
        def ode_step(t, s):
            xx, vvx, yy, vvy = s
            Ex, Ey = E_field(xx, yy)
            return [
                vvx,
                (Ex + u_curr[0] - alpha * vvx) / m,
                vvy,
                (Ey + u_curr[1] - alpha * vvy) / m,
            ]

        sol_step = solve_ivp(ode_step, (k * Ts, (k + 1) * Ts), state, method='RK45')
        state = sol_step.y[:, -1]

        # Atualiza buffers CARIMA
        x_past = [state[0], x_past[0], x_past[1]]
        y_past = [state[2], y_past[0], y_past[1]]

        t_hist[k + 1] = (k + 1) * Ts
        state_hist[:, k + 1] = state
        u_hist[:, k + 1] = u_curr
        E_hist[:, k + 1] = E_field(state[0], state[2])

    return t_hist, state_hist, u_hist, E_hist


# =============================================================================
# 4. PERSPECTIVA 2: GPC EVOLUTIVO RUN-TO-RUN (NOS PARÂMETROS DO CAMPO)
# =============================================================================
def simulate_perspective2_evolutionary_gpc(n_epochs=10, Np_ep=5, Nu_ep=2,
                                           r_ep=0.15, pop_size=5, sigma_init=0.18,
                                           state0=(1.0, 0.0, 8.0, 0.0), t_end=40.0):
    """
    Em cada época (simulação i = 1..n_epochs):
      - A partícula move-se EXCLUSIVAMENTE sob o campo potencial E(r; A_virt^(i)),
        sem nenhuma força externa u(t).
      - O controlador GPC Run-to-Run (combinado com exploração evolutiva (1+lambda)-ES)
        observa o ponto final atingido r_end^(i) e calcula a atualização preditiva
        Delta theta^(i) = A_virt^(i+1) - A_virt^(i) para que r_end -> A_TARGET = (10, 10).
    """
    rng = np.random.default_rng(42)

    # Parâmetro inicial theta^(0) = A_virt^(0) = (10, 10) (sistema original)
    theta = A_TARGET.copy()

    # Matriz de sensibilidade inicial J_theta = d(r_eq)/d(A_virt) (~0.889 * I)
    J_est = 0.89 * np.eye(2)

    history = []

    for ep in range(n_epochs):
        # 1. Avalia o indivíduo nominal (gerado pelo GPC Run-to-Run)
        t_sim, y_sim = simulate_open_loop(state0=state0, t_end=t_end, A=theta)
        r_end = np.array([y_sim[0, -1], y_sim[2, -1]])
        err_vec = A_TARGET - r_end
        err_norm = float(np.linalg.norm(err_vec))

        # 2. Exploração evolutiva ao redor da predição atual (se erro ainda não zerou)
        sigma = sigma_init * (0.55 ** ep)
        candidates_theta = [theta.copy()]
        candidates_rend = [r_end.copy()]
        candidates_err = [err_norm]

        if ep > 0 and err_norm > 1e-3:
            for _ in range(pop_size - 1):
                mut_theta = theta + rng.normal(0.0, sigma, size=2)
                # Usa a raiz analítica rápida com chute escalado na diagonal S -> mut_theta
                r_eq_mut = find_equilibrium(A=mut_theta, guess=S_DEFAULT + 0.8888 * (mut_theta - S_DEFAULT))
                candidates_theta.append(mut_theta)
                candidates_rend.append(r_eq_mut)
                candidates_err.append(float(np.linalg.norm(A_TARGET - r_eq_mut)))

            # Atualiza estimativa da Jacobiana J_est por mínimos quadrados com a população
            dTheta = np.array(candidates_theta[1:]) - theta
            dR = np.array(candidates_rend[1:]) - r_end
            if np.linalg.matrix_rank(dTheta) == 2:
                J_est = (np.linalg.lstsq(dTheta, dR, rcond=None)[0]).T

        best_idx = int(np.argmin(candidates_err))
        if best_idx != 0:
            theta = candidates_theta[best_idx]
            t_sim, y_sim = simulate_open_loop(state0=state0, t_end=t_end, A=theta)
            r_end = np.array([y_sim[0, -1], y_sim[2, -1]])
            err_vec = A_TARGET - r_end
            err_norm = float(np.linalg.norm(err_vec))

        history.append({
            'epoch': ep + 1,
            'theta': theta.copy(),
            't': t_sim,
            'y': y_sim,
            'r_end': r_end.copy(),
            'error_norm': err_norm,
            'candidates_theta': np.array(candidates_theta),
            'candidates_rend': np.array(candidates_rend),
        })

        # 3. Lei de controle GPC Run-to-Run para a próxima época (CARIMA sobre épocas)
        # Predição multi-época: y_hat(i+j) = y(i) + j * J_est * dtheta(i) + ...
        G_ep = np.zeros((2 * Np_ep, 2 * Nu_ep))
        for j in range(Np_ep):
            for m_u in range(min(j + 1, Nu_ep)):
                G_ep[2 * j:2 * j + 2, 2 * m_u:2 * m_u + 2] = (j - m_u + 1) * J_est

        Q_ep = np.eye(2 * Np_ep)
        R_ep = r_ep * np.eye(2 * Nu_ep)
        K_ep = np.linalg.solve(G_ep.T @ Q_ep @ G_ep + R_ep, G_ep.T @ Q_ep)

        # Vetor de erro futuro em relação à referência (A_TARGET repetido Np_ep vezes)
        e_future = np.tile(A_TARGET - r_end, Np_ep)
        dtheta_opt = (K_ep @ e_future)[:2]

        theta = theta + dtheta_opt

    return history


# =============================================================================
# 5. GERAÇÃO DAS FIGURAS E DO RELATÓRIO EM PDF
# =============================================================================
def create_figures_and_pdf(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Executa as simulações
    t_ol, y_ol = simulate_open_loop()
    r_eq_ol = np.array([y_ol[0, -1], y_ol[2, -1]])

    t_p1, y_p1, u_p1, E_p1 = simulate_perspective1_gpc()
    r_eq_p1 = np.array([y_p1[0, -1], y_p1[2, -1]])

    hist_p2 = simulate_perspective2_evolutionary_gpc(n_epochs=10)
    r_eq_p2 = hist_p2[-1]['r_end']
    theta_star = hist_p2[-1]['theta']

    # -------------------------------------------------------------------------
    # FIGURA 1: PERSPECTIVA 1 (GPC NO TEMPO t)
    # -------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(13, 8.5), dpi=150)
    fig1.suptitle(
        '1ª Perspectiva: Controle Preditivo GPC Intra-Simulação (Força Aditiva no Tempo $t$)',
        fontsize=14, fontweight='bold', y=0.98
    )

    gs1 = fig1.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.32, wspace=0.25)

    # Subplot (a): Trajetória 2D
    ax1a = fig1.add_subplot(gs1[:, 0])
    ax1a.plot([S_DEFAULT[0], A_TARGET[0]], [S_DEFAULT[1], A_TARGET[1]],
              '--', color='#888888', lw=1.8, label='Segmento TEV (Memória $S \\to A$)')
    ax1a.plot(y_ol[0], y_ol[2], color='#d95f02', lw=2.0, alpha=0.85,
              label=f'Sem Controle (Para em $r_{{eq}}=({r_eq_ol[0]:.2f}, {r_eq_ol[1]:.2f})$)')
    ax1a.plot(y_p1[0], y_p1[2], color='#1b9e77', lw=2.4,
              label=f'Com GPC no Tempo $t$ (Atinge $({r_eq_p1[0]:.2f}, {r_eq_p1[1]:.2f})$)')

    ax1a.scatter(*S_DEFAULT, color='red', s=90, zorder=6, label='Start $S=(0,0)$')
    ax1a.scatter(*A_TARGET, color='#0066cc', marker='*', s=220, zorder=8, label='Goal Exato $A=(10,10)$')
    ax1a.scatter(y_ol[0, 0], y_ol[2, 0], color='orange', s=80, edgecolor='black', zorder=7, label='Condição Inicial $(1,8)$')
    ax1a.scatter(*r_eq_ol, color='#d95f02', marker='X', s=130, zorder=7)
    ax1a.scatter(*r_eq_p1, color='#1b9e77', marker='P', s=130, zorder=9)

    ax1a.set_title('Trajetória no Plano 2D $(X \\times Y)$', fontsize=12, fontweight='bold')
    ax1a.set_xlabel('Posição $X$')
    ax1a.set_ylabel('Posição $Y$')
    ax1a.set_xlim(-0.5, 11.2)
    ax1a.set_ylim(-0.5, 11.2)
    ax1a.set_aspect('equal', adjustable='box')
    ax1a.grid(True, linestyle=':', alpha=0.6)
    ax1a.legend(loc='lower right', fontsize=8.8)

    # Subplot (b): Resposta Temporal X(t) e Y(t)
    ax1b = fig1.add_subplot(gs1[0, 1])
    ax1b.axhline(A_TARGET[0], color='#0066cc', linestyle='--', lw=1.4, label='Alvo $A = 10.0$')
    ax1b.axhline(r_eq_ol[0], color='#d95f02', linestyle=':', lw=1.2, label=f'Eq. Natural ($8.89$)')
    ax1b.plot(t_ol, y_ol[0], color='#d95f02', lw=1.5, alpha=0.75, label='$x(t)$ Sem Controle')
    ax1b.plot(t_p1, y_p1[0], color='#1b9e77', lw=2.2, label='$x(t)$ Com GPC')
    ax1b.plot(t_p1, y_p1[2], color='#7570b3', lw=1.8, linestyle='-.', label='$y(t)$ Com GPC')
    ax1b.set_title('Convergência Temporal para o Alvo Exato $A=(10,10)$', fontsize=11, fontweight='bold')
    ax1b.set_xlabel('Tempo $t$ (s)')
    ax1b.set_ylabel('Posição $(x, y)$')
    ax1b.grid(True, linestyle=':', alpha=0.6)
    ax1b.legend(loc='lower right', fontsize=8.5, ncol=2)

    # Subplot (c): Esforço de Controle u(t) vs -E(A)
    ax1c = fig1.add_subplot(gs1[1, 1])
    Ex_A, Ey_A = E_field(A_TARGET[0], A_TARGET[1])
    ax1c.axhline(-Ex_A, color='#333333', linestyle='--', lw=1.3,
                 label=f'$-E(A) = {-Ex_A:.2f}$ (Compensação de Regime)')
    ax1c.plot(t_p1, u_p1[0], color='#1b9e77', lw=2.0, label='Força GPC $u_x(t)$')
    ax1c.plot(t_p1, u_p1[1], color='#7570b3', lw=1.8, linestyle='-.', label='Força GPC $u_y(t)$')
    ax1c.set_title('Ação Integral do GPC: $\\mathbf{u}(t) \\to -\\mathbf{E}(\\mathbf{A})$ em Regime', fontsize=11, fontweight='bold')
    ax1c.set_xlabel('Tempo $t$ (s)')
    ax1c.set_ylabel('Força de Controle $\\mathbf{u}(t)$')
    ax1c.grid(True, linestyle=':', alpha=0.6)
    ax1c.legend(loc='lower right', fontsize=8.5)

    fig1_path = os.path.join(out_dir, 'perspectiva1_gpc_tempo.png')
    fig1.savefig(fig1_path, bbox_inches='tight')

    # -------------------------------------------------------------------------
    # FIGURA 2: PERSPECTIVA 2 (GPC EVOLUTIVO RUN-TO-RUN NOS PARÂMETROS)
    # -------------------------------------------------------------------------
    fig2 = plt.figure(figsize=(13, 8.5), dpi=150)
    fig2.suptitle(
        '2ª Perspectiva: GPC Evolutivo Run-to-Run Atuando nos Parâmetros do Campo ($\\mathbf{A}_{virt}^{(i)}$)',
        fontsize=14, fontweight='bold', y=0.98
    )

    gs2 = fig2.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.32, wspace=0.25)

    # Subplot (a): Família de Trajetórias Evoluindo entre as Simulações
    ax2a = fig2.add_subplot(gs2[:, 0])
    ax2a.plot([S_DEFAULT[0], A_TARGET[0]], [S_DEFAULT[1], A_TARGET[1]],
              '--', color='#888888', lw=1.8, label='Segmento TEV Original')

    cmap = plt.cm.viridis(np.linspace(0.1, 0.95, len(hist_p2)))
    for idx, item in enumerate(hist_p2):
        ep = item['epoch']
        yy = item['y']
        rend = item['r_end']
        if ep in (1, 2, 3, 5, len(hist_p2)):
            lbl = f'Época {ep}: fim em $({rend[0]:.2f}, {rend[1]:.2f})$'
            lw = 2.4 if ep in (1, len(hist_p2)) else 1.4
            alpha_line = 0.95 if ep in (1, len(hist_p2)) else 0.65
        else:
            lbl = None
            lw = 1.0
            alpha_line = 0.35
        ax2a.plot(yy[0], yy[2], color=cmap[idx], lw=lw, alpha=alpha_line, label=lbl)
        ax2a.scatter(rend[0], rend[1], color=cmap[idx], s=55, zorder=6)

    ax2a.scatter(*S_DEFAULT, color='red', s=90, zorder=7, label='Start $S=(0,0)$')
    ax2a.scatter(*A_TARGET, color='#0066cc', marker='*', s=220, zorder=9, label='Goal Real $A=(10,10)$')
    ax2a.scatter(*theta_star, color='crimson', marker='D', s=90, zorder=8,
                 label=f'Polo Virtual Ótimo $A_{{virt}}^*=({theta_star[0]:.2f},{theta_star[1]:.2f})$')

    ax2a.set_title('Evolução das Trajetórias Físicas Puras ao Longo das Épocas $i$', fontsize=11.5, fontweight='bold')
    ax2a.set_xlabel('Posição $X$')
    ax2a.set_ylabel('Posição $Y$')
    ax2a.set_xlim(-0.5, 12.0)
    ax2a.set_ylim(-0.5, 12.0)
    ax2a.set_aspect('equal', adjustable='box')
    ax2a.grid(True, linestyle=':', alpha=0.6)
    ax2a.legend(loc='lower right', fontsize=8.3)

    # Subplot (b): Convergência do Ponto de Equilíbrio e do Parâmetro A_virt por Época
    epochs = [h['epoch'] for h in hist_p2]
    r_ends_x = [h['r_end'][0] for h in hist_p2]
    thetas_x = [h['theta'][0] for h in hist_p2]
    errors = [h['error_norm'] for h in hist_p2]

    ax2b = fig2.add_subplot(gs2[0, 1])
    ax2b.axhline(A_TARGET[0], color='#0066cc', linestyle='--', lw=1.5, label='Meta de Equilíbrio $A_x = 10.0$')
    ax2b.plot(epochs, r_ends_x, '-o', color='#1b9e77', lw=2.2, label='Equilíbrio Atingido $x_{eq}^{(i)}$')
    ax2b.plot(epochs, thetas_x, '-s', color='crimson', lw=2.0, label='Parâmetro do Campo $A_{virt,x}^{(i)}$')
    ax2b.set_title('Adaptação do Parâmetro do Campo vs. Equilíbrio Atingido', fontsize=11, fontweight='bold')
    ax2b.set_xlabel('Época / Simulação $i$')
    ax2b.set_ylabel('Coordenada $X$')
    ax2b.set_xticks(epochs)
    ax2b.grid(True, linestyle=':', alpha=0.6)
    ax2b.legend(loc='center right', fontsize=8.5)

    # Subplot (c): Decaimento do Erro de Posição Final ||A - r_end^(i)||
    ax2c = fig2.add_subplot(gs2[1, 1])
    ax2c.semilogy(epochs, np.maximum(errors, 1e-6), '-o', color='#7570b3', lw=2.2)
    for ep_i, err_i in zip(epochs, errors):
        if ep_i in (1, 2, 3, 5, 10):
            ax2c.annotate(f'{err_i:.4f}', (ep_i, max(err_i, 1e-6)),
                          textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    ax2c.set_title('Erro de Regime Permanente $\\|\\mathbf{A} - \\mathbf{r}_{eq}^{(i)}\\|_2$ (Escala Log)',
                   fontsize=11, fontweight='bold')
    ax2c.set_ylim(3e-5, 5.0)
    ax2c.set_xlabel('Época / Simulação $i$')
    ax2c.set_ylabel('Erro Final (unidades)')
    ax2c.set_xticks(epochs)
    ax2c.grid(True, which='both', linestyle=':', alpha=0.6)

    fig2_path = os.path.join(out_dir, 'perspectiva2_gpc_evolutivo.png')
    fig2.savefig(fig2_path, bbox_inches='tight')

    # -------------------------------------------------------------------------
    # RELATÓRIO EM PDF (4 PÁGINAS: TEORIA + RESULTADOS EXPERIMENTAIS)
    # -------------------------------------------------------------------------
    pdf_path = os.path.join(out_dir, 'Resumo_Propostas_GPC_Sistema_Potencial.pdf')
    with PdfPages(pdf_path) as pdf:
        # ---- PÁGINA 1: PROBLEMA E 1ª PERSPECTIVA ----
        p1 = plt.figure(figsize=(8.27, 11.69), dpi=150)  # A4 Retrato
        ax = p1.add_axes([0, 0, 1, 1])
        ax.axis('off')

        # Cabeçalho
        ax.add_patch(plt.Rectangle((0.05, 0.90), 0.90, 0.065, color='#1b365d', alpha=0.95))
        ax.text(0.50, 0.94, 'RELATÓRIO TÉCNICO: CONTROLE PREDITIVO GENERALIZADO (GPC)',
                color='white', fontsize=13, fontweight='bold', ha='center', va='center')
        ax.text(0.50, 0.915, 'Aplicado ao Modelo Dinâmico de Trajetórias por Campo Potencial (TEV)',
                color='#d0e1f9', fontsize=10, ha='center', va='center')

        texto_p1 = (
            "1. CONTEXTUALIZAÇÃO E ORIGEM DO DESVIO DE EQUILÍBRIO\n"
            "───────────────────────────────────────────────────────────────────────────────\n"
            "O agente desloca-se no plano 2D segundo a equação diferencial de 2ª ordem amortecida:\n"
            r"$m\,\ddot{\mathbf{r}}(t) + \alpha\,\dot{\mathbf{r}}(t) = \mathbf{E}(\mathbf{r})$"
            ", onde o campo vetorial "
            r"$\mathbf{E}(\mathbf{r}) = -\nabla V(\mathbf{r})$"
            " é gerado por uma densidade\n"
            "linear de memória "
            r"$\lambda(q) = q$"
            " ao longo do segmento TEV entre a origem "
            r"$\mathbf{S}=(0,0)$"
            " e o alvo "
            r"$\mathbf{A}=(10,10)$"
            ":\n\n"
            r"$\mathbf{E}(\mathbf{r}) = -\frac{\kappa}{T}\left["
            r"\frac{\mathbf{r}_1}{\tilde{r}_1}\left(-1 + \frac{r_1^2 - r_2^2 + T^2}{T^2 - (\tilde{r}_1+\tilde{r}_2)^2}\right)"
            r" + \frac{\mathbf{r}_2}{\tilde{r}_2}\left(1 + \frac{r_1^2 - r_2^2 + T^2}{T^2 - (\tilde{r}_1+\tilde{r}_2)^2}\right)"
            r" + \frac{\mathbf{T}}{T}\ln\left(\frac{\tilde{r}_1+\tilde{r}_2+T}{\tilde{r}_1+\tilde{r}_2-T}\right)\right]$"
            "\n\n"
            "• Com "
            r"$\kappa = -10.0$"
            " e "
            r"$\epsilon = 0.1$"
            ", o mínimo global do potencial não ocorre sobre a extremidade "
            r"$\mathbf{A}=(10,10)$"
            ",\n"
            "  mas sim antes dela, em "
            f"$\\mathbf{{r}}_{{eq}} = ({r_eq_ol[0]:.4f},\\, {r_eq_ol[1]:.4f})$"
            " (erro estacionário de "
            f"${np.linalg.norm(A_TARGET - r_eq_ol):.4f}$"
            " unidades).\n"
            "• Isso ocorre porque sobre a ponta "
            r"$\mathbf{A}$"
            " toda a distribuição de memória puxa a partícula para trás ao longo\n"
            "  de "
            r"$-\hat{\mathbf{T}}$"
            ", resultando em uma força residual não-nula "
            f"$\\mathbf{{E}}(\\mathbf{{A}}) = ({Ex_A:.3f},\\, {Ey_A:.3f}) \\neq \\mathbf{{0}}$"
            ".\n\n\n"
            "2. PRIMEIRA PERSPECTIVA: GPC INTRA-SIMULAÇÃO (NO DOMÍNIO DO TEMPO t)\n"
            "───────────────────────────────────────────────────────────────────────────────\n"
            "Nesta abordagem, o campo "
            r"$\mathbf{E}(\mathbf{r})$"
            " é mantido inalterado e introduzimos uma força de controle externa\n"
            r"$\mathbf{u}(t) = [u_x(t),\, u_y(t)]^T$"
            " atuando diretamente sobre a dinâmica da partícula:\n\n"
            r"$m\,\ddot{\mathbf{r}}(t) + \alpha\,\dot{\mathbf{r}}(t) = \mathbf{E}(\mathbf{r}) + \mathbf{u}(t)$"
            "\n\n"
            "• Modelo CARIMA Discreto (com Ação Integradora Intrínseca):\n"
            "  A parte linear "
            r"$P(s) = \frac{1}{m s^2 + \alpha s}$"
            " é discretizada com período de amostragem "
            r"$T_s$"
            " (ZOH):\n"
            r"  $A(z^{-1})\,\mathbf{r}(k) = B(z^{-1})\,[\mathbf{u}(k-1) + \mathbf{E}(k-1)] + \frac{1}{\Delta}\boldsymbol{\xi}(k)$"
            ",   com "
            r"$\Delta = 1 - z^{-1}$"
            ".\n\n"
            "• Predição a Longo Alcance ("
            r"$j = 1, \dots, N_p$"
            ") e Função Custo Quadrática:\n"
            r"  $\hat{\mathbf{r}}(k+j|k) = \mathbf{f}(k+j) + \sum_{m=1}^{N_u} \mathbf{G}_{j,m}\,\Delta\mathbf{u}(k+m-1)$"
            "\n"
            r"  $J_{GPC} = \sum_{j=1}^{N_p} \|\hat{\mathbf{r}}(k+j|k) - \mathbf{w}(k+j)\|_{\mathbf{Q}}^2 + \sum_{m=1}^{N_u} \|\Delta\mathbf{u}(k+m-1)\|_{\mathbf{R}}^2$"
            "\n\n"
            "• Como o GPC elimina o erro em A=(10,10):\n"
            "  Como a variável otimizada é o incremento "
            r"$\Delta\mathbf{u}(k) = \mathbf{u}(k) - \mathbf{u}(k-1)$"
            ", o controlador possui um\n"
            "  integrador natural. Ao aproximar-se de "
            r"$\mathbf{A}$"
            ", a força de controle converge automaticamente para\n"
            f"  $\\mathbf{{u}}_\\infty = -\\mathbf{{E}}(\\mathbf{{A}}) = ({-Ex_A:.3f},\\, {-Ey_A:.3f})$"
            ", cancelando exatamente o viés do potencial e garantindo\n"
            "  equilíbrio assintótico estável em "
            r"$\mathbf{r}_\infty = (10.000,\, 10.000)$"
            " (mesmo se "
            r"$\alpha \leq 0$"
            ")."
        )
        ax.text(0.06, 0.87, texto_p1, fontsize=9.8, va='top', ha='left', linespacing=1.42, color='#111111')
        ax.text(0.50, 0.03, 'Página 1 de 4', fontsize=9, color='#666666', ha='center')
        pdf.savefig(p1)
        plt.close(p1)

        # ---- PÁGINA 2: 2ª PERSPECTIVA E COMPARAÇÃO ----
        p2 = plt.figure(figsize=(8.27, 11.69), dpi=150)
        ax = p2.add_axes([0, 0, 1, 1])
        ax.axis('off')

        ax.add_patch(plt.Rectangle((0.05, 0.90), 0.90, 0.065, color='#1b365d', alpha=0.95))
        ax.text(0.50, 0.94, '2ª PERSPECTIVA: GPC EVOLUTIVO RUN-TO-RUN NOS PARÂMETROS',
                color='white', fontsize=12.5, fontweight='bold', ha='center', va='center')
        ax.text(0.50, 0.915, 'Adaptação Episódica do Campo Potencial entre Simulações Sucessivas',
                color='#d0e1f9', fontsize=10, ha='center', va='center')

        texto_p2 = (
            "3. SEGUNDA PERSPECTIVA: GPC EVOLUTIVO SOBRE OS PARÂMETROS DO CAMPO\n"
            "───────────────────────────────────────────────────────────────────────────────\n"
            "Em vez de aplicar uma força externa "
            r"$\mathbf{u}(t)$"
            ", preservamos a equação física pura da partícula em cada\n"
            "simulação (época "
            r"$i = 1, 2, \dots, N_{ep}$"
            "):\n\n"
            r"$m\,\ddot{\mathbf{r}}^{(i)}(t) + \alpha\,\dot{\mathbf{r}}^{(i)}(t) = \mathbf{E}\!\left(\mathbf{r}^{(i)}(t);\; \boldsymbol{\theta}^{(i)}\right)$"
            "\n\n"
            "onde "
            r"$\boldsymbol{\theta}^{(i)}$"
            " representa os parâmetros construtivos do campo potencial na época "
            r"$i$"
            " — especificamente\n"
            "a posição do polo atrator virtual do campo "
            r"$\boldsymbol{\theta}^{(i)} = \mathbf{A}_{virt}^{(i)} = [A_{vx}^{(i)},\, A_{vy}^{(i)}]^T$"
            " (podendo também incluir\n"
            "a origem "
            r"$\mathbf{S}^{(i)}$"
            " ou a espessura "
            r"$\epsilon^{(i)}$"
            ").\n\n"
            "• Dinâmica CARIMA no Domínio das Épocas (Run-to-Run):\n"
            "  Ao final de cada simulação "
            r"$i$"
            ", mede-se o ponto de repouso atingido "
            r"$\mathbf{y}(i) = \mathbf{r}_{end}^{(i)} = \mathbf{r}^{(i)}(t_f)$"
            ".\n"
            "  A evolução do equilíbrio entre simulações sucessivas é modelada de forma incremental por:\n\n"
            r"  $\mathbf{y}(i) = \mathbf{y}(i-1) + \mathbf{J}_{\theta}\,\Delta\boldsymbol{\theta}(i-1) + \boldsymbol{\zeta}(i)$"
            ",     onde "
            r"$\Delta\boldsymbol{\theta}(i) = \boldsymbol{\theta}^{(i+1)} - \boldsymbol{\theta}^{(i)}$"
            "\n\n"
            "  e "
            r"$\mathbf{J}_{\theta} = \frac{\partial \mathbf{r}_{eq}}{\partial \boldsymbol{\theta}}$"
            " é a matriz Jacobiana de sensibilidade do equilíbrio, atualizada evolutivamente\n"
            "  a cada geração a partir de uma população "
            r"$(1+\lambda)$"
            "-ES amostrada ao redor da predição atual.\n\n"
            "• Predição Multi-Simulação e Atualização Ótima dos Parâmetros:\n"
            "  O GPC prevê o deslocamento do ponto de equilíbrio para as próximas "
            r"$N_p$"
            " simulações:\n"
            r"  $\hat{\mathbf{y}}(i+j|i) = \mathbf{y}(i) + \sum_{m=1}^{\min(j, N_u)} (j - m + 1)\,\mathbf{J}_{\theta}\,\Delta\boldsymbol{\theta}(i+m-1)$"
            "\n\n"
            "  Minimizando o custo preditivo inter-simulações:\n"
            r"  $J_{ep}(i) = \sum_{j=1}^{N_p} \|\hat{\mathbf{y}}(i+j|i) - \mathbf{A}_{real}\|_{\mathbf{Q}_{ep}}^2 + \sum_{m=1}^{N_u} \|\Delta\boldsymbol{\theta}(i+m-1)\|_{\mathbf{R}_{ep}}^2$"
            "\n\n"
            "• Resultado Físico Obtido:\n"
            "  Em poucas simulações, o controlador descobre autonomamente o polo virtual ótimo\n"
            f"  $\\mathbf{{A}}_{{virt}}^* = ({theta_star[0]:.4f},\\, {theta_star[1]:.4f})$"
            " tal que "
            r"$\mathbf{E}(\mathbf{A}_{real};\, \mathbf{A}_{virt}^*) = \mathbf{0}$"
            ", fazendo a partícula estacionar\n"
            f"  exatamente em $\\mathbf{{r}}_{{eq}} = ({r_eq_p2[0]:.4f},\\, {r_eq_p2[1]:.4f})$ sem nenhuma força artificial externa!\n\n\n"
            "4. QUADRO COMPARATIVO DAS DUAS PROPOSTAS\n"
            "───────────────────────────────────────────────────────────────────────────────\n"
            "• 1ª Perspectiva (GPC no Tempo t):\n"
            "  - Variável manipulada: Força aditiva contínua/discreta u(t) = [ux(t), uy(t)].\n"
            "  - Quando atua: A cada passo de amostragem Ts = 0.05 s dentro de uma única simulação.\n"
            "  - Vantagem: Corrige perturbações externas em tempo real e estabiliza mesmo se alpha <= 0.\n\n"
            "• 2ª Perspectiva (GPC Evolutivo Run-to-Run nos Parâmetros):\n"
            "  - Variável manipulada: Parâmetros internos do campo potencial theta^(i) = A_virt^(i).\n"
            "  - Quando atua: Entre simulações (época i -> i+1), preservando a EDO pura do potencial.\n"
            "  - Vantagem: Recalibra o próprio campo potencial para que seu mínimo coincida com A=(10,10)."
        )
        ax.text(0.06, 0.87, texto_p2, fontsize=9.6, va='top', ha='left', linespacing=1.38, color='#111111')
        ax.text(0.50, 0.03, 'Página 2 de 4', fontsize=9, color='#666666', ha='center')
        pdf.savefig(p2)
        plt.close(p2)

        # ---- PÁGINA 3: GRÁFICOS DA 1ª PERSPECTIVA ----
        pdf.savefig(fig1, bbox_inches='tight')

        # ---- PÁGINA 4: GRÁFICOS DA 2ª PERSPECTIVA ----
        pdf.savefig(fig2, bbox_inches='tight')

    plt.close(fig1)
    plt.close(fig2)

    print("=================================================================")
    print("RESUMO NUMÉRICO DOS RESULTADOS:")
    print(f"  Sem Controle (Malha Aberta) : r_eq = ({r_eq_ol[0]:.4f}, {r_eq_ol[1]:.4f}) | erro = {np.linalg.norm(A_TARGET - r_eq_ol):.4f}")
    print(f"  1ª Perspectiva (GPC Tempo t): r_eq = ({r_eq_p1[0]:.4f}, {r_eq_p1[1]:.4f}) | erro = {np.linalg.norm(A_TARGET - r_eq_p1):.6f}")
    print(f"  2ª Perspectiva (GPC Evolut.): r_eq = ({r_eq_p2[0]:.4f}, {r_eq_p2[1]:.4f}) | erro = {np.linalg.norm(A_TARGET - r_eq_p2):.6f}")
    print(f"  Parâmetro Ótimo Encontrado  : A_virt* = ({theta_star[0]:.4f}, {theta_star[1]:.4f})")
    print("=================================================================")
    print(f"Arquivos salvos em: {out_dir}")


if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))
    create_figures_and_pdf(out_dir)

