# main.py
import numpy as np
from scipy.constants import c
from ris_viz import RISLocalizationViz

def main():
    N = 16            # Number of RIS elements
    fc = 28e9         # 28 GHz
    U = 10            # Number of transmissions
    V = 64            # Number of subcarriers
    N0 = 1e-10        # Noise variance
    P = 0.1           # Transmit power (Watts)

    ris = RISLocalizationViz(N, fc, U, V, N0, P)

    # Range of AoAs and powers to sweep
    theta_range = np.linspace(-np.pi/3, np.pi/3, 50)
    power_range = np.linspace(-5, 5, 30)  # in dBm

    # Positions
    ue_pos = np.array([5.0, 5.0])   # User Equipment
    bs_pos = np.array([1.0, 1.0])   # Base Station
    ris_pos = np.array([0.0, 0.0])  # RIS assumed at origin

    np.random.seed(42)
    phi = np.exp(1j * 2 * np.pi * np.random.rand(N))
    tau = np.linalg.norm(ue_pos - bs_pos) / c

    aeb_vals = np.zeros_like(theta_range)
    crlb_vals = np.zeros_like(theta_range)
    mcrlb_vals = np.zeros_like(theta_range)

    for i, theta in enumerate(theta_range):
        alpha = np.sqrt(P) * (1 + 1j)
        fim_mm = ris.compute_fim_mm(alpha, theta, tau, phi)
        fim_tm = ris.compute_fim_tm(alpha, theta, tau, phi, ue_pos, bs_pos, ris_pos)

        aeb_vals[i] = ris.compute_aeb(fim_mm)
        crlb_vals[i] = ris.compute_crlb_theta(fim_mm)
        mcrlb_vals[i] = ris.compute_mcrlb(fim_mm, fim_tm, theta_true=theta, theta_est=theta)

    ris.plot_error_bounds(theta_range, aeb_vals, crlb_vals, mcrlb_vals)

    error_map = np.zeros((len(power_range), len(theta_range)))

    for i_p, p_dbm in enumerate(power_range):
        # Convert dBm to Watts
        P_lin = 10 ** ((p_dbm - 30) / 10.0)
        for j, theta in enumerate(theta_range):
            alpha = np.sqrt(P_lin) * (1 + 1j)
            fim = ris.compute_fim_mm(alpha, theta, tau, phi)
            error_map[i_p, j] = ris.compute_aeb(fim)

    ris.plot_heatmap(theta_range, power_range, error_map)

    try:
        ris.plot_interactive_error_bounds(theta_range, aeb_vals, crlb_vals, mcrlb_vals)
        ris.plot_interactive_heatmap(theta_range, power_range, error_map)
    except Exception as e:
        print("Interactive plots skipped:", e)

    mid_theta = theta_range[len(theta_range) // 2]
    fim_example = ris.compute_fim_mm(np.sqrt(P) * (1 + 1j), mid_theta, tau, phi)
    ris.plot_fim_heatmap(fim_example)

if __name__ == "__main__":
    main()

