import numpy as np
from scipy.constants import c
from ris_viz import RISLocalizationViz


def main():
    # System parameters
    N = 16
    fc = 28e9
    U = 10
    V = 64
    N0 = 1e-10
    P = 0.1

    ris = RISLocalizationViz(N, fc, U, V, N0, P)

    theta_range = np.linspace(-np.pi / 3, np.pi / 3, 50)
    power_range = np.linspace(-5, 5, 30)

    aeb_values = np.zeros_like(theta_range)
    crlb_values = np.zeros_like(theta_range)
    mcrlb_values = np.zeros_like(theta_range)

    p = np.array([5.0, 5.0])
    pb = np.array([1.0, 1.0])

    np.random.seed(42)
    phi = np.exp(1j * 2 * np.pi * np.random.rand(N))

    for i, theta in enumerate(theta_range):
        alpha = np.sqrt(P) * (1 + 1j)
        tau = np.linalg.norm(p - pb) / c

        fim_mm = ris.compute_fim_mm(alpha, theta, tau, phi)
        fim_tm = ris.compute_fim_tm(alpha, theta, tau, phi, p, pb)

        aeb_values[i] = ris.compute_aeb(fim_mm)
        crlb_values[i] = np.sqrt(np.abs(np.real(ris.safe_inverse(fim_mm)[2, 2])))
        mcrlb_values[i] = ris.compute_mcrlb(fim_mm, fim_tm, theta, theta)

    ris.plot_results(theta_range, aeb_values, crlb_values, mcrlb_values)

    error_bounds = np.zeros((len(power_range), len(theta_range)))
    for i, power in enumerate(power_range):
        for j, theta in enumerate(theta_range):
            P_lin = 10 ** (power / 10) / 1000
            alpha = np.sqrt(P_lin) * (1 + 1j)
            tau = np.linalg.norm(p - pb) / c
            fim_mm = ris.compute_fim_mm(alpha, theta, tau, phi)
            error_bounds[i, j] = ris.compute_aeb(fim_mm)

    ris.plot_heatmap(theta_range, power_range, error_bounds)
    ris.plot_interactive_error_bounds(theta_range, aeb_values, crlb_values, mcrlb_values)
    ris.plot_interactive_heatmap(theta_range, power_range, error_bounds)


if __name__ == "__main__":
    main()
