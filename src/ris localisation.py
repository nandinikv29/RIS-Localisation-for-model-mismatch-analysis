import numpy as np
from scipy.constants import c


class RISLocalization:
    def __init__(self, N, fc, U, V, N0, P):
        self.N = N  # Number of RIS elements
        self.fc = fc  # Carrier frequency
        self.lambda_c = c / fc  # Wavelength
        self.U = U  # Number of transmissions
        self.V = V  # Number of subcarriers
        self.N0 = N0  # Noise variance
        self.P = P  # Transmission power

    def compute_steering_vector(self, theta, wavelength=None):
        if wavelength is None:
            wavelength = self.lambda_c
        n = np.arange(-(self.N - 1) / 2, (self.N) / 2)
        return np.exp(-1j * np.pi * n * np.sin(theta))

    def compute_steering_vector_derivative(self, theta):
        n = np.arange(-(self.N - 1) / 2, (self.N) / 2)
        return -1j * np.pi * n * np.cos(theta) * np.exp(-1j * np.pi * n * np.sin(theta))

    def compute_delay(self, tau, v, delta_f):
        return np.exp(-1j * 2 * np.pi * (self.fc + v * delta_f) * tau)

    def compute_channel_gain(self, p, pb):
        norm_p = max(np.linalg.norm(p), 1e-10)  # Avoid division by zero
        norm_pb = max(np.linalg.norm(pb), 1e-10)
        return (self.lambda_c**2 / (4 * np.pi)**2) / (norm_p * norm_pb)

    def regularize_matrix(self, matrix, epsilon=1e-10):
        """Add small values to diagonal for numerical stability"""
        return matrix + epsilon * np.eye(matrix.shape[0])

    def compute_fim_mm(self, alpha, theta, tau, phi):
        fim = np.zeros((3, 3), dtype=complex)
        for u in range(self.U):
            for v in range(self.V):
                ap = self.compute_steering_vector(theta)
                apb = self.compute_steering_vector(theta)
                dap = self.compute_steering_vector_derivative(theta)

                d_real = apb.T @ np.diag(phi) @ ap * self.compute_delay(tau, v, 0)
                d_imag = 1j * d_real
                d_theta = alpha * apb.T @ np.diag(phi) @ dap * self.compute_delay(tau, v, 0)

                derivatives = np.array([d_real, d_imag, d_theta])
                fim += 2 / self.N0 * np.real(np.outer(derivatives, derivatives.conj()))
        return self.regularize_matrix(fim)

    def compute_fim_tm(self, alpha, theta, tau, phi, p, pb):
        fim = np.zeros((3, 3), dtype=complex)
        for u in range(self.U):
            for v in range(self.V):
                wavelength = c / (self.fc + v * 0)
                ap = self.compute_steering_vector(theta, wavelength)
                apb = self.compute_steering_vector(theta, wavelength)
                dap = self.compute_steering_vector_derivative(theta)

                alpha_v = self.compute_channel_gain(p, pb)

                d_real = apb.T @ np.diag(phi) @ ap * self.compute_delay(tau, v, 0)
                d_imag = 1j * d_real
                d_theta = alpha_v * apb.T @ np.diag(phi) @ dap * self.compute_delay(tau, v, 0)

                derivatives = np.array([d_real, d_imag, d_theta])
                fim += 2 / self.N0 * np.real(np.outer(derivatives, derivatives.conj()))
        return self.regularize_matrix(fim)

    def safe_inverse(self, matrix):
        """Compute inverse with fallback pseudo-inverse"""
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

    def compute_aeb(self, fim):
        inv_fim = self.safe_inverse(fim)
        return np.sqrt(np.abs(np.real(inv_fim[2, 2])))

    def compute_mcrlb(self, fim_mm, fim_tm, theta_true, theta_0):
        inv_A = self.safe_inverse(fim_mm)
        mcrlb_var = np.real(inv_A @ fim_tm @ inv_A)
        bias = (theta_true - theta_0) ** 2
        return np.sqrt(np.abs(mcrlb_var[2, 2]) + bias)
