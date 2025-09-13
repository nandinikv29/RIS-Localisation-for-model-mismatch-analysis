import numpy as np
from scipy.constants import c

class RISLocalization:

    def __init__(self, N, fc, U, V, N0, P, d=None):
        """
        N : int, Number of RIS elements
        fc : float, Carrier frequency (Hz)
        U : int, Number of transmissions
        V : int, Number of subcarriers
        N0 : float, Noise variance (linear scale)
        P : float, Transmit power (Watts)
        d : float, RIS element spacing (default = λ/2)
        """
        self.N = int(N)
        self.fc = float(fc)
        self.lambda_c = c / self.fc  # Wavelength
        self.U = int(U)
        self.V = int(V)
        self.N0 = float(N0)
        self.P = float(P)
        self.d = self.lambda_c / 2 if d is None else float(d)
        
    # Steering Vector Computations

    def _element_indices(self):
        return np.arange(self.N) - (self.N - 1) / 2

    def steering_vector(self, theta, wavelength=None):
        """Compute array steering vector for a given AoA."""
        if wavelength is None:
            wavelength = self.lambda_c
        n = self._element_indices()
        phase_shift = -1j * 2 * np.pi * self.d * n * np.sin(theta) / wavelength
        return np.exp(phase_shift)

    def steering_vector_derivative(self, theta, wavelength=None):
        """Derivative of steering vector w.r.t theta."""
        if wavelength is None:
            wavelength = self.lambda_c
        n = self._element_indices()
        a = self.steering_vector(theta, wavelength)
        factor = -1j * 2 * np.pi * self.d * n * np.cos(theta) / wavelength
        return factor * a

    # Channel Modeling

    def compute_delay(self, tau, v, delta_f=0.0):
      
        return np.exp(-1j * 2 * np.pi * (self.fc + v * delta_f) * tau)

    def two_hop_channel_gain(self, ue_pos, bs_pos, ris_pos=None):
        
        if ris_pos is None:
            ris_pos = np.zeros_like(ue_pos)

        dist_bs_to_ris = max(np.linalg.norm(bs_pos - ris_pos), 1e-10)
        dist_ris_to_ue = max(np.linalg.norm(ue_pos - ris_pos), 1e-10)

        g_bs_ris = (self.lambda_c / (4 * np.pi * dist_bs_to_ris)) ** 2
        g_ris_ue = (self.lambda_c / (4 * np.pi * dist_ris_to_ue)) ** 2

        return g_bs_ris * g_ris_ue

    def regularize(self, matrix, eps=1e-12):
     
        return matrix + eps * np.eye(matrix.shape[0])

    def safe_inverse(self, matrix, eps=1e-12, rcond=1e-12):
   
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
          
            reg = self.regularize(matrix, eps)
            try:
                return np.linalg.inv(reg)
            except np.linalg.LinAlgError:
                return np.linalg.pinv(matrix, rcond=rcond)

    # FIM Computations

    def compute_fim_mm(self, alpha, theta, tau, phi, delta_f=0.0):
     
        fim = np.zeros((3, 3), dtype=float)
        a_theta = self.steering_vector(theta)
        da_theta = self.steering_vector_derivative(theta)

        for u in range(self.U):
            for v in range(self.V):
                delay = self.compute_delay(tau, v, delta_f)
                signal = np.vdot(a_theta, phi * a_theta) * delay
                signal_theta = alpha * np.vdot(a_theta, phi * da_theta) * delay

                derivatives = np.array([signal, 1j * signal, signal_theta], dtype=complex)
                fim += (2.0 / self.N0) * np.real(np.outer(derivatives, derivatives.conj()))

        return self.regularize(fim)

    def compute_fim_tm(self, alpha, theta, tau, phi, ue_pos, bs_pos, ris_pos=None, delta_f=0.0):
     
        fim = np.zeros((3, 3), dtype=float)

        gain = np.sqrt(self.two_hop_channel_gain(ue_pos, bs_pos, ris_pos))
        alpha_scaled = alpha * gain

        a_theta = self.steering_vector(theta)
        da_theta = self.steering_vector_derivative(theta)

        for u in range(self.U):
            for v in range(self.V):
                delay = self.compute_delay(tau, v, delta_f)
                signal = np.vdot(a_theta, phi * a_theta) * delay
                signal_theta = alpha_scaled * np.vdot(a_theta, phi * da_theta) * delay

                derivatives = np.array([signal, 1j * signal, signal_theta], dtype=complex)
                fim += (2.0 / self.N0) * np.real(np.outer(derivatives, derivatives.conj()))

        return self.regularize(fim)

    # Error Bound Calculations
    
    def compute_aeb(self, fim):
        
        inv_fim = self.safe_inverse(fim)
        return np.sqrt(np.abs(np.real(inv_fim[2, 2])))

    def compute_crlb_theta(self, fim):
   
        inv_fim = self.safe_inverse(fim)
        return np.sqrt(np.abs(np.real(inv_fim[2, 2])))

    def compute_mcrlb(self, fim_mm, fim_tm, theta_true, theta_est):
     
        inv_fim_mm = self.safe_inverse(fim_mm)
        middle = inv_fim_mm @ fim_tm @ inv_fim_mm
        variance_term = np.real(middle[2, 2])
        bias_term = (theta_true - theta_est) ** 2
        return np.sqrt(np.abs(variance_term) + bias_term)
