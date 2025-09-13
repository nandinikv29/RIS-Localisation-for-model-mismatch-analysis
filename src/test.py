import numpy as np
import pytest
from ris_localization import RISLocalization

# Reusable setup function
@pytest.fixture
def ris_instance():
    return RISLocalization(N=16, fc=28e9, U=10, V=64, N0=1e-10, P=0.1)

@pytest.fixture
def common_test_data(ris_instance):
    np.random.seed(42)
    alpha = 1 + 1j
    theta = np.random.uniform(-np.pi/3, np.pi/3)  # Random AoA within range
    tau = 0.001  # Arbitrary delay
    phi = np.exp(1j * 2 * np.pi * np.random.rand(ris_instance.N))
    return alpha, theta, tau, phi


def test_fim_mm_shape_and_values(ris_instance, common_test_data):
    alpha, theta, tau, phi = common_test_data

    fim = ris_instance.compute_fim_mm(alpha, theta, tau, phi)

    # FIM should be a 3x3 matrix
    assert fim.shape == (3, 3), "FIM must be 3x3 matrix"

    # Diagonal should be approximately non-negative (floating-point safe)
    assert np.all(np.real(np.diag(fim)) > -1e-12), "FIM diagonal has unexpected negative values"

    # FIM should be Hermitian (symmetric in real terms)
    assert np.allclose(fim, fim.T.conj(), atol=1e-10), "FIM is not Hermitian"


def test_safe_inverse_matches_pinv(ris_instance):
    # Random positive-definite matrix
    matrix = np.array([[2, 0.5], [0.5, 1.5]], dtype=float)

    safe_inv = ris_instance.safe_inverse(matrix)
    pinv = np.linalg.pinv(matrix)

    # Both inverses should match closely
    assert np.allclose(safe_inv, pinv, atol=1e-10), "Safe inverse does not match pseudo-inverse"


def test_compute_aeb_non_negative(ris_instance, common_test_data):
    alpha, theta, tau, phi = common_test_data

    fim = ris_instance.compute_fim_mm(alpha, theta, tau, phi)
    aeb = ris_instance.compute_aeb(fim)

    # AEB should always be non-negative
    assert aeb >= 0, "AEB must be non-negative"


def test_compute_mcrlb(ris_instance, common_test_data):
    alpha, theta, tau, phi = common_test_data

    # Compute two FIMs for MCRLB
    fim_mm = ris_instance.compute_fim_mm(alpha, theta, tau, phi)
    fim_tm = ris_instance.compute_fim_tm(alpha, theta, tau, phi, p=np.array([5.0, 5.0]), pb=np.array([1.0, 1.0]))

    mcrlb_value = ris_instance.compute_mcrlb(fim_mm, fim_tm, theta_true=theta, theta_0=theta + 0.01)

    # MCRLB should be finite and non-negative
    assert np.isfinite(mcrlb_value), "MCRLB must be finite"
    assert mcrlb_value >= 0, "MCRLB must be non-negative"


def test_plot_functions_no_error(ris_instance, common_test_data):
    """This test just ensures plot functions run without errors.
    It does NOT check correctness of the visual output."""
    import matplotlib
    matplotlib.use('Agg') 

    from ris_viz import RISLocalizationViz

    viz = RISLocalizationViz(
        ris_instance.N, ris_instance.fc, ris_instance.U,
        ris_instance.V, ris_instance.N0, ris_instance.P
    )

    theta_range = np.linspace(-np.pi/3, np.pi/3, 10)
    aeb_values = np.random.rand(10)
    crlb_values = np.random.rand(10)
    mcrlb_values = np.random.rand(10)
    power_range = np.linspace(-5, 5, 5)
    error_bounds = np.random.rand(5, 10)

    # Just call functions to ensure they run without error
    viz.plot_results(theta_range, aeb_values, crlb_values, mcrlb_values)
    viz.plot_heatmap(theta_range, power_range, error_bounds)
