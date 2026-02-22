"""
Tests for model_hamiltonians module.
"""
import pytest
import numpy as np
from src.model_hamiltonians import get_hamiltonian, get_dissipators
from src.pauli_algebra import QuantumOperator, Dissipator


class TestGetHamiltonian:
    """Tests for get_hamiltonian function."""

    def test_get_hamiltonian_single_qubit(self):
        """Test Hamiltonian with single-qubit terms."""
        N = 2
        terms = {
            "Bx": np.array([0.5, 0.3]),
            "By": np.array([0.2, 0.4]),
            "Bz": np.array([0.1, 0.2])
        }
        H = get_hamiltonian(N, terms)
        assert isinstance(H, QuantumOperator)
        assert H.N == N

    def test_get_hamiltonian_two_qubit(self):
        """Test Hamiltonian with two-qubit interactions."""
        N = 2
        J = np.array([[0, 0.5], [0.5, 0]])
        terms = {"Jxx": J}
        H = get_hamiltonian(N, terms)
        assert isinstance(H, QuantumOperator)
        assert H.N == N

    def test_get_hamiltonian_mixed(self):
        """Test Hamiltonian with mixed terms."""
        N = 2
        terms = {
            "Bz": np.array([0.1, 0.2]),
            "Jzz": np.array([[0, 0.5], [0.5, 0]])
        }
        H = get_hamiltonian(N, terms)
        assert isinstance(H, QuantumOperator)

    def test_get_hamiltonian_invalid_field_length(self):
        """Test Hamiltonian with invalid field length."""
        N = 2
        terms = {"Bx": np.array([0.5])}  # Wrong length
        with pytest.raises(ValueError):
            get_hamiltonian(N, terms)

    def test_get_hamiltonian_invalid_coupling_shape(self):
        """Test Hamiltonian with invalid coupling shape."""
        N = 2
        terms = {"Jxx": np.array([[0, 0.5]])}  # Wrong shape
        with pytest.raises(ValueError):
            get_hamiltonian(N, terms)

    def test_get_hamiltonian_coupling_diagonal(self):
        """Test Hamiltonian with non-zero diagonal coupling."""
        N = 2
        J = np.array([[0.5, 0.5], [0.5, 0.5]])  # Non-zero diagonal
        terms = {"Jxx": J}
        with pytest.raises(ValueError):
            get_hamiltonian(N, terms)

    def test_get_hamiltonian_with_cutoff(self):
        """Test Hamiltonian with cutoff."""
        N = 3
        terms = {
            "Jzz": np.array([[0, 0.5, 0.3], [0.5, 0, 0.4], [0.3, 0.4, 0]])
        }
        H = get_hamiltonian(N, terms, cutoff=1)
        assert isinstance(H, QuantumOperator)

    def test_get_hamiltonian_with_noise(self):
        """Test Hamiltonian with noise."""
        N = 2
        terms = {"Bz": np.array([0.1, 0.2])}
        H = get_hamiltonian(N, terms, add_noise=0.01)
        assert isinstance(H, QuantumOperator)

    def test_get_hamiltonian_add_noise_does_not_mutate_caller(self):
        """Regression: add_noise must not mutate caller's terms dict."""
        N = 2
        terms = {"Bz": np.array([0.1, 0.2])}
        orig = terms["Bz"].copy()
        np.random.seed(42)
        get_hamiltonian(N, terms, add_noise=0.01)
        np.testing.assert_array_equal(terms["Bz"], orig, "Caller's terms dict was mutated")

    def test_get_hamiltonian_flip_left_to_right(self):
        """Test Hamiltonian with flip_left_to_right."""
        N = 2
        terms = {"Bz": np.array([0.1, 0.2])}
        H = get_hamiltonian(N, terms, flip_left_to_right=True)
        assert isinstance(H, QuantumOperator)


class TestGetDissipators:
    """Tests for get_dissipators function."""

    def test_get_dissipators_single(self):
        """Test getting single dissipator."""
        N = 2
        Gx = np.array([[0, 0.1], [0.1, 0]])
        dissipation_rates = {"Gx": Gx}
        dissipators = get_dissipators(N, dissipation_rates)
        assert isinstance(dissipators, list)
        assert len(dissipators) > 0
        assert all(isinstance(d, Dissipator) for d in dissipators)

    def test_get_dissipators_multiple(self):
        """Test getting multiple dissipators."""
        N = 2
        Gx = np.array([[0, 0.1], [0.1, 0]])
        Gy = np.array([[0, 0.2], [0.2, 0]])
        Gz = np.array([[0, 0.3], [0.3, 0]])
        dissipation_rates = {"Gx": Gx, "Gy": Gy, "Gz": Gz}
        dissipators = get_dissipators(N, dissipation_rates)
        assert isinstance(dissipators, list)
        assert len(dissipators) > 0

    def test_get_dissipators_with_cutoff(self):
        """Test dissipators with cutoff."""
        N = 3
        Gz = np.array([[0, 0.1, 0.2], [0.1, 0, 0.3], [0.2, 0.3, 0]])
        dissipation_rates = {"Gz": Gz}
        dissipators = get_dissipators(N, dissipation_rates, cutoff=1)
        assert isinstance(dissipators, list)

    def test_get_dissipators_invalid_type(self):
        """Test dissipators with invalid dissipation type."""
        N = 2
        Gx = np.array([[0, 0.1], [0.1, 0]])
        dissipation_rates = {"Ginvalid": Gx}
        with pytest.raises(ValueError):
            get_dissipators(N, dissipation_rates)

    def test_get_dissipators_invalid_shape(self):
        """Test dissipators with invalid matrix shape."""
        N = 2
        Gx = np.array([[0, 0.1]])  # Wrong shape
        dissipation_rates = {"Gx": Gx}
        with pytest.raises(ValueError):
            get_dissipators(N, dissipation_rates)
