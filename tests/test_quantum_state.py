"""
Tests for quantum_state module.
"""
import pytest
import numpy as np
from src.quantum_state import QuantumState, getQuantumState_QuTip
from src.pauli_algebra import QuantumOperator, PauliOperator
from src.data_statistics import DataEntry


class TestQuantumState:
    """Tests for QuantumState class."""

    def test_init(self):
        """Test QuantumState initialization."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        assert state.N == 2
        assert state.excitations == "00"
        assert state.basis == "zz"

    def test_init_default_basis(self):
        """Test initialization with default basis."""
        state = QuantumState(N=2, excitations="00")
        assert state.basis == "zz"

    def test_excitations_setter_valid(self):
        """Test excitations setter with valid input."""
        state = QuantumState(N=2, excitations="00")
        state.excitations = "11"
        assert state.excitations == "11"

    def test_excitations_setter_invalid_length(self):
        """Test excitations setter with invalid length."""
        state = QuantumState(N=2, excitations="00")
        with pytest.raises(TypeError):
            state.excitations = "0"  # Wrong length

    def test_excitations_setter_invalid_chars(self):
        """Test excitations setter with invalid characters."""
        state = QuantumState(N=2, excitations="00")
        with pytest.raises(ValueError):
            state.excitations = "02"  # Invalid char

    def test_basis_setter_valid(self):
        """Test basis setter with valid input."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        state.basis = "xx"
        assert state.basis == "xx"

    def test_basis_setter_invalid(self):
        """Test basis setter with invalid input."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        with pytest.raises(ValueError):
            state.basis = "aa"  # Invalid chars

    def test_state_preparation_setter(self):
        """Test state_preparation setter."""
        state = QuantumState(N=2, excitations="00")
        state_prep = [["Rx0", np.pi/2]]
        state.state_preparation = state_prep
        assert state.state_preparation == state_prep

    def test_state_preparation_setter_invalid_ion(self):
        """Test state_preparation setter with invalid ion index."""
        state = QuantumState(N=2, excitations="00")
        with pytest.raises(ValueError):
            state.state_preparation = [["Rx2", np.pi/2]]  # Ion index too large

    def test_state_preparation_label_setter(self):
        """Test state_preparation_label setter."""
        state = QuantumState(N=2, excitations="00")
        state.state_preparation_label = "test"
        assert state.state_preparation_label == "test"

    def test_state_preparation_label_setter_invalid(self):
        """Test state_preparation_label setter with invalid input."""
        state = QuantumState(N=2, excitations="00")
        with pytest.raises(ValueError):
            state.state_preparation_label = "Test"  # Not lowercase

    def test_state_preparation_error_setter(self):
        """Test state_preparation_error setter."""
        state = QuantumState(N=2, excitations="00")
        state.state_preparation_error = 0.1
        assert state.state_preparation_error == 0.1

    def test_state_preparation_error_setter_invalid(self):
        """Test state_preparation_error setter with invalid input."""
        state = QuantumState(N=2, excitations="00")
        with pytest.raises(TypeError):
            state.state_preparation_error = 1.5  # > 1

    def test_copy(self):
        """Test copy method."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        state_copy = state.copy()
        assert state_copy.N == state.N
        assert state_copy.excitations == state.excitations
        assert state_copy is not state

    def test_eq(self):
        """Test equality."""
        state1 = QuantumState(N=2, excitations="00", basis="zz")
        state2 = QuantumState(N=2, excitations="00", basis="zz")
        state3 = QuantumState(N=2, excitations="11", basis="zz")
        assert state1 == state2
        assert state1 != state3

    def test_hash(self):
        """Test hash method."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        hash_val = hash(state)
        assert isinstance(hash_val, int)

    def test_str(self):
        """Test string representation."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        s = state.str()
        assert isinstance(s, str)
        assert "z0" in s or "0" in s

    def test_to_QuTip(self):
        """Test conversion to QuTip state."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        qt_state = state.to_QuTip(apply_state_preparation=False)
        assert qt_state is not None

    def test_evaluate_exact_expvals(self):
        """Test exact expectation value evaluation."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        qop = QuantumOperator(N=2, terms={"ZZ": 1.0})
        expvals = state.evaluate_exact_expvals(qop)
        assert isinstance(expvals, dict)
        assert "ZZ" in expvals

    def test_evaluate_exact_expvals_pauli_operator(self):
        """Test exact expectation value with PauliOperator."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        pop = PauliOperator(N=2, pauli_type="ZZ", coeff=1.0)
        expvals = state.evaluate_exact_expvals(pop)
        assert isinstance(expvals, dict)

    def test_extend_to_larger_system(self):
        """Test extension to larger system."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        state_ext = state.extend_to_larger_system(extension_factor=2)
        assert state_ext.N == 4
        assert state_ext.excitations == "0000"

    def test_extend_to_larger_system_invalid(self):
        """Test extension with invalid factor."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        with pytest.raises(TypeError):
            state.extend_to_larger_system(extension_factor=0)

    def test_is_eigenstate_eigenstate(self):
        """Test is_eigenstate returns True for eigenstate."""
        pytest.importorskip("tqdm")
        state = QuantumState(N=2, excitations="00", basis="zz")
        qop = QuantumOperator(N=2, terms={"ZZ": 1.0})
        assert state.is_eigenstate(qop) is True

    def test_is_eigenstate_non_eigenstate(self):
        """Test is_eigenstate returns False for non-eigenstate."""
        pytest.importorskip("tqdm")
        state = QuantumState(N=2, excitations="00", basis="zz")
        state.state_preparation = [["Rx0", np.pi / 2]]
        qop = QuantumOperator(N=2, terms={"ZZ": 1.0})
        assert state.is_eigenstate(qop) is False

    def test_get_state_preparation_from_data(self):
        """Test get_state_preparation_from_data static method."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        entry = DataEntry(Nions=2, initial_state=state, simulation_time=0.0)
        prep = QuantumState.get_state_preparation_from_data(entry, nshots=0)
        assert isinstance(prep, list)
        assert len(prep) == 6

    def test_get_mixed_state_from_data(self):
        """Test get_mixed_state_from_data static method."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        entry = DataEntry(Nions=2, initial_state=state, simulation_time=0.0)
        rho = QuantumState.get_mixed_state_from_data(entry, nshots=0)
        assert isinstance(rho, QuantumState)
        assert rho.N == 2
        assert rho.qutip_state is not None

    def test_getQuantumState_QuTip(self):
        """Test getQuantumState_QuTip standalone function."""
        psi = getQuantumState_QuTip(excitations="00", basis="zz")
        assert psi is not None
        psi_x = getQuantumState_QuTip(excitations="10", basis="zx")
        assert psi_x is not None

    # def test_split_into_basis(self):
    #     """Test splitting into basis."""
    #     state = QuantumState(N=1, excitations="0", basis="z")
    #     basis_states = [
    #         QuantumState(N=1, excitations="0", basis="z"),
    #         QuantumState(N=1, excitations="1", basis="z")
    #     ]
    #     coeffs = state.split_into_basis(basis_states)
    #     assert isinstance(coeffs, list)
    #     assert len(coeffs) == 2

    # def test_split_into_basis_invalid(self):
    #     """Test splitting with invalid basis."""
    #     state = QuantumState(N=1, excitations="0", basis="z")
    #     with pytest.raises(TypeError):
    #         state.split_into_basis("invalid")
