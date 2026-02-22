"""
Shared pytest fixtures for hamiltonian-liouvillian-learning tests.
"""
import pytest
import numpy as np
from src.quantum_state import QuantumState
from src.pauli_algebra import QuantumOperator, PauliOperator


@pytest.fixture
def state_2q():
    """Two-qubit state |00⟩ in Z basis."""
    return QuantumState(N=2, excitations="00", basis="zz")


@pytest.fixture
def qop_2q_xx():
    """Two-qubit QuantumOperator with XX term."""
    return QuantumOperator(N=2, terms={"XX": 1.0})


@pytest.fixture
def qop_2q_mixed():
    """Two-qubit QuantumOperator with multiple terms."""
    return QuantumOperator(N=2, terms={"XX": 1.0, "YY": 0.5, "ZZ": 0.3})
