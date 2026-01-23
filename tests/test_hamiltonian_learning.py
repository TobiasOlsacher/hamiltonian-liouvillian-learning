"""
Tests for hamiltonian_learning module.
"""
import pytest
import numpy as np
from src.hamiltonian_learning import Result, Constraint, Ansatz
from src.quantum_state import QuantumState
from src.pauli_algebra import QuantumOperator
from tqdm import tqdm



class TestResult:
    """Tests for Result class."""

    def test_init(self):
        """Test Result initialization."""
        result = Result()
        assert result.ansatz_operator is None
        assert result.learning_method is None

    def test_init_with_kwargs(self):
        """Test Result initialization with kwargs."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        result = Result(ansatz_operator=qop, learning_method="test")
        assert result.ansatz_operator == qop
        assert result.learning_method == "test"


class TestConstraint:
    """Tests for Constraint class."""

    def test_init(self):
        """Test Constraint initialization."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        constraint = Constraint(
            initial_state=state,
            simulation_times=[0.0, 1.0],
            constraint_operator=qop
        )
        assert constraint.initial_state == state
        assert constraint.simulation_times == [0.0, 1.0]
        assert constraint.constraint_operator == qop

    def test_init_invalid_initial_state(self):
        """Test Constraint initialization with invalid initial_state."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        with pytest.raises(TypeError):
            Constraint(
                initial_state="invalid",
                simulation_times=[0.0, 1.0],
                constraint_operator=qop
            )

    def test_init_invalid_simulation_times(self):
        """Test Constraint initialization with invalid simulation_times."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        with pytest.raises(TypeError):
            Constraint(
                initial_state=state,
                simulation_times="invalid",
                constraint_operator=qop
            )

    def test_init_invalid_constraint_operator(self):
        """Test Constraint initialization with invalid constraint_operator."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        with pytest.raises(TypeError):
            Constraint(
                initial_state=state,
                simulation_times=[0.0, 1.0],
                constraint_operator="invalid"
            )

    def test_copy(self):
        """Test copy method."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        constraint = Constraint(
            initial_state=state,
            simulation_times=[0.0, 1.0],
            constraint_operator=qop
        )
        constraint_copy = constraint.copy()
        assert constraint_copy.initial_state == constraint.initial_state
        assert constraint_copy is not constraint


class TestAnsatz:
    """Tests for Ansatz class."""

    def test_init(self):
        """Test Ansatz initialization."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        ansatz = Ansatz(Nions=2, ansatz_operator=qop)
        assert ansatz.ansatz_operator == qop
        assert ansatz.Nions == 2

    def test_init_invalid_operator(self):
        """Test Ansatz initialization with invalid operator."""
        with pytest.raises(ValueError):
            # The ansatz_operator setter will raise a ValueError if given invalid type
            ansatz = Ansatz(Nions=2, ansatz_operator="invalid")

    def test_init_with_dissipators(self):
        """Test Ansatz initialization with dissipators."""
        from src.pauli_algebra import Dissipator
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        diss = Dissipator(N=2, diss_type=["XX", "YY"], coeff=0.1)
        ansatz = Ansatz(Nions=2, ansatz_operator=qop, ansatz_dissipators=[diss])
        diss_norm = diss.copy()
        diss_norm.coeff = 1
        assert ansatz.ansatz_operator == qop
        assert ansatz.ansatz_dissipators == [diss_norm]

    def test_copy(self):
        """Test copy method."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        ansatz = Ansatz(Nions=2, ansatz_operator=qop)
        ansatz_copy = ansatz.copy()
        assert ansatz_copy.ansatz_operator == ansatz.ansatz_operator
        assert ansatz_copy.Nions == ansatz.Nions
        assert ansatz_copy is not ansatz
