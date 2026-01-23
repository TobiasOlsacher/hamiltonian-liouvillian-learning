"""
Tests for quantum_simulator module.
"""
import pytest
import numpy as np
from src.quantum_simulator import QuantumSimulator
from src.quantum_state import QuantumState
from src.pauli_algebra import QuantumOperator, Dissipator
from src.data_statistics import DataSet


class TestQuantumSimulator:
    """Tests for QuantumSimulator class."""

    def test_init(self):
        """Test QuantumSimulator initialization."""
        sim = QuantumSimulator(Nions=2)
        assert sim.Nions == 2
        assert sim.data_set is not None

    def test_init_invalid_Nions(self):
        """Test initialization with invalid Nions."""
        with pytest.raises(TypeError):
            QuantumSimulator(Nions=0)

    def test_init_with_dataset(self):
        """Test initialization with custom dataset."""
        dataset = DataSet(Nions=2)
        sim = QuantumSimulator(Nions=2, data_set=dataset)
        assert sim.data_set == dataset

    def test_init_with_hamiltonian(self):
        """Test initialization with Hamiltonian."""
        H = QuantumOperator(N=2, terms={"XX": 1.0})
        sim = QuantumSimulator(Nions=2, hamiltonian=H)
        assert sim.hamiltonian == H

    def test_init_with_dissipators(self):
        """Test initialization with dissipators."""
        diss = Dissipator(N=2, diss_type=("XX", "YY"), coeff=0.1)
        sim = QuantumSimulator(Nions=2, dissipators=[diss])
        assert sim.dissipators == [diss]

    def test_rotating_frame_setter(self):
        """Test rotating_frame setter."""
        sim = QuantumSimulator(Nions=2)
        H = QuantumOperator(N=2, terms={"ZZ": 1.0})
        sim.rotating_frame = H
        assert sim.rotating_frame == H

    def test_rotating_frame_setter_invalid_type(self):
        """Test rotating_frame setter with invalid type."""
        sim = QuantumSimulator(Nions=2)
        with pytest.raises(TypeError):
            sim.rotating_frame = "invalid"

    def test_rotating_frame_setter_wrong_Nions(self):
        """Test rotating_frame setter with wrong Nions."""
        sim = QuantumSimulator(Nions=2)
        H = QuantumOperator(N=3, terms={"ZZZ": 1.0})
        with pytest.raises(ValueError):
            sim.rotating_frame = H

    def test_n_error_batch_setter(self):
        """Test n_error_batch setter."""
        sim = QuantumSimulator(Nions=2)
        sim.n_error_batch = 100
        assert sim.n_error_batch == 100

    def test_n_error_batch_setter_invalid(self):
        """Test n_error_batch setter with invalid input."""
        sim = QuantumSimulator(Nions=2)
        with pytest.raises(TypeError):
            sim.n_error_batch = -1

    def test_shot_to_shot_fluctuation_rate_setter(self):
        """Test shot_to_shot_fluctuation_rate setter."""
        sim = QuantumSimulator(Nions=2)
        sim.shot_to_shot_fluctuation_rate = 0.01
        assert sim.shot_to_shot_fluctuation_rate == 0.01

    def test_shot_to_shot_fluctuation_rate_setter_invalid(self):
        """Test shot_to_shot_fluctuation_rate setter with invalid input."""
        sim = QuantumSimulator(Nions=2)
        with pytest.raises(TypeError):
            sim.shot_to_shot_fluctuation_rate = -0.1

    def test_copy(self):
        """Test copy method."""
        sim = QuantumSimulator(Nions=2)
        sim_copy = sim.copy()
        assert sim_copy.Nions == sim.Nions
        assert sim_copy is not sim

    def test_str(self):
        """Test string representation."""
        sim = QuantumSimulator(Nions=2)
        s = sim.str()
        assert isinstance(s, str)
