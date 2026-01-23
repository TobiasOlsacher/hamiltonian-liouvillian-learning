"""
Tests for data_statistics module.
"""
import pytest
import numpy as np
from src.data_statistics import DataEntry, DataSet
from src.quantum_state import QuantumState
from src.pauli_algebra import QuantumOperator, PauliOperator


class TestDataEntry:
    """Tests for DataEntry class."""

    def test_init(self):
        """Test DataEntry initialization."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        entry = DataEntry(
            Nions=2,
            initial_state=state,
            simulation_time=1.0
        )
        assert entry.Nions == 2
        assert entry.initial_state == state
        assert entry.simulation_time == 1.0

    def test_init_invalid_Nions(self):
        """Test initialization with invalid Nions."""
        with pytest.raises(TypeError):
            DataEntry(Nions=0)

    def test_Nions_setter(self):
        """Test Nions setter."""
        entry = DataEntry(Nions=2)
        entry.Nions = 3
        assert entry.Nions == 3

    def test_Nions_setter_invalid(self):
        """Test Nions setter with invalid input."""
        entry = DataEntry(Nions=2)
        with pytest.raises(TypeError):
            entry.Nions = -1

    def test_measurements_setter(self):
        """Test measurements setter."""
        entry = DataEntry(Nions=2)
        measurements = {"XX": [0, 1, 2]}
        entry.measurements = measurements
        assert entry.measurements == measurements

    def test_exact_expvals_default(self):
        """Test default exact_expvals."""
        entry = DataEntry(Nions=2)
        assert "II" in entry.exact_expvals
        assert entry.exact_expvals["II"] == [1]

    def test_exact_expvals_custom(self):
        """Test custom exact_expvals."""
        entry = DataEntry(Nions=2, exact_expvals={"XX": [0.5]})
        assert "XX" in entry.exact_expvals
        assert entry.exact_expvals["XX"] == [0.5]

    def test_copy(self):
        """Test copy method."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        entry = DataEntry(
            Nions=2,
            initial_state=state,
            simulation_time=1.0
        )
        entry_copy = entry.copy()
        assert entry_copy.Nions == entry.Nions
        assert entry_copy is not entry

    def test_hash(self):
        """Test hash method."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        entry = DataEntry(Nions=2, initial_state=state, simulation_time=1.0)
        hash_val = hash(entry)
        assert isinstance(hash_val, int)

    def test_str(self):
        """Test string representation."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        entry = DataEntry(Nions=2, initial_state=state, simulation_time=1.0)
        s = entry.str()
        assert isinstance(s, str)


class TestDataSet:
    """Tests for DataSet class."""

    def test_init(self):
        """Test DataSet initialization."""
        dataset = DataSet(Nions=2)
        assert dataset.Nions == 2
        assert isinstance(dataset.data, dict)

    def test_add_data_entry(self):
        """Test adding entry to dataset."""
        dataset = DataSet(Nions=2)
        state = QuantumState(N=2, excitations="00", basis="zz")
        entry = DataEntry(Nions=2, initial_state=state, simulation_time=1.0)
        dataset.add_data_entry(entry)
        assert len(dataset.data) > 0
        # Check that entry is stored with correct key
        assert (state, 1.0) in dataset.data

    def test_add_data_entry_invalid_Nions(self):
        """Test adding entry with wrong Nions."""
        dataset = DataSet(Nions=2)
        state = QuantumState(N=3, excitations="000", basis="zzz")
        entry = DataEntry(Nions=3, initial_state=state, simulation_time=1.0)
        with pytest.raises(ValueError):
            dataset.add_data_entry(entry)

    def test_add_data_entry_combines_duplicates(self):
        """Test that adding duplicate entries combines them."""
        dataset = DataSet(Nions=2)
        state = QuantumState(N=2, excitations="00", basis="zz")
        entry1 = DataEntry(Nions=2, initial_state=state, simulation_time=1.0, measurements={"XX": [0, 1]})
        entry2 = DataEntry(Nions=2, initial_state=state, simulation_time=1.0, measurements={"XX": [2, 3]})
        dataset.add_data_entry(entry1)
        dataset.add_data_entry(entry2)
        # Should have only one entry (combined)
        assert len(dataset.data) == 1
        # Combined entry should have more measurements
        combined = dataset.data[(state, 1.0)]
        assert len(combined.measurements["XX"]) == 4

    def test_copy(self):
        """Test copy method."""
        dataset = DataSet(Nions=2)
        dataset_copy = dataset.copy()
        assert dataset_copy.Nions == dataset.Nions
        assert dataset_copy is not dataset

    def test_str(self):
        """Test string representation."""
        dataset = DataSet(Nions=2)
        s = dataset.str()
        assert isinstance(s, str)
