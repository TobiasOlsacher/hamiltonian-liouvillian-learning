"""
Tests for measurement_setting module.
"""
import pytest
from src.measurement_setting import MeasurementSetting
from src.quantum_state import QuantumState
from src.pauli_algebra import QuantumOperator


class TestMeasurementSetting:
    """Tests for MeasurementSetting class."""

    def test_init(self):
        """Test MeasurementSetting initialization."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        msetting = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            measurement_basis="zz",
            nshots=100
        )
        assert msetting.initial_state == state
        assert msetting.simulation_time == 1.0
        assert msetting.measurement_basis == "zz"
        assert msetting.nshots == 100

    def test_initial_state_setter(self):
        """Test initial_state setter."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        msetting = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            measurement_basis="zz"
        )
        new_state = QuantumState(N=2, excitations="11", basis="zz")
        msetting.initial_state = new_state
        assert msetting.initial_state == new_state

    def test_initial_state_setter_invalid(self):
        """Test initial_state setter with invalid input."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        msetting = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            measurement_basis="zz"
        )
        with pytest.raises(TypeError):
            msetting.initial_state = "invalid"

    def test_simulation_time_setter(self):
        """Test simulation_time setter."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        msetting = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            measurement_basis="zz"
        )
        msetting.simulation_time = 2.0
        assert msetting.simulation_time == 2.0

    def test_simulation_time_setter_invalid(self):
        """Test simulation_time setter with invalid input."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        msetting = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            measurement_basis="zz"
        )
        with pytest.raises(TypeError):
            msetting.simulation_time = "invalid"

    def test_measurement_basis_setter(self):
        """Test measurement_basis setter."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        msetting = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            measurement_basis="zz"
        )
        msetting.measurement_basis = "xx"
        assert msetting.measurement_basis == "xx"

    def test_nshots_setter(self):
        """Test nshots setter."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        msetting = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            measurement_basis="zz",
            nshots=100
        )
        msetting.nshots = 200
        assert msetting.nshots == 200

    def test_nshots_setter_invalid(self):
        """Test nshots setter with invalid input."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        msetting = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            measurement_basis="zz"
        )
        with pytest.raises(TypeError):
            msetting.nshots = -2  # Invalid (must be >= 0 or -1)

    def test_exact_observables_setter(self):
        """Test exact_observables setter."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        msetting = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            exact_observables=qop
        )
        assert msetting.exact_observables == qop
        assert msetting.nshots == 0  # Should be set to 0 when exact_observables is set

    def test_exact_observables_and_basis_conflict(self):
        """Test that exact_observables and measurement_basis cannot both be set."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        with pytest.raises(ValueError):
            MeasurementSetting(
                initial_state=state,
                simulation_time=1.0,
                measurement_basis="zz",
                exact_observables=qop
            )

    def test_eq(self):
        """Test equality."""
        state1 = QuantumState(N=2, excitations="00", basis="zz")
        state2 = QuantumState(N=2, excitations="00", basis="zz")
        msetting1 = MeasurementSetting(
            initial_state=state1,
            simulation_time=1.0,
            measurement_basis="zz"
        )
        msetting2 = MeasurementSetting(
            initial_state=state2,
            simulation_time=1.0,
            measurement_basis="zz"
        )
        msetting3 = MeasurementSetting(
            initial_state=state1,
            simulation_time=2.0,
            measurement_basis="zz"
        )
        assert msetting1 == msetting2
        assert msetting1 != msetting3

    def test_ne(self):
        """Test inequality."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        msetting1 = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            measurement_basis="zz"
        )
        msetting2 = MeasurementSetting(
            initial_state=state,
            simulation_time=2.0,
            measurement_basis="zz"
        )
        assert msetting1 != msetting2

    def test_hash(self):
        """Test hash method."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        msetting = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            measurement_basis="zz",
            nshots=100
        )
        hash_val = hash(msetting)
        assert isinstance(hash_val, int)

    def test_copy(self):
        """Test copy method."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        msetting = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            measurement_basis="zz",
            nshots=100
        )
        msetting_copy = msetting.copy()
        assert msetting_copy.initial_state == msetting.initial_state
        assert msetting_copy.simulation_time == msetting.simulation_time
        assert msetting_copy.measurement_basis == msetting.measurement_basis
        assert msetting_copy.nshots == msetting.nshots
        assert msetting_copy is not msetting

    def test_str(self):
        """Test string representation."""
        state = QuantumState(N=2, excitations="00", basis="zz")
        msetting = MeasurementSetting(
            initial_state=state,
            simulation_time=1.0,
            measurement_basis="zz",
            nshots=100
        )
        s = msetting.str()
        assert isinstance(s, str)
        assert "t=" in s or "time" in s.lower()
