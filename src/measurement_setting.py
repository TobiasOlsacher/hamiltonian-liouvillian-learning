"""
This module contains methods for measurement settings.
classes:
    - MeasurementSetting
        A class to define a measurement setting for a quantum experiment.
"""
from __future__ import annotations
import copy
from src.pauli_algebra import QuantumOperator
from src.quantum_state import QuantumState



class MeasurementSetting:
    """
    Measurement setting of the quantum simulator.

    This class defines a measurement setting for a quantum experiment.
    The attributes measurement_basis and exact_observables cannot be set at the same time.

    Attributes
    ----------
    initial_state : QuantumState
        initial state of the system
    simulation_time : float
        time at which the measurement is performed
    measurement_basis : str
        measurement basis 
    nshots : int
        number of shots for the measurement
    exact_observables : QuantumOperator, optional
        quantum operator for which the exact expectation values are calculated
        All coefficients of the QuantumOperator are set to 1.
        Default is None.
    """
    def __init__(self, 
                initial_state: QuantumState | None = None,
                simulation_time: float | None = None,
                measurement_basis: str | None = None,
                nshots: int | None = None,
                exact_observables: QuantumOperator | None = None,
                ):
        """
        Initialize a MeasurementSetting.

        Parameters
        ----------
        initial_state : QuantumState
            initial state of the system
        simulation_time : float
            time at which the measurement is performed
        measurement_basis : str
            measurement basis 
        nshots : int
            number of shots for the measurement
        exact_observables : QuantumOperator, optional
            quantum operator for which the exact expectation values are calculated
            All coefficients of the QuantumOperator are set to 1.
            Default is None.        
        """
        self.initial_state = initial_state
        self.simulation_time = simulation_time
        self.measurement_basis = measurement_basis
        self.nshots = nshots
        self.exact_observables = exact_observables
    ### ------------------------------ ###
    ### MeasurementSetting attributes ###
    ### ------------------------------ ###
    @property
    def initial_state(self):
        return self._initial_state
    @initial_state.setter
    def initial_state(self, value):
        if not isinstance(value, QuantumState):
            raise TypeError("initial_state is not a QuantumState object.")
        self._initial_state = value
    @property
    def simulation_time(self):
        return self._simulation_time
    @simulation_time.setter
    def simulation_time(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("simulation_time is not a number.")
        self._simulation_time = value
    @property
    def measurement_basis(self):
        return self._measurement_basis
    @measurement_basis.setter
    def measurement_basis(self, value):
        # check if value is a string
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("measurement_basis is not a string.")
        self._measurement_basis = value
    @property
    def nshots(self):
        return self._nshots
    @nshots.setter
    def nshots(self, value):
        if value is not None:
            if not (value >= 0 or value == -1):
                raise TypeError("nshots={}, but must be a non-negative integer or -1.".format(value))
        self._nshots = value
    @property
    def exact_observables(self):
        return self._exact_observables
    @exact_observables.setter
    def exact_observables(self, value):
        if value is not None:
            if not isinstance(value, QuantumOperator):
                raise TypeError("exact_observables is not a QuantumOperator object.")
        # check if measurement_basis is not set
        if self.measurement_basis is not None and value is not None:
            raise ValueError("measurement_basis and exact_observables cannot be set at the same time.")
        # set nshots = 0 if exact_observables is set
        if value is not None:
            self.nshots = 0
        if value is not None:   
            value.set_coeffs_one()
        self._exact_observables = value
    ### --------------------------- ###
    ### MeasurementSetting methods ###
    ### --------------------------- ###
    def __eq__(self, 
                other: MeasurementSetting
                ) -> bool:
        """
        Returns True if self and other have the same initial_state, simulation_time and measurement_basis.
        Nshots and exact_expvals are not considered.
        """
        if not isinstance(other, MeasurementSetting):
            return False
        return self.initial_state == other.initial_state and self.simulation_time == other.simulation_time and self.measurement_basis == other.measurement_basis
    
    def __ne__(self, 
                other: MeasurementSetting
                ) -> bool:
        """
        Returns True if self and other have different initial_state, simulation_time or measurement_basis.
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.initial_state, self.simulation_time, self.measurement_basis, self.nshots))

    def copy(self) -> MeasurementSetting:
        """ 
        Return a copy of the MeasurementSetting.
        
        Returns
        -------
        MeasurementSetting
            A new instance with the same attributes as this measurement setting.
        """
        return MeasurementSetting(
            initial_state = self.initial_state.copy() if self.initial_state is not None else None,
            simulation_time = self.simulation_time,
            measurement_basis = self.measurement_basis,
            nshots = self.nshots,
            exact_observables = self.exact_observables.copy() if self.exact_observables is not None else None
        )

    def __str__(self):
        return self.str()
    def str(self) -> str:
        """
        Return the string representation of the MeasurementSetting.
        
        Returns
        -------
        str
            String representation of the MeasurementSetting.
        """
        time_str = "{:.4f}".format(self.simulation_time)
        state_str = self.initial_state.str()
        basis_str = self.measurement_basis
        qop_str = None
        if self.exact_observables is not None:
            qop_str = self.exact_observables.str()
        msetting_str = "psi0={}, t={}, basis={}, nshots={}, qop={}".format(state_str,time_str,basis_str,self.nshots,qop_str)
        return msetting_str