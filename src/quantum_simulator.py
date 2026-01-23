"""
This module contains methods for simulating an experimental quantum simulator.
classes:
    - QuantumSimulator
        A class to simulate quantum dynamics.
"""
from __future__ import annotations
import numpy as np
import qutip as qt
import itertools as it
from icecream import ic
import copy
import time as tm
from tqdm import tqdm
from src.pauli_algebra import PauliOperator, QuantumOperator, Dissipator, get_expval_from_Zproduct_state
from src.data_statistics import DataEntry, DataSet
from src.quantum_state import QuantumState
from src.measurement_setting import MeasurementSetting



class QuantumSimulator:
    """
    This class defines a QuantumSimulator object.
    
    Attributes
    ----------
    Nions : int
        number of ions
    data_set : DataSet
        measurement data in a DataSet
        Default is DataSet(Nions=self.Nions).
    hamiltonian : QuantumOperator 
        Hamiltonian of the system
        Default is None.
    dissipators : list of Dissipators
        Dissipators that define the Liouvillian of the system.
        Default is None.
    rotating_frame : QuantumOperator
        simulation_time dependet rotation that is applied to final state 
        after time-evolution (before measurement),
        i.e. psi(T) = exp(i*rotating_frame*T) * psi(T)
        Default is None.
    measurement_error : QuantumOperator
        measurement error
        Default is None.
    n_error_batch : int
        Simulation with errors is divided into batches,
        each batch consisting of n_error_batch shots.
        Default is None.
    shot_to_shot_fluctuation_rate : float
        relative drift in the parameters at each update (after each batch)
        Default is None.
    """
    def __init__(self, 
                Nions: int, 
                data_set: DataSet | None = None,
                hamiltonian: QuantumOperator | None = None,
                dissipators: list[Dissipator] | None = None,
                rotating_frame: QuantumOperator | None = None,
                measurement_error: QuantumOperator | None = None,
                n_error_batch: int | None = None,
                shot_to_shot_fluctuation_rate: float | None = None,
                ):
        if not isinstance(Nions, int) or Nions <= 0:
            raise TypeError("Nions is not a positive integer.")
        self._Nions = Nions
        self.data_set = data_set
        if self.data_set is None:
            self.data_set = DataSet(Nions=self.Nions)      
        self.hamiltonian = hamiltonian
        self.dissipators = dissipators
        self.rotating_frame = rotating_frame
        self.measurement_error = measurement_error
        self.n_error_batch = n_error_batch
        self.shot_to_shot_fluctuation_rate = shot_to_shot_fluctuation_rate
    ### ---------------------------- ###
    ### QuantumSimulator attributes ###
    ### ---------------------------- ###
    @property
    def Nions(self):
        return self._Nions
    @property
    def rotating_frame(self):
        return self._rotating_frame
    @rotating_frame.setter
    def rotating_frame(self, value):
        if value is not None:
            if not isinstance(value, QuantumOperator):
                raise TypeError("rotating_frame is not a QuantumOperator object.")
            if value.N != self.Nions:
                raise ValueError("rotating_frame has wrong number of ions.")
        self._rotating_frame = value
    @property
    def n_error_batch(self):
        return self._n_error_batch
    @n_error_batch.setter
    def n_error_batch(self, value):
        if value is not None:
            if not isinstance(value, int) or value <= 0:
                raise TypeError("n_error_batch is not a positive integer.")
        self._n_error_batch = value
    @property
    def shot_to_shot_fluctuation_rate(self):
        return self._shot_to_shot_fluctuation_rate
    @shot_to_shot_fluctuation_rate.setter
    def shot_to_shot_fluctuation_rate(self, value):
        if value is not None:
            if not isinstance(value, (int, float)) or value < 0:
                raise TypeError("shot_to_shot_fluctuation_rate is not a non-negative number.")
        self._shot_to_shot_fluctuation_rate = value
    ### ------------------------- ###
    ### QuantumSimulator methods ###
    ### ------------------------- ###
    def copy(self) -> QuantumSimulator:
        """ 
        Return a copy of the quantum simulator.

        Returns
        -------
        QuantumSimulator
            A new instance with the same attributes as this quantum simulator,
            except for the data_set attribute.
        """
        return QuantumSimulator(
            Nions = self.Nions,
            hamiltonian = self.hamiltonian.copy() if self.hamiltonian is not None else None,
            dissipators = [diss.copy() for diss in self.dissipators] if self.dissipators is not None else None,
            rotating_frame = self.rotating_frame.copy() if self.rotating_frame is not None else None,
            measurement_error = self.measurement_error.copy() if self.measurement_error is not None else None,
            n_error_batch = self.n_error_batch,
            shot_to_shot_fluctuation_rate = self.shot_to_shot_fluctuation_rate
        )

    def str(self) -> str:
        """
        Return the string representation of the QuantumSimulator.
        """
        return "QuantumSimulator(Nions={}, hamiltonian={}, dissipators={}, rotating_frame={}, measurement_error={}, n_error_batch={}, shot_to_shot_fluctuation_rate={})".format(self.Nions, self.hamiltonian, self.dissipators, self.rotating_frame, self.measurement_error, self.n_error_batch, self.shot_to_shot_fluctuation_rate)
    def __str__(self) -> str:
        return self.str()

    def __repr__(self) -> str:
        return self.str()

    def get_final_states(self, 
                        initial_state: QuantumState, 
                        times: list[float], 
                        save_states: bool = True, 
                        no_output: bool = True,
                        ) -> list[qt.Qobj] | None:
        """
        Adds final states to the DataSet of the QuantumSimulator.

        Final states are calculated for initial_state at each time in times.

        Parameters
        ----------
        initial_state : QuantumState
            initial state
        times : list of float
            times at which the final states are calculated
        save_states : bool 
            if True, final states are saved to the DataSet of self
            if False, final states are returned
            Default is True.
        no_output : bool
            if True, no output is printed
            Default is True.

        Returns
        -------
        list of QuTip states 
            list of final states as QuTip quantum states
            only returned if save_states is False
        """
        # check if errors are set
        if self.n_error_batch is not None and save_states:
            raise ValueError("Cannot save final state because n_error_batch is set (state preparation errors or shot-to-shot fluctuations). Use simulate() instead.")  
        # remove duplicate times and sort times
        times = list(set(times))
        times.sort()
        if times == [0] and initial_state.state_preparation_error is None and initial_state.state_preparation is None:
            print("Warning: times = [0] and state_preparation_error = None. Returning initial state.")
        #     #TODO add state preparation to initial state
        #     tevolved_states = [initial_state]
        # else:
        #-------------------------------------
        ### STEP 1 ### get final state for each time
        tevolved_states = getFinalState_QuTip(self.hamiltonian, initial_state, times, self.dissipators, self.rotating_frame)
        ## display memory usage of final states
        if not no_output:
            mem = np.sum([np.round(state.full().nbytes/1024,2) for state in tevolved_states])
            pstr_get_final_states = "Total memory usage of final states is {} kB for initial state {} and ntimes={}".format(mem, initial_state.str(), len(times))
            ic(pstr_get_final_states)
        #-------------------------------------
        ### STEP 2 ### save final states to data_set or return them
        if save_states:
            print("Saving final states to data_set.")
            ic("Saving final states to data_set.")
            for tinx in range(len(times)):
                time = times[tinx]
                data_entry = DataEntry(Nions=self.Nions, initial_state=initial_state, simulation_time=time, final_state=tevolved_states[tinx]) 
                self.data_set.add_data_entry(data_entry)
        else:
            return tevolved_states

    def get_exact_expvals(self, 
                        initial_state: QuantumState, 
                        times: list[float], 
                        qop: QuantumOperator, 
                        save_data: bool = False,
                        ) -> DataSet:
        """
        Calculates exact expectation values from stored final states.

        Returns exact expectation values as DataSet.
        Expectation values are calculated from saved final states in the DataSet of self.
        Expectation values are calcuated for each term in the quantum_operator object qop.
        initial_state is the initial state of the system.
        times is float or a list of times at which the exact expectation values are calculated.

        Parameters
        ----------
        initial_state : QuantumState
            initial state of the system
        - times : float or list of floats
            times at which the exact expectation values are calculated
        - qop : QuantumOperator
            quantum operator for which the expectation values are calculated
        - save_data : bool
            if True, adds the exact expectation values to the data_set attribute of the quantum_simulator object.
            Default is False.

        Returns
        -------
        DataSet
            data_set object containing the exact expectation values
        """
        # check if times is a float:
        if isinstance(times, float):
            times = [times]
        if self.n_error_batch is not None:
            raise ValueError("Cannot get exact expvals because n_error_batch is set (state preparation errors or shot-to-shot fluctuations). Use simulate() instead.") 
        # setup data_set
        dataset = DataSet(Nions=self.Nions)
        # split into paulis
        pops_tmp = [PauliOperator(N=qop.N, pauli_type=pop.pauli_type) for pop in qop.terms.values() if pop.pauli_type!="I"*qop.N]
        for time in times:
            # get final state from data_set
            final_state = self.data_set.data[(initial_state, time)].final_state
            # evaluate expectation value
            pops_vals = get_expvals_QuTip(final_state, pops_tmp)
            # add to data_set
            exact_expvals = {pops_tmp[inx].pauli_type : [pops_vals[inx]] for inx in range(len(pops_tmp))}
            data_entry = DataEntry(Nions=self.Nions, initial_state=initial_state, simulation_time=time, exact_expvals=exact_expvals)
            dataset.add_data_entry(data_entry)
            if save_data:
                self.data_set.add_data_entry(data_entry)
        return dataset

    # run a simulation experiment
    # returns measurement data 
    # parallize via https://superfastpython.com/multiprocessing-in-python/
    # def simulate(self, initial_state, times, bases, nshots, **kwargs):
    def simulate(self, 
                initial_state: QuantumState, 
                measurement_settings: list[MeasurementSetting], 
                save_data: bool = False,
                no_output: bool = True,
                ) -> DataSet:
        """
        Simulates a quantum experiment. 
        Returns measurement data as DataSet.
        nshots is the number of shots for each measurement.

        Parameters
        ----------
        initial_state : QuantumState
            initial state of the experiment
        measurement_settings : list
            list of MeasurementSettings for the experiment
        save_data : bool
            if True, all data is also added to the DataSet of self.
            Default is False.
        no_output : bool 
            if True, no output is printed
            Default is True.

        Returns
        -------
        DataSet
            data set containing the measurement data
        """
        # if there are no n_error_batch is None (no state_preparation_error or shot_to_shot_fluctuations),
        # then we can reuse saved final states and also save the missing final states
        # otherwise we do not save the final states
        if self.n_error_batch is None:
            measurement_data = self.simulate_ideal(initial_state, measurement_settings, save_data=save_data, no_output=no_output)
        else:
            measurement_data = self.simulate_errors(initial_state, measurement_settings, save_data=save_data, no_output=no_output)
        return measurement_data

    def simulate_ideal(self, 
                    initial_state: QuantumState, 
                    measurement_settings: list[MeasurementSetting], 
                    save_data: bool = False,
                    no_output: bool = True,
                    print_timers: bool = False,
                    ) -> DataSet:
        """
        Simulate experiment without experimental imperfections
        
        Sampling is done from stored final states.

        Parameters
        ----------
        initial_state : QuantumState
            initial state of the system
        measurement_settings : list
            list of MeasurementSettings for the experiment, 
        save_data : bool
            if True, all data is also added to the DataSet of self.
            Default is False
        no_output : bool
            if False, output from get_final_states() is printed
            Default is True
        print_timers : bool
            if True, prints the time for each step
            Default is False
        
        Returns
        -------
        DataSet
            DataSet containing the measurement data
        """
        if print_timers:
            ic("--------")
            ic("enter simulate_ideal()")
            tm0 = tm.time()
        #--------------------------------
        ### STEP 1 ### sort measurement settings by simulation_time (transform measurement settings into dict of time:list_of_measurement_settings)
        measurement_settings_dict = {}
        times = []
        for setting in measurement_settings:
            time = setting.simulation_time
            times.append(time)
            # add element to dict if not already present
            if time not in measurement_settings_dict:
                measurement_settings_dict[time] = []
            measurement_settings_dict[time].append(setting)
        times = np.sort(list(set(times)))
        # print number of measurement settings at each time
        nbases_str = "".join(["t={}:nsett={}, ".format(time, len(measurement_settings_dict[time])) for time in times])
        if not no_output:
            pstr_simulate_ideal = "number of measurement settings at each time: {}".format(nbases_str)
            ic(pstr_simulate_ideal)
        tm1 = tm.time()
        if print_timers:
            pstr_simulate_ideal = "... time for sorting measurement settings: {}".format(tm1-tm0)
            ic(pstr_simulate_ideal)
        #--------------------------------
        ### STEP 2 ### get required final states
        ## check if ALL final states are already stored in data_set (for all times)
        found = True
        for time in times:
            if (initial_state, time) not in self.data_set.data:
                found = False
                break
        # if found, get them from the data_set
        if found:
            final_states = [self.data_set.data[(initial_state, time)].final_state for time in times]
        ## if not, calculate them (do not add them to the data_set)
        if not found:
            final_states = self.get_final_states(initial_state, times, save_states=False, no_output=no_output)
        tm2 = tm.time()
        if print_timers:
            pstr_simulate_ideal = "...time for getting final states: {}".format(tm2-tm1)
            ic(pstr_simulate_ideal)
        #--------------------------------
        ### STEP 3 ### get data from final states
        # create empty data_set to be filled with measurement data
        dataset = DataSet(Nions=self.Nions)
        ## loop over times
        for tinx, time in enumerate(times):
            tm21 = tm.time()
            final_state = final_states[tinx]
            settings = measurement_settings_dict[time]
            #--------------------------------
            ### determine required operator and bases
            qop = QuantumOperator(N=self.Nions)
            tmp_basis_dict = {}
            for setting in settings:
                # add exact observables to qop
                if setting.exact_observables is not None:
                    qop += setting.exact_observables
                # add measurement basis to tmp_basis_dict
                if setting.nshots>0:
                    if setting.measurement_basis not in tmp_basis_dict or tmp_basis_dict[setting.measurement_basis]<setting.nshots:
                            tmp_basis_dict[setting.measurement_basis] = setting.nshots
            #--------------------------------
            ### get the exact expectation values from the final state
            exact_expvals = {}
            if qop.terms != {}:
                pops = [PauliOperator(N=qop.N, pauli_type=pop.pauli_type) for pop in qop.terms.values()] # if pop.pauli_type!="I"*qop.N]
                pops_vals = get_expvals_QuTip(final_state, pops)
                # add to data_set
                exact_expvals = {pops[inx].pauli_type : [pops_vals[inx]] for inx in range(len(pops))}
                data_entry = DataEntry(Nions=self.Nions, initial_state=initial_state, simulation_time=time, exact_expvals=exact_expvals)
                dataset.add_data_entry(data_entry)
            tm22 = tm.time()
            if print_timers: # and tinx in [0,1]:
                pstr_simulate_ideal = "...time for getting exact expvals (tinx={}, nterms={}): {}".format(tinx, len(exact_expvals.keys()), tm22-tm21)
                ic(pstr_simulate_ideal)
                del exact_expvals
            #--------------------------------
            ### get the measurement data 
            measurements = SampleBitstrings_QuTip(final_state, tmp_basis_dict, measurement_error=self.measurement_error)
            data_entry = DataEntry(Nions=self.Nions, initial_state=initial_state, simulation_time=time, measurements=measurements)
            del measurements
            # add measurements to data set if save_data is True
            if save_data:
                self.data_set.add_data_entry(data_entry)
            dataset.add_data_entry(data_entry)
            # delete data entry
            del data_entry
            tm23 = tm.time()
            if print_timers: # and tinx in [0,1]:
                pstr_simulate_ideal = "...time for getting measurements (tinx={}, nbases={}): {} [total est {}]".format(tinx, len(tmp_basis_dict.keys()), tm23-tm22, (tm23-tm22)*len(times))
                ic(pstr_simulate_ideal)
        if print_timers:
            ic("--------")
        # --------------------------------
        return dataset

    def simulate_errors(self, 
                        initial_state: QuantumState, 
                        times: list[float], 
                        bases: list[str] | dict, 
                        nshots: int | dict, 
                        qop: QuantumOperator | None = None,
                        ) -> DataSet:
        """
        Simulate experiment with experimental imperfections

        Cannot sample from stored final states.
        Used only if self.n_error_batch is not None
        defines batch Hamiltonian and calls simulate_ideal() for each batch.

        Parameters
        ----------
        initial_state : QuantumState
            initial state of the system
        times : list of floats
            times at which the final state is measured
        bases : list of strings or dict
            If bases is a list of strings, then it is the measurement bases for each time.
            If bases is a dict, then the keys are time and the values are the measurement basis for that time.
        nshots : int or dict
            If nshots is an int, then it is the number of shots for each measurement basis and for each time.
            If nshots is a dict, then the keys are (time,basis) and the values are the number of shots.
        qop : QuantumOperator
            quantum operator to be measured at each time
            (calculates exact expvals for each term in qop)
            Default is None.

        Returns
        -------
        DataSet
            DataSet containing the measurement data
        """
        raise NotImplementedError("simulate_errors() is not implemented yet.")
        # create empty data_set to be filled with measurement data
        dataset = DataSet(Nions=self.Nions)
        if self.shot_to_shot_fluctuation_rate is not None:
            nshots_batch = self.n_error_batch
            niter = nshots//nshots_batch
            nshots_rest = nshots - niter*nshots_batch
        else:
            niter = 0
            nshots_rest = nshots
        # create copy of the quantum_simulator object
        qsim_batch = self.copy()
        for iter in range(niter+1):
            if iter==niter:
                nshots_batch = nshots_rest
            # add shot to shot fluctuations to ideal Hamiltonian
            if self.shot_to_shot_fluctuation_rate is not None:
                batch_coeffs = self.hamiltonian.coeffs()*np.random.normal(1, self.shot_to_shot_fluctuation_rate, len(self.hamiltonian.coeffs()))
                qsim_batch.hamiltonian.set_coeffs(batch_coeffs)
            # add state preparation error
            initial_state_batch = initial_state
            if self.state_preparation_error is not None:
                initial_state_batch = [0 for bit in initial_state]
                for exc in range(self.Nions):
                    initial_state_batch[exc] = int(initial_state[exc])
                    if np.random.rand() < self.state_preparation_error:
                        initial_state_batch[exc] = 1-int(initial_state[exc])
                initial_state_batch = "".join([str(exc) for exc in initial_state_batch])
            # simulate the quantum system for the current batch
            measurement_data = qsim_batch.simulate_ideal(initial_state_batch, times, bases, nshots_batch, **kwargs) # qop=qop)
            dataset.add_data_set(measurement_data)
        # scramble the data
        dataset.scramble_data()
        return dataset

    def simulate_dataset(self, 
                        dataset: DataSet, 
                        nshots_replace: int | None = None,
                        add_exact_expvals: QuantumOperator | None = None,
                        ) -> DataSet:
        """
        Simulate a dataset.

        Calls self.simulate() for the same states, times and measurement bases as in the dataset,
        and returns the simulated dataset.
        
        Parameters
        ----------
        dataset : DataSet
            dataset to be simulated
        nshots_replace : int 
            if given, replaces the number of shots for each  measurement with nshots_replace
            Default is None.
        add_exact_expvals : QuantumOperator 
            if given, adds exact expectation values of the given QuantumOperator to the simulated dataset
            Default is None.

        Returns
        -------
        DataSet
            data set that contains the simulated measurement data
        """
        ### STEP 1 ### create dict of required measurement settings
        measurement_settings = dataset.get_measurement_settings(nshots_replace=nshots_replace, add_exact_expvals=add_exact_expvals)
        #-------------------------------------------------------------
        ### STEP 2 ### call simulate() for the required measurement settings
        dataset_simulated = DataSet(Nions=self.Nions)
        # for sinx, state in enumerate(measurement_settings):
        for sinx, state in tqdm(enumerate(measurement_settings), desc="Processing states for simulate_dataset()"):
            time1 = tm.time()
            dataset_state = self.simulate(initial_state=state, measurement_settings=measurement_settings[state])
            time2 = tm.time()
            if sinx==0:
                print("Time for first state:",time2-time1)
            dataset_simulated.add_data_set(dataset_state)
        #-------------------------------------------------------------
        return dataset_simulated

    def get_classical_shadow(self, 
                            initial_state: QuantumState, 
                            simulation_times: list[float], 
                            nshots_shadow: int, 
                            nshots_per_basis: int = 1, 
                            nbases_cutoff: int | None = None,
                            required_operator_initial_time: QuantumOperator | None = None, 
                            required_operator_all_times: QuantumOperator | None = None,
                            nshots_ratio_integrand: float | None = None,
                            print_timers: bool = False,
                            skip_initial_time: bool = False,
                            save_data: bool = False,
                            no_output: bool = True,
                            ) -> DataSet:
        """
        Creates a classical shadow for the quantum state at each time in simulation_times.

        Parameters
        ----------
        initial_state : QuantumState
            initial state of the experiment
        simulation_times : list of floats
            times at which the classical shadow is created
        nshots_shadow : int
            total number of shots used for each time in simulation_times
        nshots_per_basis : int
            number of shots taken for each randomly sampled measurement basis
            Default is ``1``.
        nbases_cutoff : int
            if set, the number of randomly sampled measurement bases at each time is limited to nbases_cutoff
            Default is ``None``.
        required_operator_initial_time : QuantumOperator 
            quantum operator for which the exact expectation values are calculated at the initial time
            Default is None.
        required_operator_all_times : QuantumOperator
            quantum operator for which the exact expectation values are calculated at all times
            Default is None.
        nshots_ratio_integrand : float
            if set, the number of shots used is reduced by nshots_ratio_integrand 
            for all times except the first and the last.
            Default is ``None``.
        print_timers : bool 
            If True, print timers for each step.
            Default is False.
        skip_initial_time : bool
            If True, no measurements are taken at the initial time (simulation_times[0]).
            Default is False.
        save_data : bool
            If True, the data is saved to a file.
            Default is False.
        no_output : bool
            If True, no output is printed.
            Default is True.

        Returns
        -------
        DataSet
            data_set object containing the measurement data for each classical shadow (at each time)
        """
        if print_timers:
            ic("--------")
            ic("enter get_classical_shadow()")
            tm0 = tm.time()
        ##--------------------------------
        ### STEP 1 ### set required measurement settings
        N = self.Nions
        basis_options = ["X","Y","Z"]
        all_bases = ["".join(p) for p in it.product(basis_options, repeat=N)]
        ### create list of measurement settings
        measurement_settings = []
        for tinx, time in enumerate(simulation_times):
            ### step 1 ### set total number of shots at given time
            nshots = nshots_shadow
            ## reduce nshots by nshots_ratio_integrand
            if nshots_ratio_integrand is not None and time not in [simulation_times[0],simulation_times[-1]]:
                nshots = int(np.ceil(nshots_shadow * nshots_ratio_integrand))
            ## set nshots=1 if skip_initial_time is True and time is the initial time
            if skip_initial_time and time==simulation_times[0]:
                nshots = 1
            #----------
            ### step 2 ### determine required measurement bases (minimum 1 basis)
            if nshots>0:
                ### randomly sample measurement bases
                numbers = np.arange(3**N)
                ## choose random subset of numbers if nbases_cutoff is set (no replacement)
                if nbases_cutoff is not None and len(numbers) > nbases_cutoff:
                    numbers = np.random.choice(numbers, size=nbases_cutoff, replace=False)
                ## sample measurement bases with replacement from numbers
                ndraws = 1+nshots//nshots_per_basis
                draws = np.random.choice(numbers, size=ndraws, replace=True)
                # count how often each basis is sampled
                frequencies = np.bincount(draws, minlength=3**N)
                frequencies_dict = {all_bases[inx]:frequencies[inx] for inx in range(len(frequencies)) if frequencies[inx]>0}
                ### add measurment settings for each basis
                for base, nmeas in frequencies_dict.items():
                    ms = MeasurementSetting(initial_state=initial_state, simulation_time=time, measurement_basis=base, nshots=nmeas*nshots_per_basis)
                    measurement_settings.append(ms)
            #----------
            ### step 3 ### add setting for exact expectation values if required
            # get required operator
            req_op = None
            if tinx==0 and required_operator_initial_time is not None:
                req_op = required_operator_initial_time
            elif required_operator_all_times is not None:
                req_op = required_operator_all_times
            # add setting for exact expectation values to measurement_settings
            if req_op is not None:
                ms = MeasurementSetting(initial_state=initial_state, simulation_time=time, exact_observables=req_op)
                measurement_settings.append(ms)
            #----------
        tm1 = tm.time()
        if print_timers:
            pstr_get_classical_shadow = "time for setting measurement settings: {}".format(tm1-tm0)
            ic(pstr_get_classical_shadow)
        #--------------------------------
        ### STEP 2 ### create classical shadow
        dataset = self.simulate(initial_state, measurement_settings, save_data=save_data, no_output=no_output)
        # ## add empty data entries for times that were skipped
        # if skip_initial_time:
        #     if (initial_state, simulation_times[0]) not in dataset.data:
        #         data_entry = DataEntry(Nions=self.Nions, initial_state=initial_state, simulation_time=simulation_times[0])
        #         dataset.add_data_entry(data_entry)
        tm2 = tm.time()
        if print_timers:
            pstr_get_classical_shadow = "time for simulate(): {}".format(tm2-tm1)
            ic(pstr_get_classical_shadow)
        #--------------------------------
        return dataset

def getFinalState_QuTip(Hamiltonian: QuantumOperator,
                        initial_state: QuantumState,
                        times: list[float] | np.ndarray,
                        dissipators: list[Dissipator] | None = None,
                        rotating_frame: QuantumOperator | None = None,
                        ) -> list[qt.Qobj]:
    """ 
    Get the final state(s) via QuTip time-evolution.

    Simulates time-evolution with (Ham,Diss)-Liouvillian (QuTip)
    and returns the time-evolved quantum state at each time in times
    starting from the initial_state.

    Parameters
    ----------
    Hamiltonian : QuantumOperator
        Hamiltonian of the system
    initial_state : QuantumState
        initial state of the system
    times : list[float] or np.ndarray
        final times for which to calculate the time-evolved state
    dissipators : list of Dissipators 
        Dissipators that define the Liouville of the system.
        Default is None.
    rotating_frame : QuantumOperator 
        rotating frame applied to the final state
        i.e., psi(T) = exp(1j * rotating_frame * T) * psi(T)
        Default is None.

    Returns
    -------
    list of QuTip QuantumState objects
        time-evolved final state for each time in times
    """
    ### STEP 1 ### check inputs
    # check if either Hamiltonian or dissipators are set
    if Hamiltonian is None and dissipators is None:
        raise ValueError("Neither Hamiltonian nor dissipators set.")
    # check times
    if not isinstance(times, (list, np.ndarray)):
        ic(times)
        raise ValueError("times must be a list or numpy array.")
    # -------------------------------------
    ### STEP 2 ### create QuTip objects for Hamiltonian and dissipators
    ## Hamiltonian
    Hamiltonian_QuTip = sum([pauliop.coeff * getPauliOp_QuTip(pauliop.pauli_type) for pauliop in Hamiltonian.terms.values()])
    ## dissipator
    dissipators_QuTip = None
    if dissipators is not None:
        dissipators_QuTip = [to_QuTip(diss) for diss in dissipators]
    # -------------------------------------
    ### STEP 3 ### setup initial state with state preparation 
    # initial state (can be vector or density matrix)
    psi0 = initial_state.to_QuTip() 
    # -------------------------------------
    ### STEP 4 ### simulate time evolution with QuTip
    ## add t0=0 to times if not present
    # QuTip needs t0=0 in times
    addtzero = False
    times_QuTip = times
    if 0 not in times:
        times_QuTip = np.append(0,times)
        addtzero = True
    ### simulate time evolution
    mesolveOptions = qt.Options(store_states=True) #, atol=1e-10) 
    res = qt.mesolve(Hamiltonian_QuTip, psi0, times_QuTip, c_ops=dissipators_QuTip, options=mesolveOptions) #, progress_bar="tqdm") #, progress_bar=None, _safe_mode=True)
    tevolved_states = res.states
    if addtzero:
        tevolved_states = tevolved_states[1:]
    ### apply optional rotating frame
    if rotating_frame is not None:
        rotating_frame_qutip = to_QuTip(rotating_frame)
        tevolved_states = [(1j* rotating_frame_qutip* times[inx]).expm()*psi for inx, psi in enumerate(tevolved_states)]
    # -------------------------------------
    return tevolved_states

def getPauliOp_QuTip(PauliStr) -> qt.Qobj:
    """
    Create a QuTip Pauli product operator.

    Returns QuTip Pauli product operator
    for given PauliStr of the form "X1X2X3...XN",
    where X1...XN are Pauli chars ("X","Y","Z","M","P","I").

    Parameters
    ----------
    PauliStr : str
        Pauli product operator string (e.g. "X1X2X3...XN")

    Returns
    -------
    QuantumOperator
        QuTip Pauli product operator
    """
    pauliop = [qt.identity(2) for inx in range(len(PauliStr))]
    for pinx, pchar in enumerate(PauliStr):
        if pchar == "X":
            pauliop[pinx] = qt.sigmax()
        elif pchar == "Y":
            pauliop[pinx] = qt.sigmay()
        elif pchar == "Z":
            pauliop[pinx] = qt.sigmaz()
        elif pchar == "M":
            pauliop[pinx] = qt.sigmam()
        elif pchar == "P":
            pauliop[pinx] = qt.sigmap()
        elif pchar != "I":
            raise ValueError("Pauli operator {} not recognized".format(pchar))
    return qt.tensor(pauliop)

def getQuantumGate_QuTip(GateList: list, 
                        Nions: int,
                        ) -> qt.Qobj:
    """
    Create a QuTip unitary product operator from a list of gates.

    Returns QuTip unitary product operator
    for given list of tuples of the form "[[gatestr,gateangle],...]".
    Here, gatestr can be any of ("H","Rx%","Ry%","Rz%") where % is the ion index,
    and gateangle is a floats.
    Gates are applied from left to right, i.e. the first gate in the list is applied first.

    Parameters
    ----------
    GateList : list
        list of tuples of unitary operator chars and floats (e.g. [["Rx",0.1],["Ry",0.2],["Rz",0.3]])
    Nions : int
        number of ions

    Returns
    -------
    QuTip QuantumOperator
        QuTip unitary product operator
    """
    gates = []
    # get gates as QuTip operators
    for gatetuple in GateList:
        gstr = gatetuple[0]
        gateval = gatetuple[1]
        if gstr[:2] == "Rx":
            # gateval is a rotation angle
            ioninx = int(gstr[2:])
            gate = qt.tensor([qt.rx(gateval) if inx==ioninx else qt.identity(2) for inx in range(Nions)])
            # gate = qt.tensor([qt.rx(gateval) for inx in range(Nions)])
        elif gstr[:2] == "Ry":
            # gateval is a rotation angle
            ioninx = int(gstr[2:])
            gate = qt.tensor([qt.ry(gateval) if inx==ioninx else qt.identity(2) for inx in range(Nions)])
            # gate = qt.tensor([qt.ry(gateval) for inx in range(Nions)])
        elif gstr[:2] == "Rz":
            # gateval is a rotation angle
            ioninx = int(gstr[2:])
            gate = qt.tensor([qt.rz(gateval) if inx==ioninx else qt.identity(2) for inx in range(Nions)])
            # gate = qt.tensor([qt.rz(gateval) for inx in range(Nions)])
        elif gstr == "qop":
            # gateval is a quantum operator
            qop_QuTip = to_QuTip(gateval)
            gate = (-1j*qop_QuTip).expm()
        elif gstr == "bellpairs":
            # gateval is a list of pairs of qubits
            # gate = qt.tensor([qt.cnot()*qt.tensor([qt.hadamard_transform(),qt.identity(2)]) for pair in gateval])
            from qutip.qip.circuit import QubitCircuit, Gate
            # create the quantum circuit
            qc = QubitCircuit(Nions)
            for pairinx, pair in enumerate(gateval):
                qc.add_gate("SNOT", targets=pair[0])
                qc.add_gate("CNOT", controls=pair[0], targets=pair[1])
            # convert to QuTip gate
            gate = qc.propagators()[0]
        else:
            raise ValueError("Gate {} not recognized".format(gstr))
        ### append gate to list ###     
        gates.append(gate)
    # multiply all gates together from right to left
    total_gate = gates[0]
    for gate in gates[1:]:
        total_gate = gate*total_gate
    return total_gate



#---------------------#
### other functions ###
#---------------------#
def to_QuTip(operator: QuantumOperator | PauliOperator | Dissipator) -> qt.Qobj:
    """
    Converts a QuantumOperator or Dissipator to a QuTip operator.

    Returns a QuTip operator. 
    Input can be a QuantumOperator, a PauliOperator or a Dissipator.

    Parameters
    ----------
    operator : QuantumOperator or PauliOperator or Dissipator 
        operator to be converted to QuTip operator

    Returns
    -------
    QuTip operator
        The corresponding QuTip operator.
    """
    # ------------------------------
    ## convert Pauli operator
    if isinstance(operator, PauliOperator):
        operator = operator.to_quantum_operator()
    # ------------------------------
    ## convert Quantum operator
    if isinstance(operator, QuantumOperator):
        operator_QuTip = qt.Qobj()
        for pstr, pop in operator.terms.items():
            operator_QuTip += pop.coeff * getPauliOp_QuTip(pstr)  
    # ------------------------------
    ## convert Dissipator    
    if isinstance(operator, Dissipator):
        pop1_type = operator.diss_type[0]
        pop2_type = operator.diss_type[1]
        pop1 = PauliOperator(N=operator.N, pauli_type=pop1_type)
        pop2 = PauliOperator(N=operator.N, pauli_type=pop2_type)
        dissipator_QuTip1 = qt.lindblad_dissipator(to_QuTip(pop1), to_QuTip(pop2))
        dissipator_QuTip2 = qt.lindblad_dissipator(to_QuTip(pop2), to_QuTip(pop1))
        operator_QuTip = operator.coeff * (dissipator_QuTip1 + dissipator_QuTip2)
        # check if coeff is real and positive
        if not np.isreal(operator.coeff) or operator.coeff < 0:
            print("Dissipator = {}, but must have a nonnegative rate.".format(operator.str()))
    # --------------------------------
    return operator_QuTip

def get_expvals_QuTip(qstate: qt.Qobj | str,
                    qops: QuantumOperator | PauliOperator | list,
                    ) -> list[float]:
    """
    Returns expectation values of QuantumOperators qops 
    in a QuTip quantum state qstate.

    Parameters
    ----------
    qstate : QuTip quantum state
        quantum state
    qops : list of QuantumOperator or PauliOperator
        quantum operators

    Returns
    -------
    list of floats
        expectation values of qops in qstate
    """
    # check if qops is a quantum operator or Pauli operator
    if isinstance(qops, QuantumOperator) or isinstance(qops, PauliOperator):
        qops = [qops]
    # check if qops is a list of quantum operators or pauli operators
    if not isinstance(qops, list) or not all(isinstance(item, QuantumOperator) or isinstance(item, PauliOperator) for item in qops):
        raise ValueError("qops is not a list of QuantumOperators or PauliOperators.")
    # get expectation values
    expvals = []
    if isinstance(qstate, str):
        for qop in qops:
            expval = get_expval_from_Zproduct_state(qop,qstate)
            expvals.append(expval)
    elif isinstance(qstate, qt.Qobj):
        for qop in qops:
            operator = to_QuTip(qop)
            expval = qt.expect(operator, qstate)
            expvals.append(expval)
    else:
        raise ValueError("qstate is not a valid QuTip state or product state.")
    return expvals

## TODO: parallelize this function???
def SampleBitstrings_QuTip(qstate: qt.Qobj, 
                        basis_dict: dict, 
                        measurement_error: float | None = None,
                        print_timers: bool = False,
                        ) -> dict:
    """
    Sample nshots bitstrings from quantum state.

    Samples bitstrings from qstate
    in measurement basis=[x1x2x3...] where xi in [X,Y,Z]. 
    Returns bitstrings as integers.

    Parameters
    ----------
    qstate : QuTip quantum state
        quantum state to be sampled from
    basis_dict : dict
        dictionary of basis:nshots pairs,
        where basis is a string of Pauli operators (e.g. "XYZ")
        and nshots is the number of bitstrings to sample
    measurement_error : float
        probability of measurement error
        Default is None.
    print_timers : bool
        if True, print times for each step
        Default is False.

    Returns
    -------
    dict
        dictionary of basis:measurements pairs, 
        where measurements is a list of sampled bitstrings
    """
    if print_timers:
        pstr_SampleBitstrings_QuTip = "enter SampleBitstrings_QuTip(), nbases = {}".format(len(basis_dict.keys()))
        ic(pstr_SampleBitstrings_QuTip)
        timers_rot = []
        timers_meas = []
    #--------------------------------
    old_version = True
    if old_version:
        ### STEP 1 ### get probabilities from rotated state
        measurements = {}
        # TODO: parallelize this loop???
        for binx, basis in enumerate(basis_dict.keys()):
            ## get rotated state
            tm11 = tm.time()
            qstate_rot = changeMeasBasis_QuTip(qstate.copy(), basis, measurement_error=measurement_error)
            tm12 = tm.time()
            if print_timers:
                timers_rot.append(tm12-tm11)
                if binx==len(basis_dict.keys())//10:
                    pstr_SampleBitstrings_QuTip = "...avg time for rotation: {} [total est: {}]".format(np.mean(timers_rot), np.mean(timers_rot)*len(basis_dict.keys()))
                    ic(pstr_SampleBitstrings_QuTip)
            ## get probabilities
            if qstate.type == "ket":
                probs = np.abs(qstate_rot.data.toarray().flatten())**2
            elif qstate.type == "oper":
                probs = np.real(qstate_rot.diag())
            # check if probabilities sum to 1
            probs_sum = np.sum(np.abs(probs))
            # if np.abs(probs_sum - 1) > 1e-6:
            #     print("State probabilities sum to {} instead of 1".format(probs_sum))
            if np.abs(probs_sum - 1) > 1e-2:
                raise ValueError("State probabilities sum to {} instead of 1".format(probs_sum))
            # normalize probabilities
            probs = np.abs(probs)/probs_sum
            ## get measured bitstrings
            bitvals = np.random.choice(2**len(basis), size=basis_dict[basis], p=np.flip(probs))
            measurements[basis] = bitvals
            tm13 = tm.time()
            if print_timers:
                timers_meas.append(tm13-tm12)
                if binx==len(basis_dict.keys())//10:
                    pstr_SampleBitstrings_QuTip = "...avg time for sampling: {} [total est: {}]".format(np.mean(timers_meas), np.mean(timers_meas)*len(basis_dict.keys()))
                    ic(pstr_SampleBitstrings_QuTip)
        # if print_timers:
        #     tm2 = tm.time()
        #     pstr_SampleBitstrings_QuTip = "...time for multiplication: {}".format(tm2-tm1)
        #     ic(pstr_SampleBitstrings_QuTip)
    #--------------------------------
    new_version = False
    if new_version:
        for binx, basis in enumerate(basis_dict.keys()):
            ## get rotated state
            qstate_rot = changeMeasBasis_QuTip_mesolve(qstate.copy(), basis, measurement_error=measurement_error)
            ## get probabilities
            if qstate.type == "ket":
                probs = np.abs(qstate_rot.data.toarray().flatten())**2
            elif qstate.type == "oper":
                probs = np.real(qstate_rot.diag())
            # check if probabilities sum to 1
            probs_sum = np.sum(np.abs(probs))
            # if np.abs(probs_sum - 1) > 1e-6:
            #     print("State probabilities sum to {} instead of 1".format(probs_sum))
            if np.abs(probs_sum - 1) > 1e-2:
                raise ValueError("State probabilities sum to {} instead of 1".format(probs_sum))
            # normalize probabilities
            probs = np.abs(probs)/probs_sum
            ## get measured bitstrings
            bitvals = np.random.choice(2**len(basis), size=basis_dict[basis], p=np.flip(probs))
            measurements[basis] = bitvals
        # if print_timers:
        #     tm3 = tm.time()
        #     pstr_SampleBitstrings_QuTip = "...time for mesolve: {}".format(tm3-tm2)
        #     ic(pstr_SampleBitstrings_QuTip)
    #--------------------------------
    if print_timers:
        pstr_SampleBitstrings_QuTip = "exit SampleBitstrings_QuTip()"
        ic(pstr_SampleBitstrings_QuTip)
    return measurements

# TODO: this is the bottleneck for shadow method
def changeMeasBasis_QuTip(qstate: qt.Qobj, 
                        measBasis: str, 
                        measurement_error: float | None = None,
                        use_flips: bool = False,
                        print_timers: bool = False,
                        ):
    """
    Rotates quantum state qstate from computational
    to measurement basis given by measBasis.
    measurement_error is the relative rotation error. 
    There are 2 options to perform the rotations:
    we use the one that requires larger rotation angles,
    but no additional sigmax gates (see below).

    Parameters
    ----------
    qstate : QuTip quantum state
        quantum state to be rotated
    measBasis : str
        measurement basis in which to rotate qstate
    measurement_error : float
        probability of measurement error
        Default is None.
    use_flips : bool
        if True, use sigmax gates to flip qubits
        after rotation
        Default is False.
    print_timers : bool
        if True, print times for each step
        Default is False.

    Returns
    -------
    QuTip quantum state
        rotated quantum state
    """
    tm0 = tm.time()
    ### STEP 1 ### define single-qubit rotation operators 
    if measurement_error is None:
        rotopX = (-1j*(np.pi*3/4) * qt.sigmay()).expm()
        rotopY = (-1j*(np.pi*1/4) * qt.sigmax()).expm()
    else:
        rotopX = (-1j*(np.pi*3/4)*(1+measurement_error*np.random.normal()) * qt.sigmay()).expm()
        rotopY = (-1j*(np.pi*1/4)*(1+measurement_error*np.random.normal()) * qt.sigmax()).expm()
    #--------------------------------
    ### STEP 2 ### define rotation operator
    rotop = qt.tensor([rotopX if bchar == "X" else rotopY if bchar == "Y" else qt.identity(2) for bchar in measBasis])
    if print_timers:
        tm1 = tm.time()
        pstr_changeMeasBasis_QuTip = "...time for setting rotation operator: {}".format(tm1-tm0)
        ic(pstr_changeMeasBasis_QuTip)
    #--------------------------------
    ### STEP 3 ### apply rotations simultaneously (slightly slower?)
    if qstate.type == "ket":
        qstate = rotop*qstate
    if qstate.type == "oper":
        qstate = rotop*qstate*rotop.dag()
    if print_timers:
        tm2 = tm.time()
        pstr_changeMeasBasis_QuTip = "...time for rotation: {}".format(tm2-tm1)
        ic(pstr_changeMeasBasis_QuTip)
    #--------------------------------
    return qstate

#TODO this is not working
def changeMeasBasis_QuTip_mesolve(qstate: qt.Qobj, 
                                measBasis: str, 
                                measurement_error: float | None = None,
                                ) -> qt.Qobj:
    """
    Rotates quantum state qstate from computational
    to measurement basis given by measBasis.

    measurement_error is the relative rotation error. 
    There are 2 options to perform the rotations:
    we use the one that requires larger rotation angles,
    but no additional sigmax gates (see below).

    Parameters
    ----------
    qstate : QuTip quantum state
        quantum state to be rotated
    measBasis : str
        measurement basis in which to rotate qstate
    measurement_error : float 
        probability of measurement error
        Default is None.

    Returns
    -------
    QuTip quantum state
        rotated quantum state
    """
    ### STEP 1 ### define generator for single-qubit rotation
    if measurement_error is None:
        rotopX = (np.pi*3/4) * qt.sigmay()
        rotopY = (np.pi*1/4) * qt.sigmax()
    else:
        rotopX = (np.pi*3/4)*(1+measurement_error*np.random.normal()) * qt.sigmay()
        rotopY = (np.pi*1/4)*(1+measurement_error*np.random.normal()) * qt.sigmax()
    #--------------------------------
    ### STEP 2 ### define generator for rotation operator
    #TODO this is the bottleneck!
    rotop_list = []
    for inx, bchar in enumerate(measBasis):
        if bchar == "X":
            rotop_list.append(qt.expand_operator(rotopX, len(measBasis), targets=[inx]))
        elif bchar == "Y":
            rotop_list.append(qt.expand_operator(rotopY, len(measBasis), targets=[inx]))
    rotop = sum(rotop_list)
    #--------------------------------
    # ### STEP 2 ### define generator for rotation operator
    # #TODO this is the bottleneck!
    # rotop = None  
    # for inx, bchar in enumerate(measBasis):
    #     if bchar == "X":
    #         if rotop is None:
    #             rotop = qt.expand_operator(rotopX, N=len(measBasis), dims=[2]*len(measBasis), targets=[inx])
    #         else:
    #             rotop += qt.expand_operator(rotopX, N=len(measBasis), dims=[2]*len(measBasis), targets=[inx])
    #     elif bchar == "Y":
    #         if rotop is None:
    #             rotop = qt.expand_operator(rotopY, N=len(measBasis), dims=[2]*len(measBasis), targets=[inx])
    #         else:
    #             rotop += qt.expand_operator(rotopY, N=len(measBasis), dims=[2]*len(measBasis), targets=[inx])
    #--------------------------------
    # ### STEP 2 ### define generator for rotation operator
    # #TODO this is the bottleneck!
    # rotop = qt.qeye(2**len(measBasis))  # Identity matrix of the appropriate dimension
    # for inx, bchar in enumerate(measBasis):
    #     if bchar == "X":
    #         rotop += qt.tensor([qt.identity(2) if inx != jinx else rotopX for jinx in range(len(measBasis))])
    #     elif bchar == "Y":
    #         rotop += qt.tensor([qt.identity(2) if inx != jinx else rotopY for jinx in range(len(measBasis))])
    #--------------------------------
    ### STEP 3 ### apply rotations using mesolve
    # apply rotations
    qstate = qt.mesolve(rotop, qstate, [1], [], options=qt.Options(store_states=True)).states[-1]
    #--------------------------------
    return qstate

def get_fidelity_QuTip(rho1: QuantumState | qt.Qobj,
                    rho2: QuantumState | qt.Qobj,
                    ) -> float:
    """
    Get the fidelity between two quantum states.

    Parameters
    ----------
    state1 : QuantumState or QuTip.Qobj
        First quantum state
    state2 : QuantumState or QuTip.Qobj
        Second quantum state

    Returns
    -------
    float
        Fidelity between the two quantum states
    """
    ## convert to QuTip.Qobj if necessary
    if isinstance(rho1, QuantumState):
        rho1 = rho1.to_QuTip()
    if isinstance(rho2, QuantumState):
        rho2 = rho2.to_QuTip()
    ## get fidelity as Tr[sqrt(sqrt(rho1)*rho2*sqrt(rho1))]^2
    fidelity = qt.metrics.fidelity(rho1, rho2)
    ## get fidelity2 as Tr(rho1*rho2)
    # check if state is a density matrix
    
    #     ic(rho1.full(), rho2.full())
    #     fidelity2 = np.trace(np.einsum("ij,ji", rho1.full(), rho2.full()))
    # else:
    # get inner product of vectors rho1 and rho2
    fidelity2 = np.abs(np.dot(np.conjugate(np.array(rho1.full()).flatten()), np.array(rho2.full()).flatten()))
    # ic(fidelity, fidelity2)
    if not np.isclose(fidelity, fidelity2):
        raise ValueError("Fidelities 1 and 2 are not equal, but {} and {}".format(fidelity, fidelity2))
    return fidelity

def get_random_product_states(Nions: int,
                            nstates: int,
                            number_of_excitations: int | None = None,
                            state_preparation: str | None = None,
                            state_preparation_label: str | None = None,
                            state_preparation_error: float | None = None,
                            chosen_bases: list[str] = ["x","y","z"],
                            ) -> list:
    """
    Returns list of nstates QuantumStates on Nions qubits,
    each representing a product state with random excitations
    in a randomly chosen product basis.

    Parameters
    ----------
    Nions : int
        number of ions
    nstates : int
        number of states (chosen randomly)
    number_of_excitations : int 
        number of excitations in state
        Default is None.
    state_preparation : str
        After initialization, a product of single-qubit unitary operators 
        defined by a list of strings is applied to the initial state.
        Default is None.
    state_preparation_label : str
        Label for state preparation
        Default is None.
    state_preparation_error : float
        probability of a bit-flip error in the initial state
        Default is None.
    chosen_bases : list
        list of bases to choose from for each ion
        Default is ["x","y","z"].

    Returns
    -------
    list
        list of nstates QuantumStates
    """
    ### STEP 1 ### sample nstates random excitations
    if number_of_excitations is None:
        excitations = np.random.choice(["0","1"], size=(nstates,Nions), replace=True)
    else:
        up_indices = np.random.choice(range(Nions), size=(nstates,number_of_excitations), replace=True)
        excitations = np.zeros((nstates,Nions), dtype=str)
        excitations.fill("0")
        for inx in range(nstates):
            excitations[inx][up_indices[inx]] = "1"
    excitations = ["".join(exc) for exc in excitations]
    #--------------------------------
    ### STEP 2 ### sample nstates random product bases
    if state_preparation is None:
        all_bases = chosen_bases
    else:
        all_bases = ["z"]
    bases = np.random.choice(all_bases, size=(nstates,Nions), replace=True)
    bases = ["".join(base) for base in bases]
    #--------------------------------
    ### STEP 3 ### convert excitations and bases to QuantumStates
    states = []
    for inx in range(nstates):
        state = QuantumState(N=Nions, excitations=excitations[inx], basis=bases[inx], state_preparation=state_preparation, state_preparation_label=state_preparation_label, state_preparation_error=state_preparation_error)
        states.append(state)
    #--------------------------------
    return states

def get_product_states(Nions: int,
                    nstates: int,
                    max_nexc: int | None = None,
                    nexc: int | None = None,
                    state_preparation: str | None = None,
                    state_preparation_label: str | None = None,
                    state_preparation_error: float | None = None,
                    ) -> list:
    """
    Returns list of nstates QuantumStates on Nions qubits,
    each representing a product state with random excitations
    in the computational basis.

    Parameters
    ----------
    Nions : int 
        number of ions
    nstates : int
        number of states (chosen randomly)
    max_nexc : int 
        maximum number of allowed excitations in state
        Default is Nions.
    nexc : int
        number of excitations in state
        Default is None.
    state_preparation : str
        After initialization, a product of single-qubit unitary operators 
        defined by a list of strings is applied to the initial state.
        Default is None.
    state_preparation_label : str 
        Label for state preparation
        Default is None.
    state_preparation_error : float
        probability of a bit-flip error in the initial state
        Default is None.
        
    Returns
    -------
    list
        list of nstates QuantumStates
    """
    if max_nexc is None:
        max_nexc = Nions
    # --------------------------------------------------------
    ### STEP 1 ### get all allowed combinations of excitations
    combs = []
    if nexc is None:
        for i in range(max_nexc+1): 
            combs.extend(list(it.combinations(range(Nions),i))) #Nions//2-i)))
    else:
        combs = list(it.combinations(range(Nions),nexc))
    # --------------------------------------------------------
    ### STEP 2 ### convert combinations to QuantumStates
    all_states = []
    for comb in combs:
        exc_state = np.array(["0"]*Nions)
        exc_state[list(comb)] = "1"
        exc_state = "".join(exc_state)
        state = QuantumState(N=Nions, excitations=exc_state, state_preparation_label=state_preparation_label, state_preparation=state_preparation, state_preparation_error=state_preparation_error)
        all_states.append(state)
    # --------------------------------------------------------
    ### STEP 3 ### choose random subset of all QuantumStates with lenght nstates
    # check if there are enough states
    if nstates > len(all_states):
        raise ValueError("nstates={} is larger than the number of available states {}".format(nstates,len(all_states)))
    # choose random subset of states
    states = np.random.choice(all_states, size=nstates, replace=False)
    # --------------------------------------------------------
    return states

def get_measurement_settings(state: QuantumState, 
                            simulation_times: list, 
                            bases: list, 
                            nshots: dict, 
                            required_operator: QuantumOperator | None = None, 
                            nshots_ratio_integrand: bool = False,
                            ) -> list: 
    """
    Returns the measurement settings for a given state.

    Parameters
    ----------
    state : QuantumState
        quantum state for which to get the measurement settings
    simulation_times : list
        list of simulation times
    bases : list
        list of measurement bases
    nshots : dict
        dictionary of (time,base):nshots pairs
    required_operator : QuantumOperator
        required operator for exact expectation values
    nshots_ratio_integrand : bool
        if True, divide nshots by number of simulation times for all times except initial and final time

    Returns
    -------
    list
        list of measurement settings
    """
    measurement_settings = []
    for time in simulation_times:
        for base in bases:
            if nshots_ratio_integrand and time not in [simulation_times[0],simulation_times[-1]]:
                nshots_tmp = nshots[(time,base)]/len(simulation_times)
            else:
                nshots_tmp = nshots[(time,base)]
            ms = MeasurementSetting(initial_state=state, simulation_time=time, measurement_basis=base, nshots=nshots_tmp)
            measurement_settings.append(ms)
        if required_operator is not None:
            ms = MeasurementSetting(initial_state=state, simulation_time=time, exact_observables=required_operator)
            measurement_settings.append(ms)
    # --------------------------------------------------------
    return measurement_settings


