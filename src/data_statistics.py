"""
This module contains classes and methods for 
storage and statistical analysis of measurement data.
classes:
    - DataEntry
        A class for storing and analyzing measurement data for a given measurement setting.
    - DataSet
        A class for storing and analyzing measurement data for multiple measurement settings.
"""
from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt
from icecream import ic
import time as tm
from src.pauli_algebra import PauliOperator, QuantumOperator
from src.quantum_state import QuantumState
from src.measurement_setting import MeasurementSetting
# from memory_profiler import profile

class DataEntry:
    """
    Data entry for storing measurement data.

    Attributes
    ----------
    Nions : int
        Number of qubits in the system.
    initial_state : QuantumState
        Initial state of the quantum simulator
    simulation_time: float
        Simulation time of the quantum simulator
    measurements : dict, optional
        Dictionary of basis:shot pairs where 
        basis is a string of chars in ["X", "Y", "Z"] and
        shots is a list of bitstrings in the form of unsigned integers.
        Default is ``{}``.
    exact_expvals : dict, optional
        Exact expectation values is a dictionary of term:expval pairs where
        term is a string of chars in ["X", "Y", "Z", "I"] and
        expval is a list of floats.
        Default is ``{"I"*self.Nions:[1]}``.
    final_state : QuTip quantum state, optional
        Final state of the quantum simulator as QuTip quantum state
        Default is None.
    """
    def __init__(self, 
                Nions: int,
                initial_state: QuantumState | None = None,
                simulation_time: float | None = None,
                measurements: dict = {},
                exact_expvals: dict | None = None,
                final_state = None
                ):
        """
        Initialize a data entry.

        Parameters
        ----------
        Nions : int
            Number of qubits in the system
        initial_state : QuantumState
            Initial state of the quantum simulator
        simulation_time: float
            Simulation time of the quantum simulator
        measurements : dict, optional
            Dictionary of basis:shot pairs where 
            basis is a string of chars in ["X", "Y", "Z"] and
            shots is a list of bitstrings in the form of unsigned integers.
            Default is ``{}``.
        exact_expvals : dict, optional
            Exact expectation values is a dictionary of term:expval pairs where
            term is a string of chars in ["X", "Y", "Z", "I"] and
            expval is a list of floats.
            Default is ``{"I"*self.Nions:[1]}``.
        final_state : QuTip quantum state, optional
            Final state of the quantum simulator as QuTip quantum state
            Default is None.
        """
        if not isinstance(Nions, int) or Nions <= 0:
            raise TypeError("Nions is not a positive integer.")
        self._Nions = Nions
        self.initial_state = initial_state
        self.simulation_time = simulation_time
        self.measurements = measurements
        if exact_expvals is not None:
            self.exact_expvals = exact_expvals
        else:
            self.exact_expvals = {"I"*self.Nions:[1]}
        self.final_state = final_state
    ### --------------------- ###
    ### DataEntry attributes ###
    ### --------------------- ###
    @property
    def Nions(self):
        return self._Nions
    @Nions.setter
    def Nions(self, Nions):
        if not isinstance(Nions, int) or Nions <= 0:
            raise TypeError("Nions is not a positive integer.")
        self._Nions = Nions
    @property
    def initial_state(self):
        return self._initial_state
    @initial_state.setter
    def initial_state(self, initial_state):
        # check if initial_state is a QuantumState
        if initial_state is not None:
            if not isinstance(initial_state, QuantumState):
                raise TypeError("initial_state is not a QuantumState object.")
            if initial_state.N != self.Nions:
                raise ValueError("initial_state has a different number of ions than Nions.")
        self._initial_state = initial_state
    @property
    def simulation_time(self):
        return self._simulation_time
    @simulation_time.setter
    def simulation_time(self, simulation_time):
        # check if simulation_time is a non-negative real number
        if simulation_time is not None:
            if simulation_time < 0:
                raise ValueError("simulation_time is negative.")
            self._simulation_time = float(simulation_time)
        else:
            self._simulation_time = None
    @property
    def measurements(self):
        return self._measurements
    @measurements.setter
    def measurements(self, measurements):
        # check if measurements is a dict of basis:shots pairs where shots is a np.array of integers
        if measurements is not None:
            if not isinstance(measurements, dict):
                raise TypeError("measurements is not a dict.")
            for key in measurements.keys():
                ## transform measurements[key] to a list of unsigned integers
                if self.Nions <= 8:
                    measurements[key] = np.array(measurements[key], dtype=np.uint8)
                elif self.Nions <= 16:
                    measurements[key] = np.array(measurements[key], dtype=np.uint16)
                elif self.Nions <= 32:
                    measurements[key] = np.array(measurements[key], dtype=np.uint32)
                elif self.Nions <= 64:
                    measurements[key] = np.array(measurements[key], dtype=np.uint64)
                else:
                    raise ValueError("Nions={} is too large (max=64).".format(self.Nions))
        self._measurements = measurements
    @property
    def exact_expvals(self):
        return self._exact_expvals
    @exact_expvals.setter
    def exact_expvals(self, exact_expvals):
        # check if exact_expvals is a dict of basis:expval pairs where expval is a float or a list of floats
        if exact_expvals is not None:
            if not isinstance(exact_expvals, dict):
                raise TypeError("exact_expvals is not a dict.")
        # add "I"*Nions term if not there
        Istr = "I"*self.Nions
        if Istr not in exact_expvals:
            exact_expvals[Istr] = [1]
        self._exact_expvals = exact_expvals
    @property
    def final_state(self):
        return self._final_state
    @final_state.setter
    def final_state(self, final_state):
        if final_state is not None:
            # check if final state is a bitstring
            if isinstance(final_state, str):
                # check if final state is a valid bitstring
                if len(final_state) != self.Nions:
                    raise ValueError("final_state is not of length Nions.")
                if not all([x in ["0", "1"] for x in final_state]):
                    raise ValueError("final_state is not a valid bitstring.")
        # check if final state is a ket or a density matrix
        elif hasattr(final_state, 'type') and final_state.type != "ket" and final_state.type != "oper":
            raise TypeError("final_state is not a ket or a density matrix.")
        # check if dimension of final state is 2^Nions or 2^Nions x 2^Nions
        elif hasattr(final_state, 'shape') and final_state.shape[0] != 2**self.Nions:
            raise ValueError("final_state is not a state of Nions qubits.")
        self._final_state = final_state
    ### ------------------ ###
    ### DataEntry methods ###
    ### ------------------ ###
    def __str__(self):
        return self.str()
    def str(self,
            print_bases: bool = False) -> str:
        """
        Return a string representation of the DataEntry.

        Parameters
        ----------
        print_bases : bool, optional
            If True, prints the bases of the measurements.
            Default is False.

        Returns
        -------
        str
            string representation of the DataEntry
        """
        nshots_total = self.get_nruns()
        state_str = "[{}(t={})] with {} measurements".format(self.initial_state.str(),self.simulation_time, nshots_total)
        strout = state_str 
        if print_bases:
            meas_str = ["{}:{}, ".format(basis,len(meas)) for basis, meas in self.measurements.items()]
            strout = strout + ": \n [" + "".join(meas_str) + "]"
        return strout

    def copy(self) -> DataEntry:
        """
        Return a copy of the DataEntry.

        Returns
        -------
        DataEntry
            A new instance with the same attributes as this DataEntry.
        """
        return DataEntry(Nions = self.Nions, initial_state = self.initial_state, simulation_time = self.simulation_time, measurements = self.measurements.copy(), exact_expvals = self.exact_expvals.copy(), final_state = self.final_state)

    def get_size(self) -> int:
        """
        Return size of data entry in bytes.

        Returns
        -------
        int
            size of data entry in bytes
        """
        size = 0
        size += self.Nions.__sizeof__()
        size += self.initial_state.__sizeof__()
        size += self.simulation_time.__sizeof__()
        size += self.measurements.__sizeof__()
        size += self.exact_expvals.__sizeof__()
        size += self.final_state.__sizeof__()
        return size

    def combine(self, 
                other: DataEntry,
                ) -> DataEntry:
        """
        Combine two Data_entries into one.

        Parameters
        ----------
        other : DataEntry  
            DataEntry to be combined with this DataEntry.

        Returns
        -------
        DataEntry
            Combination of the two Data_entries
        """
        ### STEP 0 ### validate input 
        # check if other is a DataEntry object
        if not isinstance(other, DataEntry):
            raise TypeError("other is not a DataEntry object.")
        # check if other has the same number of ions as self
        if other.Nions != self.Nions:
            raise ValueError("other has a different number of ions than self.")    
        # check if other has the same initial state as self
        if other.initial_state != self.initial_state:
            raise ValueError("other has a different initial state than self.")
        # check if other has the same simulation time as self
        if other.simulation_time != self.simulation_time:
            raise ValueError("other has a different simulation time than self.")
        # -------------------------------------------
        ### STEP 1 ### combine data entries 
        data_entry_combined = self.copy()
        if other.measurements is not None:
            for key in other.measurements:
                if key in self.measurements:
                    data_entry_combined.measurements[key] = np.append(self.measurements[key], other.measurements[key])
                else:
                    data_entry_combined.measurements[key] = other.measurements[key]
        if other.exact_expvals is not None:
            for key in other.exact_expvals:
                if key in self.exact_expvals:  
                    data_entry_combined.exact_expvals[key] = np.append(self.exact_expvals[key], other.exact_expvals[key])
                else:
                    data_entry_combined.exact_expvals[key] = other.exact_expvals[key]
        if other.final_state is not None:
            data_entry_combined.final_state = other.final_state
        return data_entry_combined
    
    def get_nruns(self) -> int:
        """
        Return total number of measurements in the data entry.

        Returns
        -------
        int
            total number of measurements
        """
        nruns = 0
        for basis in self.measurements.keys():
            nruns += len(self.measurements[basis])
        return nruns

    def get_frequencies(self, 
                        nshots: int | None = None
                        ) -> dict:
        """
        Return the statistical frequencies ("histogram") of the measurements.

        Parameters
        ----------
        nshots : int, optional
            number of shots used to evaluate frequencies
            if nshots=None all available measurements are used
            Default is None.

        Returns
        -------
        dict
            dict of basis:freq pairs where freq is a np.array of 2^N integers
        """
        # check if measurements is not None
        if self.measurements is None:
            raise ValueError("No measurements in data_entry.")
        # calculate frequencies
        freq = {}
        for key in self.measurements.keys():
            measurements = self.measurements[key]
            if nshots is not None:
                measurements = measurements[0:nshots]
            # count the number of occurrences of each measurement
            freq[key] = np.bincount(measurements, minlength=2**self.Nions)
        return freq

    def evaluate_observable(
                self, 
                qop_list: list,
                nshots: int,
                evaluate_variance: bool = True,
                Gaussian_noise: bool = False,
                n_resampling: tuple = (0,1), 
                resampling_replace: bool = False, 
                min_nshots_per_term: int = 1, 
                print_timers: bool = False
                ):
        """ 
        Evaluate the sample mean and variance of a list of observables.

        For each observable in a list of QuantumOperators,
        calculates the sample mean of the expectation value 
        of that observable from the DataEntry, using nshots number of shots. 
        Optionally also evaluates the sample variance of the mean, i.e. var(O) = var(shots)/nshots.

        If individual terms of a QuantumOperator can be measured in parallel, 
        shots coming from the same measurement are used in order to keep the correlations 
        between the terms.
        If the total number of available shots are different for individual terms, 
        the number of shots used is cut to the maximum number of shots that 
        is available for all terms in the QuantumOperator.

        NOTE: get_shots() takes roughly 10 times longer than evaluating expectation values

        Parameters
        ----------
        qop_list : list of QuantumOperators or PauliOperators
            Observables to be evaluated.
        nshots : int or list of ints
            Number of shots used.
            If nshots=-1, all available shots are used.
            If nshots=0, the exact expectation value is returned.
            If nshots is a list, a list of expectation values is returned. 
        evaluate_variance : bool or str, optional
            If True, evaluate the variance of the mean of the observable via var(O) = var(shots)/nshots.
            If "binomial", evaluate the variance of the mean of the observable via var(O)=(1-<O>*2)/nshots.
            If False, the variance is not returned.
            Default is True.
        Gaussian_noise : bool
            if True, adds independent Gaussian noise to the exact measurements.
            Standard deviation is chosen as var/nshots.
            All terms in qop are assumed to be independent.
            Default is False.
        n_resampling : tuple (float,int) 
            Number of samples on which the mean (and variance) of the observable is evaluated.
            n_resampling[0] is the ratio of measurements excluded from the sample.
            n_resampling[1] is the number of samples taken.
            Default is ``(0,1)``.
        resampling_replace : bool 
            if True, resampling is done with replacement.
            Default is False.
        min_nshots_per_term : int 
            Minimum number of shots used for each term in each observable.
            If the number of shots for a term is smaller than min_nshots_per_term,
            the term is excluded from the evaluation (value is set to None).
            This is used to reduce shot noise.
            Default is ``1``.
        print_timers : bool 
            If True, print timers for each step.
            Default is False.

        Returns
        -------
        expval_list : np.array of real numbers
            Expectation values of each observable in qop_list, for each nshots.
            Dimension of expval is (len(qop_list), len(nshots), n_resampling[1]).
        variance_list : np.array of real numbers, optional
            Variance of of each observable in qop_list, for each nshots.
            Dimension of variance is (len(qop_list), len(nshots), n_resampling[1]).
            Only returned if evaluate_variance=True.
        """
        if print_timers:
            ic("enter DataEntry.evaluate_observable()")
            tm1 = tm.time()
        # -----------------------------------------------------
        ### STEP 1 ### setup nshots and observables
        ## transform nshots to list
        if isinstance(nshots, int):
            nshots = [nshots]
        ## get maximum number of shots
        nshots_max = np.max(nshots)
        if -1 in nshots:
            nshots_max = -1
        if Gaussian_noise:
            nshots_max = 0
        ## transform qop_list to quantum operators and remove zero terms
        qop_list = [(op.to_quantum_operator()).remove_zero_coeffs() for op in qop_list]
        if print_timers:
            tm2 = tm.time()
            pstr_entry_eval_obs = "setup in evaluate_observable() time: {}".format(tm2-tm1)
            ic(pstr_entry_eval_obs)
        # -----------------------------------------------------
        ### STEP 2 ### get shots for each term in qop_list if nshots_max!=0
        qop_shots_list = None
        qop_shots = None
        if nshots_max!=0:
            ## get required terms from qop_list
            required_terms = list({key for qop in qop_list for key in qop.terms.keys()})
            if print_timers:
                tm11 = tm.time()
                pstr_entry_eval_obs = "... of the following get required terms time: {}".format(tm11-tm2)
                ic(pstr_entry_eval_obs)
            ## get dict of term:shots pairs for all required terms 
            allterms_shots = self.get_shots(required_terms, nshots=nshots_max)
            if print_timers:
                tm12 = tm.time()
                pstr_entry_eval_obs = "... of the following get_shots function time: {}".format(tm12-tm11)
                ic(pstr_entry_eval_obs)
            ## get shots for each individual qop in qop_list
            qop_shots_list = [[] for qop in qop_list]
            for oinx, qop in enumerate(qop_list):
                terms_shots_list = [allterms_shots[term] for term in qop.terms]
                ## check if there are enough shots for each term
                if not np.any(terms_shots_list==None) and np.all([len(term_shots)>=min_nshots_per_term for term_shots in terms_shots_list]):
                    ### cut terms_shots_list to minimum length
                    # this throws away part of the data! 
                    # this is ok because it only throws away data if the individual
                    # QuantumOperator consists of several terms that have different number of shots
                    if nshots_max==-1:
                        minshots = np.min([len(term_shots) for term_shots in terms_shots_list])
                        terms_shots_list = np.array([term_shots[:minshots] for term_shots in terms_shots_list])
                    # evaluate individual shots of observable qop
                    terms_shots_list = np.array(terms_shots_list)
                    qop_shots = np.einsum("i,ij",qop.coeffs(),terms_shots_list)
                    # check if qops_shots contains nan or none
                    if np.any(np.isnan(qop_shots)) or np.any(qop_shots==None):
                        raise ValueError("qop_shots contains nan or None for qop {} in DataEntry {} at time {}".format(qop.str(), self.str(), self.simulation_time))
                    qop_shots_list[oinx] = qop_shots
            ## delete unused variables
            del required_terms, allterms_shots, terms_shots_list, qop_shots     
        if print_timers:
            tm3 = tm.time()
            pstr_entry_eval_obs = "get_shots in evaluate_observable() time: {}".format(tm3-tm2)
            ic(pstr_entry_eval_obs)
        # -----------------------------------------------------
        ### STEP 3 ### evaluate sample mean and variance for each nshots
        expval_list = np.zeros((len(qop_list),len(nshots),n_resampling[1])) #, dtype=complex)
        variance_list = None
        if evaluate_variance:
            variance_list = np.zeros((len(qop_list),len(nshots),n_resampling[1])) #, dtype=complex)
        for nsinx, nshotsx in enumerate(nshots):
            # ---------------------
            ### no resampling
            if nshotsx==0 or (Gaussian_noise and nshotsx>0):
                for qopinx, qop in enumerate(qop_list):
                    ### evaluate expectation value and variance
                    if nshotsx==0 and self.simulation_time==0:
                        expvals_dict = self.initial_state.evaluate_exact_expvals(qop)
                        expval = np.sum([qop.terms[term_key].coeff*np.mean(expvals_dict[term_key]) for term_key in qop.terms])
                        expval_list[qopinx,nsinx,:] = expval
                        if evaluate_variance:
                            varval = np.sum([qop.terms[term_key].coeff**2*(1-np.mean(expvals_dict[term_key])**2) for term_key in qop.terms])
                            variance_list[qopinx,nsinx,:] = varval
                    elif nshotsx==0 and self.simulation_time!=0:
                        expval = np.sum([qop.terms[term_key].coeff*np.mean(self.exact_expvals[term_key]) for term_key in qop.terms])
                        expval_list[qopinx,nsinx,:] = expval
                        if evaluate_variance:
                            varval = np.sum([qop.terms[term_key].coeff**2*(1-np.mean(self.exact_expvals[term_key])**2) for term_key in qop.terms])
                            variance_list[qopinx,nsinx,:] = varval
                    elif Gaussian_noise:
                        expval = np.sum([qop.terms[term_key].coeff*np.mean(self.exact_expvals[term_key]) for term_key in qop.terms])
                        varval = np.sum([qop.terms[term_key].coeff**2*(1-np.mean(self.exact_expvals[term_key])**2) for term_key in qop.terms])
                        # check if varval is positive
                        if varval < 0:
                            if varval < -1e-6:
                                raise ValueError("varval is {} for qop {}, time {} and nshots {}".format(varval, qop.str(), self.simulation_time, nshots))
                            varval = 0
                        # TODO: var assuming all terms are independent
                        varval = varval/(nshotsx*(1-n_resampling[0]))
                        errors = np.random.normal(0, np.sqrt(varval), size=n_resampling[1])
                        expvals = errors + expval
                        expval_list[qopinx,nsinx,:] = expvals
                        if evaluate_variance:
                            variance_list[qopinx,nsinx,:] = varval
            # ---------------------
            ### with resampling (Bootstrapping)
            elif nshotsx==int(nshotsx) and nshotsx>0:
                for sampinx in range(n_resampling[1]):
                    sample_indices = np.random.choice(range(nshotsx), size=int(np.ceil(nshotsx*(1-n_resampling[0]))), replace=resampling_replace)
                    for qopinx, qop in enumerate(qop_list):
                        if len(qop_shots_list[qopinx])<nshotsx:
                            raise ValueError("Only {} shots available instead of {} for qop {} in DataEntry {} at time {} with bases {}".format(len(qop_shots_list[qopinx]), nshotsx, qop.str(), self.str(), self.simulation_time, self.measurements.keys()))
                        else:
                            sample = qop_shots_list[qopinx][0:nshotsx][sample_indices]
                            expval_list[qopinx,nsinx,sampinx] = np.mean(sample)
                            if evaluate_variance:
                                if evaluate_variance=="binomial":
                                    variance_list[qopinx,nsinx,sampinx] = (1-np.mean(sample)**2)/len(sample)
                                else:
                                    variance_list[qopinx,nsinx,sampinx] = np.var(sample, ddof=1)/len(sample)
            # ---------------------
            ### use all available shots
            elif nshotsx==-1:
                for sampinx in range(n_resampling[1]):
                    for qopinx, qop in enumerate(qop_list):
                        maxnshots = len(qop_shots_list[qopinx])
                        if maxnshots==0:
                            expval_list[qopinx,nsinx,sampinx] = None
                            if evaluate_variance:
                                variance_list[qopinx,nsinx,sampinx] = None
                        else:
                            sample_indices = np.random.choice(range(maxnshots), size=int(np.ceil(maxnshots*(1-n_resampling[0]))), replace=resampling_replace)
                            sample = qop_shots_list[qopinx][sample_indices]
                            expval_list[qopinx,nsinx,sampinx] = np.mean(sample)
                            if evaluate_variance:
                                if evaluate_variance=="binomial":
                                    varval = (1-np.mean(sample)**2)/len(sample)
                                else:
                                    varval = np.var(sample, ddof=1)/len(sample)
                                variance_list[qopinx,nsinx,sampinx] = varval
            # ---------------------
            else:
                raise ValueError("nshots={}, but must be a positive integer or -1 or 0.".format(nshotsx))
        if print_timers:
            tm4 = tm.time()
            pstr_entry_eval_obs = "get mean and var in evaluate_observable() time: {}".format(tm4-tm3)
            ic(pstr_entry_eval_obs)
            ic("end DataEntry.evaluate_observable()")
        # -----------------------------------------------------
        return expval_list, variance_list

    # @profile
    def get_shots(self, 
                terms: list, 
                nshots: int = -1, 
                print_timers: bool = False
                ) -> dict:
        """
        Return a dict of term_key:shots pairs for all terms.

        Each shot is an individual evaluation of the corresponding term
        using the measured bitstrings stored in the DataEntry self.
        If the term can be measured from multiple bases stored in self, 
        shots are returned for all available bases.
        If term_key is "I"*N, 1 is returned for each shot.
        (This is because all terms are valid for "I"*N, 
        but non of the entries of each shot is used.
        Then np.prod([])=1 for each shot.)

        Parameters
        ----------
        terms : list of strings
            List of term_keys for which to evaluate the shots.
        nshots : int, optional
            Number of shots to be used for evaluation.
            If nshots=-1, all available shots are used.
            Default is ``-1``.
        print_timers : bool, optional
            If True, print timers for each step.
            Default is False.

        Returns
        -------
        dict
            Dict of term_key:shots for each term in qop.terms.
        """
        if print_timers:
            ic("--------")
            ic("enter NEW version of get_shots")
            tm3 = tm.time()
        # -----------------------------------------------------
        ### STEP 1 ### unpack measurements
        # unpack measurements into shape=(nbases, nmeasurements, Nions)
        measurements_unpacked = unpack_measurements(self.measurements, num_bits=self.Nions)
        tm31 = tm.time()
        if print_timers:
            pstr_entry_get_shots = "... of the following time for unpack measurements: {}".format(tm31-tm3)
            ic(pstr_entry_get_shots)
        # -----------------------------------------------------
        ### STEP 2 ### evaluate all terms
        bases_array = np.array([list(base.upper()) for base in self.measurements.keys()])
        shots_dict = {term: np.array([]) for term in terms}
        for tinx, term in enumerate(terms):
            # list of Pauli chars in term, shape=(Nions,)
            term_array = np.array(list(term))
            # relevant indices where term is not "I", shape = (Nions,)
            relevant_indices = np.where(term_array != "I")[0]
            # bases that are equal to term in relevant indices, shape=(nbases,)
            match_array = np.all(bases_array[:, relevant_indices] == term_array[relevant_indices], axis=1)
            # shape measurements = (nbases, nmeasurements, Nions)
            product = np.prod(measurements_unpacked[match_array][:, :, relevant_indices], axis=-1, dtype=np.int8).flatten()
            # store shots in dictionary
            shots_dict[term] = product[product!=0][0:nshots]
        tm32 = tm.time()
        if print_timers:
            pstr_entry_get_shots = "... of the following time for evaluate products: {}".format(tm32-tm31)
            ic(pstr_entry_get_shots)
        tm4 = tm.time()
        if print_timers:
            pstr_entry_get_shots = "Total time for get_shots NEW: {}".format(tm4-tm3)
            ic(pstr_entry_get_shots)
            ic("--------")
        # -----------------------------------------------------
        return shots_dict

    def divergence(self, 
                other: DataEntry, 
                method: str = "ls", 
                nshots: int | None = None,
                ):
        """
        Return the divergence between two data entries.

        The divergence is calculated from the distribution of measurements outcomes,
        using the method specified by the method argument.

        Parameters
        ----------
        method : str, optional
            the method used to calculate the divergence
            Default is "ls".
            options are
                - "ls" (least squares)
                - "ls_vec" (least squares vectorized)
                - "kl" (Kullback-Leibler)
                - "js" (Jensen-Shannon)
                - "ml" (maximum likelihood)
        nshots : int, optional
            the number of shots to use for the least squares method
            Default is None.

        Returns
        -------
        float or list of floats
            divergence
        """
        # check if other is a data entry
        if not isinstance(other, DataEntry):
            raise TypeError("other is not a data entry.")
        # check if other has the same number of ions
        if self.Nions != other.Nions:
            raise ValueError("other has a different number of ions.")
        # check if other has the same initial state
        if self.initial_state != other.initial_state:
            raise ValueError("other has a different initial state.")
        # compare measurements for each basis
        div = 0
        if method=="ls_vec":
            div = []
        for basis in self.measurements.keys():
            # get distribution of bitstrings
            dist1 = np.bincount(self.measurements[basis][0:nshots], minlength = 2**self.Nions) / len(self.measurements[basis][0:nshots])
            dist2 = np.bincount(other.measurements[basis][0:nshots], minlength = 2**self.Nions) / len(other.measurements[basis][0:nshots])
            # compare distributions using method
            if method=="ls": # least-squares
                div_tmp = np.sum((dist1-dist2)**2)**0.5
                div += div_tmp 
            elif method=="ls_vec": # least-squares vector
                div_tmp = dist1-dist2
                div.extend(div_tmp)
            elif method=="kl": # Kullback-Leibler
                div_tmp = np.sum(dist1*np.log(dist1/dist2))
                div += div_tmp 
            elif method=="js": # Jensen-Shannon
                div_tmp = 0.5*np.sum(dist1*np.log(dist1/(0.5*(dist1+dist2)))) + 0.5*np.sum(dist2*np.log(dist2/(0.5*(dist1+dist2))))
                div += div_tmp 
            elif method=="ml": # maximum likelihood
                div_tmp = np.sum(dist1*np.log(dist1/dist2)) + np.sum(dist2*np.log(dist2/dist1))
                div += div_tmp 
        return div
    
    def crop_measurements(self,
                        nshots: int,
                        ) -> None:
        """
        Crop the measurements in the DataEntry to the first nshots entries.

        Parameters
        ----------
        nshots : int
            number of shots to keep
        """
        for basis in self.measurements:
            self.measurements[basis] = self.measurements[basis][0:nshots]

    def sample_measurements(self, nshots, random=False, seed=None):
        """
        Sample nshots measurements from the DataEntry without replacement.

        The probability of sampling a measurement in a given basis is 
        proportional to the number of shots in that basis.

        Parameters
        ----------
        nshots : int
            Total number of measurements to be sampled from the DataEntry.
        random : bool, optional
            if True, samples measurements randomly
            if False, samples measurements sequentially
            Default is False.
        seed : int, optional
            Random seed for sampling. Default is None.

        Returns
        -------
        DataEntry
            DataEntry with sampled measurements
        """
        # -----------------------------------------------
        ### STEP 1 ### choose number of shots for each basis
        nshots_total = self.get_nruns()
        probs_per_basis = np.array([len(self.measurements[basis])/nshots_total for basis in self.measurements])
        basis_samples = np.random.choice(range(len(self.measurements)), size=nshots, p=probs_per_basis)
        basis_counts = np.bincount(basis_samples, minlength=len(self.measurements))
        nshots_per_basis = {}
        for binx, basis in enumerate(self.measurements):
            nshots_per_basis[basis] = basis_counts[binx]
        # -----------------------------------------------
        ### STEP 2 ### create data_entry with sampled data
        sampled_data = DataEntry(Nions=self.Nions, initial_state=self.initial_state, simulation_time=self.simulation_time)
        sampled_data.exact_expvals = self.exact_expvals
        ## sample measurements
        if nshots > nshots_total:
            measurements = self.measurements
            sampled_data.measurements = measurements
        else:
            for basis in self.measurements:
                nshots_basis = nshots_per_basis[basis]
                if nshots_basis > len(self.measurements[basis]):
                    sampled_data.measurements[basis] = self.measurements[basis]
                else:
                    if random:
                        if seed is not None:
                            np.random.seed(seed)
                        sampled_indices = np.random.choice(range(len(self.measurements[basis])), size=nshots_basis, replace=False)
                        sampled_measurements = self.measurements[basis][sampled_indices]
                        sampled_data.measurements[basis] = sampled_measurements
                    else:
                        sampled_data.measurements[basis] = self.measurements[basis][0:nshots_basis]
        # -----------------------------------------------
        return sampled_data

    def fix_quantum_states(self) -> None:
        """
        Check if initial_states are QuantumStates and convert them if necessary.
        """
        if not isinstance(self.initial_state, QuantumState):
            excitations = self.initial_state
            state = QuantumState(N=self.Nions, excitations=excitations)
            self.initial_state = state

    def add_exact_expvals_at_0(self, qop) -> None:
        """
        Add the exact expectation value of the observable qop at time ``0`` to the DataEntry.

        Parameters
        ----------
        QuantumOperator
            observable for which to add the exact expectation value
        """
        # check if simulation time is 0
        if self.simulation_time == 0:
            # check if qop is a QuantumOperator
            if not isinstance(qop, QuantumOperator):
                raise TypeError("qop is not a QuantumOperator object.")
            state = self.initial_state
            exact_expvals = state.evaluate_exact_expvals(qop)
            self.exact_expvals.update(exact_expvals)

    def extend_to_larger_system(self, extension_factor) -> DataEntry:
        """
        Extend the DataEntry from Nions to Nions*extension_factor.

        The larger system consists of extension_factor non-interacting 
        subsystems each of size Nions.
        The measurements are generated by randomly concatenating 
        measurements from the original DataEntry.
        This reduces the total number of shots for each measurement basis 
        by a factor of extension_factor.

        Parameters
        ----------
        extension_factor : int
            number of non-interacting subsystems

        Returns
        -------
        DataEntry
            DataEntry extended to the larger system but with reduced number of shots.
        """
        ### STEP 0 ### check if extension_factor is a positive integer
        if not isinstance(extension_factor, int):
            raise TypeError("extension_factor is not an integer.")
        if extension_factor <= 0:
            raise ValueError("extension_factor is not positive.")
        #--------------------------------
        ### STEP 1 ### setup extended data entry
        extended_data_entry = DataEntry(Nions=self.Nions*extension_factor)
        extended_data_entry.simulation_time = self.simulation_time
        #--------------------------------
        ### STEP 2 ### get extended initial state
        # excitations
        excitations_extended = self.initial_state.excitations * extension_factor
        # basis
        basis_extended = self.initial_state.basis * extension_factor
        # set initial state
        initial_state_extended = QuantumState(N=self.Nions*extension_factor, excitations=excitations_extended, basis=basis_extended, state_preparation=self.initial_state.state_preparation, state_preparation_label=self.initial_state.state_preparation_label)
        # add initial state to extended data entry
        extended_data_entry.initial_state = initial_state_extended
        #--------------------------------
        ### STEP 3 ### extend measurements
        ### extract all measurements and bases
        all_measurements = np.array([[basis, int(measurement)] for basis in self.measurements for measurement in self.measurements[basis]])
        np.random.shuffle(all_measurements)
        n_measurements_total = len(all_measurements)
        n_measurements_reduced = n_measurements_total//extension_factor
        ### get extended measurements
        measurements_extended = {}
        for minx in range(n_measurements_reduced):
            measurements = all_measurements[minx*extension_factor:(minx+1)*extension_factor]
            if len(measurements) != extension_factor:
                break
            # get extended basis
            basis_extended = "".join(measurements[:,0])
            # get extended measurements
            unpacked_bits = unpackbits(np.array(measurements[:,1],dtype=int), self.Nions)
            unpacked_bit_extended = np.concatenate(unpacked_bits)
            packed_bits_extended = packbits(unpacked_bit_extended)
            # add to measurements_extended
            if basis_extended not in measurements_extended:
                measurements_extended[basis_extended] = np.array([packed_bits_extended])
            else:
                measurements_extended[basis_extended] = np.append(measurements_extended[basis_extended], packed_bits_extended)
        # add measurements to extended data entry
        extended_data_entry.measurements = measurements_extended
        #--------------------------------
        ### STEP 4 ### extend exact expectation values
        exact_expvals_extended = {}
        for term_key in self.exact_expvals:
            value = self.exact_expvals[term_key]
            ### in-subsystem terms
            for inx in range(extension_factor):
                # extend term_key
                term_key_extended = "I"*self.Nions*inx + term_key + "I"*self.Nions*(extension_factor-inx-1)
                # update expectation values
                exact_expvals_extended[term_key_extended] = value
        # add exact expectation values to extended data entry
        extended_data_entry.exact_expvals = exact_expvals_extended
        #-------------------------------- 
        return extended_data_entry       



class DataSet:
    """
    DataSet for storing multiple DataEntrys in a dictionary.

    A DataSet stores multiple DataEntrys in a dictionary,
    where the key is the tuple (initial_state, simulation_time).

    Attributes
    ----------
    Nions : int
        number of qubits of the system
    data : dict, optional
        dictionary of key:DataEntry pairs
        key is a tuple (initial_state, simulation_time)
        Default is ``{}``.
    """
    def __init__(self, Nions, **kwargs):
        """
        Initialize a DataSet.

        Parameters
        ----------
        Nions : int
            number of qubits of the system
        data : dict, optional
            dictionary of key:DataEntry pairs
            key is a tuple (initial_state, simulation_time)
            Default is ``{}``.
        """
        self._Nions = Nions
        self.data = kwargs.get("data", {})
    ### ------------------- ###
    ### DataSet attributes ###
    ### ------------------- ###
    @property
    def Nions(self):    
        return self._Nions
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, data):
        # check if data is a dictionary of data_entry objects
        if data is not None:
            if not isinstance(data, dict):
                raise TypeError("data is not a dictionary.")
            for key, value in data.items():
                if not isinstance(value, DataEntry):
                    raise TypeError("data is not a dictionary of data_entry objects.")
        self._data = data
    ### ---------------- ###
    ### DataSet methods ###
    ### ---------------- ###
    def __str__(self):
        return self.str()
    def str(self, 
            print_bases: bool = False
            ) -> str:
        """
        Return a string representation of the DataSet.

        Parameters
        ----------
        print_bases : bool, optional
            If True, prints the bases of the measurements.
            Default is False.

        Returns
        -------
        str
            string representation fo the DataSet
        """
        nshots_total = self.get_nruns()
        title_str = "N={} with {} Data_entries and total number of {} shots:".format(self.Nions, len(self.data), nshots_total)
        entries_str = [entry.str(print_bases=print_bases) for entry in self.data.values()]
        return "\n".join([title_str]+entries_str)

    def copy(self) -> DataSet:
        """
        Return a copy of the data set.

        Returns
        -------
        DataSet
            A new instance with the same attributes as this DataSet.
        """
        data = {}
        for key in self.data:
            data[key] = self.data[key].copy()
        return DataSet(Nions=self.Nions, data=data)

    def add_data_entry(self, 
                    data_entry: DataEntry
                    ) -> None:
        """
        Add a DataEntry to the DataSet.

        Parameters
        ----------
        data_entry : DataEntry
            DataEntry to be added to self
        """
        # check if data_entry is a valid data_entry object
        if not isinstance(data_entry, DataEntry):
            raise ValueError("DataEntry is not a valid DataEntry object.")
        # check if number of ions is the same in the data set and the data_entry
        if self.Nions != data_entry.Nions:
            raise ValueError("Nions is not the same in the DataSet (N={}) and the DataEntry (N={}).".format(self.Nions, data_entry.Nions))
        # check if initial_state is a QuantumState()
        if not isinstance(data_entry.initial_state, QuantumState):
            excitations = data_entry.initial_state
            state = QuantumState(N=self.Nions, excitations=excitations)
            data_entry.initial_state = state
        # check if there is already an entry of the same type
        # if yes, combine the two entries
        # else, add new entry
        if (data_entry.initial_state, data_entry.simulation_time) in self.data:
            data_entry_old = self.data[(data_entry.initial_state, data_entry.simulation_time)]
            data_entry_combined = data_entry_old.combine(data_entry)
            self.data[(data_entry.initial_state, data_entry.simulation_time)] = data_entry_combined
        else:
            self.data[(data_entry.initial_state, data_entry.simulation_time)] = data_entry

    def add_data_set(self, 
                    other: DataSet,
                    ) -> None:
        """
        Add each DataEntry of the DataSet other to this DataSet.

        Parameters
        ----------
        other : DataSet
            DataSet to be added
        """
        # check if other is a valid data set
        if not isinstance(other, DataSet):
            raise ValueError("other is not a valid data set.")
        # add all entries of other to self (using other as a list, not a dictionary)
        for entry in other.data.values():
            self.add_data_entry(entry)

    def scramble_data(self,
                    seed: bool = False
                    ) -> None:
        """
        Randomly shuffle the measurement shots in each DataEntry.

        Parameters
        ----------
        seed : bool, optional
            if True, sets the seed of the random number generator
            Default is False.
        """
        # loop over all data entries
        seedinx=1
        for entry in self.data.values():
            # loop over all measurement bases
            for basis in entry.measurements:
                # create copy of measurements
                measurements = entry.measurements[basis]
                # shuffle measurements
                if seed:
                    np.random.seed(seedinx)
                np.random.shuffle(measurements)
                entry.measurements[basis] = measurements
                seedinx += 1

    def get_size(self) -> int:
        """
        Return the size of the DataSet in bytes.

        Returns
        -------
        int
            size of this DataSet in bytes
        """
        size = 0
        for entry in self.data.values():
            size += entry.get_size()
        return size
    
    def get_nruns(self) -> int:
        """
        Return the total number of measurement shots in the DataSet.

        Returns
        -------
        int
            total number of measurements in this DataSet
        """
        nruns = 0
        for entry in self.data.values():
            nruns_entry = entry.get_nruns()
            nruns += nruns_entry
        return nruns

    def initial_states(self, 
                    time: float | None = None
                    ) -> list:
        """
        Return a list of all different initial states in the data set.

        Parameters
        ----------
        time : float, optional
            If given, only returns initial states for the given time.
            Default is None.

        Returns
        -------
        list of QuantumState objects
            list of all different QuantumStates in the DataSet
        """
        #----------------------------------
        ### STEP 1 ### get all initial states
        initial_states = []
        for entry in self.data.values():
            if time is None or entry.simulation_time==time:
                state = entry.initial_state
                # convert to QuantumState() if necessary
                if not isinstance(state, QuantumState):
                    state = QuantumState(N=len(state),excitations=state)
                initial_states.append(state)
        return list(set(initial_states))

    def simulation_times(self, 
                        state: QuantumState | None = None
                        ) -> list:
        """
        Return an ordered list of all different simulation times in the DataSet.

        Parameters
        ----------
        state : QuantumState, optional
            If given, only returns simulation times for the given state.
            Default is None.

        Returns
        -------
        list
            ordered list of all different simulation times in the data set
        """
        #----------------------------------
        ### STEP 1 ### get all simulation times
        simulation_times = []
        for entry in self.data.values():
            if state is None or entry.initial_state==state:
                simulation_times.append(entry.simulation_time)
        simulation_times = list(np.sort(list(set(simulation_times))))
        return simulation_times

    def measurement_bases(self, 
                        state: QuantumState | None = None, 
                        time: float | None = None
                        ) -> list:
        """
        Return a list of all different measurement bases in the DataSet, 
        and the number of shots for each basis.

        Parameters
        ----------
        state : QuantumState, optional
            If given, only returns measurement bases for the given state.
            Default is None.
        time : float, optional
            If given, only returns measurement bases for the given time.
            Default is None.

        Returns
        -------
        list of str
            list of strings that describe all different measurement bases and 
            corresponding number of shots in the DataSet
        """
        measurement_bases = []
        for entry in self.data.values():
            if state is None or entry.initial_state==state:
                if time is None or entry.simulation_time==time:
                    if entry.measurements is not None:
                        for basis_key in entry.measurements:
                            return_key = basis_key + ": ns={}".format(len(entry.measurements[basis_key]))
                            measurement_bases.append(return_key)
        return list(set(measurement_bases))

    def select_entries(self, 
                    states: list[QuantumState] | None = None, 
                    times: list[float] | None = None, 
                    bases: list[str] | None = None,
                    ) -> DataSet:
        """
        Return a DataSet with selected entries from the original data_set.

        Parameters
        ----------
        states : list of QuantumState objects, optional
            If given, only returns entries for the given initial states.
            Default is None.
        times : list of floats, optional
            If given, only returns entries for the given simulation times.
            Default is None.
        bases : list of str, optional
            If given, only returns entries for the given measurement bases.
            Default is None.

        Returns
        -------
        DataSet
            New DataSet with selected entries
        """
        #----------------------------------
        selected_data_set = DataSet(Nions=self.Nions)
        for entry in self.data.values():
            # check if entry is selected
            if states is not None and entry.initial_state not in states:
                continue
            if times is not None and entry.simulation_time not in times:
                continue
            if bases is not None and not any(key in bases for key in entry.measurements.keys()):
                continue
            # add entry to selected data set
            selected_data_set.add_data_entry(entry)
        #----------------------------------
        return selected_data_set

    def get_measurement_settings(self, 
                                nshots_replace: int | None = None, 
                                add_exact_expvals: QuantumOperator | None = None, 
                                states: list[QuantumState] | None = None, 
                                times: list[float] | None = None
                                ) -> dict:
        """
        Return measurement settings for the DataSet.

        Creates a dict of MeasurementSetting objects that can be used to 
        recreate all measurements in the DataSet using a QuantumSimulator.
        This also includes the exact expectation values.

        Parameters
        ----------
        nshots_replace : int, optional
            If given, replaces all nshots in the MeasurementSettings with nshots_replace
            Default is None.
        add_exact_expvals : QuantumOperator, optional
            If given, adds the exact expectation values of the given QuantumOperator to the MeasurementSettings
            Default is None.
        states : list of QuantumStates, optional
            If given, only returns MeasurementSettings for the given initial states.
            Default is self.initial_states().
        times : list of floats, optional
            If given, only returns MeasurementSettings for the given times.
            Default is self.simulation_times().

        Returns
        -------
        dict
            dict of state:list pairs for all measurements in the DataSet,
            where state is the initial state (QuantumState) and
            the list contains MeasurementSettings for the given state,
            including the required exact expectation values.
        """
        if states is None:
            states = self.initial_states()
        if times is None:
            times = self.simulation_times()
        #-------------------------------------------------------------
        measurement_settings = {}
        for entry in self.data.values():
            state = entry.initial_state
            time = entry.simulation_time
            if state in states and time in times:
                ### STEP 1 ### add Pauli measurements
                for basis in entry.measurements:
                    nshots = len(entry.measurements[basis])
                    if nshots_replace is not None:
                        nshots = nshots_replace
                    # set MeasurementSetting
                    setting = MeasurementSetting(initial_state=state, simulation_time=time, measurement_basis=basis, nshots=nshots)
                    # append to dict of measurement_settings
                    if state not in measurement_settings:
                        measurement_settings[state] = [setting]
                    else:
                        measurement_settings[state].append(setting)
                # ----------------------------------
                ### STEP 2 ### add exact expectation values
                exact_expvals = entry.exact_expvals
                if add_exact_expvals is not None:
                    tmp_dict = {term:1 for term in add_exact_expvals.terms.keys()}
                    exact_expvals.update(tmp_dict)
                ### convert exact_expvals to QuantumOperator if necessary
                exact_observables = None
                terms = {key:1 for key in exact_expvals if key!="I"*state.N}
                if len(terms)>0:
                    exact_observables = QuantumOperator(N=entry.Nions, terms=terms)
                ### add measurement_setting for exact_observables, if there is no measurement basis given
                if exact_observables is not None:
                    setting = MeasurementSetting(initial_state=state, simulation_time=time, exact_observables=exact_observables)
                    if state not in measurement_settings:
                        measurement_settings[state] = [setting]
                    else:
                        measurement_settings[state].append(setting)
        #-------------------------------------------------------------
        return measurement_settings 

    # @profile
    def evaluate_observable(self, 
                            state: QuantumState,
                            time: float, 
                            qop_list: list[QuantumOperator | PauliOperator],
                            nshots: int | list[int],
                            evaluate_variance: bool = False, 
                            Gaussian_noise: bool = False, 
                            use_exact_initial_values: bool = False, 
                            n_resampling: tuple[float,int] = (0,1), 
                            resampling_replace: bool = False,
                            min_nshots_per_term: int = 1
                            ):
        """ 
        Evaluate the sample mean of a list of observables from the DataSet.

        Returns the expectation value for given initial state at given time, 
        using given number of shots for each QuantumOperator in qop_list.
        Optionally, also evaluates the sample variance.

        Parameters
        ----------
        state : QuantumState
            initial state
        time : float
            simulation time
        qop_list : list of QuantumOperators or PauliOperators
            Observables to be evaluated.
            Expectation values are returned individually for each operator in qop_list.
            If the individual terms in a QuantumOperator can be measured in parallel, 
            the correlations between the measurements are taken into account.
        nshots : int or list of ints
            number of shots to use for the least squares method
            if nshots=-1, all available shots are used (cuts the shots to the minimum number of shots for all terms)
            If nshots=0, the exact expectation value is returned.
            If nshots is a list, a list of expectation values is returned.
        evaluate_variance : bool or str, optional
            If True, evaluate the variance of the observable via resampling.
            If "binomial", evaluate the variance of the observable via var(O)=(1-<O>*2)/nshots.
            Default is False.
        Gaussian_noise : bool, optional
            if True, add independent Gaussian noise with variance var/nshots
            to the exact measurements, instead of sampling measurements. 
            Default is False.
        use_exact_initial_values : bool
            if True, return exact expectation values for time=0
            Default is False.
        n_resampling : tuple
            n_resampling[0] is the ratio of measurements excluded from the sample.
            n_resampling[1] is the number of samples taken.
            Default is (0,1).
        resampling_replace : bool
            if True, samples are taken with replacement.
            Default is False
        min_nshots_per_term : int
            Minimum number of shots per term in qop_list.
            If the number of shots for a term is smaller than min_nshots_per_term,
            the term is excluded from the evaluation (value is set to None).
            Default is ``1``.

        Returns
        -------
        expval_list : np.array
            Expectation values of each observable in qop_list, for each nshots.
            Dimension of expval is (len(qop_list), len(nshots), n_resampling[1]).
        variance_list : np.array, optional
            Variance of of each observable in qop_list, for each nshots.
            Dimension of variance is (len(qop_list), len(nshots), n_resampling[1]).
            Returned only if evaluate_variance==True.
        """
        #----------------------------------
        ### STEP 1 ### get number of shots
        if use_exact_initial_values and time==0:
            nshots = np.zeros(np.shape(nshots))
        #----------------------------------
        ### STEP 2 ### get list of observables to evaluate
        if not isinstance(qop_list, list):
            raise ValueError("qop_list is {}, but must be a list of QuantumOperators or PauliOperator.".format(type(qop_list)))
        if not all([isinstance(op, QuantumOperator) or isinstance(op, PauliOperator) for op in qop_list]):
            raise ValueError("observable is not a valid list of QuantumOperators or PauliOperators.")
        #----------------------------------
        ### STEP 3 ### get data entry
        if (state, time) not in self.data:
            raise ValueError("No DataEntry for given state {} and time {}".format(state.str(), time))
        data_entry = self.data[(state, time)]
        # ----------------------------------
        ### STEP 4 ### evaluate observables
        expval_list, variance_list = data_entry.evaluate_observable(qop_list, nshots, Gaussian_noise=Gaussian_noise, n_resampling=n_resampling, resampling_replace=resampling_replace, min_nshots_per_term=min_nshots_per_term, evaluate_variance=evaluate_variance)
        #----------------------------------
        return expval_list, variance_list

    def crop_measurements(self, 
                        nshots: int,
                        states: list[QuantumState] | None = None, 
                        times: list[int] | None = None
                        ) -> None:
        """
        Delete all measurements that exceed nshots from the Data_entries
        defined by states and times.

        Parameters
        ----------
        nshots : int
            Number of shots to be kept.
        states : list of QuantumStates
            Initial states for which the measurements are to be cropped.
            Default is self.initial_states().
        times : list of int
            Times for which the measurements are to be cropped.
            Default is self.simulation_times().
        """
        if states is None:
            states = self.initial_states()
        if times is None:
            times = self.simulation_times()
        # loop over all data entries
        for key in self.data:
            state, time = key
            if state in states and time in times:
                self.data[key].crop_measurements(nshots)

    def sample_measurements(self, 
                            nshots_list: list[int],
                            states: list[QuantumState] | None = None, 
                            times: list[int] | None = None, 
                            seed: int | None = None, 
                            nshots_ratio_integrand: float | None = None, 
                            skip_initial_time: bool = False, 
                            random: bool = False
                            ) -> list[DataSet]:
        """
        Sample subset of measurements from the DataSet.

        The sample is taken from each initial state and time,
        with a given number of shots distributed among the different measurement bases.
        The probability for choosing a measurement from a given basis 
        is proportional to the total number of shots in that basis.

        Parameters
        ----------
        nshots_list : list
            List of total number of shots to be sampled for each initial state and time.
            The number of shots for each basis is determined by the ratio of the number of shots
            for the basis in the dataset to the total number of shots.
            If nshots=-1, all available shots are used.
            # If nshots=0, only the exact expectation values are returned.
        states : list of QuantumStates, optional
            Initial states for which the measurements are to be sampled.
            Default is self.initial_states().
        times : list, optional
            Times for which the measurements are to be sampled.
            Default is self.simulation_times().
        seed : int, optional
            Seed for the random number generator.
            Default is None.
        nshots_ratio_integrand : float, optional 
            If set, the number of shots is reduced by factor nshots_ratio_integrand 
            for all times except the first and last time.
            Default is None.
        skip_initial_time : bool, optional 
            If True, all measurements at the initial time (simulation_times[0])
            are included in the sampled data.
            Default is False.
        random : bool, optional
            If True, samples measurements randomly.
            If False, samples measurements sequentially.
            Default is False.

        Returns
        -------
        list of DataSet objects
            List of data sets with sampled measurements 
            for each given number of shots.
        """
        if states is None:
            states = self.initial_states()
        if times is None:
            times = self.simulation_times()
        #----------------------------------
        ### STEP 1 ### for each nshots, create a data set with sampled measurements
        sampled_data = [DataSet(Nions=self.Nions) for nshotsx in nshots_list]
        for key, data_entry in self.data.items():
            state, time = key
            ## skip key if state or time not in states or times
            if state not in states or time not in times:
                continue
            ## skip initial time
            elif skip_initial_time and time == times[0]:
                # store unsampled data
                for nsinx, nshotsx in enumerate(nshots_list):
                    sampled_data[nsinx].add_data_entry(data_entry)
            ## sample measurements
            else:
                for nsinx, nshotsx in enumerate(nshots_list):
                    nshots_sample = int(nshotsx)
                    # apply nshots_ratio_integrand if given
                    if nshots_ratio_integrand is not None and time!=times[0] and time!=times[-1]:
                        nshots_sample = int(nshotsx*nshots_ratio_integrand)
                    # create data entry with sampled measurements
                    data_entry_sample = data_entry.sample_measurements(nshots_sample, random=random, seed=seed)
                    # add data entry to sampled data set
                    sampled_data[nsinx].add_data_entry(data_entry_sample)
        #----------------------------------
        return sampled_data

    def change_keys_to_quantum_states(self, 
                                    state_preparation: list[str] | None = None, 
                                    state_preparation_label: str | None = None
                                    ) -> None:
        """
        Changes the keys of the data dictionary to QuantumStates.

        Parameters
        ----------
        state_preparation : list of str, optional
            After initialization, a product of single-qubit unitary operators
            defined by a list of strings is applied to the initial state.
            Default is None.
        state_preparation_label : str, optional
            label for the state preparation used for hashing
            Default is None.
        """
        data_keys = list(self.data.keys())
        for key in data_keys:
            state = key[0]
            # check if state is a QuantumState
            if not isinstance(state, QuantumState):
                new_state = QuantumState(N=self.Nions, excitations=state, state_preparation=state_preparation, state_preparation_label=state_preparation_label)
                new_key = (new_state, key[1])
                # replace old key with new key
                self.data[new_key] = self.data.pop(key)

    def fix_quantum_states(self) -> None:
        """
        Check if initial_states are QuantumStates and convert them if necessary.
        """
        for entry in self.data.values():
            entry.fix_quantum_states()

    def add_exact_expvals_at_0(self,
                            qop: QuantumOperator
                            ) -> None:
        """
        Adds the exact expectation values of a given QuantumOperator to the DataSet, 
        evaluated at time=0.

        Parameters
        ----------
        qop : QuantumOperator
            QuantumOperator for which to add the exact expectation values.
        """
        for entry in self.data.values():
            entry.add_exact_expvals_at_0(qop)

    def plot_time_dynamics(self, 
                        observables : list[QuantumOperator],
                        states: list[QuantumState] = None,
                        times: list = None, 
                        nshots: int = -1, 
                        errorbars: bool = True, 
                        legends: bool = True,
                        ):
        """
        Plot the dynamics of the expectation values of given observables 
        for given initial states and times using nshots shots.

        Parameters
        ----------
        observables : list of QuantumOperators
            Quantum operators for which to evaluate the expectation values.
        states : list of QuantumStates, optional
            States for which to evaluate the expectation values.
            Default is self.initial_states().
        times : list of float, optional
            Times for which to evaluate the expectation values.
            Default is self.simulation_times().
        nshots : int, optional
            Number of shots used for evaluating observables
            Default is ``-1``.
        errorbars : bool, optional
            If True, also plots errorbars of expectation values from resampling.
            Default is True.
        legends : bool, optional
            If True, adds legends to the plot.
            Default is True.
        """
        if states is None:
            states = self.initial_states()
        if times is None:
            times = self.simulation_times()
        if nshots==0:
            errorbars = False
        ### get xdata and ydata for figure
        xdata = times
        ydata = []
        var_ydata = []
        for state in states:
            ydata_state = []
            var_ydata_state = []
            for observable in observables:
                ydata_observable = []
                var_ydata_observable = []
                for time in times:
                    expval, varval = self.evaluate_observable(state, time, [observable], nshots=nshots, Gaussian_noise=False, evaluate_variance=errorbars)
                    if not errorbars:
                        varval = [[0]]
                    ydata_observable.append(expval[0][0])
                    var_ydata_observable.append(varval[0][0])
                ydata_state.append(ydata_observable)
                var_ydata_state.append(var_ydata_observable)
            ydata.append(ydata_state)
            var_ydata.append(var_ydata_state)
        ### plot figure
        fig, ax = plt.subplots()
        for sinx, state in enumerate(states):
            for opinx, observable in enumerate(observables):
                plotlabel = "{}, {}".format(state.str(), observable)
                ax.plot(xdata, ydata[sinx][opinx], label=plotlabel, linestyle="None", marker="x")
                plotcolor = plt.gca().lines[-1].get_color()
                if errorbars:
                    lolims = np.squeeze(np.subtract(ydata[sinx][opinx],np.sqrt(var_ydata[sinx][opinx])))
                    uplims = np.squeeze(np.add(ydata[sinx][opinx],np.sqrt(var_ydata[sinx][opinx])))
                    ax.fill_between(xdata, lolims, uplims, alpha=0.2, color=plotcolor)
        if legends:
            ax.legend(fontsize=8) 
        plt.xlabel("Time")
        plt.ylabel("Expectation value")
        plt.show()
        return fig, ax

    def compare_to(self, 
                other: DataSet,
                qops: list[QuantumOperator] | list[PauliOperator],
                nshots: int | list[int] = -1,
                states: list[QuantumState] | None = None, 
                times: list[float] | None = None, 
                compare_exact: bool = True,
                ):
        """
        Compares self to other DataSet by evaluating the expectation values.

        Parameters
        ----------
        other : DataSet
            DataSet to which to compare self.
        qops : list of QuantumOperator or PauliOperator
            Quantum operators for which to evaluate the expectation values.
        nshots : int or list of ints
            Number of shots to be used for evaluation.
            If nshots=-1, all available shots are used.
            If nshots=0, exact expectation values are returned.
        states : list of QuantumStates, optional
            States for which to evaluate the expectation values.
            Default is  self.initial_states()
        times : list of float, optional 
            Times for which to evaluate the expectation values.
            Default is self.simulation_times().
        compare_exact : bool
            If True, compare exact expectation values for nshots=0.
            Default is True.

        Returns
        -------
        expvals_self : np.array
            Array of expectation values of self.
        varvals_self : np.array
            Array of variances of self.
        expvals_other : np.array
            Array of expectation values of other.
        varvals_other : np.array
            Array of variances of other.
        error_distribution : np.array
            Array of errors of self.
        errorbar_flags : np.array
            Array of flags for errorbars.
            flag is 1 if error is within errorbar, 0 otherwise
        """
        if states is None:
            states = self.initial_states()
        if times is None:
            times = self.simulation_times()
        # setup data arrays
        expvals_self = np.zeros((len(qops), len(states), len(times)))
        exact_expvals_self = np.zeros((len(qops), len(states), len(times)))
        varvals_self = np.zeros((len(qops), len(states), len(times)))
        expvals_other = np.zeros((len(qops), len(states), len(times)))
        exact_expvals_other = np.zeros((len(qops), len(states), len(times)))
        varvals_other = np.zeros((len(qops), len(states), len(times)))
        error_distribution = np.zeros((len(qops), len(states), len(times)))
        errorbar_flags = np.zeros((len(qops), len(states), len(times), 3))
        # evaluate expectation values for each state and time
        for opinx, qop in enumerate(qops):
            for sinx, state in enumerate(states):
                for tinx, time in enumerate(times):
                    expval_self, varval_self = self.evaluate_observable(state, time, [qop], nshots, evaluate_variance=True)
                    expvals_self[opinx,sinx,tinx] = expval_self[0]
                    varvals_self[opinx,sinx,tinx] = varval_self[0]
                    if compare_exact:
                        exact_expval_self, exact_varval_self = self.evaluate_observable(state, time, [qop], nshots=0, evaluate_variance=True)
                        exact_expvals_self[opinx,sinx,tinx] = exact_expval_self[0]
                    expval_other, varval_other = other.evaluate_observable(state, time, [qop], nshots, evaluate_variance=True)
                    expvals_other[opinx,sinx,tinx] = expval_other[0]
                    varvals_other[opinx,sinx,tinx] = varval_other[0]
                    exact_expval_other, exact_varval_other = other.evaluate_observable(state, time, [qop], nshots=0, evaluate_variance=True)
                    exact_expvals_other[opinx,sinx,tinx] = exact_expval_other[0]
                    # compare expectation values to exact expectation values
                    error = np.abs(expval_self - exact_expval_other)
                    # check if error is smaller than errorbars
                    error_distribution[opinx,sinx,tinx] = error
                    for sigmainx in range(0,3):
                        if error > (1+sigmainx)*np.sqrt(varval_self):
                            flagval = 0
                        else:
                            flagval = 1
                        errorbar_flags[opinx,sinx,tinx,sigmainx] = flagval
        time_dynamics_self = [expvals_self, varvals_self, exact_expvals_self]
        time_dynamics_other = [expvals_other, varvals_other, exact_expvals_other]
        return time_dynamics_self, time_dynamics_other, error_distribution, errorbar_flags

    def divergence(self, 
                other: DataSet, 
                method: str = "ls",
                nshots: int | None = None
                ) -> float:
        """
        Return the divergence of two DataSet objects.

        Parameters
        ----------
        other : DataSet
        method : str
            Options are "ls" for least squares, 
            "ls_vec" for least squares vectorized, 
            "kl" for Kullback-Leibler divergence
        nshots : int
            number of shots to use for the least squares method

        Returns
        -------
        float
            divergence of the two DataSet objects
        """
        # check if other is a valid data set
        if not isinstance(other, DataSet):
            raise ValueError("other is not a valid data set.")
        # check if number of ions is the same in the data sets
        if self.Nions != other.Nions:
            raise ValueError("Nions is not the same in the data sets.")
        # get divergence
        if method=="ls_vec":
            divergence = []
        else:
            divergence = 0
        for key in self.data:
            if key in other.data:  
                if method=="ls_vec":
                    divergence.extend(self.data[key].divergence(other.data[key],method=method,nshots=nshots)) 
                else:
                    divergence += self.data[key].divergence(other.data[key],method=method,nshots=nshots)
        return divergence
    
    def compare_dynamics(self, 
                        other: DataSet,
                        qops: list[QuantumOperator], 
                        nshots: int = 0, 
                        states: list[QuantumState] | None = None, 
                        times: list[float] | None = None
                        ):
        """
        Compare the time-dynamics of two data sets.

        Parameters
        ----------
        other : DataSet
            DataSet to compare to
        qops : list of QuantumOperators
            Quantum operators for which to evaluate the expectation values.
        nshots : int, optional
            Number of shots to be used for evaluation.
            If nshots=-1, all available shots are used.
            If nshots=0, the exact expectation value is used.
            Default is ``0``.
        states : list of QuantumStates, optional
            States for which to evaluate the expectation values.
            By default all states common to both data sets are used.
        times : list of float, optional
            Times for which to evaluate the expectation values.
            By default all available times are used for each DataSet.

        Returns
        -------
        list
            time dynamics of self is a list of 3 arrays
            [times, expvals, varvals]
            where times are the simulation times and expvals and varvals 
            the corresponding expectation values and variances
        list
            time dynamics of other in analogy to time dynamics of self
        """
        ## states
        states_self = states
        states_other = states
        if states is None:
            states_self = self.initial_states()
            states_other = other.initial_states()
        # get common states
        states = list(set(states_self).intersection(set(states_other)))
        ## times
        times_self = times
        times_other = times
        if times is None:
            times_self = self.simulation_times()
            times_other = other.simulation_times()
        ### setup data arrays
        expvals_self = np.zeros((len(states), len(qops), len(times_self)))
        varvals_self = np.zeros((len(states), len(qops), len(times_self)))
        expvals_other = np.zeros((len(states), len(qops), len(times_other)))
        varvals_other = np.zeros((len(states), len(qops), len(times_other)))
        ### evaluate expectation values for each state and time
        for sinx, state in enumerate(states):
            for opinx, qop in enumerate(qops):
                ## self
                for tinx, time in enumerate(times_self):
                    expval_self, varval_self = self.evaluate_observable(state, time, [qop], nshots, evaluate_variance=True)
                    expvals_self[sinx,opinx,tinx] = expval_self[0]
                    varvals_self[sinx,opinx,tinx] = varval_self[0]
                ## other
                for tinx, time in enumerate(times_other):
                    try:
                        expval_other, varval_other = other.evaluate_observable(state, time, [qop], nshots, evaluate_variance=True)
                    except:
                        nshots_tmp = -1
                        expval_other, varval_other = other.evaluate_observable(state, time, [qop], nshots_tmp, evaluate_variance=True, Gaussian_noise=False)
                    expvals_other[sinx,opinx,tinx] = expval_other[0]
                    varvals_other[sinx,opinx,tinx] = varval_other[0]
        time_dynamics_self = [times_self, expvals_self, varvals_self]
        time_dynamics_other = [times_other, expvals_other, varvals_other]
        return time_dynamics_self, time_dynamics_other
    
    def extend_to_larger_system(self, 
                            extension_factor: int,
                            ) -> DataSet:
        """
        Extend the DataSet from Nions to Nions*extension_factor.

        The larger system consists of extension_factor non-interacting 
        subsystems each of size Nions.
        The measurements are generated by randomly concatenating 
        measurements from the original system.
        This reduces the total number of shots for each measurement basis 
        by a factor of extension_factor.
        # TODO: also randomize initial states, not only measurements

        Parameters
        ----------
        extension_factor : int
            number of non-interacting subsystems

        Returns
        -------
        DataSet
            DataSet extended to larger system but with reduced number of shots.
        """
        ### STEP 1 ### setup extended data set
        extended_data_set = DataSet(Nions=self.Nions*extension_factor)
        #----------------------------------
        ### STEP 2 ### extend data entries
        for entry in self.data.values():
            extended_data_entry = entry.extend_to_larger_system(extension_factor)
            extended_data_set.add_data_entry(extended_data_entry)
        #----------------------------------
        return extended_data_set



########################
### helper functions ###
########################
def unpack_measurements(packed_measurements: dict, 
                        num_bits: int):
    """
    Unpack integers into binary strings of length num_bits.

    Here "-1" stands for the measurement outcome "down" 
    and "1" stands for the measurement outcome "up".
    The result is padded with 0 for missing values.
    This function inverts the pack_measurements function.

    Parameters
    ----------
    packed_measurements : dict
        dictionary of basis:measurements pairs where 
        basis is a string of chars in ["X", "Y", "Z"]
        and measurements is a numpy array of integers
    num_bits : int
        Number of bits to unpack each integer into (number of atoms)

    Returns
    -------
    unpacked_measurements : np.array
        unpacked_measurements[binx, minx, qinx] is the outcome of the 
        minx-th measurement in the binx-th basis for the qinx-th qubit.
        Here -1 stands for outcome "down" and 1 stands for outcome "up".
    """
    ### STEP 0 ### setup placeholder
    placeholder = 0
    #----------------------------------
    ### STEP 1 ### unpack measurements
    # find the maximum length of the sublists
    max_len = max(len(sublist) for sublist in packed_measurements.values())
    # initialize numpy array to store the unpacked_measurements
    shape = (len(packed_measurements), max_len, num_bits)
    unpacked_measurements = np.zeros(shape, dtype=np.int8)
    for sinx, key in enumerate(packed_measurements):
        sublist = packed_measurements[key]
        # Convert sublist to a NumPy array
        sublist_array = np.array(sublist)
        # Unpack bits using the previously defined logic
        mask = 2**np.arange(num_bits, dtype=sublist_array.dtype).reshape(1, num_bits)
        unpacked_bits = (sublist_array[:, None] & mask).astype(bool).astype(int)
        # Flip to have the most significant bit first
        unpacked_bits = np.flip(unpacked_bits, axis=-1)
        # Convert the result to a list of lists, adding -1 where placeholder value should be
        sublist_result = unpacked_bits.tolist()
        sublist_result.extend([[-1] * num_bits] * (max_len - len(sublist)))
        unpacked_measurements[sinx] = sublist_result
    # ## transform result to a sparse array of np.int8
    unpacked_measurements = np.subtract(np.multiply(2, unpacked_measurements, dtype=np.int8), 1, dtype=np.int8)
    # set -1 values to 0
    unpacked_measurements[unpacked_measurements==-3] = placeholder
    #----------------------------------
    return unpacked_measurements

def pack_measurements(unpacked_measurements, 
                    input_type: str = "str"):
    """
    Pack measurements of type input_type into a numpy array of integers.

    If input_type=="str" [default], the input is an array of strings, e.g., ["1010", "0101", "1111"].
    where "0" stands for the measurement outcome "down"
    and "1" stands for the measurement outcome "up".
    If input_type=="int", the input is an array of -1 and 1.
    where -1 stands for the measurement outcome "down"
    and 1 stands for the measurement outcome "up".
    In this case the function inverts the unpack_measurements function.

    Parameters
    ----------
    unpacked_measurements : array
        unpacked_measurements is an array of dimension (nshots, num_bits)
        where -1 stands for the measurement outcome "down"
        and 1 stands for the measurement outcome "up".
    input_type : str, optional
        If "str", input is a list of bitstrings, e.g., ["1010", "0101", "1111"].
        If "int", input is a 2D-array of -1 and 1. 
        Default is ``"str"``.

    Returns
    -------
    np.array
        packed_measurements as a numpy array of integers
    """
    # ----------------------------------
    ### STEP 1 ### convert the input to a binary array [1,0,1,0,1]
    if input_type=="str":
        unpacked_measurements = np.array([list(map(int, list(bitstring))) for bitstring in unpacked_measurements])
    if input_type=="int":
        unpacked_measurements[unpacked_measurements==-1] = 0
    # ----------------------------------
    ### STEP 2 ### pack the measurements into integers
    # find the number of bits
    num_bits = unpacked_measurements.shape[-1]
    # convert the binary strings to integers
    # flip the bits to have the least significant bit first
    unpacked_measurements = np.flip(unpacked_measurements, axis=-1)
    # convert the binary strings to integers
    packed_measurements = np.sum(unpacked_measurements * 2**np.arange(num_bits), axis=-1)
    # ----------------------------------
    return np.squeeze(packed_measurements)

def packbits(unpacked_bits):
    """
    Packs a numpy array of binary strings into a numpy array of integers.

    NOTE: this function is only used inside extend_to_larger_system()

    Parameters
        unpacked_bits : np.array
            array of binary strings to be packed into an array of integers
    """
    # ensure input is a NumPy array
    unpacked_bits = np.array(unpacked_bits)
    # check that the unpacked bits are binary (0 or 1)
    if not np.array_equal(unpacked_bits, unpacked_bits.astype(bool)):
        raise ValueError("unpacked_bits array must contain only binary values (0 or 1)")
    # determine the number of bits
    num_bits = unpacked_bits.shape[-1]
    # create the mask array (powers of 2)
    mask = 2**np.arange(num_bits)[::-1]
    # dot product of unpacked bits with the mask gives the packed integers
    packed_bits = np.dot(unpacked_bits, mask)
    # reshape to match the original packed array"s shape
    return packed_bits

# NOTE: used only in extend_to_larger_system()
def unpackbits(packed_bits, 
            num_bits: int
            ):
    """
    Unpacks a numpy array of integers into a numpy array of binary strings.

    NOTE: This function is only used inside extend_to_larger_system().

    Parameters
        packed_bits : np.array with dtype int
            array of integers to be unpacked into binary strings
        num_bits : int
            number of bits to unpack to
    """
    packed_bits = np.array(packed_bits)
    if np.issubdtype(packed_bits.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    shape = list(packed_bits.shape)
    packed_bits = packed_bits.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=packed_bits.dtype).reshape([1, num_bits])
    return np.flip((packed_bits & mask).astype(bool).astype(int).reshape(shape + [num_bits]),axis=1)  
