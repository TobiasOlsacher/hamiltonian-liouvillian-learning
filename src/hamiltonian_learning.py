"""
This module contains functions to perform Hamiltonian learning.
It contains the following classes:
    - Result
        A class for storing the results of the learning algorithm.
    - Constraint
        A class for defining constraints used for learning.
    - Ansatz
        A class for defining the Hamiltonian and Liouvillian ansatz used for learning.

The learning has high memory requirement (RAM) because of all the saved result attributes (can be reduced!)
"""
from __future__ import annotations
import numpy as np
import scipy as sp
import itertools as it
import math 
import time as tm
from icecream import ic
from collections import Counter
import time
from tqdm import tqdm
from src.quantum_state import QuantumState
from src.measurement_setting import MeasurementSetting
from src.quantum_simulator import QuantumSimulator
from src.pauli_algebra import PauliOperator, QuantumOperator, Dissipator, qop_sum
import src.utility_functions as uf
from src.data_statistics import DataSet
from src.ansatz_parametrization import Parametrization



class Result:
    """
    Result of the learning algorithm.

    This class creates and stores the results of the learning algorithm.
    #TODO list of attributes and methods is outdated/incomplete

    Attributes
    ----------

    data : dict
        Dictionary of all result objects.
    """
    def __init__(self, **kwargs):
        ## ansatz properties
        self.ansatz_operator = kwargs.get("ansatz_operator", None)
        self.ansatz_dissipators = kwargs.get("ansatz_dissipators", None)
        self.parametrizations = kwargs.get("parametrizations", [None])   # list of different parametrizations
        self.parametrization_matrix = kwargs.get("parametrization_matrix", None)   # list for different parametrizations
        self.learning_method = kwargs.get("learning_method", None)   # chosen learning method
        self.nshots = kwargs.get("nshots", None)
        self.nshots_scale = kwargs.get("nshots_scale", None)
        self.ntimes = kwargs.get("ntimes", None)
        self.constraint_sample_indices = kwargs.get("constraint_sample_indices", {})   # constraint_type:list pairs where list is a list of indices_lists each defining a single samples
        self.jackknife_inflation_factors = kwargs.get("jackknife_inflation_factors", {})   # inflation factor for jackknife resampling (multiplied to variance)
        self.prior = kwargs.get("prior", None)   # prior distribution for the free parameters
        self.gamma0 = kwargs.get("gamma0", None)   # initial guess for the free parameters
        self.gamma_bounds = kwargs.get("gamma_bounds", None)   # bounds for the free parameters
        self.gamma_exact = kwargs.get("gamma_exact", None)   # if given, gamma is fixed to gamma_exact and no optimization is performed
        ## Hamiltonian
        self.operator_learned = kwargs.get("operator_learned", None)   # list of dimensions parametrizations*samples
        self.Qoperator_learned = kwargs.get("Qoperator_learned", None)   # list of dimensions parametrizations*samples
        self.learning_error = kwargs.get("learning_error", None)  # learning error for given learning method (parallel to c")
        self.learning_error_noscale = kwargs.get("learning_error_noscale", None)  # learning error for non-scale constraints
        self.svd_vals = kwargs.get("svd_vals", None)   # list of different parametrizations
        self.svd_vecs = kwargs.get("svd_vecs", None)   # list of different parametrizations
        # Bayesian
        self.posterior = kwargs.get("posterior", None)   # posterior distribution for the free parameters
        self.Gamma_eps = kwargs.get("Gamma_eps", None)   # covariance matrix of the learning error (Ec+e)
        ## dissipation
        self.dissipators_learned = kwargs.get("dissipators_learned", None)   # list of dimensions parametrizations*samples
        self.gamma_landscape_grid = kwargs.get("gamma_landscape_grid", None)   # list of different parametrizations
        self.gamma_landscape_vals = kwargs.get("gamma_landscape_vals", None)   # list of different parametrizations
        self.gamma_landscape_sol = kwargs.get("gamma_landscape_sol", None)   # list of different parametrizations



class Constraint:
    """
    Constraint for learning the ansatz parameters.

    A Constraint determines a single row of the constraint matrix M and the constraint vector b.
    It consists of the used initial state, simulation times and constraint operator.
    Then the coefficients ``c`` for the ansatz of the Hamiltonian or Liouvillian are learned
    solving the linear equation ``Mc=b``.

    Attributes
    ----------
    initial_state : QuantumState
        Initial state of the quantum system.
    simulation_times : list of floats
        List of times at which the quantum system is simulated.
        For learning method "LZHx" for x=1,2...,
        only the first and last element of simulation_times are used.
    constraint_operator : QuantumOperator
        Observable to be measured as a constraint.
        The constraint is measured at the first and last element of simulation_times.
        Only used for BAL and ZYLB method.
    nshots_ratio_integrand : float, optional
        Ratio of number of shots used for each time step in the integrand, 
        compared to the number of shots used at the end points.
        Default is ``1``.
    """
    def __init__(
            self, 
            initial_state: QuantumState | None = None, 
            simulation_times = None, 
            constraint_operator: QuantumOperator | None = None, 
            nshots_ratio_integrand: float = 1,
            ):
        """
        Initialize a Constraint.

        Parameters
        ----------
        initial_state : QuantumState
            Initial state of the quantum system.
        simulation_times : list of floats
            List of times at which the quantum system is simulated.
        constraint_operator : QuantumOperator
            Observable to be measured as a constraint.
        nshots_ratio_integrand : float, optional
            Ratio of number of shots used for each time step in the integrand, 
            compared to the number of shots used at the end points.
            Default is ``1``.
        """
        self.initial_state = initial_state
        self.simulation_times = simulation_times
        self.constraint_operator = constraint_operator
        self.nshots_ratio_integrand = nshots_ratio_integrand
    ### --------------------- ###
    ### Constraint attributes ###
    ### --------------------- ###
    @property
    def initial_state(self):
        return self._initial_state
    @initial_state.setter
    def initial_state(self, value):
        # check if value is a QuantumState object
        if value is not None:
            if not isinstance(value, QuantumState):
                raise TypeError("initial_state must be a QuantumState object")
        self._initial_state = value
    @property
    def simulation_times(self):
        return self._simulation_times
    @simulation_times.setter
    def simulation_times(self, value):
        if value is not None:
            # check if value is a list
            if not isinstance(value, (list, np.ndarray)):
                raise TypeError("simulation_times must be a list")
            # check if times are sorted
            if not np.all(np.diff(value) > 0):
                raise ValueError("simulation_times must be sorted")
            # check if there are no almost equal times
            for i in range(len(value)-1):
                if value[i+1]-value[i] < 1e-6:
                    raise ValueError("simulation_times must not have almost equal times")
        self._simulation_times = value
    @property
    def constraint_operator(self):
        return self._constraint_operator
    @constraint_operator.setter
    def constraint_operator(self, value):
        # check if value is a QuantumOperator object
        if value is not None:
            if not isinstance(value, QuantumOperator) and not isinstance(value, PauliOperator):
                raise TypeError("constraint_operator must be a QuantumOperator or PauliOperator object")
        self._constraint_operator = value
    @property
    def nshots_ratio_integrand(self):
        return self._nshots_ratio_integrand
    @nshots_ratio_integrand.setter
    def nshots_ratio_integrand(self, value):
        # check if value is a positive float
        if value is not None:
            if not isinstance(value, (int, float, np.float64)):
                raise TypeError("nshots_ratio_integrand = {}, but must be a float".format(value))
            if value <= 0:
                raise ValueError("nshots_ratio_integrand = {}, but must be positive".format(value))
        self._nshots_ratio_integrand = value
    ### ------------------ ###
    ### Constraint methods ###
    ### ------------------ ###
    def __eq__(self, other) -> bool:
        """
        Equality relation of two constraints.

        Two constraints are equal if they have the same initial state, 
        simulation times, constraint operator and shots distribution in the integral.

        Parameters
        ----------
        other : Constraint
            Constraint to compare to

        Returns
        -------
        bool
            whether self and other have the same attributes
        """
        for attr in ["initial_state", "constraint_operator", "nshots_ratio_integrand"]:
            if getattr(self, attr) != getattr(other, attr):
                return False
        for attr in ["simulation_times"]:
            if not np.allclose(getattr(self, attr), getattr(other, attr)):
                return False
        return True
    
    def __ne__(self, other) -> bool:
        """
        Inequality relation of two constraints.

        Two constraints are not equal if they differ in initial state, 
        simulation times, constraint operator or shots distribution in the integral.

        Parameters
        ----------
        other : Constraint
            Constraint to compare to

        Returns
        -------
        bool
            whether self and other have different attributes
        """
        return not self.__eq__(other)
    
    def copy(self) -> Constraint:
        """
        Return a copy of the constraint.

        Returns
        -------
        Constraint
            A new instance with the same attributes as this Constraint.
        """
        return Constraint(initial_state=self.initial_state, simulation_times=self.simulation_times, constraint_operator=self.constraint_operator, nshots_ratio_integrand=self.nshots_ratio_integrand)
    
    def __str__(self):
        return self.str()
    def str(self) -> str:
        """
        Return the string representation of the Constraint

        Returns
        -------
        str
            string representation of the constraint
        """
        state_str = self.initial_state.str()
        con_op_str = "None"
        if self.constraint_operator is not None:
            con_op_str = self.constraint_operator.str()
        time_str = "[{}to{},dt={},nt={}]".format(np.round(self.simulation_times[0],4), np.round(self.simulation_times[-1],4), np.round(self.simulation_times[1]-self.simulation_times[0],6), len(self.simulation_times))
        ratio_str = "{}".format(self.nshots_ratio_integrand)
        name = "Constraint: psi0={}, op={}, times={}, ratio={}".format(state_str, con_op_str, time_str, ratio_str)
        return name



class Ansatz:
    """
    Ansatz for Hamiltonian or Liouvillian learning.

    Attributes
    ----------
    Nions : int
        Number of qubits in the system that generates the data.
    ansatz_operator : QuantumOperator object
        Ansatz for the Hamiltonian of the system. 
        All coefficients are set to ``1`` by default.
    ansatz_dissipators : list of Dissipators
        Ansatz for the Dissipators of the Liouvillian of the system.
    parametrization : list of Parametrization objects
        List of parametrizations for the ansatz.
    data_set : DataSet
        Measurement data used for learning. 
    constraints : list of Constraint objects
        list of Constraint objects used for the learning
    constraint_tensors : dict, optional
        Constraint tensors used for learning.
        The keys of the dict are (method, nshots) for "LZHx" method and (method, nshots, ntimes) for "ZYLB" and "BAL" method.
        The values of the dict are lists of constraint tensors.
    gamma_bounds : tuple of 1D arrays
        Bounds for the optimal gamma.
        If not None, the optimal gamma is refined using a nonlinear solver.
        Default is None.
    gamma0 : 1D array
        Initial value for the optimal gamma for the iterative solver.
        If None, the iterative solver is not used.
        Default is None
    gamma_exact : 1D array 
        If given, gamma is fixed to gamma_exact and no optimization is performed.
        Default is None.
    gamma_max_nfev : int
        Maximum number of function evaluations for the solver during Dissipation correction.
        Default is ``1000``.
    prior : tuple of arrays, optional
        Prior distribution required for Bayesian learning.
        Prior[0] is the mean of the free ansatz parameters
        and Prior[1] is the covariance matrix of the free ansatz parameters.
        Default is None.
    result_keys : list of str
        List of keys to choose what is stored in the result dictionary.
        If None, all results are stored (default).
    result : dict
        Dictionary where each value is a Result object containing the results of the learning.
    print_timers : bool
        If True, print timers during learning.
        Default is False

    save_landscape : bool
        If True, saves the landscape of the optimization over disspation rates.
        Default is False
    exclude_lowest_solutions : int, optional
        Number of lowest solutions to exclude from the result for the LZH or O3KZi learning methods.
        E.g. if exclude_lowest_solutions is ``1``, we are solving for the second-lowest singular value.
        This can be useful if there exists a conserved quantity that should be excluded.
        Default is ``0``.
    scale_factor : float
        Factor by which the scale constraints are multiplied.
        Only used if scale_method is not None.
        Default is ``1``.
    get_variance_of_coefficients_from_variance_of_constraint_tensors : bool, optional
        If True, also estimates the variance of the learned coefficients
        from the variance of the constraint tensors.
        NOTE: This requires evaluate_variances==True in the get_constraint_tensors() method.
        Default is False.

    n_resampling_constraints ((float,int)) [default: (0,1)]
        parameters for resampling constraints.
        n_resampling_constraints[0] is the fraction of constraints used for resampling.
        n_resampling_constraints[1] is the number of resampling constraints.
    resampling_constraints_replace (bool) [default: False]
        If True, resampling constraints is done with replacement.
    resampling_constraints_jackknife (bool) [default: False]
        If True, n_resampling_constraints[1] is replaced by the required number for full jackknife resampling, 
        and resampling_constraints_replace is set to False.



        
        - learning_method (str)
            Learning method ("ZYLB", "BAL" or "LZHx" for integer x>=1).


        - n_resampling_measurements ((float,int)) [default: (0,1)]
            parameters for resampling measurements (with replacement).
            n_resampling_measurements[0] is the fraction of measurements used for resampling.
            n_resampling_measurements[1] is the number of resampling measurements.
        - resampling_measurements_replace (bool) [default: False]
            If True, resampling measurements is done with replacement.
        - resampling_measurements_jackknife (bool) [default: False]
            If True, n_resampling_measurements[1] are replaced by the required number for full jackknife resampling, 
            and resampling_measurements_replace is set to False.
        # - constraint_sample_indices (array) 
        #     Indices of constraints used for resampling.
        - constraint_operators (list of QuantumOperator objects)
            Constraint operators used for the learning.
            Required for learning using BAL or ZYLB method.
        - initial_states (list of QuantumStates)
            List of initial states used for the learning. 
            Each state is a QuantumState object, 
            where state.excitations is a bitstring of length Nions.
            For example ["000", "001", "010"] etc. for Nions=3.
        - simulation_times (list of list of floats)
            List of simulation times used for the learning.
            Each element of the list is a list of floats, e.g. [[0.1,0.2,0.3],[0.1,0.2,0.3]].

        - plotdata (dict)
            Dictionary containing data for plotting.
        - constraint_tensors_samples (dict)
            Dictionary of key:constraint_tensor_samples pairs.
            keys are (label, method, nshots), 
            where method in ["O3KZi", "BAL", etc.] and 
            nshots is the number of shots used for the constraint tensor 
            and label is in ["learn", "scale"] by default.
            constraint_tensor_samples is the list of len(n_resampling_constraints[1]) samples of the constraint tensors.
        - constraint_tensors_nruns (dict)
            Dictionary of key:nruns pairs.
            keys are (label, method, nshots), 
            where method in ["O3KZi", "BAL", etc.] and 
            nshots is the number of shots used for the constraint tensor 
            and label is in ["learn", "scale"] by default.
            nruns is the number of runs used for estimating constraint_tensor_samples[key]
    """
    def __init__(
            self, 
            Nions: int,
            ansatz_operator: QuantumOperator | None = None, 
            ansatz_dissipators: list | None = None,
            parametrization: Parametrization | None = None, 
            data_set: DataSet | None = None, 
            constraints: list | None = None, 
            constraint_tensors: dict | None = None,
            gamma_bounds: list | None = None,
            gamma0: np.ndarray | None = None,
            gamma_exact: np.ndarray | None = None,
            gamma_max_nfev: int = 1000,
            prior: tuple | None = None,
            result_keys: list | None = None,
            result: dict = {},
            plotdata: dict = {},
            print_timers: bool = False,
            save_landscape: bool = False,
            exclude_lowest_solutions: int = 0,
            scale_factor: float = 1.0,
            get_variance_of_coefficients_from_variance_of_constraint_tensors: bool = False,
            n_resampling_constraints: tuple = (0,1),
            resampling_constraints_replace: bool = False,
            resampling_constraints_jackknife: bool = False,
            ):
        """
        Initialize an Ansatz.
        """
        self.Nions = Nions
        self.ansatz_operator = ansatz_operator
        self.ansatz_dissipators = ansatz_dissipators
        self.parametrization = parametrization
        self.data_set = DataSet(Nions=Nions) if data_set is None else data_set
        self.constraints = constraints
        self.constraint_tensors = {} if constraint_tensors is None else constraint_tensors
        self.gamma_bounds = gamma_bounds
        self.gamma0 = gamma0
        self.gamma_exact = gamma_exact
        self.gamma_max_nfev = gamma_max_nfev
        self.prior = prior
        self.result_keys = result_keys
        self.result = result
        self.plotdata = plotdata
        self.print_timers = print_timers
        self.save_landscape = save_landscape
        self.exclude_lowest_solutions = exclude_lowest_solutions
        self.scale_factor = scale_factor
        self.get_variance_of_coefficients_from_variance_of_constraint_tensors = get_variance_of_coefficients_from_variance_of_constraint_tensors
        self.n_resampling_constraints = n_resampling_constraints
        self.resampling_constraints_replace = resampling_constraints_replace
        self.resampling_constraints_jackknife = resampling_constraints_jackknife
    ### ----------------- ###
    ### Ansatz attributes ###
    ### ----------------- ### 
    @property
    def ansatz_operator(self):
        return self._ansatz_operator
    @ansatz_operator.setter
    def ansatz_operator(self,ansatz_operator):
        if ansatz_operator is not None:
            # # check if operator is a QuantumOperator object
            if not isinstance(ansatz_operator, QuantumOperator):
                raise ValueError("Operator is not a QuantumOperator object.")
            # set coefficients of ansatz operator to 1
            ansatz_operator_copy = ansatz_operator.copy()
            ansatz_operator_copy.set_coeffs_one()     
            ansatz_operator = ansatz_operator_copy     
        self._ansatz_operator = ansatz_operator
    @property
    def ansatz_dissipators(self):
        return self._ansatz_dissipators
    @ansatz_dissipators.setter
    def ansatz_dissipators(self,ansatz_dissipators):
        if ansatz_dissipators is not None:
            # check if ansatz_dissipators is a list
            if not isinstance(ansatz_dissipators, list):
                raise ValueError("Ansatz_dissipators is not a list.")
            # check if ansatz_dissipators is a list of Dissipators
            if not all([isinstance(diss, Dissipator) for diss in ansatz_dissipators]):
                raise ValueError("Ansatz_dissipators is not a list of Dissipators.")
            # set coefficients of ansatz dissipators to 1
            ansatz_dissipators_new = []
            for diss in ansatz_dissipators:
                diss = diss.copy()
                diss.coeff = 1
                ansatz_dissipators_new.append(diss)
            ansatz_dissipators = ansatz_dissipators_new
        self._ansatz_dissipators = ansatz_dissipators
    ### -------------- ###
    ### Ansatz methods ###
    ### -------------- ###
    def copy(self) -> Ansatz:
        """
        Return a copy of the Ansatz.

        Returns
        -------
        A new instance with the same attributes as the ansatz.
        """
        return Ansatz(Nions=self.Nions, ansatz_operator=self.ansatz_operator, ansatz_dissipators=self.ansatz_dissipators, parametrization=self.parametrization, data_set=self.data_set, constraints=self.constraints, constraint_tensors=self.constraint_tensors)

    def add_terms(self, qop: QuantumOperator) -> None:
        """
        Add terms to the ansatz operator. 
        
        The coefficients of the added terms are set to 1.

        Parameters
        ----------
        qop : QuantumOperator
            The operator to be added to the current ansatz operator.
        """
        # check if qoperator is a QuantumOperator object
        if not isinstance(qop, QuantumOperator):
            raise ValueError("qop is not a QuantumOperator object.")
        # add terms
        self.ansatz_operator = self.ansatz_operator + qop

    def add_data(self, data_set: DataSet) -> None:
        """
        Add data to the ansatz data_set.

        Parameters
        ----------
        data_set : DataSet
            The data_set to be added to the current ansatz data_set.
        """
        # check if data_set is a DataSet object
        if not isinstance(data_set, DataSet):
            raise ValueError("DataSet is not a DataSet object.")
        # add data
        self.data_set.add_data_set(data_set)

    def delete_data(self) -> None:
        """
        Delete the data_set attribute of the Ansatz.
        """
        self.data_set = DataSet(self.Nions)

    def get_data(
            self, 
            Qsim: QuantumSimulator,
            method: str,
            nshots: int,
            constraints: list | None = None,
            suggested_measurement_bases: list | None = None, 
            num_cpus: int = 1, 
            ) -> None:
        """
        Add data for chosen method to the ansatz DataSet.

        The data is simulated using the quantum simulator Qsim.

        Parameters
        ----------
        Qsim : QuantumSimulator
            The QuantumSimulator object used to generate the measurement data.
        method : str
            The learning method for which the data is generated.
            Options are "O3KZi", "BAL", "ZYLB", "O3KZd" and "LZH".
        nshots : int
            (Maximal) number of shots taken for each measurement basis.
            If nshots is ``0``, only exact expectation values are generated.
        constraints : list
            List of Constraints for which to generate the constraint tensor and vector.
            Default is self.constraints. 
        suggested_measurement_bases : list of Pauli strings, optional
            Suggested list of measurement bases for the data generation.
            If possible, the suggested bases are used for the data generation.
            If None or insufficient, measurement bases are determined automatically.
            Default is None.
        num_cpus : int 
            Number of cpus to use for parallelization.
            Default is ``1``.
        """
        if constraints is None:
            constraints = self.constraints
        if nshots > 1000:
            print(f"Warning: Sampling data points for nshots={nshots}. May take a while.")
        #--------------------------------------------------------
        ### STEP 1 ### determine required measurement settings
        measurement_settings = self.get_measurement_settings(method, constraints, nshots, suggested_measurement_bases=suggested_measurement_bases)
        #--------------------------------------------------------
        ### STEP 2 ### get the data
        if num_cpus==1:
            # sequential version
            for sinx, state in enumerate(measurement_settings.keys()):
                tm1 = tm.time()
                new_data_set = parallel_simulate(state, measurement_settings=measurement_settings, Qsim=Qsim) 
                self.add_data(new_data_set)
                tm2 = tm.time()
                if sinx==0:
                    pstr_get_data = "took {} seconds to simulate the first state out of {}".format(tm2-tm1, len(measurement_settings))
                    ic(pstr_get_data)
        elif num_cpus>1:
            # parallelized version
            raise NotImplementedError("parallelized version is not implemented yet.")
            print("enter parallel simulate, ncpus ={}".format(num_cpus))
            new_data_set_list = parfor(parallel_simulate, measurement_settings.keys(), measurement_settings=measurement_settings, Qsim=Qsim, num_cpus=num_cpus)
            # result = parfor(parallel_simulate, initial_states, num_cpus=num_cpus, simulation_times=alltimes, bases=measurement_bases, nshots=nshots_dict, required_operator=required_operator, Qsim=Qsim, save_states=save_states, nshots_distribution_integrand=nshots_distribution_integrand,times_integrand=times_integrand,nshots_integrand=nshots_integrand)#, state_preparation=state_preparation
            # result = mp.Process(target=parallel_simulate, args=initial_states) #, num_cpus=num_cpus)
            # result.start()
            # result.join()
            for new_data_set in new_data_set_list:
                self.add_data(new_data_set)

    def get_required_operator(
            self, 
            method: str, 
            constraint_operators: list | None = None
            ) -> tuple[QuantumOperator, QuantumOperator]:
        """
        Get the required operator for the learning method.

        The required operator depends on the learning method, 
        the ansatz operator and the constraint operators.
        It defines what measurements are required for calculating the constraint tensors.

        Parameters
        ----------
        method : str
            The learning method for which the operator is generated.
            Options are "O3KZi", "BAL", "ZYLB", "O3KZd" and "LZHx".
            LZHx adds all data up to order x.
        constraint_operators : list of QuantumOperators 
            List of constraint operators used for the learning.
            Required for learning using BAL or ZYLB method.

        Returns
        -------
        QuantumOperator
            The operator to be measured at times t=0 and t=T.
        QuantumOperator
            The operator to be measured at all times t in [0,T].
        """
        ### STEP 1 ### get required operator (depends on ansatz, method and constraints)
        required_operator_endpoints = None
        required_operator_integrand = None
        ### LZH method
        if method == "LZH":
            if self.ansatz_operator is None:
                raise ValueError("For LZH method ansatz_operator cannot be None.")
            required_operator_endpoints = self.ansatz_operator
        #--------
        ### BAL or ZYLB method
        elif method in ["BAL", "ZYLB"]:
            identity = QuantumOperator(self.Nions, terms={"I"*self.Nions:1})
            required_operator_endpoints = identity
            required_operator_integrand = identity
            if constraint_operators is None:
                raise ValueError("For BAL or ZYLB method constraint_operators cannot be None.")
            ## constraint operators
            for cinx in range(len(constraint_operators)):
                required_operator_endpoints += constraint_operators[cinx]
            ## commutators
            if self.ansatz_operator is not None:
                for constr in constraint_operators:
                    comm = constr.commutator(self.ansatz_operator)
                    if np.linalg.norm(comm.coeffs()) != 0:
                        comm.remove_zero_coeffs()
                        required_operator_integrand += comm
            ## dissipation operators
            if self.ansatz_dissipators is not None:
                # get all upper triangle pairs of dissipation operators
                for diss in self.ansatz_dissipators:
                    for constr in constraint_operators:  
                        term = diss(constr)
                        if np.linalg.norm(term.coeffs()) != 0:
                            term.remove_zero_coeffs()
                            required_operator_integrand += term
        #--------
        ### O3KZi or O3KZd method
        elif method in ["O3KZi","O3KZd"]:
            required_operator_integrand = QuantumOperator(self.Nions)
            # terms for Hamiltonian
            required_operator_endpoints = self.ansatz_operator.copy()
            ## dissipation operators
            if self.ansatz_dissipators is not None:
                # get all upper triangle pairs of dissipation operators
                for diss in self.ansatz_dissipators:
                    term = diss(self.ansatz_operator)
                    if np.linalg.norm(term.coeffs()) != 0:
                        term.remove_zero_coeffs()
                        required_operator_integrand += term
        #--------
        else:
            raise ValueError("Method {} not recognized.".format(method))
        #--------------------------------------------------------
        ### STEP 2 ### set coefficients to 1
        if required_operator_endpoints is not None:
            required_operator_endpoints.set_coeffs_one()
        if required_operator_integrand is not None:
            required_operator_integrand.set_coeffs_one()
        #--------------------------------------------------------
        return required_operator_endpoints, required_operator_integrand

    def get_required_operators_from_constraints(
            self, 
            method: str, 
            constraints: list
            ) -> tuple[dict, dict]:
        """
        Get the required operators for the learning method.

        Given a learning method and a list of Constraints,
        returns a dictionary of (initial_state, simulation_time) pairs and the
        corresponding operators that need to be measured for the given state and time.
        NOTE: This function is slow!!!

        Parameters
        ----------
        method : str
            The learning method for which the operator is generated.
            Options are "O3KZ", "BAL", "ZYLB" and "LZH".
        constraints : list of Constraints
            List of Constraint objects to be used for learning.
        
        Returns
        -------
        dict
            Required terms to be measured for each (initial_state, simulation_time) pair.
            The result is a dictionary of key:qop_list pairs where 
            key is a (initial_state, simulation_time) pair and
            qop_list is a list of QuantumOperator objects that need to be measured for the given state and time.
        """
        required_terms_endpoints = {}
        required_terms_integrand = {}
        for constraint in constraints:
            initial_state = constraint.initial_state
            simulation_times = constraint.simulation_times
            constraint_operator = constraint.constraint_operator
            #--------------------------------------------------------
            ### STEP 1 ### get required operators for given constraint operator
            required_operator_endpoints, required_operator_integrand = self.get_required_operator(method, constraint_operators=[constraint_operator])
            #--------------------------------------------------------
            ### STEP 2 ### add required operators to required_terms
            endpoints = [simulation_times[0],simulation_times[-1]]
            for time in simulation_times:
                ### add endpoint operators
                if time in endpoints:
                    if (initial_state,time) not in required_terms_endpoints.keys():
                        required_terms_endpoints[(initial_state,time)] = required_operator_endpoints
                    else:
                        required_terms_endpoints[(initial_state,time)] += required_operator_endpoints
                    ## re-set coeffs of required_operator to 1
                    required_terms_endpoints[(initial_state,time)].set_coeffs_one()
                #---------
                ### add integrand operators
                if (initial_state,time) not in required_terms_integrand.keys():
                    required_terms_integrand[(initial_state,time)] = required_operator_integrand
                else:
                    required_terms_integrand[(initial_state,time)] += required_operator_integrand
                ## re-set coeffs of required_operator to 1
                required_terms_integrand[(initial_state,time)].set_coeffs_one()
        #--------------------------------------------------------
        return required_terms_endpoints, required_terms_integrand

    # @profile
    def get_constraint_tensors(
            self, 
            constraints: list,
            method: str, 
            nshots: int | list = -1, 
            required_terms: tuple | None = None, 
            evaluate_variance: bool = False, 
            Gaussian_noise: bool = False, 
            use_exact_initial_values: bool = False, 
            min_nshots_per_term: int = 1, 
            label: str = None, 
            print_timers: bool = False, 
            ) -> None:
        """
        Calculate constraint tensors for learning.

        Adds constraint tensor and vector to the ansatz object.
        If n_resampling is set, saves constraint_tensors_samples instead of constraint_tensors.
        NOTE: For BAL method a constraint is used as soon as there is a single non-nan entry for the integral
        NOTE: The new bottleneck is RAM and calculation of the constraint tensor entries?

        Parameters
        ----------
        method : str
            Method for which to generate the constraint tensor and vector.
            Options are "ZYLB", "BAL" or "LZHx" for integer x>=1).
        nshots : int or list of ints, optional
            Number of shots used to estimate expectation values.
            For LZHx, ZYLB, O3KZd entries, each expectation value is estimated using nshots number of measurements.
            For integration in O3KZi, BAL entries, the number of measurements at each time step equals nshots/ntimes.
            If nshots=0, exact expectation values are used.
            If nshots=-1, all available measurements are used. 
            If nshots is a list, get_constraint_tensors() is called recursively for each value.
            NOTE: For ZYLB method, nshots is set to ``0`` at ``t=0``.
            Default is ``-1``.
        required_terms : tuple, optional
            Required terms to be measured for the given states and times.
            required_terms is a tuple of dicts of (state,time):qop pairs, 
            where qop is the QuantumOperator to be measured at the given state and time.
            required_terms[0] is the required terms for the endpoints.
            required_terms[1] is the required terms for the integrand.
            If None, the required terms are determined automatically.
            Default is None.
        evaluate_variance : bool, optional
            If True, evaluates the variance of the constraint tensors.
            Default is False.
        Gaussian_noise : bool, optional
            If True, the exact expectation values
            with added Gaussian noise with variance var/nshots
            are used for the constraint tensors,
            instead of using the sampled expectation values.
            Default is False.
        use_exact_initial_values : bool, optional
            If True, expectation values at t=0 are evaluated exactly.
            Default is False.
        min_nshots_per_term : int, optional
            Minimum number of shots required for estimating expectation values.
            If the number of shots is smaller than min_nshots_per_term, 
            then the corresponding expectation value and variance are set to np.nan.
            Default is ``1``.
        label : str, optional
            Label added to the key under which the constraint tensor and vector are saved.
            The key is set to (label, method, nshots[nsinx]).
            Default is method.
        print_timers : bool
            If True, print timers for different step in the function.
            Default is False.

        Fixed parameters (fixed to default values for now)
        -------------------------------------------------
        n_resampling : (float,int), optional
            Number of samples constraint tensors created,
            each of which uses a different sample of measurements for evaluating the expectation values.
            n_resampling[0] is the ratio of constraints for resampling.
            n_resampling[1] is the number of samples.
            If set, saves constraint_tensors_samples instead of constraint_tensors.
            NOTE: Fixed to default value of ``(0,1)``. 
        resampling_replace : bool, optional
            If True, resampling of measurements is done with replacement.
            NOTE: Fixed to default value of False.
        """
        n_resampling = (0,1)
        resampling_replace = False
        if isinstance(nshots, int):
            nshots = [nshots]
        if label is None:
            label = method
        if self.data_set is None:
            raise ValueError("No measurement data provided.")
        # ----------------------------------------------------------------
        ### STEP 1 ### get the constraint tensor and vector
        #################################
        ### O3KZ method for integrals ###
        #################################
        if method == "O3KZi":
            ### STEP 1 ### setup constraint tensors
            # setup constraint tensor for ansatz_operator
            shape = tuple([len(nshots)] + [n_resampling[1]] + [len(constraints)] + [len(self.ansatz_operator.terms.keys())])
            constraint_tensor_list = np.zeros(shape) 
            var_constraint_tensor_list = np.zeros(shape) 
            constraint_vector_list = [None for inx in range(len(nshots))]
            var_constraint_vector_list = [None for inx in range(len(nshots))]
            # setup constraint tensor for ansatz_dissipators
            if self.ansatz_dissipators is not None:
                nterms_diss = len(self.ansatz_dissipators)
                shape_dissipators = tuple([len(nshots)] + [n_resampling[1]] + [len(constraints)] + [len(self.ansatz_operator.terms.keys())] + [nterms_diss])
                constraint_tensor_dissipators_list = np.zeros(shape_dissipators) 
                var_constraint_tensor_dissipators_list = np.zeros(shape_dissipators) 
            # -----------------------------------------------------
            ### STEP 2 ### evaluate entries of constraint tensors
            # loop over constraints (state,times)
            for cinx, constraint in enumerate(constraints):
                state = constraint.initial_state
                times = constraint.simulation_times
                constraint_operator = constraint.constraint_operator
                # set start and end time
                time1 = times[0]
                time2 = times[-1]
                ### coherent terms (Hamiltonian)
                coherent_terms = list(self.ansatz_operator.terms.values())
                # get expectation values from data set res = (qoqs,nshots,samples)
                exp2_list, var2_list = self.data_set.evaluate_observable(state=state,time=time2,qop_list=coherent_terms,nshots=nshots,evaluate_variance=True,Gaussian_noise=Gaussian_noise,use_exact_initial_values=use_exact_initial_values,n_resampling=n_resampling,resampling_replace=resampling_replace,min_nshots_per_term=min_nshots_per_term)
                exp1_list, var1_list = self.data_set.evaluate_observable(state=state,time=time1,qop_list=coherent_terms,nshots=nshots,evaluate_variance=True,Gaussian_noise=Gaussian_noise,use_exact_initial_values=use_exact_initial_values,n_resampling=n_resampling,resampling_replace=resampling_replace,min_nshots_per_term=min_nshots_per_term)
                o3kzval_list = np.subtract(exp2_list, exp1_list)
                var_o3kzval_list = np.add(var2_list, var1_list)  
                # change dimensions of o3kzval_list (qoqs,nshots,samples) -> (nshots,samples,qops)
                # change dimensions of o3kzval_list from (0,1,2) to (1,2,0)
                o3kzval_list = np.transpose(o3kzval_list, (1,2,0))
                var_o3kzval_list = np.transpose(var_o3kzval_list, (1,2,0))
                # save to constraint tensor
                constraint_tensor_list[:,:,cinx,:] = np.real(o3kzval_list)
                var_constraint_tensor_list[:,:,cinx,:] = np.real(var_o3kzval_list)
                # -----------------------------------------------------
                ### terms for dissipators (Lindblad operators) ###
                if self.ansatz_dissipators is not None:
                    ### get nshots for integrand
                    nshots_integrand = nshots
                    nshots_integrand = np.multiply(nshots,constraint.nshots_ratio_integrand)
                    nshots_integrand[nshots_integrand<0] = -1
                    nshots_integrand[(nshots_integrand > 0) & (nshots_integrand < 1)] = 1
                    nshots_integrand = nshots_integrand.astype(int)
                    ### Get all required terms
                    required_terms_diss = qop_sum([diss(self.ansatz_operator) for diss in self.ansatz_dissipators])
                    required_terms_diss.remove_zero_coeffs()
                    required_terms_diss.set_coeffs_one()
                    # add identity term
                    identity = QuantumOperator(self.Nions, terms={"I"*self.Nions:1})
                    required_terms_diss = required_terms_diss + identity
                    required_terms_diss = list(required_terms_diss.terms.values())
                    ### pre-evaluate all required terms
                    expvals_list = []
                    varvals_list = []
                    for time in times:
                        expvals, varvals = self.data_set.evaluate_observable(state=state,time=time,qop_list=required_terms_diss,nshots=nshots_integrand,evaluate_variance=True,Gaussian_noise=Gaussian_noise,use_exact_initial_values=use_exact_initial_values,n_resampling=n_resampling,resampling_replace=resampling_replace,min_nshots_per_term=min_nshots_per_term)
                        expvals_list.append(expvals)
                        varvals_list.append(varvals)
                    # change dimensions of expvals_list from (0,1,2,3) -> (1,2,3,0)
                    expvals_list = np.transpose(expvals_list, (1,2,3,0))  # (ntimes, nterms, nshots, nsamples) to (nterms, nshots, nsamples, ntimes)
                    varvals_list = np.transpose(varvals_list, (1,2,3,0))  # (ntimes, nterms, nshots, nsamples) to (nterms, nshots, nsamples, ntimes)
                    ### pre-evaluate all integrals
                    evaluated_terms = {}
                    for terminx, term in enumerate(required_terms_diss):
                        eval_list = np.zeros((len(nshots),n_resampling[1]))
                        var_eval_list = np.zeros((len(nshots),n_resampling[1]))
                        for nsinx in range(len(nshots)):
                            for sampinx in range(n_resampling[1]):
                                # var(integral(integrand)) = sum(var(integrand)) * dt**2
                                var_evalval = np.sum(varvals_list[terminx,nsinx,sampinx])*(times[1]-times[0])**2
                                evalval = -1* sp.integrate.simpson(expvals_list[terminx,nsinx,sampinx],x=times)
                                # check if evalval is real
                                if not np.isclose(evalval.imag,0):
                                    print("Imaginary part of evalval is not zero, but " + str(np.round(evalval.imag,3)) + ".")
                                eval_list[nsinx,sampinx] = np.real(evalval)
                                var_eval_list[nsinx,sampinx] = np.real(var_evalval)
                        # add to evaluated terms
                        evaluated_terms[term.pauli_type] = [eval_list,var_eval_list]
                    ### evaluate constraint matrix elements
                    for terminx, term in enumerate(self.ansatz_operator.terms.values()):
                        for dissinx, diss in enumerate(self.ansatz_dissipators):
                            dissterm = diss(term)
                            if dissterm.is_zero():
                                continue
                            dissterm.remove_zero_coeffs()
                            ## evaluate dissterm from evaluated terms
                            mo3kzval_list = np.sum([np.multiply(evaluated_terms[term.pauli_type][0],term.coeff) for term in dissterm.terms.values()],axis=0)
                            var_mo3kzval_list = np.sum([np.multiply(evaluated_terms[term.pauli_type][1],term.coeff) for term in dissterm.terms.values()],axis=0)
                            # check if values are real
                            if not np.isclose(mo3kzval_list.imag,0):
                                print("Imaginary part of mo3kzval is not zero, but " + str(np.round(mo3kzval_list.imag,3)) + ".")
                            # add to constraint tensor
                            constraint_tensor_dissipators_list[:,:,cinx,terminx,dissinx] = np.real(mo3kzval_list)
                            var_constraint_tensor_dissipators_list[:,:,cinx,terminx,dissinx] = np.real(var_mo3kzval_list)
        # -----------------------------------------------------
        ##################
        ### BAL method ###
        ##################
        # NOTE: The bottleneck is evaluating the constraint tensor? and RAM?
        elif method == "BAL":
            tm1 = tm.time()
            # -----------------------------------------------------
            ### STEP 1 ### setup constraint vector and tensor
            ## get number of free parameters
            nterms = 0
            if self.ansatz_operator is not None:
                nterms += len(self.ansatz_operator.terms.keys())
            if self.ansatz_dissipators is not None:
                nterms += len(self.ansatz_dissipators)
            ## setup constraint vector
            shape_vec = tuple([len(nshots)] + [n_resampling[1]] + [len(constraints)])
            constraint_vector_list = np.zeros(shape_vec) 
            var_constraint_vector_list = np.zeros(shape_vec) 
            ## setup constraint tensor
            shape = tuple([len(nshots)] + [n_resampling[1]] + [len(constraints)] + [nterms])
            constraint_tensor_list = np.zeros(shape) 
            var_constraint_tensor_list = np.zeros(shape) 
            if print_timers:
                tm2 = tm.time()
                pstr_get_con_tens = "time for setup tensors and get nruns = {}".format(tm2-tm1)
                ic(pstr_get_con_tens)
            # -----------------------------------------------------
            ### STEP 2 ### determine all required terms for BAL method (NOTE: this is the bottleneck)
            if required_terms is not None:
                pstr_get_con_tens = "use provided required terms for BAL method"
                ic(pstr_get_con_tens)
                required_terms_endpoints, required_terms_integrand = required_terms
            else:
                pstr_get_con_tens = "automatically determine required terms for BAL method"
                ic(pstr_get_con_tens)
                # required_terms are dicts of (state,time):qop pairs
                required_terms_endpoints, required_terms_integrand = self.get_required_operators_from_constraints(method, constraints)
            ### check if all required terms have coeff 1
            for key, qop in required_terms_endpoints.items():
                if not np.allclose(qop.coeffs(),1):
                    raise ValueError("Required term for BAL method does not have coeff ``1``.")
            for key, qop in required_terms_integrand.items():
                if not np.allclose(qop.coeffs(),1):
                    raise ValueError("Required term for BAL method does not have coeff ``1``.")
            if print_timers:
                tm3 = tm.time()
                pstr_get_con_tens = "time get required terms = {}".format(tm3-tm2)
                ic(pstr_get_con_tens)
            # -----------------------------------------------------
            ### STEP 3 ### pre-evaluate all required terms for endpoints
            # structure of expvals_endpoints: expvals_endpoints[(state,time)] = expvals
            # expvals shape = (qops,nshots,samples)
            expvals_endpoints = {}
            varvals_endpoints = {}
            for key, qop in required_terms_endpoints.items():
                state = key[0]
                time = key[1]
                qop_terms = list(qop.terms.values())
                # expvals shape = (qops,nshots,samples)
                expvals, varvals = self.data_set.evaluate_observable(state=state,time=time,qop_list=qop_terms,nshots=nshots,Gaussian_noise=Gaussian_noise,use_exact_initial_values=use_exact_initial_values,n_resampling=n_resampling,resampling_replace=resampling_replace,evaluate_variance=evaluate_variance,min_nshots_per_term=min_nshots_per_term)
                expvals_endpoints[key] = {term.pauli_type:expvals[term_inx,:,:] for term_inx,term in enumerate(qop_terms)}
                if evaluate_variance:
                    varvals_endpoints[key] = {term.pauli_type:varvals[term_inx,:,:] for term_inx,term in enumerate(qop_terms)}  
            if print_timers:
                tm4 = tm.time()
                pstr_get_con_tens = "time for pre-evaluate required terms for endpoints = {}".format(tm4-tm3)
                ic(pstr_get_con_tens)
            ## delete unused variables
            del required_terms_endpoints, expvals, varvals
            # -----------------------------------------------------
            ### STEP 4 ### evaluate entries of constraint vector
            for cinx, constraint in enumerate(constraints):
                state = constraint.initial_state
                times = constraint.simulation_times
                constraint_operator = constraint.constraint_operator
                # set start and end time
                time1 = times[0]
                time2 = times[-1]
                # get expectation values from data set shape exp2_list = (nshots,samples)
                exp2_list = np.sum([term.coeff*expvals_endpoints[(state,time2)][term.pauli_type] for term in constraint_operator.terms.values()],axis=0)
                exp1_list = np.sum([term.coeff*expvals_endpoints[(state,time1)][term.pauli_type] for term in constraint_operator.terms.values()],axis=0)
                balval_list = np.subtract(exp2_list, exp1_list)
                # save to constraint tensor
                constraint_vector_list[:,:,cinx] = np.real(balval_list)
                ### get variances from data set shape var2_list = (nshots,samples)
                if evaluate_variance:
                    var2_list = np.sum([term.coeff**2*varvals_endpoints[(state,time2)][term.pauli_type] for term in constraint_operator.terms.values()],axis=0)
                    var1_list = np.sum([term.coeff**2*varvals_endpoints[(state,time1)][term.pauli_type] for term in constraint_operator.terms.values()],axis=0)
                    # TODO variances wrong?
                    var_balval_list = np.add(var2_list, var1_list)
                    var_constraint_vector_list[:,:,cinx] = np.real(var_balval_list)
            if print_timers:
                tm5 = tm.time()
                pstr_get_con_tens = "time for evaluate entries of constraint vector = {}".format(tm5-tm4)
                ic(pstr_get_con_tens)
            ## delete unused variables
            del expvals_endpoints, exp2_list, exp1_list, balval_list
            if evaluate_variance:
                del varvals_endpoints, var2_list, var1_list
            # -----------------------------------------------------
            ### STEP 5 ### pre-evaluate all required terms for the integrand
            ## get list of all required nshots_ratio_integrand
            nshots_ratio_integrand_list = []
            for constraint in constraints:
                if constraint.nshots_ratio_integrand not in nshots_ratio_integrand_list:
                    nshots_ratio_integrand_list.append(constraint.nshots_ratio_integrand)
            if len(nshots_ratio_integrand_list) > 1:
                raise ValueError("Different nshots_ratio_integrand values for different constraints not implemented yet, nshots_ratio_integrand={}".format(nshots_ratio_integrand_list))
            nshots_ratio_integrand = nshots_ratio_integrand_list[0]
            ## get nshots for integrand
            nshots_integrand = nshots
            nshots_integrand = np.multiply(nshots,nshots_ratio_integrand)
            nshots_integrand[nshots_integrand<0] = -1
            nshots_integrand[(nshots_integrand > 0) & (nshots_integrand < 1)] = 1
            nshots_integrand = nshots_integrand.astype(int)
            ## evaluate terms for integrand
            # structure of expvals_integrand: expvals_integrand[(state,time)] = expvals
            # shape expvals[term] = (nshots,samples)
            expvals_integrand = {}
            varvals_integrand = {}
            keyinx = 1
            for key, qop in required_terms_integrand.items():
                state = key[0]
                time = key[1]
                qop_terms = list(qop.terms.values())
                # expvals shape = (qops,nshots,samples)
                tm11 = tm.time()
                expvals, varvals = self.data_set.evaluate_observable(state=state,time=time,qop_list=qop_terms,nshots=nshots_integrand,Gaussian_noise=Gaussian_noise,use_exact_initial_values=use_exact_initial_values,n_resampling=n_resampling,resampling_replace=resampling_replace,evaluate_variance=evaluate_variance,min_nshots_per_term=min_nshots_per_term)
                if print_timers:
                    tm12 = tm.time()
                    pstr_get_con_tens = "key: {}({}), time for evaluate required terms for integrand = {}, rest time estimate: {}".format(key[0].str(), key[1] ,tm12-tm11,(tm12-tm11)*(len(required_terms_integrand)-keyinx))
                    ic(pstr_get_con_tens)
                expvals_integrand[key] = {term.pauli_type:expvals[term_inx,:,:] for term_inx,term in enumerate(qop_terms)}
                if evaluate_variance:
                    varvals_integrand[key] = {term.pauli_type:varvals[term_inx,:,:] for term_inx,term in enumerate(qop_terms)}
                keyinx += 1
            if print_timers:
                tm6 = tm.time()
                pstr_get_con_tens = "time for pre-evaluate required terms for integrand = {}".format(tm6-tm5)
                ic(pstr_get_con_tens)
            ## delete unused variables
            del required_terms_integrand, expvals, varvals
            # -----------------------------------------------------
            ### STEP 6 ### evaluate entries of constraint tensor
            for cinx, constraint in enumerate(constraints):
                if cinx==0:
                    tm61 = tm.time()
                state = constraint.initial_state
                times = constraint.simulation_times
                times_integrand = times #times[1:]
                constraint_operator = constraint.constraint_operator
                #-------------------
                ### Get all required terms
                # add identity term
                identity = QuantumOperator(self.Nions, terms={"I"*self.Nions:1})
                required_terms_constraint = identity
                if self.ansatz_operator is not None:
                    required_terms_constraint += abs(constraint_operator.commutator(self.ansatz_operator))
                if self.ansatz_dissipators is not None:
                    required_terms_constraint += qop_sum([abs(diss(constraint_operator)) for diss in self.ansatz_dissipators])
                required_terms_constraint.remove_zero_coeffs()
                required_terms_constraint.set_coeffs_one()
                required_terms_constraint = list(required_terms_constraint.terms.values())
                #-------------------
                ### pre-evaluate all integrals
                # evaluated_integrals[term.pauli_type] = [eval_list,var_eval_list]
                # shape eval_list = (nshots,samples)
                evaluated_integrals = {}
                for terminx, term in enumerate(required_terms_constraint):
                    eval_list = np.zeros((len(nshots),n_resampling[1]))
                    var_eval_list = np.zeros((len(nshots),n_resampling[1]))
                    for nsinx in range(len(nshots)):
                        for sampinx in range(n_resampling[1]):
                            ## get expvals and varvals for integrand
                            expvals = np.array([expvals_integrand[(state,time)][term.pauli_type][nsinx,sampinx] for time in times_integrand])
                            if evaluate_variance:
                                varvals = np.array([varvals_integrand[(state,time)][term.pauli_type][nsinx,sampinx] for time in times_integrand])
                            ## remove nan entries from integrand values and corresponding times
                            non_nan_indices = np.where(~np.isnan(expvals.real) & ~np.isnan(expvals.imag))[0]
                            times_integrand = np.array(times)[non_nan_indices]
                            #--------
                            ### evaluate integral 
                            if len(non_nan_indices) == 0:
                                evalval = np.nan
                            else:
                                integrand_values = expvals[non_nan_indices]
                                # evaluate integral
                                # evalval = sp.integrate.simpson(integrand_values,x=times_integrand)
                                evalval = np.trapz(integrand_values, times_integrand)
                                # check if mbalval is real
                                if not np.isclose(evalval.imag,0):
                                    print("Imaginary part of evalval is not zero, but " + str(np.round(evalval.imag,3)) + ".")
                                evalval = np.real(evalval)
                            # add to eval_list
                            eval_list[nsinx,sampinx] = evalval
                            #--------
                            ### evaluate variance of integral
                            if evaluate_variance:
                                if len(non_nan_indices) == 0:
                                    var_evalval = np.nan
                                else:
                                    integrand_variances = varvals[non_nan_indices]
                                    # var(integral(integrand)) = sum(var(integrand)) * dt**2
                                    var_evalval = np.sum(integrand_variances)*(times[1]-times[0])**2
                                    # check if var_evalval is real
                                    if not np.isclose(var_evalval.imag,0):
                                        print("Imaginary part of var_evalval is not zero, but " + str(np.round(var_evalval.imag,3)) + ".")
                                    var_evalval = np.real(var_evalval)
                                var_eval_list[nsinx,sampinx] = var_evalval
                    ## add to evaluated terms
                    evaluated_integrals[term.pauli_type] = [eval_list,var_eval_list]
                ## delete unused variables
                del required_terms_constraint, eval_list, expvals, integrand_values, non_nan_indices, times_integrand
                if evaluate_variance:
                    del var_eval_list, varvals, integrand_variances
                # -------------------
                ### coherent terms in constraint tensor ###
                opinx = 0
                if self.ansatz_operator is not None:
                    for terminx, term in enumerate(self.ansatz_operator.terms.values()):
                        commterm = -1j*constraint_operator.commutator(term)
                        if commterm.is_zero():
                            opinx +=1
                            continue
                        commterm.remove_zero_coeffs()
                        ## evaluate commterm from evaluated terms
                        mbalval_list = np.sum([np.multiply(evaluated_integrals[tmpterm.pauli_type][0],tmpterm.coeff) for tmpterm in commterm.terms.values()],axis=0)
                        constraint_tensor_list[:,:,cinx,opinx] = np.real(mbalval_list)
                        if evaluate_variance:
                            var_mbalval_list = np.sum([np.multiply(evaluated_integrals[tmpterm.pauli_type][1],tmpterm.coeff) for tmpterm in commterm.terms.values()],axis=0)
                            var_constraint_tensor_list[:,:,cinx,opinx] = np.real(var_mbalval_list)
                        opinx +=1
                # -------------------
                ### terms for dissipators (Lindblad operators) ###
                if self.ansatz_dissipators is not None:
                    for dissinx, diss in enumerate(self.ansatz_dissipators):
                        dissterm = diss(constraint_operator)
                        # check if dissterm is zero
                        if dissterm.is_zero():
                            opinx +=1
                            continue
                        dissterm.remove_zero_coeffs()
                        ## evaluate dissterm from evaluated terms
                        mbalval_list = np.sum([np.multiply(evaluated_integrals[tmpterm.pauli_type][0],tmpterm.coeff) for tmpterm in dissterm.terms.values()],axis=0)
                        constraint_tensor_list[:,:,cinx,opinx] = np.real(mbalval_list)
                        if evaluate_variance:
                            var_mbalval_list = np.sum([np.multiply(evaluated_integrals[tmpterm.pauli_type][1],tmpterm.coeff) for tmpterm in dissterm.terms.values()],axis=0)
                            var_constraint_tensor_list[:,:,cinx,opinx] = np.real(var_mbalval_list)
                        opinx +=1
                if cinx==20:
                    tm62 = tm.time()
                    pstr_get_con_tens = "... time for evaluate single row of constraint tensor={}, total time estimate={}".format((tm62-tm61)/20,(tm62-tm61)*len(constraints)/20)
            if print_timers:
                tm7 = tm.time()
                pstr_get_con_tens = "time for evaluate entries of constraint tensor = {}".format(tm7-tm6)
                ic(pstr_get_con_tens)
                ic("--------------------------------")
        # -----------------------------------------------------
        ###################
        ### ZYLB method ###
        ###################
        # TODO we evaluate exact expectation values at 0 for ZYLB method
        elif method == "ZYLB":
            raise ValueError("ZYLB method not implemented yet. Has to be updated for n_resampling.")
            # setup constraint tensor and vector
            nterms=0
            if self.ansatz_operator is not None:
                nterms += len(self.ansatz_operator.terms.keys())
            if ansatz_dissipators is not None:
                nterms += len(ansatz_dissipators)
            constraint_tensor_list = np.zeros((len(nshots),len(initial_states)*len(simulation_times)*len(constraint_operators),nterms), dtype=complex)
            constraint_vector_list = np.zeros((len(nshots),len(initial_states)*len(simulation_times)*len(constraint_operators)), dtype=complex)
            var_constraint_tensor_list = np.zeros((len(nshots),len(initial_states)*len(simulation_times)*len(constraint_operators),nterms), dtype=complex)
            var_constraint_vector_list = np.zeros((len(nshots),len(initial_states)*len(simulation_times)*len(constraint_operators)), dtype=complex)
            for sinx, state in enumerate(initial_states):
                for tinx, times in enumerate(simulation_times):
                    for cinx, constraint in enumerate(constraint_operators):
                        ### constraint vector ###
                        # TODO: function for estimating derivative below
                        tmp_expvals = np.transpose([self.data_set.evaluate_observable(state,time,[constraint],nshots=nshots,Gaussian_noise=Gaussian_noise,use_exact_initial_values=True) for time in simulation_times[tinx]])
                        bzylbval_list = np.array([uf.estimate_derivative(tmp_expvals[nsinx],simulation_times[tinx]) for nsinx in range(len(nshots))])
                        #TODO variance estimation wrong 
                        # tmp_varvals = np.transpose([self.data_set.evaluate_observable(state,time,constraint,nshots=nshots,evaluate_variance=True,Gaussian_noise=Gaussian_noise) for time in simulation_times[tinx]])
                        # check if bbalval_list is real
                        if not np.allclose(np.round(bzylbval_list.imag,0),0):
                            print("Imaginary part of ZYLB constraint vector element is not zero, but " + str(np.round(bzylbval_list.imag,3)) + ".")
                        bzylbval_list = np.real(bzylbval_list)
                        # var_bzylbval_list = np.real(var_bzylbval_list)
                        # fill bbalval_list into constraint vector
                        constraint_vector_list[:,sinx*len(simulation_times)*len(constraint_operators)+tinx*len(constraint_operators)+cinx] = bzylbval_list
                        # var_constraint_vector_list[:,sinx*len(constraint_operators)+cinx] = var_bzylbval_list
                    
                        ### coherent terms in constraint tensor ###
                        opinx=0
                        for term_key in self.ansatz_operator.terms.keys():
                            # required operator for coherent terms
                            comm = -1j * constraint.commutator(self.ansatz_operator.terms[term_key])
                            # times for derivative estimation
                            expvals_list = np.transpose([self.data_set.evaluate_observable(state,time,[comm],nshots=nshots,Gaussian_noise=Gaussian_noise,use_exact_initial_values=True) for time in simulation_times[tinx]])
                            for nsinx in range(len(nshots)):
                                mzylbval = expvals_list[nsinx,0]
                                # var_mzylbval = -1j * varvals_list[nsinx,0]
                                # check if mbalval is real
                                if not np.isclose(mzylbval.imag,0):
                                    print("Imaginary part of ZYLB constraint matrix element is not zero, but " + str(np.round(mzylbval.imag,3)) + ".")
                                mzylbval = np.real(mzylbval)
                                # var_mzylbval = np.real(var_mzylbval)
                                # add to constraint matrix
                                constraint_tensor_list[nsinx,sinx*len(simulation_times)*len(constraint_operators)+tinx*len(constraint_operators)+cinx,opinx] = mzylbval
                                # var_constraint_tensor_list[nsinx,sinx*len(constraint_operators)+cinx,opinx] = var_mzylbval
                            opinx += 1

                        ## dissipative terms in constraint matrix ###
                        if ansatz_dissipators is not None:
                            for diss in ansatz_dissipators:
                                dissterm = diss(constraint)

                                # # required operator for dissipative terms
                                # qop_diss = diss.to_quantum_operator()
                                # # qop_diss = QuantumOperator(self.Nions)
                                # qop_diss.add_term(diss)
                                # qop_dissdag = qop_diss.dagger()
                                # term1 = qop_dissdag * constraint.commutator(qop_diss)
                                # term2 = qop_dissdag.commutator(constraint) * qop_diss
                                # dissterm = term1 + term2

                                expvals_list = np.transpose([self.data_set.evaluate_observable(state,time,[dissterm],nshots=nshots,Gaussian_noise=Gaussian_noise,use_exact_initial_values=True) for time in simulation_times[tinx]])
                                for nsinx in range(len(nshots)):
                                    # estimate derivative at time t=0
                                    mzylbval = expvals_list[nsinx,0]
                                    # var_mzylbval = 1/2 * varvals_list[nsinx,0]
                                    # check if mbalval is real
                                    if not np.isclose(mzylbval.imag,0):
                                        print("Imaginary part of ZYLB constraint matrix element is not zero, but " + str(np.round(mzylbval.imag,3)) + ".")
                                    mzylbval = np.real(mzylbval)
                                    # var_mzylbval = np.real(var_mzylbval)
                                    # add to constraint matrix
                                    constraint_tensor_list[nsinx,sinx*len(simulation_times)*len(constraint_operators)+tinx*len(constraint_operators)+cinx,opinx] = mzylbval
                                    # var_constraint_tensor_list[nsinx,sinx*len(constraint_operators)+cinx,opinx] = var_mzylbval
                                opinx += 1
        # -----------------------------------------------------
        ###################################
        ### O3KZ method for derivatives ###
        ###################################
        elif method == "O3KZd":
            raise NotImplementedError("Method O3KZd not implemented yet.")
            # setup constraint tensor for ansatz_operator
            shape = [len(nshots)] + [len(initial_states)*len(simulation_times)] + [len(self.ansatz_operator.terms.keys())]
            shape = tuple(shape)
            constraint_tensor_list = np.zeros(shape, dtype=complex)
            var_constraint_tensor_list = np.zeros(shape, dtype=complex)
            constraint_vector_list = [None for inx in range(len(nshots))]
            var_constraint_vector_list = [None for inx in range(len(nshots))]
            # setup constraint tensor for ansatz_dissipators
            if ansatz_dissipators is not None:
                shape_dissipators = [len(nshots)] + [len(initial_states)*len(simulation_times)] + [len(self.ansatz_operator.terms.keys())] + [len(ansatz_dissipators)]
                shape_dissipators = tuple(shape_dissipators)
                constraint_tensor_dissipators_list = np.zeros(shape_dissipators, dtype=complex)
                var_constraint_tensor_dissipators_list = np.zeros(shape_dissipators, dtype=complex)
            # evaluate entries of constraint tensors
            for sinx, state in enumerate(initial_states):
                for tinx, times in enumerate(simulation_times):
                    ### coherent terms (Hamiltonian)
                    for terminx, term in enumerate(self.ansatz_operator.terms.values()):
                        # expectation values for evaluating derivative
                        expvals_list = []
                        varvals_list = []
                        for time in times:
                            expvals, varvals = self.data_set.evaluate_observable(state,time,term,nshots=nshots,evaluate_variance=True,Gaussian_noise=Gaussian_noise)
                            expvals_list.append(expvals)
                            varvals_list.append(varvals)
                        expvals_list = np.transpose(expvals_list)
                        varvals_list = np.transpose(varvals_list)
                        for nsinx in range(len(nshots)):
                            o3kzval = estimate_derivative(expvals_list[nsinx],times)
                            var_o3kzval = 0  #TODO add variance of derivative
                            # check if mbalval is real
                            if not np.isclose(o3kzval.imag,0):
                                print("Imaginary part of O3KZd constraint matrix element is not zero, but " + str(np.round(o3kzval.imag,3)) + ".")
                            o3kzval = np.real(o3kzval)
                            var_o3kzval = np.real(var_o3kzval)
                            # add to constraint matrix
                            constraint_tensor_list[nsinx,sinx*len(simulation_times)+tinx,terminx] = o3kzval
                            var_constraint_tensor_list[nsinx,sinx*len(simulation_times)+tinx,terminx] = var_o3kzval
                    ### terms for dissipators (Lindblad operators)
                    if ansatz_dissipators is not None:
                        for terminx, term in enumerate(self.ansatz_operator.terms.values()):
                            for dissinx, diss in enumerate(ansatz_dissipators):
                                # required operator for dissipative terms (all term.coeff are 1)
                                qop_term = QuantumOperator(self.Nions, terms={term.pauli_type: term.coeff})
                                qop_diss = QuantumOperator(self.Nions, terms={diss.pauli_type: term.coeff})
                                qop_dissdag = qop_diss.dagger()
                                term1 = qop_dissdag * qop_term.commutator(qop_diss)
                                term2 = qop_dissdag.commutator(qop_term) * qop_diss
                                dissterm = term1 + term2
                                # #TODO: remove identities
                                # dissterm = dissterm.remove_identity()
                                # evaluate expectation values
                                time = times[0]   # choose first time for evaluating derivative
                                expvals, varvals = self.data_set.evaluate_observable(state,time,dissterm,nshots=nshots,evaluate_variance=True,Gaussian_noise=Gaussian_noise)
                                for nsinx in range(len(nshots)):
                                    var_mo3kzval = 0 # TODO add variance
                                    mo3kzval = -1/2 * expvals_list[nsinx]
                                    # check if mbalval is real
                                    if not np.isclose(mo3kzval.imag,0):
                                        print("Imaginary part of O3KZd constraint matrix element is not zero, but " + str(np.round(mo3kzval.imag,3)) + ".")
                                    mo3kzval = np.real(mo3kzval)
                                    var_mo3kzval = np.real(var_mo3kzval)
                                    # add to constraint matrix
                                    constraint_tensor_dissipators_list[nsinx,sinx*len(simulation_times)+tinx,terminx,dissinx] = mo3kzval
                                    var_constraint_tensor_dissipators_list[nsinx,sinx*len(simulation_times)+tinx,terminx,dissinx] = var_mo3kzval
        # -----------------------------------------------------
        ##################
        ### LZH method ###
        ##################
        elif method[0:3] == "LZH":
            raise ValueError("LZH method not implemented yet (has to be checked).")
            order = int(method[3:])
            # setup constraint tensor and vector
            shape = tuple([len(nshots)] + [len(initial_states)*len(simulation_times)] + [len(self.ansatz_operator.terms.keys())]*order)
            constraint_tensor_list = np.zeros(shape, dtype=complex)
            var_constraint_tensor_list = np.zeros(shape, dtype=complex)
            constraint_vector_list = [None for inx in range(len(nshots))]
            var_constraint_vector_list = [None for inx in range(len(nshots))]
            for sinx, state in enumerate(initial_states):
                for tinx, times in enumerate(simulation_times):
                    shape = [len(nshots)] + [len(self.ansatz_operator.terms.keys())]*order
                    constraint_tensor = np.zeros(shape, dtype=complex)
                    var_constraint_tensor = np.zeros(shape, dtype=complex)
                    # loop over product of terms in ansatz operator (for LZHx for x>1)
                    for term in it.product(enumerate(self.ansatz_operator.terms.keys()),repeat=order):
                        term_keys = [term_key for opinx, term_key in term]
                        opinxs = [opinx for opinx, term_key in term]
                        # get the required product operator
                        prod = math.prod([self.ansatz_operator.terms[term_key] for term_key in term_keys])
                        if prod.is_identity():
                            continue
                        # set start and end time
                        time1 = simulation_times[tinx][0]
                        time2 = simulation_times[tinx][-1]
                        # get expectation values from data set
                        exp2_list, var2_list = self.data_set.evaluate_observable(state,time2,[prod],nshots=nshots,evaluate_variance=True,Gaussian_noise=Gaussian_noise,use_exact_initial_values=use_exact_initial_values)
                        exp1_list, var1_list = self.data_set.evaluate_observable(state,time1,[prod],nshots=nshots,evaluate_variance=True,Gaussian_noise=Gaussian_noise,use_exact_initial_values=use_exact_initial_values)
                        lzhval_list = np.subtract(exp2_list, exp1_list)
                        var_lzhval_list = np.add(var2_list, var1_list)  
                        for nsinx in range(len(nshots)):
                            inxtuple = tuple([nsinx] + opinxs)
                            constraint_tensor[inxtuple] = lzhval_list[nsinx]
                            var_constraint_tensor[inxtuple] = var_lzhval_list[nsinx]
                    for nsinx in range(len(nshots)):
                        constraint_tensor_list[nsinx,sinx*len(simulation_times)+tinx] = constraint_tensor[nsinx]
                        var_constraint_tensor_list[nsinx,sinx*len(simulation_times)+tinx] = var_constraint_tensor[nsinx]
        else:
            raise ValueError("Method {} not recognized.".format(method))
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        ### STEP 2 ### add the constraint tensor and vector to the ansatz
        for nsinx in range(len(nshots)):
            if method[:3] == "LZH":
                tmpkey = (label, method, nshots[nsinx])
                # self.constraint_tensors[tmpkey] = [[constraint_tensor_list[nsinx,nsamp], var_constraint_tensor_list[nsinx,nsamp]] for nsamp in range(n_resampling[1])]
                self.constraint_tensors[tmpkey] = [constraint_tensor_list[nsinx, 0], var_constraint_tensor_list[nsinx, 0]]
            elif method in ["BAL","ZYLB"]:
                tmpkey = (label, method, nshots[nsinx]) 
                # self.constraint_tensors[tmpkey] = [[constraint_tensor_list[nsinx,nsamp],constraint_vector_list[nsinx,nsamp],var_constraint_tensor_list[nsinx,nsamp],var_constraint_vector_list[nsinx,nsamp]] for nsamp in range(n_resampling[1])]
                self.constraint_tensors[tmpkey] = [constraint_tensor_list[nsinx, 0], constraint_vector_list[nsinx, 0], var_constraint_tensor_list[nsinx, 0], var_constraint_vector_list[nsinx, 0]]
            elif method in ["O3KZi","O3KZd"]:
                tmpkey = (label, method, nshots[nsinx]) 
                if self.ansatz_dissipators is not None:
                    # self.constraint_tensors[tmpkey] = [[constraint_tensor_list[nsinx,nsamp],constraint_tensor_dissipators_list[nsinx,nsamp],var_constraint_tensor_list[nsinx,nsamp],var_constraint_tensor_dissipators_list[nsinx,nsamp]] for nsamp in range(n_resampling[1])]
                    self.constraint_tensors[tmpkey] = [constraint_tensor_list[nsinx, 0], constraint_tensor_dissipators_list[nsinx, 0], var_constraint_tensor_list[nsinx, 0], var_constraint_tensor_dissipators_list[nsinx, 0]]
                else:
                    # self.constraint_tensors[tmpkey] = [[constraint_tensor_list[nsinx,nsamp],None,var_constraint_tensor_list[nsinx,nsamp],None] for nsamp in range(n_resampling[1])]
                    self.constraint_tensors[tmpkey] = [constraint_tensor_list[nsinx, 0], None, var_constraint_tensor_list[nsinx, 0], None]

    def load_constraint_tensors(
            self,
            method: str,
            nshots: int,
            label: str | None = None,
            MHexact: bool = False,
            MDexact: bool = False,
            ) -> dict:
        """ 
        Load constraint tensors from the ansatz.

        Returns constraint_tensors with key = (label,method,nshots),
        from either ansatz.constraint_tensors or ansatz.constraint_tensors_samples.

        Parameters
        ----------
        method : str
            Choice of learning method.
            Options are "O3KZi","O3KZd","ZYLB","BAL".
        nshots : int
            Number of shots for constraint tensors and vectors.
        label : str
            Label for the constraint tensor.
            Default is method.
        MHexact : bool
            If True, the exact Hamiltonian constraints are loaded (no shot noise on MH)
            Only used for method="O3KZi" or method="O3KZd".
            Default is False.
        MDexact : bool
            If True, the dissipation correction is evaluated from exact expectation values (no shot noise on MD)
            Only used for method="O3KZi" or method="O3KZd".
            Default is False.

        Returns
        -------
        dict
            Loaded constraint tensors as a dict
            with entry method:tensors_list,
            where tensors_list is a list of all loaded constraint tensor.
        """
        if label is None:
            label = method
        ## get key for loading constraint tensors
        if method in ["O3KZi","O3KZd","ZYLB","BAL"]:
            key = (label,method,nshots)
        else:
            raise ValueError("Method {} not recognized.".format(method))
        #-----------------------------------------------------
        ### STEP 1 ### load constraint tensors from ansatz
        constraint_tensors = {method:self.constraint_tensors[key]}
        #-----------------------------------------------------
        ### STEP 2 ### optionally replace by exact constraint tensors
        ## load exact constraint tensors
        if MHexact or MDexact:
            exact_constraint_tensors = self.load_constraint_tensors(method=method, nshots=0, label=label)
        ## replace constraint tensors
        if MHexact and method=="O3KZi":
            for inx in range(len(exact_constraint_tensors[method])):
                constraint_tensors[method][inx][0] = exact_constraint_tensors[method][inx][0]
                constraint_tensors[method][inx][2] = exact_constraint_tensors[method][inx][2]
        if MDexact and method=="O3KZi":
            for inx in range(len(exact_constraint_tensors[method])):
                constraint_tensors[method][inx][1] = exact_constraint_tensors[method][inx][1]
                constraint_tensors[method][inx][3] = exact_constraint_tensors[method][inx][3]
        if MHexact and MDexact and method=="BAL":
            constraint_tensors[method] = exact_constraint_tensors[method]
        #-----------------------------------------------------
        return constraint_tensors

    def learn(
            self,
            learn_method: str,
            parametrizations: list | None = None,
            learn_label: str = "learn",
            nshots: int = -1,
            scale_method: str | None = None,
            scale_label: str = "scale",
            nshots_scale: int | None = None,
            scale_factor: int = 1,
            diss_method: str | None = None, 
            diss_label: str = "diss",
            nshots_diss: int | None = None,
            normalize_constraints: bool = False,
            MHexact: bool = False,
            MDexact: bool = False,
            MSexact: bool = False,
            num_cpus: int = 1,
            exclude_lowest_solutions: int = 0,
            ) -> None:
        """
        Learn the coefficients of the ansatz operator.

        Learn the ansatz parameters from ansatz.constraint_tensors 
        and add the solution to ansatz.results.

        Parameters
        ----------
        learn_method : str
            Choice of learning method.
            Options are "O3KZi", "ZYLB", "BAL" or "LZHx" for integer x>=1.
            If method is "LZHx", all constraint tensors up to order x are used.
        parametrizations : list of Parametrization objects
            Parametrizations used for learning.
            None is equivalent to a free parametrization.
        learn_label : str, optional
            Label of the constraint tensors used for learning.
            Default is "learn".
        nshots : int
            Number of shots used to estimate each constraint tensor elements. (nshots=ntimes*nshots_per_time).
            For LZHx, ZYLB, O3KZd entries, each expectation value is estimated with nshots shots.
            For integration in O3KZi, BAL entries, the number of shots at each time step equals nshots/ntimes.
            Default is ``-1``.
        normalize_constraints : bool, optional
            If True, each row in (M,b) is normalized by the maximum norm of the row of M.
            NOTE: This option only applies to the constraint tensors used for learning,
            not to the scale reconstruction and the dissipation learning.
            Default is False.
        scale_method : str 
            If set, the scale is reconstructed from the data set using the given method.
            Options are "BAL" or "ZYLB"
            Default is None.
        scale_label : str, optional
            Label of the constraint tensors used for scale reconstruction.
            Default is "scale".
        nshots_scale : int
            Analog to nshots, nshots_scale are number of measurement shots used for scale reconstruction.
            Default is nshots.
        scale_factor : float
            Factor by which the scale constraints are multiplied.
            Only used if scale_method is not None.
            Default is ``1``.
        diss_method : str
            If set, the dissipation is reconstructed from the data set using the given method.
            Options are "BAL" or "ZYLB"
            Default is None.
        diss_label : str, optional
            Label of the constraint tensors used for dissipation learning.
            Default is "diss".
        nshots_diss : int
            Analog to nshots, nshots_diss are number of measurement shots used for dissipation learning.
            Default is nshots.
        MHexact : bool, optional
            If True, the exact LZH tensor is used for learning (no shot noise on MH)
            Only used for method="O3KZi" or method="O3KZd".
            Default is False.
        MDexact : bool, optional
            If True, the dissipation correction is evaluated from exact expectation values (no shot noise on MD)
            Only used for method="O3KZi" or method="O3KZd".
            Default is False
        MSexact : bool
            If True, the scale correction is evaluated from exact expectation values (no shot noise on Mscale)
            Only used for method="O3KZi" or method="O3KZd".
            Default is False.
        num_cpus : int 
            Number of cpus used for parallelization.
            Default is 1.
        exclude_lowest_solutions : int, optional
            Number of lowest solutions to exclude from the result for the LZH or O3KZi learning methods.
            E.g. if exclude_lowest_solutions=1, we are solving for the second-lowest singular value.
            This can be useful if there exists a conserved quantity that should be excluded.
            Default is 0.


            - operator_exact (QuantumOperator object) [default: None]
                Exact Hamiltonian of the system.
                If given, the corresponding errors (exact, fit) are calculated and stored in the result object.   
            - prior (tuple of arrays) [default: None]
                Prior distribution for the free parameters
                Prior[0] is the mean and Prior[1] is the covariance matrix.


            - add_lsq_solution (bool) [default: False]
                If True, also learns the coefficients using a least squares solver.
            - get_variance_new (bool) [default: False]
                If True, also estimates the variance of the learned coefficients
                from the variance of the constraint tensors.
                NOTE: This requires evaluate_variances==True in the get_constraint_tensors() method.
        """
        if scale_method is not None and nshots_scale is None:
            nshots_scale = nshots
        if diss_method is not None and nshots_diss is None:
            nshots_diss = nshots
        pstr_learn = "enter learn method {} with nshots {} and {} cpus".format(learn_method, nshots, num_cpus)
        ic(pstr_learn)
        #------------------------------------------------------------
        ### STEP 1 ### get constraint tensors and vectors
        ## get constraint tensors for learning
        constraint_tensors = self.load_constraint_tensors(learn_method, nshots, label=learn_label, MHexact=MHexact, MDexact=MDexact)
        if normalize_constraints:
            constraint_tensors = Ansatz.normalize_constraint_tensors(constraint_tensors)
        ## get constraint tensors for scale reconstruction
        constraint_tensors_scale = None
        if scale_method is not None:
            constraint_tensors_scale = self.load_constraint_tensors(scale_method, nshots_scale, label=scale_label, MHexact=MSexact, MDexact=MSexact)
        ## get constraint tensors for separate dissipator learning
        constraint_tensors_diss = None
        if diss_method is not None:
            constraint_tensors_diss = self.load_constraint_tensors(diss_method, nshots_diss, label=diss_label, MHexact=MDexact, MDexact=MDexact)
        #------------------------------------------------------------
        ### STEP 2 ### get sample indices for resampling rows of constraint tensors (only rows of tensors for learning are sampled)
        constraint_sample_indices, jackknife_inflation_factors = self.get_constraint_sample_indices(label=learn_label)
        #------------------------------------------------------------
        ### STEP 3 ### solve learning equation including the overall scale (sequential or parallel)
        if num_cpus == 1:
            ## sequential version (loop over constraint tensors samples)
            allkeys = list(constraint_sample_indices.keys())
            nsamples_tot = max([len(constraint_sample_indices[key]) for key in allkeys])
            result_dict_list = []
            for sample_inx in tqdm(range(nsamples_tot), desc="Processing samples for errorbars"):
                # get sample indices for a single constraint tensor
                constraint_sample_indices_tmp = {key: [constraint_sample_indices[key][np.mod(sample_inx, len(constraint_sample_indices[key]))]] for key in constraint_sample_indices}
                # draw single sample of constraint tensors (learn, scale, diss)
                tensors_sample = Ansatz.draw_constraint_tensor_samples(constraint_tensors, constraint_sample_indices_tmp, constraint_tensors_scale=constraint_tensors_scale, constraint_tensors_diss=constraint_tensors_diss) 
                # learn coefficients for the chosen sample
                result_dict = self.learn_sampled_coefficients(tensors_sample, parametrizations=parametrizations)
                # save result for current sample
                result_dict_list.append(result_dict)
        else:
            ## parallel version
            raise NotImplementedError("parallel version requires qutip parfor")
            from qutip import parfor
            print("enter parallel learning, ncpus={}".format(num_cpus))
            result_dict_list = parfor(parallel_learn_sampled_coefficients, list(range(nsamples_tot)), ansatz=self, constraint_tensors=constraint_tensors, **kwargs)
        #------------------------------------------------------------
        ### STEP 4 ### store results in result object
        result = Result()
        # combine results into dictionary of lists (each list contains results for all samples)
        result_dict = {key: [list(it.chain(*group)) 
            for group in zip(*(data[key] for data in result_dict_list))]
                for key in result_dict_list[0].keys() }
        # store results in result object
        for key in result_dict:
            setattr(result, key, result_dict[key])
        # store result in ansatz
        self.result[(learn_method, nshots)] = result

    def learn_sampled_coefficients(
        self,
        constraint_tensors_tuple: list,
        parametrizations: list,
        ) -> dict:
        """
        Return the learning result for given constraint tensors.

        The result depends on the given sample of constraint tensors,
        the given parametrizations and the given scale- or diss-constraints.
        The learning is performed using the static solve_learning_equation() method.

        Parameters
        ----------
        constraint_tensors_tuple : tuple of dicts
            Constraint tensors as a tuple of 3 dictionaries.
            The first dictionary are the constraint tensors for learning.
            The second dictionary are the constraint tensors for scale reconstruction.
            The third dictionary are the constraint tensors for dissipation learning. (O3KZ method only)
            The keys of each dictionary can be any of "O3KZi", "O3KZd", "ZYLB" or "BAL".
        parametrizations : list of Parametrization objects
            List of parametrizations to be used for learning.

        Returns
        -------
        dict
            Learning results stored as a dictionary,
            where the keys are the attributes of the Result object and
            the value is a list with one element for each parametrization.
            Each element is a list that contains only a single element for the sample for given constraints.
            This is necessary for parallelization (see parallel_learn_sampled_coefficients()).
        """
        ### STEP 1 ### create dictionary for all results
        result_keys = self.result_keys
        if self.result_keys is None:
            # result_keys = ["operator_learned","solution_of_linear_system","parametrization_matrix_tupel","dissipators_learned","learning_error","svd_vals","svd_vecs","Theta","Phi","condition_number","error_bound"]
            result_keys = ["operator_learned", "var_operator_learned", "operator_learned_lsq", "var_operator_learned_lsq", "solution_of_linear_system", "solution_of_linear_system_lsq", "parametrization_matrix_tupel", "dissipators_learned", "var_dissipators_learned", "dissipators_learned_lsq", "var_dissipators_learned_lsq", "learning_error", "learning_error_lsq", "svd_vals", "svd_vecs", "Theta", "Phi", "condition_number", "error_bound"]
            other_result_keys = ["gamma_landscape_grid","gamma_landscape_vals","gamma_landscape_sols","posterior","Gamma_eps","chi2_error","complementary_learning_error","complementary_chi2_error","learning_error_exact","chi2_error_exact","complementary_learning_error_exact","complementary_chi2_error_exact","learning_error_fit","chi2_error_fit","complementary_learning_error_fit","complementary_chi2_error_fit"]
            result_keys += other_result_keys
            # if kwargs["constraint_tensors_scale"] is not None:
            scale_result_keys = ["Qoperator_learned","operator_learned_scaled","var_operator_learned_scaled","operator_learned_combined","var_operator_learned_combined","solution_of_linear_system_combined","parametrization_matrix_tupel_combined","operator_learned_combined_scaled","var_operator_learned_combined_scaled","dissipators_learned_combined","var_dissipators_learned_combined","learning_error_combined","Theta_combined","Phi_combined","condition_number_combined","error_bound_combined","svd_vals_combined","svd_vecs_combined","learning_error_noscale","learning_error_scale","complementary_learning_error_combined","gamma_landscape_grid_combined","gamma_landscape_vals_combined","gamma_landscape_sols_combined"]
            result_keys += scale_result_keys
            # if kwargs["constraint_tensors_diss"] is not None:
            diss_result_keys = ["learning_error_validation","operator_learned_diss_separate","var_operator_learned_diss_separate","solution_of_linear_system_diss_separate","parametrization_matrix_tupel_diss_separate","dissipators_learned_separate","var_dissipators_learned_separate","learning_error_diss_separate"]
            result_keys += diss_result_keys
        results_dict = {key:[[] for par in range(len(parametrizations))] for key in result_keys}
        #------------------------------------------------------------
        ### STEP 2 ### get all results for each parametrization
        constraint_tensors = constraint_tensors_tuple[0]
        constraint_tensors_scale = constraint_tensors_tuple[1]
        constraint_tensors_diss = constraint_tensors_tuple[2]
        for parinx, param in enumerate(parametrizations):
            ### step 1 ### learning without scale constraints ###
            results_dict_learn = self.solve_learning_equation(constraint_tensors=constraint_tensors, parametrization=param) 
            ## store results without scale
            all_results_dict_learn = results_dict_learn
            for key, val in all_results_dict_learn.items():
                if key in result_keys:
                    results_dict[key][parinx].append(val)
            #------------------------------------------------------------
            ### step 2 ### separate scale reconstruction ###
            if constraint_tensors_scale is not None:
                operator_learned_scaled, scale_mean, scale_var = Ansatz.get_overall_scale(results_dict_learn["operator_learned"], constraint_tensors_scale, dissipators=results_dict_learn["dissipators_learned"])
                ## store results with scale
                all_results_dict_scale = {"operator_learned_scaled":operator_learned_scaled}
                for key, val in all_results_dict_scale.items():
                    if key in result_keys:
                        results_dict[key][parinx].append(val)
            #------------------------------------------------------------
            ### step 3 ### separate dissipation reconstruction ###
            if constraint_tensors_diss is not None:
                ## fix parametrization of Hamiltonian to previously learned operator
                operator_fixed = results_dict["operator_learned"].copy()
                param_diss = param.copy()
                param_diss.fix_parameters(fixed_Hamiltonian=operator_fixed)
                ## solve learning equation with diss tensors and fixed Hamiltonian
                results_dict_diss = self.solve_learning_equation(constraint_tensors=constraint_tensors_diss, parametrization=param_diss)
                ## store results with dissipation constraints
                all_results_dict_diss_separate = {"operator_learned_diss_separate":results_dict_diss["operator_learned"],"solution_of_linear_system_diss_separate":results_dict_diss["solution_of_linear_system"],"parametrization_matrix_tupel_diss_separate":results_dict_diss["parametrization_matrix_tupel"],"dissipators_learned_separate":results_dict_diss["dissipators_learned"],"learning_error_diss_separate":results_dict_diss["learning_error"]}
                for key, val in all_results_dict_diss_separate.items():
                    if key in result_keys:
                        results_dict[key][parinx].append(val)
        # ------------------------------------------------------------
        return results_dict

    def solve_learning_equation(
            self,
            constraint_tensors,
            parametrization,
            ) -> dict:
        """
        Solve the learning equation.

        Solves the learning equation for given constraint tensors using the given parametrization.
        For linear parametrization, the learning equation is solved using the static learn_free_parameters() method.
        For nonlinear parametrization, the learning equation is solved using the static learn_all_parameters() method.

        Parameters
        ----------
        ansatz_operator : QuantumOperator
            Ansatz operator.
        constraint_tensors : dict
            Dictionary of key:constraint_tensor pairs.
            key can be any of "O3KZi", "O3KZd", "ZYLB", "BAL" or "LZHx" for integer x>=1.
            If there is more than one key, constraint tensors are stacked vertically (along the first axis).
        parametrization : Parametrization
            Parametrization for the ansatz operator.

        Returns
        -------
        dict
            Dictionary that contains the results.
            Keys and values are 
                - operator_learned (QuantumOperator)
                    Learned ansatz operator.
                - var_operator_learned (QuantumOperator)
                    Variance of the learned ansatz operator.
                - solution_of_linear_system (1D-array)
                    Solution of the linear system for the learned coefficients (parametrized coefficients)
                - parametrization_matrix_tupel (pair of 2D-arrays)
                    Matrix that maps the learned coefficients to the parametrized coefficients.
                    First matrix is the parametrization, second matrix is the regularization.
                - dissipators_learned (list of Dissipators)
                    Learned ansatz Dissipators.
                - learning_error (1D-array)
                    Learning error.
                    If b is None, the learning error is the lowest singular value of M.
                    If b is not None, the learning error is (Mc-b)/norm(c).
                - svd_vals (1D-array)
                    Singular values of M if b is None, otherwise singular values of (M,-b).
                - svd_vecs (2D-array)   
                    Singular vectors of M if b is None, otherwise singular vectors of (M,-b).
                - and other stuff
        """
        ### STEP 1 ### learn parameters based on type of parametrization 
        if parametrization is None or len(parametrization.get_nonlinear_parameters())==0:
            # learn parameters of linear parametrization
            result_dict = self.learn_free_parameters(constraint_tensors=constraint_tensors, parametrization=parametrization) 
        else:
            # learn parameters of nonlinear parametrization
            result_dict = self.learn_all_parameters(constraint_tensors=constraint_tensors, parametrization=parametrization)
        #------------------------------------------------------------
        ### STEP 2 ### construct learned Hamiltonian
        operator_learned = None
        var_operator_learned = None
        if self.ansatz_operator is not None:
            if result_dict["learned_free_parameters"] is not None:
                terms = {term_key: result_dict["learned_free_parameters"][opinx] for opinx, term_key in enumerate(self.ansatz_operator.terms.keys())}
                operator_learned = QuantumOperator(N=self.ansatz_operator.N ,terms=terms) 
            if result_dict["var_learned_free_parameters"] is not None:
                var_terms = {term_key: result_dict["var_learned_free_parameters"][opinx] for opinx, term_key in enumerate(self.ansatz_operator.terms.keys())}
                var_operator_learned = QuantumOperator(N=self.ansatz_operator.N ,terms=var_terms)
        #------------------------------------------------------------
        ### STEP 3 ### construct learned dissipators
        dissipators_learned = None
        var_dissipators_learned = None
        if self.ansatz_dissipators is not None:
            if self.ansatz_operator is None:
                if result_dict["learned_free_parameters"] is not None:
                    dissipators_learned = [result_dict["learned_free_parameters"][dinx] * diss for dinx, diss in enumerate(self.ansatz_dissipators)]
                if result_dict["var_learned_free_parameters"] is not None:
                    var_dissipators_learned = [result_dict["var_learned_free_parameters"][dinx] * diss for dinx, diss in enumerate(self.ansatz_dissipators)]
            elif len(result_dict["learned_free_parameters"]) > len(self.ansatz_operator.terms):
                if result_dict["learned_free_parameters"] is not None:
                    dissipators_learned = [result_dict["learned_free_parameters"][len(self.ansatz_operator.terms)+dinx] * diss for dinx, diss in enumerate(self.ansatz_dissipators)]
                if result_dict["var_learned_free_parameters"] is not None:
                    var_dissipators_learned = [result_dict["var_learned_free_parameters"][len(self.ansatz_operator.terms)+dinx] * diss for dinx, diss in enumerate(self.ansatz_dissipators)]
            else:
                dissipators_learned = [np.nan * diss for dinx, diss in enumerate(self.ansatz_dissipators)]
                var_dissipators_learned = [np.nan * diss for dinx, diss in enumerate(self.ansatz_dissipators)]
        #------------------------------------------------------------
        ### STEP 4 ### construct learned posterior
        posterior = None
        if self.prior is not None:
            terms_bayes = {term_key: posterior[0][opinx] for opinx, term_key in enumerate(self.ansatz_operator.terms.keys())}
            operator_posterior_mean = QuantumOperator(N=self.ansatz_operator.N ,terms=terms_bayes) 
            dissipators_posterior_mean = None
            if self.ansatz_dissipators is not None and len(posterior[0])==(len(self.ansatz_operator.terms)+len(self.ansatz_dissipators[0])):
                raise ValueError("Dissipators learned in Bayesian ansatz not implemented yet.")
            posterior = [operator_posterior_mean, posterior[1]]
        #------------------------------------------------------------
        ### STEP 5 ### construct dictionary that contains all results
        result_dict.update({"operator_learned": operator_learned, 
                "var_operator_learned": var_operator_learned, 
                "dissipators_learned": dissipators_learned, 
                "var_dissipators_learned": var_dissipators_learned,
                "posterior": posterior})
        #------------------------------------------------------------
        return result_dict

    def learn_free_parameters(
            self,
            constraint_tensors,
            parametrization,
            ) -> dict:
        """
        Learn the free ansatz parameters.

        Learns the free parameters for given constraint_tensors and a linear parametrization.

        Parameters
        ----------
        constraint_tensors : dict
            Dictionary of key:constraint_tensor pairs.
            key can be any of "O3KZi","O3KZd","ZYLB","BAL" or "LZHx" for integer x>=1.
            All given constraints are used for learning.
            Constraints of different type are stacked vertically (along the first axis).
        parametrization : Parametrization
            Parametrization of the ansatz coefficients.

        Returns
        -------
        dict
            Result as dictionary containing the following keys and values
                - learned_free_parameters (1D-array)
                    Learned ansatz coefficients using singular value decomposition. (not parametrized)
                - var_learned_free_parameters (1D-array)
                    Variance of the learned ansatz coefficients using singular value decomposition. (not parametrized)
                - solution_of_linear_system (1D-array)
                    SVD-solution of the linear system for the learned coefficients (parametrized coefficients)
                - parametrization_matrix_tupel (pair of 2D-arrays)
                    Matrix that maps the learned coefficients to the parametrized coefficients.
                    First matrix is the parametrization, second matrix is the regularization.
                - learning_error (1D-array)
                    Learning error for the learned coefficients using singular value decomposition.
                    If constraint_tensors consists of multiple constraints of different type,
                    all learning_errors are concatenated into a single array.
                - svd_vals (1D-array)
                    Singular values of M if b is None, otherwise singular values of (M,-b).
                - svd_vecs (pair of 2D-arrays)
                    Left and right singular vectors of M if b is None, otherwise left and right singular vectors of (M,-b).
                - gamma_landscape_grid
                - gamma_landscape_vals
                - gamma_landscape_sols
                - posterior
                - Gamma_eps_save
                - learning_error_noscale
        """
        tm0 = time.time()
        if self.print_timers:
            ic("enter learn_free_parameters()")
        #------------------------------------------------------------
        ### STEP 1 ### regularize and parametrize constraint tensors, prior and gamma 
        # NOTE: this is the runtime bottleneck for large M and b
        if parametrization is not None:
            tm01 = time.time()
            ## parametrize constraint tensors (par = non_par @ G)
            parametrization_matrices = parametrization.get_parametrization_matrices(ansatz_operator=self.ansatz_operator, ansatz_dissipators=self.ansatz_dissipators)
            if self.print_timers:
                tm02 = time.time()
                pstr_learn_free_parameters_parametrization = "substep time for get_parametrization_matrices = {}".format(tm02-tm01)
                ic(pstr_learn_free_parameters_parametrization)
            parametrized_constraint_tensors = Ansatz.parametrize_constraint_tensors(constraint_tensors, parametrization_matrices) 
            if self.print_timers:
                tm03 = time.time()
                pstr_learn_free_parameters_parametrization = "substep time for parametrize_constraint_tensors = {}".format(tm03-tm02)
                ic(pstr_learn_free_parameters_parametrization)
            constraint_tensors = parametrized_constraint_tensors
            ## parametrize regularization (par = non_par @ G)
            tm04 = time.time()
            regularization_matrices = parametrization.get_regularization_matrices(ansatz_operator=self.ansatz_operator, ansatz_dissipators=self.ansatz_dissipators)
            if self.print_timers:
                tm05 = time.time()
                pstr_learn_free_parameters_parametrization = "substep time for get_regularization_matrices = {}".format(tm05-tm04)
                ic(pstr_learn_free_parameters_parametrization)
            if regularization_matrices is not None:
                parametrized_regularization_matrices = Ansatz.parametrize_regularization_matrices(regularization_matrices, parametrization_matrices)   #np.einsum("ij,jk",regularization_matrix,parametrization_matrices[0])
                if self.print_timers:
                    tm06 = time.time()
                    pstr_learn_free_parameters_parametrization = "substep time for parametrize_regularization_matrices = {}".format(tm06-tm05)
                    ic(pstr_learn_free_parameters_parametrization)
                regularization_matrices = parametrized_regularization_matrices
            ## parametrize prior
            if self.prior is not None:
                parametrized_prior = Ansatz.parametrize_prior(self.prior, parametrization_matrices)
                prior = parametrized_prior
            ### parametrize dissipation rates for O3KZ method
            if self.ansatz_dissipators is not None:
                ## parametrize gamma0 (par = G^T @ non-par)
                if self.gamma0 is not None:
                    gamma0 = np.einsum("ji,j", parametrization_matrices[1], self.gamma0)
                ## parametrize gamma_exact (par = G^T @ non-par)
                if self.gamma_exact is not None:
                    gamma_exact = np.einsum("ji,j", parametrization_matrices[1], self.gamma_exact)
                ## parametrize bounds (par = G^T @ non-par)
                if self.gamma_bounds is not None:
                    gamma_bounds = np.array(self.gamma_bounds)
                    lb = np.real(np.einsum("ji,j", parametrization_matrices[1], gamma_bounds[:,0]))
                    ub = np.real(np.einsum("ji,j", parametrization_matrices[1], gamma_bounds[:,1]))
                    gamma_bounds = [(np.min([lb[inx],ub[inx]]), np.max([lb[inx],ub[inx]])) for inx in range(len(lb))]
                    # check if lower bound is not larger than upper bound
                    if not all([gamma_bounds[inx][0]<=gamma_bounds[inx][1] for inx in range(len(gamma_bounds))]):
                        raise ValueError("gamma_bounds not valid (lb<!ub) \n gamma_bounds = {}".format(gamma_bounds))
                    # check if gamma0 in gamma_bounds
                    if self.gamma0 is not None:
                        if not all([gamma_bounds[inx][0]<=gamma0[inx]<=gamma_bounds[inx][1] for inx in range(len(gamma_bounds))]):
                            raise ValueError("gamma0 not in parametrized gamma_bounds. \n gamma0 = {} \n gamma_bounds = {}".format(gamma0,gamma_bounds))
        tm1 = time.time()
        if self.print_timers:
            pstr_learn_free_parameters = "... of the following time for parametrization = {}".format(tm1-tm0)
            ic(pstr_learn_free_parameters)
        #------------------------------------------------------------
        ### STEP 2 ### get optimal gamma from dissipation correction [optional]
        gamma_opt = None
        # gamma_iterations = None
        gamma_landscape_grid = None
        gamma_landscape_vals = None
        gamma_landscape_sols = None
        if self.ansatz_dissipators is not None and self.gamma0 is not None:
            gamma_opt = gamma0
            # check if there is any optimization needed for given gamma0 and gamma_bounds
            no_optimization = False
            if self.gamma_exact is not None:
                gamma_opt = gamma_exact
                no_optimization = True
            # run optimization if needed
            if any([key=="O3KZi" or key=="O3KZd" for key in list(constraint_tensors.keys())]) and len(gamma0)>0 and not no_optimization:
                gamma_opt, gamma_landscape_grid, gamma_landscape_vals, gamma_landscape_sols = Ansatz.get_optimal_gamma(constraint_tensors, gamma0=gamma0, gamma_bounds=gamma_bounds, return_steps=True, scale_factor=self.scale_factor, save_landscape=self.save_landscape, max_nfev=self.gamma_max_nfev, regularization_matrices=regularization_matrices, exclude_lowest_solutions=self.exclude_lowest_solutions)
        #------------------------------------------------------------
        ### STEP 3 ### get constraint matrix M and vector b
        ## get dissipation-corrected constraint matrix and vector
        M, b, varM, varb = Ansatz.get_constraint_matrix_and_vector(constraint_tensors, gamma=gamma_opt, scale_factor=self.scale_factor, regularization_matrices=regularization_matrices)
        ## get constraint matrix and vector without scale constraints
        M_noscale = None # TODO: unused variable M_noscale
        # if len(constraint_tensors.keys())>1 and ("BAL" in constraint_tensors.keys() or "ZYLB" in constraint_tensors.keys()):
        #     raise ValueError("BAL and ZYLB constraints not implemented yet.")
        #     M_noscale, b_noscale, varM_noscale, varb_noscale = Ansatz.get_constraint_matrix_and_vector(constraint_tensors, gamma=gamma_opt, scale_factor=0, regularization_matrices=regularization_matrices)
        tm2 = time.time()
        if self.print_timers:
            pstr_learn_free_parameters = "... of the following time for getting M and b = {}".format(tm2-tm1)
            ic(pstr_learn_free_parameters)
        #------------------------------------------------------------
        ### STEP 3 ### optional Bayesian setup
        Gamma_eps_save = [None, None, None]
        if self.prior is not None:
            tmpvec = np.add(np.diag(prior[1]),np.power(prior[0],2))
            Gamma_eps_diag = np.einsum("ij,j",varM,tmpvec)
            Gamma_eps = np.diag(Gamma_eps_diag)
            Gamma_eps_inv = np.linalg.inv(Gamma_eps)
            prior_cov_inv = np.linalg.inv(prior[1])
            chol_eps = np.linalg.cholesky(Gamma_eps_inv)  
            chol_prior = np.linalg.cholesky(prior_cov_inv)
            #check if cholesky factors are symmetric
            if not np.allclose(chol_eps, np.transpose(chol_eps)):
                raise ValueError("Cholesky decomposition of Gamma_eps is not symmetric.")
            if not np.allclose(chol_prior, np.transpose(chol_prior)):
                raise ValueError("Cholesky decomposition of prior is not symmetric.")
            M_bayes = np.einsum("ij,jk",chol_eps, M)
            M_bayes = np.concatenate((M_bayes,chol_prior), axis=0)
            b_bayes = np.zeros(np.shape(M)[0])
            if b is not None:
                b_bayes = np.einsum("ij,j",chol_eps, b)
            b_bayes = np.concatenate((b_bayes,np.einsum("ij,j",chol_prior,prior[0])), axis=0)
        tm3 = time.time()
        if self.print_timers:
            pstr_learn_free_parameters = "... of the following time for Bayesian setup = {}".format(tm3-tm2)
            ic(pstr_learn_free_parameters)
        #------------------------------------------------------------
        ### STEP 4 ### solve the linear learning equation M*sol=b
        solution_of_linear_system, learning_error, svd_vals, svd_vecs, = solve_linear_eq(M=M, b=b, exclude_lowest_solutions=self.exclude_lowest_solutions)
        ## svd solution
        learned_free_parameters = solution_of_linear_system.copy()
        var_learned_free_parameters = None       
        if self.get_variance_of_coefficients_from_variance_of_constraint_tensors:
            var_learned_free_parameters, cov_learned_free_parameters = Ansatz.get_cov_of_solution_of_linear_system(solution_of_linear_system, M, var_M=varM, var_b=varb)
        ## get learning error without scale constraints
        learning_error_noscale = None
        if M_noscale is not None:
            learning_error_noscale = np.einsum("ij,j", M_noscale, learned_free_parameters)/np.linalg.norm(learned_free_parameters)
        ## concatenate learned dissipation rates to learned_free_parameters
        if gamma_opt is not None and self.ansatz_dissipators is not None:
            learned_free_parameters = np.concatenate((learned_free_parameters, gamma_opt))
        ## get posterior distribution [optional]
        posterior = None
        if self.prior is not None:
            posterior_mean, cost_bayes, svd_vals_bayes, svd_vecs_bayes = solve_linear_eq(M=M_bayes, b=b_bayes, exclude_lowest_solutions=self.exclude_lowest_solutions)
            posterior_cov = np.linalg.inv(np.add(prior_cov_inv, np.einsum("ji,jk,kl", M, Gamma_eps_inv, M)))
            Gamma_eps_save = [np.linalg.inv(np.linalg.inv(Gamma_eps)), np.linalg.inv(np.einsum("ji,jk,kl", M, Gamma_eps_inv, M)), M]
            posterior = (posterior_mean, posterior_cov)
        ## print timers
        tm4 = time.time()
        if self.print_timers:
            pstr_learn_free_parameters = "... of the following time for solving learning equations = {}".format(tm4-tm3)
            ic(pstr_learn_free_parameters)
        #------------------------------------------------------------
        ### STEP 5 ### invert parametrization
        if parametrization is not None:
            ## invert parametrization on learned_free_parameters  (non_par = G @ par)
            if learned_free_parameters is not None:
                learned_free_parameters = Ansatz.invert_parametrization(learned_free_parameters, parametrization_matrices)
            if var_learned_free_parameters is not None:
                var_learned_free_parameters = Ansatz.invert_parametrization(var_learned_free_parameters, parametrization_matrices)
            ## invert parametrization of posterior
            if posterior is not None:
                posterior_mean = np.einsum("ij,j", parametrization_matrices[0], posterior[0])
                posterior_cov = np.einsum("ij,jk,lk", parametrization_matrices[0], posterior[1], parametrization_matrices[0])
                posterior = (posterior_mean, posterior_cov)
        tm5 = time.time()
        if self.print_timers:
            pstr_learn_free_parameters = "... of the following time for inverting parametrization = {}".format(tm5-tm4)
            ic(pstr_learn_free_parameters)
            pstr_learn_free_parameters = "Total time for learning free parameters = {}".format(tm5-tm0)
            ic(pstr_learn_free_parameters)
            ic("--------------")
        parametrization_matrix_tupel = [parametrization_matrices, regularization_matrices]
        #------------------------------------------------------------
        ### STEP 6 ### construct dictionary that contains all results
        result_dict = {
            "learned_free_parameters": learned_free_parameters,
            "var_learned_free_parameters": var_learned_free_parameters,
            "solution_of_linear_system": solution_of_linear_system,
            "parametrization_matrix_tupel": parametrization_matrix_tupel,
            "learning_error": learning_error,
            "svd_vals": svd_vals,
            "svd_vecs": svd_vecs,
            "gamma_landscape_grid": gamma_landscape_grid,
            "gamma_landscape_vals": gamma_landscape_vals,
            "gamma_landscape_sols": gamma_landscape_sols,
            "posterior": posterior,
            "Gamma_eps_save": Gamma_eps_save,
            "learning_error_noscale": learning_error_noscale,
        }
        #------------------------------------------------------------
        return result_dict

    def learn_all_parameters(
            self,
            constraint_tensors,
            parametrization,
            ) -> dict:
        """
        Learn all parameters of the ansatz.

        Learns parameters for given constraint_tensors and a nonlinear parametrization.
        The nonlinear parametrization is optimized over iterative use of learn_free_parameters().

        Parameters
        ----------
        constraint_tensors : dict
            Dictionary of key:constraint_tensor pairs.
            key can be any of "O3KZi", "O3KZd", "ZYLB", "BAL" or "LZHx" for integer x>=1.
            All given constraints are used for learning.
            Constraints of different type are stacked vertically (along the first axis).
        parametrization : Parametrization
            Parametrization for the ansatz operator.

        Returns
        -------
        dict
            Result as dictionary that contains the same keys and values
            as the dictionary returned by learn_free_parameters().
        """
        ### STEP 1 ### define costfunction for optimizing nonlinear parametrization 
        def costfct(x):
            # update parameters of parametrization
            parametrization.set_nonlinear_parameters(x)
            # learn free parameters
            result_dict = self.learn_free_parameters(constraint_tensors=constraint_tensors, parametrization=parametrization) 
            return np.real(result_dict["learning_error"]) 
        #------------------------------------------------------------
        ### STEP 2 ### get initial guess for parameters
        x0 = []
        if parametrization.functions is not None:
            for parfct in parametrization.functions:
                if parfct.parameters is not None:
                    x0.extend(parfct.parameters)
        if parametrization.regularizations is not None:
            for reg in parametrization.regularizations:
                if reg.parameters is not None:
                    x0.extend(reg.parameters)
        x0 = np.real(np.array(x0))
        bounds = parametrization.get_nonlinear_bounds()
        #------------------------------------------------------------
        ### STEP 3 ### optimize nonlinear parametrization 
        xopt = parametrization.get_exact_nonlinear_parameters()
        if len(xopt)==0:
            ## 1-dim optimizer
            if len(x0)==1:
                # TODO: changed max_nfev to be faster
                max_nfev = 9 #16
                xopt, err, landscape_grid, landscape_vals = uf.minimize(costfct, x0, bounds=bounds, method="brute", max_nfev=max_nfev) 
            ## multi-dim optimizer
            else:
                raise ValueError("Nonlinear parametrization for more than one parameter not implemented yet.")
                xopt, err, landscape_grid, landscape_vals = uf.minimize(costfct, x0, bounds=bounds, method="direct" ,max_nfev=10, xopt=xopt)
                # xopt, err, landscape_grid, landscape_vals = uf.minimize(costfct, xopt, bounds=bounds, method="lsq" ,max_nfev=10, xopt=xopt)
        #------------------------------------------------------------
        ### STEP 4 ### get optimal solution for coeffs and dissipation rates 
        parametrization.set_nonlinear_parameters(xopt)
        parametrization.nonlinear_parameters_optimized = True
        result_dict = self.learn_free_parameters(constraint_tensors=constraint_tensors, parametrization=parametrization) 
        #------------------------------------------------------------
        return result_dict

    @staticmethod
    def get_optimal_gamma(
        constraint_tensors, 
        **kwargs
        ):
        """
        Finds optimal dissipation rates for given constraint_tensors.

        The optimal dissipation rates gamma are those that minimize least squares
        error of the learning equation M_H c + M_D gamma = 0.

        Parameters
        ----------
        constraint_tensors : dict
            Dictionary of key:constraint_tensor pairs.
            key can be any of "O3KZi","O3KZd","ZYLB","BAL" or "LZHx" for integer x>=1.
        kwargs:
            - gamma0 (1D-array) [default: None]
                Initial guess for the dissipation rates used for the iterative solver.
            - gamma_bounds (list of tuples) [default: None]
                Bounds for the dissipation rates.
                If set, the optimal gamma is found using a nonlinear solver instead of the iterative solver.
            - return_steps (bool) [default: False]
                If True, also returns the steps of the iterative procedure.
            - save_landscape (bool) [default: False]
                If True, also returns the costfunction landscape.
            - scale_factor (float) [default: 1]
                Scale factor for the constraint tensors.
            - max_nfev (int) [default: 1000]
                Maximum number of function evaluations for the nonlinear solver.
            - regularization_matrices (tuple of 2D-arrays) [default: None]
                Regularization for the ansatz operator.
                regularization_matrices[0] is the regularization for the coherent terms
                regularization_matrices[1] is the regularization for the incoherent terms
            - exclude_lowest_solutions (int) [default: 0]
                Number of lowest solutions to exclude from the landscape, 
                e.g. exclude_lowest_solutions=1 we solve for the second lowest solution.
                
        Returns
        -------
        gamma_opt : 1D-array
            Optimal dissipation rates.
        gamma_landscape_grid : list of 1D-arrays
            Costfunction landscape of the brute solver. Last element is cost.
            Only returned if save_landscape=True.
        gamma_landscape_vals : list of 1D-arrays
            Costfunction landscape of the brute solver. Last element is cost.
            Only returned if save_landscape=True.
        gamma_landscape_sols : list of 1D-arrays
            Raw solution of the brute solver.
            Only returned if save_landscape=True.
        """
        scale_factor = kwargs.get("scale_factor",1)
        gamma0 = kwargs.get("gamma0",None)
        gamma_bounds = kwargs.get("gamma_bounds",None)
        save_landscape = kwargs.get("save_landscape",False)
        max_nfev = kwargs.get("max_nfev",1000)
        regularization_matrices = kwargs.get("regularization_matrices",None)
        exclude_lowest_solutions = kwargs.get("exclude_lowest_solutions",0)
        #--------------------------------------------------------------
        if gamma_bounds is None:
            raise ValueError("gamma_bounds must not be None.")
        if gamma0 is None:
            raise ValueError("gamma0 must not be None.")
        #--------------------------------------------------------------
        ### STEP 1 ### define costfunction for optimizing gamma
        def costfct(gamma):
            M, b, varM, varb = Ansatz.get_constraint_matrix_and_vector(constraint_tensors, gamma=gamma, scale_factor=scale_factor, regularization_matrices=regularization_matrices)
            if b is None:
                # get lowest singular value of M
                svd_vals = np.linalg.svd(M, compute_uv=False)
                # svd_vals = sp.linalg.svd(M,compute_uv=False,overwrite_a=True,check_finite=False)
                err = svd_vals[-1-exclude_lowest_solutions]
            else:
                # get solution of linear equation
                err = np.linalg.lstsq(M, b, rcond=None)[1]
            return err
        #--------------------------------------------------------------
        ### STEP 2 ### find optimal gamma using nonlinear solver
        ### brute solver
        gamma_landscape_sols = []
        # save exact solution
        if gamma0 is not None:
            gamma_landscape_sols.append(gamma0)
        gamma_landscape_grid = None
        gamma_landscape_vals = None
        if save_landscape:
            print("save_landscale=True, finding optimal gamma using brute solver.")
            brute_sols = []
            brute_costs = []
            brute_grids = []
            brute_vals = []
            ## all dimensions
            gamma_opt_brute, opt_cost_brute, gamma_landscape_grid_brute, gamma_landscape_vals_brute = uf.minimize(costfct, gamma0, bounds=gamma_bounds, method="brute", max_nfev=max_nfev)
            brute_sols.append(gamma_opt_brute)
            brute_costs.append(opt_cost_brute)
            brute_grids.append(gamma_landscape_grid_brute)
            brute_vals.append(gamma_landscape_vals_brute)
            ## pair dimensions
            max_nfev_brute2D = np.min([max_nfev,10000])
            pair_list = list(it.combinations(range(len(gamma0)),2))
            if pair_list is not None:
                for pair in pair_list:
                    def costfct_pair(gamma_pair):
                        gamma = gamma0.copy()
                        for inx in pair:
                            gamma[inx] = gamma_pair[pair.index(inx)]
                        return costfct(gamma)
                    gamma0_pair = [gamma0[pair[0]], gamma0[pair[1]]]
                    gamma_bounds_pair = [gamma_bounds[pair[0]], gamma_bounds[pair[1]]]
                    gamma_opt_brute2D, opt_cost_brute2D, gamma_landscape_grid_brute2D, gamma_landscape_vals_brute2D = uf.minimize(costfct_pair, gamma0_pair, bounds=gamma_bounds_pair, method="brute", max_nfev=max_nfev_brute2D)
                    brute_sol2D = gamma0.copy()
                    brute_sol2D[pair[0]] = gamma_opt_brute2D[0]
                    brute_sol2D[pair[1]] = gamma_opt_brute2D[1]
                    brute_sols.append(brute_sol2D)
                    brute_costs.append(opt_cost_brute2D)
                    brute_grids.append(gamma_landscape_grid_brute2D)
                    brute_vals.append(gamma_landscape_vals_brute2D)
            gamma_landscape_sols.append(brute_sols)
            gamma_landscape_grid = brute_grids
            gamma_landscape_vals = brute_vals
        #--------------------------------------------------------------
        ### direct solver
        gamma_opt_direct, opt_cost_direct, grid_direct, vals_direct = uf.minimize(costfct, gamma0, bounds=gamma_bounds, method="direct", max_nfev=max_nfev, eps=1e-4)
        gamma_landscape_sols.append(gamma_opt_direct)
        gamma_opt = gamma_opt_direct
        opt_cost = opt_cost_direct
        #--------------------------------------------------------------
        ### save 2D brute solution if its cost is lower than direct solution
        if save_landscape:
            min_cost_inx = np.argmin(brute_costs)
            if brute_costs[min_cost_inx]+1e-6 < opt_cost_direct:
                gamma_opt = brute_sols[min_cost_inx]
                opt_cost = brute_costs[min_cost_inx]
                gamma_landscape_sols.append(gamma_opt)
        ### check if gamma_opt < cost(gamma0)
        if costfct(gamma0)+1e-6 < opt_cost:
            gamma_opt = gamma0
            gamma_landscape_sols.append(gamma0)
        #--------------------------------------------------------------
        ## check if gamma_opt inside gamma_bounds
        eps = 1e-6
        if not all([gamma_bounds[inx][0]-eps<=gamma_opt[inx]<=gamma_bounds[inx][1]+eps for inx in range(len(gamma_bounds))]):
            print("gamma_opt not in gamma_bounds. \n gamma_opt = {} \n gamma_bounds = {}".format(gamma_opt,gamma_bounds))
            for inx in range(len(gamma_opt)):
                if gamma_opt[inx]<gamma_bounds[inx][0]:
                    gamma_opt[inx] = gamma_bounds[inx][0]
                if gamma_opt[inx]>gamma_bounds[inx][1]:
                    gamma_opt[inx] = gamma_bounds[inx][1]
            raise ValueError("gamma_opt not in gamma_bounds. \n gamma_opt = {} \n gamma_bounds = {}".format(gamma_opt,gamma_bounds))
        ## whether to return grid
        if not save_landscape:
            gamma_landscape_grid = None
            gamma_landscape_vals = None
            gamma_landscape_sols = None
        return gamma_opt, gamma_landscape_grid, gamma_landscape_vals, gamma_landscape_sols

    @staticmethod
    def get_constraint_matrix_and_vector(
            constraint_tensors,
            gamma=None,
            scale_factor=1,
            regularization_matrices=None,
            ):
        """
        Get M and b from constraint tensors.

        The learning equation is defined by the constraint matrix M and constraint vector b, i.e., M*sol = b.
        This function returns M and b from a dictionary of constraint tensors
        and includes the dissipation correction if given.

        Parameters
        ----------
        constraint_tensors : dict
            Dictionary of key:constraint_tensor pairs.
            key can be any of "O3KZi", "O3KZd", "ZYLB", "BAL" or "LZHx" for integer x>=1.
        gamma : 1D-array, optional
            Dissipation rates for the dissipation correction.
            Default is None.
        scale_factor : float, optional
            Scale factor for the dissipation correction.
            Default is ``1``.
        regularization_matrices : tuple of 2D-np.arrays, optional
            Regularization for the constraint matrix.
            regularization_matrices[0] is the regularization for the coherent terms
            regularization_matrices[1] is the regularization for the incoherent terms
            Default is None.

        Returns
        -------
        2D-array
            Constraint matrix M.
        1D-array
            Constraint vector b. Can be None if no vector is given.
        2D-array
            Variance of constraint matrix M.
        1D-array
            Variance of constraint vector b. Can be None if no vector is given.
        """
        ### STEP 1 ### create list of all constraint matrices and vectors
        Mcomb = []
        varMcomb = []
        bcomb = []
        varbcomb = []
        bstr = False
        for key in list(constraint_tensors.keys()):
            # initialize optional variables
            b = None
            varb = None
            ## get constraint tensors
            if key=="LZH1":
                M = constraint_tensors[key][0]
                varM = constraint_tensors[key][1]
            elif key in ["BAL","ZYLB"]:
                M = np.multiply(scale_factor,constraint_tensors[key][0])
                b = np.multiply(scale_factor,constraint_tensors[key][1])
                varM = np.multiply(scale_factor,constraint_tensors[key][2])
                varb = np.multiply(scale_factor,constraint_tensors[key][3])
            elif key in ["O3KZi","O3KZd"]:
                M = constraint_tensors[key][0]
                R = constraint_tensors[key][1]
                varM = constraint_tensors[key][2]
                varR = constraint_tensors[key][3]
                if R is not None and gamma is not None:
                    # apply dissipation correction
                    Mdiss = np.einsum("ijk,k",R,gamma)
                    M = np.add(M, Mdiss)
                    varMdiss = np.einsum("ijk,k",varR,np.power(gamma,2))
                    varM = np.add(varM, varMdiss)
            elif key[0:3]=="LZH":
                raise ValueError("Higher order LZH method not implemented yet.")
            else:
                raise ValueError("Constraint tensor type {} not recognized.".format(key))
            # add to combined matrix
            Mcomb.append(M)
            varMcomb.append(varM)
            if b is not None:
                bstr = True
                bcomb.append(b)
                varbcomb.append(varb)
            else:
                bcomb.append(np.zeros(np.shape(M)[0]))
                varbcomb.append(np.zeros(np.shape(M)[0]))
        #------------------------------------------------------------
        ### STEP 2 ### combine all constraints into single matrix and vector
        b = None
        varb = None
        ## case for only a single constraint tensor
        if len(Mcomb)==1:
            M = Mcomb[0]
            varM = varMcomb[0]
            if bstr:
                b = bcomb[0]
                varb = varbcomb[0]
        # ------------
        ## case for several constraint tensors
        else:
            # check if all constraint tensors have the same number of columns
            if not all([np.shape(M)[1]==np.shape(Mcomb[0])[1] for M in Mcomb]):
                if gamma is None:
                    raise ValueError("gamma must be given for dissipation correction.")
                # get minimum number of rows
                mincols = np.min([np.shape(M)[1] for M in Mcomb])
                for Minx, Mtmp in enumerate(Mcomb):
                    varMtmp = varMcomb[Minx]
                    if np.shape(Mtmp)[1]>mincols:
                        # cut M to minimum number of rows
                        Mcomb[Minx] = Mtmp[:,:mincols]
                        varMcomb[Minx] = varMcomb[Minx][:,:mincols]
                        # add gamma correction -M*gamma to b
                        bcorr = np.multiply(-1,np.einsum("ij,j",Mtmp[:,mincols:],gamma))
                        var_bcorr = np.einsum("ij,j",varMtmp[:,mincols:],np.power(gamma,2))
                        bcomb[Minx] = np.add(bcomb[Minx],bcorr)
                        varbcomb[Minx] = np.add(varbcomb[Minx],var_bcorr)  #TODO: check if this is correct
            # combine all constraint tensors
            M = np.concatenate(Mcomb, axis=0)
            varM = np.concatenate(varMcomb, axis=0)
            if bstr:
                b = np.concatenate(bcomb, axis=0)
                varb = np.concatenate(varbcomb, axis=0)
        #------------------------------------------------------------
        ### STEP 3 ### add regularization
        if regularization_matrices is not None: 
            if "O3KZi" in list(constraint_tensors.keys()):
                #TODO add regularization for incoherent terms not implemented
                regularization_matrix = regularization_matrices[0]
                M = np.vstack((M,regularization_matrix))
                # TODO: varM hast wrong dimensions!!
                varM = np.vstack((varM,np.zeros(np.shape(regularization_matrix))))
                if bstr:
                    b = np.concatenate((b,np.zeros(np.shape(regularization_matrix)[0])))
                    varb = np.concatenate((varb,np.zeros(np.shape(regularization_matrix)[0])))
            else:
                regularization_matrix = Parametrization.combine_regularization_matrices(regularization_matrices)
                M = np.vstack((M,regularization_matrix))
                varM = np.vstack((varM,np.zeros(np.shape(regularization_matrix))))
                if bstr:
                    b = np.concatenate((b,np.zeros(np.shape(regularization_matrix)[0])))
                    varb = np.concatenate((varb,np.zeros(np.shape(regularization_matrix)[0])))
        #------------------------------------------------------------
        ### STEP 4 ### remove all rows that contain nan values
        ## get indices of nan rows
        nan_rows = np.any(np.isnan(M),axis=1)
        if bstr:
            nan_rows_b = np.isnan(b)
            nan_rows = np.logical_or(nan_rows,nan_rows_b)
        ## remove nan rows from M and b
        M = M[~nan_rows]
        varM = varM[~nan_rows]
        if bstr:
            b = b[~nan_rows]
            varb = varb[~nan_rows]
        #------------------------------------------------------------
        return M, b, varM, varb

    @staticmethod
    def apply_parametrization(coefficients, parametrization_matrices):
        """
        Apply parametrization to coefficients.

        The parametrization is defined by the parametrization matrices G_H and G_D
        for the ansatz operator and the ansatz_dissipators, respectively.
        The parametrized coefficients are calculated via parametrized_coefficients = G^T @ coefficients.

        Parameters
        ----------
        coefficients : 1D-array
            Coefficients of the ansatz operator.
        parametrization_matrices : list of 2D-arrays
            Parametrization matrices to be applied.
            First element is the parametrization matrix for the ansatz operator.
            Second element is the parametrization matrix for the ansatz_dissipators.

        Returns
        -------
        1D-array
            Parametrized coefficients c_par = G^T @ c.
        """
        # get parametrization matrices
        parametrization_matrix_H = parametrization_matrices[0]
        parametrization_matrix_D = parametrization_matrices[1]
        # check shapes of input
        if parametrization_matrix_D is not None and len(coefficients) == np.shape(parametrization_matrix_H)[0] + np.shape(parametrization_matrix_D)[0]:
            parametrization_matrix = Parametrization.combine_parametrization_matrices(parametrization_matrices)
        elif len(coefficients) == np.shape(parametrization_matrix_H)[0]:
            parametrization_matrix = parametrization_matrix_H
        else:
            raise ValueError("Shape of coefficients {} does not match shape of parametrization matrices GH{}, GD{}.".format(np.shape(coefficients),np.shape(parametrization_matrix_H),np.shape(parametrization_matrix_D)))
        # invert parametrization
        try:
            parametrized_coefficients = np.einsum("ji,j",parametrization_matrix,coefficients)  #G^T @ sol_resamp
        except:
            print("parametrization_matrix",np.shape(parametrization_matrix))
            print("coefficients",np.shape(coefficients))
            raise ValueError("Error in applying parametrization to coefficients.")
        return parametrized_coefficients

    @staticmethod
    def invert_parametrization(learned_parameters,parametrization_matrices): 
        """
        Invert parametrization on learned parameters.

        Inverts the parametrization on the learned parameters
        to get the coefficients of the ansatz operator. 
        The parametrization is defined by the parametrization matrices G_H and G_D
        for the ansatz operator and the ansatz_dissipators, respectively.
        The inverted coefficients are calculated via non_par = G @ par.

        Parameters
        ----------
        learned_parameters : 1D-array
            Learned parameters for which to invert the parametrization.
        parametrization_matrix : 2D-array
            Parametrization matrix for the parametrization.

        Returns
        -------
        1D-array
            The learned ansatz coefficients (non_par = G @ par).
        """
        parametrization_matrix_H = parametrization_matrices[0]
        parametrization_matrix_D = parametrization_matrices[1]
        #------------------------------------------------------------
        ### STEP 1 ### check shapes of input
        if parametrization_matrix_H is None:
            parametrization_matrix = parametrization_matrix_D
        elif parametrization_matrix_D is None or len(learned_parameters) == np.shape(parametrization_matrix_H)[1]:
            parametrization_matrix = parametrization_matrix_H
        elif len(learned_parameters) == np.shape(parametrization_matrix_H)[1] + np.shape(parametrization_matrix_D)[1]:
            parametrization_matrix = Parametrization.combine_parametrization_matrices(parametrization_matrices)
        else:
            raise ValueError("Shape of learned parameters {} does not match shape of parametrization matrices GH{}, GD{}.".format(np.shape(learned_parameters),np.shape(parametrization_matrix_H),np.shape(parametrization_matrix_D)))
        #------------------------------------------------------------
        ### STEP 2 ### invert parametrization on learned parameters
        learned_coefficients = np.einsum("ij,j",parametrization_matrix,learned_parameters)  # non_par = G @ par
        #------------------------------------------------------------
        return learned_coefficients

    @staticmethod
    def parametrize_prior(prior, parametrization_matrices):
        """
        Apply parametrization to prior for Bayesian learning.

        Parametrize the prior using the given parametrization matrix.
        mu_par = G^T @ mu
        Gamma_par = G^T @ Gamma @ G

        Parameters
        ----------
        prior : tuple of arrays
            Prior distribution for the free parameters
            prior[0] is the mean and prior[1] is the covariance matrix.
        parametrization_matrices : list of 2D-arrays
            Parametrization matrices.
            First element is the parametrization matrix for the ansatz_operator.
            Second element is the parametrization matrix for the ansatz_dissipators.

        Returns
        -------
        tuple of arrays
            Parametrized prior distribution for the free parameters.
            parametrized_prior[0] is the mean and parametrized_prior[1] is the covariance matrix.
        """
        parametrization_matrix_H = parametrization_matrices[0]
        parametrization_matrix_D = parametrization_matrices[1]
        # ------------------------------------------------------------
        ### STEP 1 ### parametrize prior
        parametrized_prior_mean = np.einsum("ji,j",parametrization_matrix_H,prior[0])
        parametrized_prior_covariance = np.einsum("ji,jk,kl",parametrization_matrix_H, prior[1], parametrization_matrix_H)
        parametrized_prior = (parametrized_prior_mean, parametrized_prior_covariance)
        # ------------------------------------------------------------
        ### STEP 2 ### TODO: parametrize dissipators?!
        # ------------------------------------------------------------
        return parametrized_prior

    # NOTE: this function is slow and should be optimized
    @staticmethod
    def parametrize_constraint_tensors(constraint_tensors, parametrization_matrices) -> dict:
        """
        Parametrize the constraint tensors.

        The constraint tensors are parametrized using 
        the given parametrization matrix (T_par = T @ G).

        Parameters
        ----------
        constraint_tensors : dict
            Dictionary of key:constraint_tensor pairs.
            key can be any of "ZYLB", "BAL" or "LZHx" for integer x>=1.
        parametrization_matrices : list
            List of tuples of parametrization matrices.
            First element is the parametrization matrix for the ansatz_operator.
            Second element is the parametrization matrix for the ansatz_dissipators.

        Returns
        -------
        dict
            The parametrized constraint tensors as dictionary of key:par_tensor pairs,
            where par_tensor are the parametrized constraint tensors (T_par = T @ G).
        """
        parametrization_matrix_H = parametrization_matrices[0]
        parametrization_matrix_D = parametrization_matrices[1]
        #------------------------------------------------------------
        parametrized_constraint_tensors = {}
        for key in constraint_tensors.keys():
            #------------------------------------------------------------
            ### OPTION 1 ### LZHx method for x >= 1 ###
            if key[0:3]=="LZH":
                order = int(key[3:])
                tensor = constraint_tensors[key][0]
                var_tensor = constraint_tensors[key][1]
                vector = None
                # parametrize constraint tensor
                constraint_tensor_subscripts = "m" + "".join(chr(97 + i) for i in range(0,2*order,2))
                parametrization_matrix_subscripts = "".join(f",{chr(97 + 2*j)}{chr(97 + 2*j + 1)}" for j in range(order))
                parametrized_constraint_tensor_subscripts = "m" + "".join(chr(97 + i) for i in range(1,2*order,2))
                subscripts = f"{constraint_tensor_subscripts}{parametrization_matrix_subscripts}->{parametrized_constraint_tensor_subscripts}"
                parametrization_matrix_arr = [parametrization_matrix_H]*order
                parametrization_matrix2_arr = [np.power(parametrization_matrix_H,2)]*order
                parametrized_tensor = np.einsum(subscripts,tensor,*parametrization_matrix_arr)
                # parametrize variance of constraint tensor
                parametrized_var_tensor = None
                if var_tensor is not None:
                    parametrized_var_tensor = np.einsum(subscripts,var_tensor,*parametrization_matrix2_arr)
                # combine tensors 
                parametrized_tensor_list = [parametrized_tensor, parametrized_var_tensor]
            #------------------------------------------------------------
            ### OPTION 2 ### BAL and ZYLB method ###
            elif key in ["BAL", "ZYLB"]:
                tensor = constraint_tensors[key][0]
                vector = constraint_tensors[key][1]
                var_tensor = constraint_tensors[key][2]
                var_vector = constraint_tensors[key][3]
                # get parametrization matrix
                if parametrization_matrix_H is None:
                    parametrization_matrix = parametrization_matrix_D
                elif parametrization_matrix_D is None or np.shape(tensor)[1] == np.shape(parametrization_matrix_H)[0]:
                    parametrization_matrix = parametrization_matrix_H
                elif np.shape(tensor)[1] == np.shape(parametrization_matrix_H)[0] + np.shape(parametrization_matrix_D)[0]:
                    parametrization_matrix = Parametrization.combine_parametrization_matrices(parametrization_matrices)
                else:
                    raise ValueError("Shape of constraint tensor {} does not match shape of parametrization matrices GH{}, GD{}.".format(np.shape(tensor),np.shape(parametrization_matrix_H),np.shape(parametrization_matrix_D)))
                # parametrize constraint tensor
                parametrized_tensor = np.einsum("ma,ab->mb",tensor,parametrization_matrix)
                # parametrize variance of constraint tensor
                parametrized_var_tensor = None
                if var_tensor is not None:
                    parametrized_var_tensor = np.einsum("ma,ab->mb",var_tensor,np.power(parametrization_matrix,2))
                # combine tensors and vectors
                parametrized_tensor_list = [parametrized_tensor, vector, parametrized_var_tensor, var_vector]
            #------------------------------------------------------------
            ### OPTION 3 ### O3KZi and O3KZd method ###
            elif key in ["O3KZi", "O3KZd"]:
                tensor = constraint_tensors[key][0]
                tensor_diss = constraint_tensors[key][1]
                var_tensor = constraint_tensors[key][2]
                var_tensor_diss = constraint_tensors[key][3]
                # parametrize constraint tensor
                parametrized_tensor = np.einsum("ma,ab->mb",tensor,parametrization_matrix_H)
                # parametrize variance of constraint tensor
                parametrized_var_tensor = None
                if var_tensor is not None:
                    parametrized_var_tensor = np.einsum("ma,ab->mb",var_tensor,np.power(parametrization_matrix_H,2))
                # parametrize dissipative constraint tensor
                parametrized_tensor_diss = None
                parametrized_var_tensor_diss = None
                if parametrization_matrix_D is not None and np.shape(parametrization_matrix_D)[1]>0: # and np.linalg.norm(parametrization_matrix_D)>1e-5:
                    parametrized_tensor_diss = np.einsum("cd,mca,ab->mdb",parametrization_matrix_H,tensor_diss,parametrization_matrix_D)
                    if var_tensor_diss is not None:
                        parametrized_var_tensor_diss = np.einsum("cd,mca,ab->mdb",np.power(parametrization_matrix_H,2),var_tensor_diss,np.power(parametrization_matrix_D,2))
                # combine tensors and vectors
                parametrized_tensor_list = [parametrized_tensor, parametrized_tensor_diss, parametrized_var_tensor, parametrized_var_tensor_diss]
            #------------------------------------------------------------
            else:
                raise ValueError("Constraint tensor type {} not recognized.".format(key))
            #------------------------------------------------------------
            ### save parametrized constraint tensors
            parametrized_constraint_tensors[key] = parametrized_tensor_list
        #------------------------------------------------------------
        return parametrized_constraint_tensors

    # NOTE: this function is slow and should be optimized
    @staticmethod
    def parametrize_regularization_matrices(regularization_matrices, parametrization_matrices):
        """
        Parametrize the regularization matrices.
        
        The regularization matrices are parametrized 
        using the given parametrization matrix (T_par = T @ G).

        Parameters
        ----------
        regularization_matrices : tuple of np.arrays
            regularization_matrices[0] is the regularization matrix for the coherent terms.
            regularization_matrices[1] is the regularization matrix for the incoherent terms.
        parametrization_matrices : list
            List of tuples of parametrization matrices.
            First element is the parametrization matrix for the ansatz_operator.
            Second element is the parametrization matrix for the ansatz_dissipators.

        Returns
        -------
        tuple of np.arrays
            The parametrized regularization matrices.
            parametrized_regularization_matrices[0] is the parametrized regularization matrix for the coherent terms (T_par = T @ G).
            parametrized_regularization_matrices[1] is the parametrized regularization matrix for the incoherent terms (T_par = T @ G).
        """
        regularization_matrix_H = regularization_matrices[0]
        regularization_matrix_D = regularization_matrices[1]
        #-----------------------------------------------
        ### STEP 1 ### coherent terms
        parametrized_regularization_matrix_H = None
        if regularization_matrix_H is not None and parametrization_matrices[0] is not None:
            parametrized_regularization_matrix_H = np.einsum("ij,jk",regularization_matrix_H,parametrization_matrices[0])
        #-----------------------------------------------
        ### STEP 2 ### incoherent terms
        parametrized_regularization_matrix_D = None
        if regularization_matrix_D is not None and parametrization_matrices[1] is not None:
            parametrized_regularization_matrix_D = np.einsum("ij,jk",regularization_matrix_D,parametrization_matrices[1])
        #-----------------------------------------------
        return [parametrized_regularization_matrix_H, parametrized_regularization_matrix_D]

    @staticmethod
    def get_overall_scale(
            operator_unscaled: QuantumOperator,
            constraint_tensors_scale: dict,
            dissipators: list | None = None,
            ):
        """
        Learn the overall scale using separate constraints.

        Returns the overall scale for operator_unscaled using constraint_tensors_scale.

        Parameters
        ----------
        operator_unscaled : QuantumOperator
            Unscaled operator for which to calculate the overall scale.
        constraint_tensors_scale : dict
            constraint tensors for the overall scale (either "BAL" or "ZYLB")
            constraint_tensors[0] is the constraint matrix
            constraint_tensors[1] is the constraint vector
        dissipators : list of Dissipators, optional
            (Scaled) learned Dissipators for dissipation correction
            used for the O3KZ methods.
            Default is None.

        Returns
        -------
        QuantumOperator
            The scaled quantum operator (scale_mean times the normalized operator_unscaled)
        float
            Mean value of the overall scale.
            Mean is taken over all constraints (rows of the constraint matrix).
        float
            Variance of the overall scale.
            Variance is taken over all constraints (rows of the constraint matrix).
        """
        if len(list(constraint_tensors_scale.keys())) > 1:
            raise ValueError("Only one constraint tensor type is allowed for learning the overall scale.")
        method_scale = list(constraint_tensors_scale.keys())[0]
        if method_scale not in ["BAL"]:
            raise ValueError("method_scale is {}, but only BAL is implemented.".format(method_scale))
        #--------------------------------------------------------
        ### STEP 1 ### get constraint matrix M and vector b
        Mtot = constraint_tensors_scale[method_scale][0]
        MH = Mtot[:,:len(operator_unscaled.coeffs())]
        MD = Mtot[:,len(operator_unscaled.coeffs()):]
        b = constraint_tensors_scale[method_scale][1]
        ## apply dissipation correction to constraint vector b
        if dissipators is not None:
            dissipation_rates = np.array([diss.coeff for diss in dissipators])
            bcorr = np.multiply(-1,np.einsum("ij,j",MD,dissipation_rates))
            b = np.add(b,bcorr)
        #--------------------------------------------------------
        ### STEP 2 ### get overall scale
        operator_unscaled = operator_unscaled/np.linalg.norm(operator_unscaled.coeffs())
        coefficients = operator_unscaled.coeffs()
        div = np.einsum("ij,j",MH,coefficients)  
        ## get scale for each constraint      
        scale_list = []
        for inx in range(len(div)):
            if np.abs(div[inx])<1e-5:
                print("Warning: div[{}] = {}".format(inx,div[inx]))
                print("norm Mscale",np.linalg.norm(MH))
                # continue
            if np.abs(b[inx])<1e-3:
                print("Warning: b[{}] = {}".format(inx,b[inx]))
                print("norm bscale",np.linalg.norm(b))
                # continue
            scale = b[inx]/div[inx]
            scale_list.append(scale)
        if len(scale_list)==0:
            print("b/div failed for div = {} and b = {}".format(div,b))
            raise ValueError("No scale found.")
        #--------------------------------------------------------
        ### STEP 3 ### apply scale to operator
        scale_list = np.array(scale_list)
        scale_mean = np.mean(scale_list)
        scale_var = np.var(scale_list)
        operator_scaled = scale_mean*operator_unscaled
        #--------------------------------------------------------
        return operator_scaled, scale_mean, scale_var

    @staticmethod
    def get_cov_of_solution_of_linear_system(c, M, var_M=None, var_b=None, cov_M=None, cov_b=None, print_timers=False):
        """
        Get variance of solution from variance of M.

        Computes the variance and covariance of the least-squares solution c = (M^T M)^(-1) M^T b = Minv b
        from either the variances or the covariance of M and b.
        The covariance of c is given by cov_c = Minv cov_b Minv^T + Minv cov_M Minv^T.

        Parameters
        ----------
        c : n x 1 numpy array
            least-squares solution
        M : m x n numpy array
            constraint matrix
        var_M : mxn numpy array, optional
            variance of M
        var_b : m x 1 numpy array, optional
            variance of b
        cov_M : m*n x m*n numpy array, optional
            covariance of vec(M) (stacked columns)
        cov_b : m x m numpy array, optional
            covariance of b
        print_timers : bool, optional
            whether to print timers
            Default is False

        Returns
        -------
        n x 1 numpy array
            variance of c 
        n x n numpy array
            covariance of c 
        """
        ### STEP 1 ### setup 
        tm0 = time.time()
        ## get covariance matrices
        if cov_M is None:
            if var_M is not None:
                cov_M = np.diag(var_M.flatten())
            else:
                raise ValueError("Either var_M or cov_M must be provided.")
        if cov_b is None:
            if var_b is not None:
                cov_b = np.diag(var_b)
            else:   
                raise ValueError("Either var_b or cov_b must be provided.")
        ## precompute terms
        MTM = M.T @ M
        MTM_inv = np.linalg.inv(MTM)
        MTM_inv_MT = MTM_inv @ M.T
        M_MTM_inv = M @ MTM_inv # == MTM_inv_MT.T if MTM_inv is symmetric   
        if print_timers:
            tm1 = time.time()
            print("... of the following time for setup = {}".format(tm1-tm0)) 
        # ------------------------------------------------------
        ### STEP 2 ### contribution from cov(b)
        cov_c_b = MTM_inv_MT @ cov_b @ M_MTM_inv
        if print_timers:
            tm2 = time.time()
            print("... of the following time for cov_b = {}".format(tm2-tm1))
        # ------------------------------------------------------
        ### STEP 3 ### contribution from cov(M)
        In = np.eye(M.shape[0])
        K1 = np.kron(c.T, In)
        K2 = K1.T
        Inner_Term = K1 @ cov_M @ K2
        cov_c_M = MTM_inv_MT @ Inner_Term @ M_MTM_inv
        if print_timers:
            tm3 = time.time()
            print("... of the following time for cov_M = {}".format(tm3-tm2))
        # ------------------------------------------------------
        ### STEP 4 ### combine contributions
        cov_c = cov_c_b + cov_c_M
        var_c = np.diag(cov_c)
        # ------------------------------------------------------
        return var_c, cov_c

    def get_constraint_sample_indices(
            self,
            label: str  = "learn",
            # replace: bool = False,
            # sample_ratio: float = 0,
            # n_samples: int = 1,
            # jackknife: bool = False,
            ):
        """
        Return indices for sampling constraint tensors for errorbars.

        Only the constraint tensors of the given label are sampled,
        the others are not sampled, but all rows are used.
        The rows of the constraint tensors are sampled in order to 
        calculate errorbars for the learned ansatz coefficients via bootstrap/jackknife resampling.
        By default, the samples are drawn without replacement.
        For bootstrap, the number of samples is set by n_samples
        and the ratio of excluded rows is set sample_ratio.
        If jackknife is True, all possible samples with 1 row excluded are drawn and the 
        corresponding jackknife inflation factor is calculated.

        Parameters
        ----------
        label : str, optional
            Only get sample indices for constraint_tensors with the given label (e.g., "learn", "scale", "diss").
            Other constraint tensors are not sampled, but all rows are used.
            If None, all constraint tensors are sampled.
            Default is ``"learn"``, i.e., only constraint tensors used for learning are sampled (not for scale or dissipation reconstruction).
        # replace : bool, optional
        #     If True, the samples are drawn with replacement.
        #     Default is False.
        # sample_ratio : float, optional
        #     Ratio of rows excluded from the constraint tensor for each sample.
        #     Default is ``0``.
        # n_samples : int, optional
        #     Number of samples drawn for each constraint tensor.
        #     Default is ``1``.
        # jackknife : bool, optional
        #     If True, all possible samples for given sample_ratio are drawn.
        #     Default is False.
            
        Returns
        -------
        dict
            Sampling indices as dictionary where 
            the key is (label,method) and the value is the list of row indices to be sampled.
        dict
            Inflation factor for jackknife resampling as dictionary where 
            the key is (label,method) and the value is the inflation factor.
        """
        replace = self.resampling_constraints_replace
        sample_ratio = self.n_resampling_constraints[0]
        n_samples = self.n_resampling_constraints[1]
        jackknife = self.resampling_constraints_jackknife
        #-----------------------------------------------------
        if jackknife:
            if sample_ratio == 0:
                raise ValueError("Jackknife resampling requires sample_ratio > 0.")
            if replace:
                raise ValueError("Jackknife resampling requires replace=False.")
        #-----------------------------------------------------
        ### STEP 1 ### take samples of indices of rows of constraint tensors
        constraint_sample_indices = {}
        jackknife_inflation_factors = {}
        # loop over all stored constraint tensors
        for key in self.constraint_tensors.keys():
            method = key[1]
            # get number of rows of the constraint tensor
            number_constraints = np.shape(self.constraint_tensors[key][0])[0]
            #-----------------------------------------------------
            ### step 1 ### store indices for sampled constraint tensors
            if label is None or key[0]==label:
                # check if samples are already stored for given (label,method) pair (to avoid repeated sampling for different nshots)
                if (label,method) not in constraint_sample_indices.keys():
                    if not jackknife:
                        constraints_per_sample = math.floor(number_constraints*(1-sample_ratio))
                        sample_indices = [np.random.choice(np.arange(number_constraints), size=constraints_per_sample, replace=replace) for inx in range(n_samples)]
                        jackknife_inflation_factor = 1 #len(sample_indices)/(len(sample_indices)-1)   # adjust for degrees of freedom
                    if jackknife:
                        constraints_per_sample = number_constraints - 1
                        sample_indices = list(it.combinations(np.arange(number_constraints), constraints_per_sample))
                        jackknife_inflation_factor = constraints_per_sample/(number_constraints-constraints_per_sample) 
                    # sort sample_indices
                    sample_indices = [np.sort(sample) for sample in sample_indices]
                    constraint_sample_indices[(label,method)] = sample_indices
                    jackknife_inflation_factors[(label,method)] = jackknife_inflation_factor
            #-----------------------------------------------------
            ### step 2 ### store all indices for unsampled constraint tensors
            else:
                # check if samples are already stored for given (label,method) pair (to avoid repeaded sampling for different nshots)
                if (label,method) not in constraint_sample_indices.keys():
                    all_indices = np.arange(number_constraints)
                    constraint_sample_indices[(label,method)] = [all_indices for inx in range(n_samples)]
                    jackknife_inflation_factors[(label,method)] = 1
        #-----------------------------------------------------
        ### STEP 2 ### check number of samples taken
        allkeys = list(constraint_sample_indices.keys())
        n_constraints = max([len(constraint_sample_indices[key]) for key in allkeys])
        pstr_resampling = "number of samples taken for errorbars is {}".format(n_constraints)
        ic(pstr_resampling)
        #-----------------------------------------------------
        return constraint_sample_indices, jackknife_inflation_factors

    def get_constraint_operators_for_bases(self, bases, max_support=None):
        """ 
        Return all possible constraint operators for given bases.

        Returns all possible constraint operators for BAL or ZYLB method,
        that can be measured in given bases.

        Parameters
        ----------
        bases : list
            List of measurement bases.
        max_support : int, optional
            Maximum number of qubits on which the constraint operator is non-identity.
            Default is self.Nions

        Returns
        -------
        list of QuantumOperators
            List of all constraint operators that can be measured in bases.
        """
        if max_support is None:
            max_support = self.Nions
        # -----------------------------------------------------
        # get all possible Pauli operators for Nions qubits, that are non-I on maximum r qubits, and I on the rest
        all_qops = [QuantumOperator(self.Nions, terms={"".join(pstr): 1}) for pstr in list(it.product(["I","X","Y","Z"], repeat=self.Nions)) if np.sum([1 for p in pstr if p!="I"])<=max_support and np.sum([1 for p in pstr if p=="I"])<self.Nions]
        ## get all constraints that can be measured in bases
        constraint_operators = []
        for constr in all_qops:
            required_operator = constr
            if self.ansatz_operator is not None:
                comm = constr.commutator(self.ansatz_operator)
                required_operator += comm
            if self.ansatz_dissipators is not None:
                # get all upper triangle pairs of dissipation operators
                for diss in self.ansatz_dissipators:
                    term = diss(constr)
                    required_operator += term
            # check if bases are sufficient
            required_operator_nozero = required_operator.remove_zero_coeffs()
            can_be_measured = required_operator_nozero.can_be_measured(bases,no_output=True)
            if can_be_measured:
                constraint_operators.append(constr)
        # -----------------------------------------------------
        return constraint_operators

    def get_bases_for_constraints(
            self, 
            constraint_operators, 
            bases_given: list | None = None, 
            replace_I: str = "I",
            ):
        """
        Returns required bases for given constraints.

        Used for BAL or ZYLB method.

        Parameters
        ----------
        constraint_operators : list of QuantumOperators
            List of constraint operators to be measured.
        bases_given : list of strings, optional
            List of bases, that should be extended to become sufficient for measuring the constraints.
            Default is None.
        replace_I : str, optional
            Replace all "I" in each basis with the given string.
            This is for choosing a default measurement basis where it is irrelevant for measuring the constraints.
            If replace_I=="common", "I" is replaced with the most common char in each basis.
            Default is "I".

        Returns
        -------
        list of strings
            List of measurement bases required for measuring the constraints.
        list of list of strings
            Clique cover as returned by QuantumOperator.get_measurement_bases().
            It is a list of cliques that cover all terms in the QuantumOperator that must be measured.
        """
        # -----------------------------------------------------
        ### get required operator
        required_operator = constraint_operators[0]
        for cinx in range(1,len(constraint_operators)):
            required_operator += constraint_operators[cinx]
        if self.ansatz_operator is not None:
            for constr in constraint_operators:
                comm = constr.commutator(self.ansatz_operator)
                required_operator += comm
        if self.ansatz_dissipators is not None:
            # get all upper triangle pairs of dissipation operators
            for diss in self.ansatz_dissipators:
                for constr in constraint_operators:  
                    term = diss(constr)
                    required_operator += term
        # -----------------------------------------------------
        ### determine minimal set of measurement bases
        required_operator_nozero = required_operator.copy()
        required_operator_nozero.remove_zero_coeffs()
        # check if bases are sufficient
        bases, clique_cover = required_operator_nozero.get_measurement_bases(bases_given=bases_given, no_output=True)
        ### replace "I" in bases
        if replace_I=="common":
            # replace by most common non-I char
            bases_new = bases.copy()
            for binx, b in enumerate(bases):
                # get counts
                counts = Counter(b)
                # remove "I"
                counts.pop("I", None)
                # get most common char
                most_common = max(counts, key=counts.get)
                bases_new[binx] = b.replace("I",most_common)
            bases = bases_new
        else:
            bases = [b.replace("I",replace_I) for b in bases]
        # -----------------------------------------------------
        return bases, clique_cover

    @staticmethod
    def normalize_constraint_tensors(constraint_tensors) -> dict:
        """
        Normalize the rows of the constraint tensors.
        
        Each constraint tensor is normalized by the maximum norm of the rows of M.
        This makes sure that different constraint tensors have the same contribution to the 
        full constraint matrix M and constraint vector b.
        TODO: Only impelemented for BAL method so far.

        Parameters
        ----------
        constraint_tensors : dict
            Dictionary with entry method:tensors_list,
            where method is the learning method and tensors_list is a list of all loaded constraint tensors.

        Returns
        -------
        dict
            The normalized constraint tensors stored in a dictionary,
            where keys are the learning methods and values are the normalized constraint tensors.
        """
        for key in constraint_tensors.keys():
            ## for BAL method
            if key == "BAL":
                for inx in range(len(constraint_tensors[key])):
                    M = constraint_tensors[key][inx][0]
                    b = constraint_tensors[key][inx][1]
                    max_norm_per_row = np.squeeze(np.max(np.abs(M),axis=1))
                    M_new = M / max_norm_per_row[:,np.newaxis]
                    b_new = b / max_norm_per_row
                    constraint_tensors[key][inx][0] = M_new
                    constraint_tensors[key][inx][1] = b_new
            # TODO: normalize constraints for other methods
            else:
                raise NotImplementedError("Normalization of constraints for method {} not implemented yet.".format(key))
        # -----------------------------------------------------
        return constraint_tensors

    @staticmethod
    def combine_constraint_tensors(constraint_tensors_list: list) -> dict:
        """
        Combine list of constraint tensors into a single dictionary.

        Combines all constraint tensors in constraint_tensors_list into
        a single dictionary with the same keys.
        Constraint tensors of differenty type are stored with separate keys.
        Constraint tensors of the same type are concatenated
        along the direction of the constraints.

        Parameters
        ----------
        constraint_tensors_list : list
            List of dictionaries with constraint tensors.

        Returns
        -------
        dict
            Combined constraint tensors.
        """
        combined_constraint_tensors = {}
        for constraint_tensor in constraint_tensors_list:
            for key in constraint_tensor.keys():
                tmp_tensor = constraint_tensor[key].copy()
                if key not in combined_constraint_tensors.keys():
                    combined_constraint_tensors[key] = tmp_tensor
                else:   
                    # concatenate constraint tensors according to their type
                    if key in ["O3KZi,O3KZd"]:
                        combined_tensor = combined_constraint_tensors[key]
                        combined_tensor[0] = np.concatenate((combined_tensor[0],tmp_tensor[0]),axis=0)
                        combined_tensor[1] = np.concatenate((combined_tensor[1],tmp_tensor[1]),axis=0)
                        combined_tensor[2] = np.concatenate((combined_tensor[2],tmp_tensor[2]),axis=0)
                        combined_tensor[3] = np.concatenate((combined_tensor[3],tmp_tensor[3]),axis=0)
                    elif key in ["ZYLB","BAL"]:
                        combined_tensor = combined_constraint_tensors[key]
                        combined_tensor[0] = np.concatenate((combined_tensor[0],tmp_tensor[0]),axis=0)
                        combined_tensor[1] = np.concatenate((combined_tensor[1],tmp_tensor[1]),axis=0)
                        combined_tensor[2] = np.concatenate((combined_tensor[2],tmp_tensor[2]),axis=0)
                        combined_tensor[3] = np.concatenate((combined_tensor[3],tmp_tensor[3]),axis=0)
                    else:
                        raise ValueError("Key {} not recognized.".format(key))
        # -----------------------------------------------------
        return combined_constraint_tensors


    def get_measurement_settings(
            self, 
            method: str,
            constraints: list,
            nshots: int,
            suggested_measurement_bases: list | None = None
            ) -> dict:
        """ 
        Get dict of measurement settings for given constraints.

        TODO: Not all bases are needed at the endpoints, some only for the integral.
        Currently all bases are measured at all times.

        Parameters
        ----------
        method : str
            The learning method for which the data is generated.
            Options are "O3KZi", "BAL", "ZYLB", "O3KZd" and "LZHx".
            LZHx adds all data up to order x.
        constraints : list
            list of Constraints for which to generate the constraint tensor and vector,
            where each constraint contains:
                - initial_state (QuantumState)
                    initial state of the system
                - simulation_times (list)
                    times at which the constraint is evaluated
                    If method=="LZHx", only the first and last element of simulation_times are used.
                - constraint_operator (QuantumOperator)
                    operator for which the expectation value is calculated
                    If method in ["O3KZi","O3KZd","LZHx"], constraint_operator is not used.
                - nshots_ratio_integrand (float) [default: 1]
                    Ratio of number of shots used for each time step in the integrand,
                    compared to the number of shots used at the end points.
        nshots : int
            Number of shots used for each measurement setting.
        suggested_measurement_bases : list of Pauli strings, optional
            Suggested list of measurement bases for the data generation.
            If possible, the suggested bases are used for the data generation.
            If None or insufficient, measurement bases are determined automatically.
            Default is None.

        Returns
        -------
        dict
            The measurement settings for the given constraints as a
            dict of state:measurement_settings_list pairs 
            where measurement_settings_list is a list of measurement settings.
        """
        #--------------------------------------------------------
        ### set dict of state:measurement_settings pairs
        measurement_settings = {}   
        for constraint in constraints:
            times = constraint.simulation_times
            constraint_operator = constraint.constraint_operator
            state = constraint.initial_state
            ## add state to measurement_settings if not already there
            if state not in measurement_settings.keys():
                measurement_settings[state] = []
            ## get required operator (usually same operator needed for all times)
            required_operator_endpoints, required_operator_integrand = self.get_required_operator(method,constraint_operators=[constraint_operator])
            required_operator = required_operator_endpoints
            if required_operator_integrand is not None:
                required_operator = required_operator_endpoints+required_operator_integrand
            ## check if bases are sufficient
            can_be_measured = False
            if suggested_measurement_bases is not None:
                can_be_measured, required_bases = required_operator.can_be_measured(suggested_measurement_bases, return_required_bases=True)
            ## add bases if necessary
            if not can_be_measured and nshots>0:
                print("Measurement bases {} are not sufficient to measure all terms in the required operator {}.".format(suggested_measurement_bases, required_operator.str()))
                measurement_bases, clique_cover = required_operator.get_measurement_bases(bases_given=suggested_measurement_bases)
                can_be_measured, required_bases = required_operator.can_be_measured(measurement_bases, return_required_bases=True)
                print("Required bases are: {}".format(required_bases))
            #--------------------------------------------------------
            ### get measurement settings for given constraint
            for time in times:
                ### add measurement settings for finite number of shots in given basis
                if nshots>0:
                    ## set nshots and required bases (integrand or endpoints)
                    required_bases_tmp = []
                    if time in [times[0],times[-1]]:
                        nshots_tmp = nshots ## TODO not all operators in required_operator have to be measured at endpoints!!
                        # can_be_measured_tmp, required_bases_tmp = required_operator_endpoints.can_be_measured(required_bases, return_required_bases=True)
                        can_be_measured_tmp, required_bases_tmp = required_operator.can_be_measured(required_bases, return_required_bases=True)
                    elif required_operator_integrand is not None:
                        nshots_tmp = int(np.ceil(nshots*constraint.nshots_ratio_integrand))
                        can_be_measured_tmp, required_bases_tmp = required_operator_integrand.can_be_measured(required_bases, return_required_bases=True)
                    ### add measurement settings for each basis
                    for basis in required_bases_tmp:
                        setting = MeasurementSetting(initial_state=constraint.initial_state, measurement_basis=basis, simulation_time=time, nshots=nshots_tmp)
                        # check if setting is already in measurement_settings
                        if setting not in measurement_settings[state]:
                            measurement_settings[state].append(setting)
                        else:
                            # check if nshots is larger
                            setting_in_list = measurement_settings[state][measurement_settings[state].index(setting)]
                            if setting_in_list.nshots < setting.nshots:
                                # lists are mutable, so measurement_settings is updated automatically
                                setting_in_list.nshots = setting.nshots
                ### add measurement settings for exact expectation values
                elif nshots==0:
                    ## set required operator (integrand or endpoints)
                    if time in [times[0],times[-1]]:
                        #TODO: excluded operators that are not needed at endpoints (should be fine)
                        required_operator_tmp = required_operator_endpoints
                        # required_operator_tmp = required_operator
                    else:
                        required_operator_tmp = required_operator_integrand
                    ## add measurement settings for required operator (exact expectation values)
                    if required_operator_tmp is not None:
                        setting = MeasurementSetting(initial_state=constraint.initial_state, simulation_time=time, exact_observables=required_operator_tmp)
                        # check if setting is already in measurement_settings
                        if setting not in measurement_settings[state]:
                            measurement_settings[state].append(setting)
                        else:
                            # add exact expectation values
                            setting_in_list = measurement_settings[state][measurement_settings[state].index(setting)]
                            setting_in_list.exact_observables += setting.exact_observables
        # -----------------------------------------------------
        return measurement_settings

    def get_nruns(
            self, 
            method: str, 
            constraints: list, 
            nshots: int, 
            suggested_measurement_bases: list | None = None,
            ) -> int:
        """
        Get required number of runs.

        Calculate the required number of runs of the experiment, 
        given the learning method, the constraints and a list of nshots.

        Parameters
        ----------
        method : str
            The learning method for which the data is generated.
            Options are "O3KZi", "BAL" and "ZYLB".
        constraints : list
            List of Constraints for which to generate the constraint tensor and vector.
        nshots : int
            Number of shots used for each measurement setting.
        suggested_measurement_bases : list of Pauli strings 
            Suggested list of measurement bases for the data generation.
            If possible, the suggested bases are used for the data generation.
            If None or insufficient, measurement bases are determined automatically.
            Default is None.

        Returns
        -------
        int
            Required number of runs of the experiment.
        """
        ### get measurement settings
        measurement_settings_learn = self.get_measurement_settings(method, constraints, nshots, suggested_measurement_bases=suggested_measurement_bases) #, MHexact=MHexact, MDexact=MDexact)
        ### get nruns from measurement settings
        nruns_parts = [msett.nshots for tmpkey in measurement_settings_learn.keys() for msett in measurement_settings_learn[tmpkey]]
        nruns_parts_filtered = [x for x in nruns_parts if x is not None]
        nruns = np.sum(nruns_parts_filtered)
        # -----------------------------------------------------
        return nruns
    
    @staticmethod
    def draw_constraint_tensor_samples(
            constraint_tensors: dict, 
            constraint_sample_indices: dict,
            constraint_tensors_scale: dict | None = None,
            constraint_tensors_diss: dict | None = None,
        ) -> list:
        """
        Sample the rows of the constraint tensors.

        If constraint_sample_indices is empty, returns the unsampled tensors.
        Otherwise returns samples of the constraint tensors,
        where the rows of the constraint tensors are chosen according to constraint_sample_indices
        for each constraint tensor type.
        NOTE: This function takes a lot of RAM (for large constraint tensors).
        NOTE: When sampling constraints, in the case of BAL constraints, the constraints for different 
        constraint operators are not statistically independent (come from the same measurement).

        Parameters
        ----------
        constraint_tensors : dict
            Dictionary of key:constraint_tensor pairs.
            key can be any of "O3KZi","O3KZd","ZYLB" or "BAL".
        constraint_sample_indices : dict
            Dictionary of key:indices pairs.
            key can be any of "O3KZi", "O3KZd", "ZYLB" or "BAL".
            indices is a list of indices for sampling the rows ("constraints") of the constraint tensors.
        constraint_tensors_scale : dict, optional
            Additional constraints used for scale reconstruction.
            keys can be "BAL" or "ZYLB".
            Default is None.
        constraint_tensors_diss : dict, optional
            Additional constraints used for separate dissipation learning.
            keys can be "BAL" or "ZYLB".
            Default is None.

        Returns
        -------
        tuple of dicts
            Sampled constraint tensors as a tuple of 3 dictionaries.
            The first dictionary are the constraint tensors for learning.
            The second dictionary are the constraint tensors for scale reconstruction.
            The third dictionary are the constraint tensors for dissipation learning. (O3KZ method only)
        """
        unsampled_tensors = [constraint_tensors, constraint_tensors_scale, constraint_tensors_diss]
        #--------------------------------------------------------
        ### STEP 1 ### return unsampled_tensors if constraint_sample_indices is None or empty
        if constraint_sample_indices is None or len(constraint_sample_indices.keys()) == 0:
            return unsampled_tensors
        #--------------------------------------------------------
        ### STEP 2 ### sample constraint tensors according to constraint_sample_indices
        labels = ["learn", "scale", "diss"]
        sampled_tensors = [None, None, None]
        for dictinx, tensor_dict in enumerate(unsampled_tensors):
            tmplabel = labels[dictinx]
            if tensor_dict is not None:
                sampled_tensors[dictinx] = {}
                ### add sample for each type of constraint tensor
                for key, tensor in tensor_dict.items():
                    ## get sample indices (in case there are no sample_indices for the given constraint_tensor all constraints are included in the sample)
                    if (tmplabel,key) not in constraint_sample_indices.keys():
                        tmp_sample_indices = np.arange(len(tensor[0]))
                    else:
                        tmp_sample_indices = constraint_sample_indices[(tmplabel,key)]
                    ## get sample
                    entry = [None for tmp in tensor]
                    for tmp_inx, tmp_tensor in enumerate(tensor):
                        if tmp_tensor is not None:
                            entry[tmp_inx] = tmp_tensor[tmp_sample_indices[0]]
                    sampled_tensors[dictinx][key] = entry
        return sampled_tensors



### --------------- ###
### other functions ###
### --------------- ###
def get_times(
        alltimes, 
        ntimesteps: int, 
        ntimes: int, 
        split_times: bool = False,
    ) -> list:
    """
    Get list of times for learning.

    Creates multiple lists of times that can be used to define constraints for learning.
    Each constraint consists of a list of times, where the first element is the
    initial time and the last element is the final time of the simulation.
    Times in between are used for approximating the time-integrals required for the BAL or O3KZ method.

    Parameters
    ----------
    alltimes : list
        List of all times for which there are measurements.
    ntimesteps : int
        Number of times from alltimes that are used for learning.
        If ntimesteps==len(alltimes), all times are used.
        If ntimesteps<len(alltimes), ntimesteps times are taken equally spaced from alltimes.
    ntimes : int
        Number of time constraints created.
        Each time constraint is a list of times, where the first and last time are the endpoints.
        The times in between are used for the integral.
    split_times : bool, optional
        If False, time constraints are chosen starting always at alltimes[0] and ending at variable times.
        If True, time constraints are chosen subsequently, starting from the last time of the previous constraint.
        Default is False.

    Returns
    -------
    list
        List of multiple lists of times for learning.
        List of lists, where each list corresponds to a single constaint.
        Within the list of times, the first element is the initial time and the last element is the final time.
        Times in between are used for the integral.
    """
    if len(alltimes) < ntimesteps:
        raise ValueError("len(alltimes)={}, but must be >= ntimesteps={}".format(len(alltimes),ntimesteps))
    if ntimes+1 > ntimesteps:
        raise ValueError("ntimes+1 = {}, but must be <= ntimesteps = {}".format(ntimes+1,ntimesteps))
    #---------------------------------------------------------
    ### STEP 1 ### get times for learning
    ## get ntimesteps equally spaced times from alltimes
    alltimes_cut = alltimes.copy()
    if ntimesteps != len(alltimes):
        stepsize = (len(alltimes)-1) / (ntimesteps-1)
        indices = [round(i*stepsize) for i in range(ntimesteps)]
        alltimes_cut = alltimes_cut[indices]
    ## get ntimes equally spaced times from alltimes_cut
    stepsize = (len(alltimes_cut)-1) / ntimes
    endtimes_indices = [round(i*stepsize) for i in range(0,ntimes+1)]
    ## get times for learning
    times = []
    for einx in range(len(endtimes_indices)-1):
        if split_times:
            times.append(alltimes_cut[endtimes_indices[einx]:endtimes_indices[einx+1]+1])
        else:
            times.append(alltimes_cut[:endtimes_indices[einx+1]+1])
    #---------------------------------------------------------
    ### STEP 2 ### test if times are correct
    # check if number of time constraints is correct
    if len(times) != ntimes:
        print("time:",times)
        raise ValueError("Number of times is not correct. Expected {}, but got {}.".format(ntimes,len(times)))
    # check if total number of timesteps is correct
    if split_times:
        ntimes_tot = np.sum([len(t)-1 for t in times])+1
    elif not split_times:
        ntimes_tot = np.max([len(t) for t in times])
    if ntimes_tot != ntimesteps:
        print("time:",times)
        raise ValueError("Number of timesteps is not correct. Expected {}, but got {} for times={}.".format(ntimesteps,ntimes_tot,times))
    #---------------------------------------------------------
    return times

def sort_constraints_by_states_and_times(constraints) -> dict:
    """
    Sort constraints by initial state and simulation time.

    Sorts a list of Constraints by initial states and simulation times.
    Returns a dictionary where keys are (initial_state, time) pairs and values are lists of Constraints.

    Parameters
    ----------
    constraints : list of Constraint objects
        List of constraints to sort.

    Returns
    -------
    dict
        Sorted constrainst as dictionary.
        Dictionary of initial_state:constraints pairs, 
        where initial_state is a QuantumState and 
        constraints is a list of Constraints for the given initial state.
    """
    constraints_by_states = {}
    for cinx, constraint in enumerate(constraints):
        state = constraint.initial_state
        if state not in constraints_by_states.keys():
            constraints_by_states[state] = []
        constraints_by_states[state].append(constraint)
    return constraints_by_states

def solve_linear_eq(
        M: np.ndarray | None = None,
        b: np.ndarray | None = None,
        exclude_lowest_solutions: int = 0,
        ):
    """
    Solves the linear equation M*sol=b.

    If b is not provided, the solution is the right-singular-vector corresponding to the lowest singular value of M.
    If b is provided, the solution is right-singular-vector corresponding to the lowest singular value of (M,-b).
    Additionally returns the svd spectrum and vectors of M if b=None, or (M,-b) if b is provided.

    Parameters
    ----------
    M : 2D np.array
        matrix that defines the system of equations
    b : 1D np.array
        vector that defines the system of equations
    exclude_lowest_solutions (int) [default: 0]
        Number of lowest singular values and corresponding vectors to exclude, 
        e.g, if exclude_lowest_solutions=1 we solve for the second-lowest singular value and vector.

    Returns
    -------
    1D np.array
        solution of the linear equation
    1D np.array
        error of the solution (M*sol-b)/norm(sol)
    1D np.array
        If b is None, the singular value spectrum of M,
        otherwise None.
    pair of 2D np.arrays
        If b is None, the left and right singular vectors of M,
        otherwise None..
    """
    # check if constraint matrix and vector are provided
    if M is None:
        raise ValueError("No constraint matrix provided.")
    # check if M has more rows than columns
    if M.shape[0]<M.shape[1]:
        raise ValueError("Constraint matrix must have more rows ({}) than columns ({}).".format(M.shape[0],M.shape[1]))
    #--------------------------------------------------------
    ### STEP 1 ### solve linear equation M*sol=0 using singular value decomposition
    if b is None:
        # get lowest singular value and right-singular-vector of Mlzh
        u, s, vh = sp.linalg.svd(M, full_matrices=True)
        svd_vals = s
        svd_vecs = [np.transpose(np.conjugate(u)),vh]
        # get the right-singular-vector corresponding to the lowest singular value
        sol = vh[-1-exclude_lowest_solutions,:]
        err = np.einsum("ij,j",M,sol)   # norm(sol)=1
    #--------------------------------------------------------
    ### STEP 2 ### solve linear equation M*sol=b using least-squares
    # NOTE: sp.linalg.lstsq is faster than np.linalg.lstsq
    if b is not None:
        svd_vals = None
        svd_vecs = None
        # sol, residuals, rank, s = np.linalg.lstsq(M,b,rcond=None) 
        sol, residuals, rank, s = sp.linalg.lstsq(M, b) #, rcond=None) 
        err = np.subtract(np.einsum("ij,j", M, sol), b) #/ np.linalg.norm(sol_lsq)  #(M @ sol - b) / ||sol||
    #--------------------------------------------------------
    return sol, err, svd_vals, svd_vecs

def solve_nonlinear_eq(constraint_tensors, **kwargs):
    """
    Solves the nonlinear equation for LZHx method with x>=1.
    # TODO: fix this function 

    Solves the k-th order polynomial equations given by constraint tensors T^{(k)}.
    Returns solution sol that minimizes 
    sum_k sum_{i1,...i_k} T^{(k)}_{m,i1,..ik}*sol_i1*...*sol_ik=0 
    under the condition that ||sol||=1.
    
    Parameters
    ----------
    constraint_tensors : dict
        Dictionary of key:constraint_tensor pairs.
        key can be any of "BAL" or "LZHx" for integer x>=1.
    c0 : np.array, optional
        initial guess for the solution
        Default is np.ones().

    Returns
    -------
    np.array
        solution of the nonlinear equation
    np.array
        residual error of the solution (sum_{i1,...i_k} M_{m,i1,..ik}*sol_i1*...*sol_ik)
    """
    raise NotImplementedError("This function is not implemented yet.")
    # get kwargs
    c0 = kwargs.get("c0",None)
    # define system of polynomial equations
    def polynomial_system(c):
        error = Ansatz.evaluate_learning_error(c, constraint_tensors, combined=True)
        # concatenate all vecs in errors
        # errors = np.concatenate(errors)
        # print("errors1", errors)
        # add constraint to fix norm of solution to 1
        error_norm = np.linalg.norm(c)-1
        # print("error_norm", error_norm)
        error = np.concatenate((error,[error_norm]))
        # print("errors",errors)
        return error
    def cost(c):
        return np.real(polynomial_system(c))
    if c0 is None:
        keylist = list(constraint_tensors.keys())
        key0 = keylist[0]
        cdims = constraint_tensors[key0].shape[1]
        c0 = np.ones(cdims)
    sol, err, landscape = uf.minimize(cost,c0)
    learning_error = Ansatz.evaluate_learning_error(sol, constraint_tensors, combined=True)
    return sol, learning_error



########################
### helper functions ###
########################
def parallel_simulate(state, measurement_settings, Qsim):
    """
    Call Qsim.simulate() in parallel.

    Function used for parallelization in get_data method.
    Parallelization is over initial states.

    Parameters
    ----------
    state : QuantumState
        Initial state for which to simulate the data.
    measurement_settings : dict of state:MeasurementSetting pairs
        Dictionary with QuantumStates as keys and MeasurementSetting objects as values.
    Qsim : QuantumSimulator
        Quantum simulator used for simulating the data.

    Returns
    -------
    DataSet
        Data set with simulated data.
    """
    new_data_set = Qsim.simulate(state, measurement_settings=measurement_settings[state])
    return new_data_set

def parallel_learn_sampled_coefficients(sample_inx, ansatz, constraint_tensors, **kwargs):
    """
    TODO: FIX THIS FUNCTION
    Function used for parallelization in learn_sampled_coefficients method.
    Parallelization over constraint tensors.
    Tensors is a tuple of [constraint_tensors, complementary_constraint_tensors].
    """
    constraint_sample_indices_tmp = {key: [ansatz.constraint_sample_indices[key][sample_inx]] for key in ansatz.constraint_sample_indices}
    tensors_samples = Ansatz.draw_constraint_tensor_samples(constraint_tensors, constraint_sample_indices_tmp, constraint_tensors_scale=kwargs["constraint_tensors_scale"], constraint_tensors_diss=kwargs["constraint_tensors_diss"]) 
    tensors = tensors_samples[0]
    result_dict = Ansatz.learn_sampled_coefficients(tensors, **kwargs)
    return result_dict
