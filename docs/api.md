# API Reference

This document provides an overview of the core classes in the hamiltonian-liouvillian-learning package, organized by functional area.

---

## Core Quantum Objects

### pauli_algebra

#### `PauliOperator`

Represents a tensor product of Pauli matrices acting on a fixed number of qubits. Each operator is defined by a Pauli string (e.g., "XZI") and a complex coefficient. Supports algebraic operations like multiplication, addition, and commutators.

**Key attributes:**
- `N`: Number of qubits
- `type`: Pauli string representation (e.g., "XZI")
- `coeff`: Complex coefficient

#### `QuantumOperator`

Represents a linear combination of Pauli operators, forming a quantum operator as a sum of Pauli terms. Used to represent Hamiltonians, observables, and other quantum operators.

**Key attributes:**
- `N`: Number of qubits
- `terms`: Dictionary mapping Pauli strings to their coefficients

#### `Dissipator`

Represents a dissipator (superoperator) in the Lindblad form of the master equation. Defined by a pair of Pauli operators and a dissipation rate coefficient.

**Key attributes:**
- `N`: Number of qubits
- `type`: Tuple of two Pauli strings
- `coeff`: Dissipation rate (complex number)

#### `CollectiveSpinOperator`

Represents a collective spin operator, which is a sum of identical single-qubit Pauli operators across all qubits.

**Key attributes:**
- `N`: Number of qubits
- `type`: Type of spin operator ("X", "Y", or "Z")
- `coeff`: Coefficient

### quantum_state

#### `QuantumState`

Represents the initial state of a quantum system for simulation. Can be defined as a product state in a specified basis, with optional state preparation operations.

**Key attributes:**
- `N`: Number of qubits
- `excitations`: Bitstring representation of the state
- `basis`: Basis in which the state is defined
- `state_preparation`: Optional list of unitary gates to apply
- `state_preparation_error`: Optional probability of bit-flip errors

---

## Simulation

### quantum_simulator

#### `QuantumSimulator`

Simulates quantum dynamics and measurements. Evolves quantum states according to a Hamiltonian and optional dissipators, and performs measurements according to specified settings.

**Key attributes:**
- `Nions`: Number of qubits/ions
- `hamiltonian`: QuantumOperator representing the system Hamiltonian
- `dissipators`: List of Dissipator objects for open system dynamics
- `data_set`: DataSet storing measurement results
- `rotating_frame`: Optional rotating frame transformation
- `measurement_error`: Optional measurement error model

**Key methods:**
- `simulate(initial_state, measurement_settings, save_data=False, no_output=True)`: Simulates a quantum experiment and returns measurement data as DataSet.
  
  **Parameters:**
  - `initial_state` (QuantumState): Initial state of the experiment.
  - `measurement_settings` (list): List of MeasurementSetting objects for the experiment.
  - `save_data` (bool, optional): If `True`, all data is also added to the DataSet of `self`. Default is `False`.
  - `no_output` (bool, optional): If `True`, no output is printed. Default is `True`.
  
  **Returns:** `DataSet` - Data set containing the measurement data

### measurement_setting

#### `MeasurementSetting`

Defines a single measurement configuration for a quantum experiment, specifying the initial state, simulation time, measurement basis, and number of shots.

**Key attributes:**
- `initial_state`: QuantumState for the measurement
- `simulation_time`: Time at which measurement is performed
- `measurement_basis`: Measurement basis string (e.g., "XYZ")
- `nshots`: Number of measurement shots
- `exact_observables`: Optional QuantumOperator for exact expectation values

---

## Data Management

### data_statistics

#### `DataEntry`

Stores measurement data for a single measurement setting, including measurement outcomes, exact expectation values, and associated quantum state information.

**Key attributes:**
- `Nions`: Number of qubits
- `initial_state`: QuantumState used
- `simulation_time`: Time of measurement
- `measurements`: Dictionary of measurement basis to bitstring outcomes
- `exact_expvals`: Dictionary of exact expectation values
- `final_state`: Optional final quantum state

#### `DataSet`

Manages collections of DataEntry objects, providing methods to organize, filter, and analyze measurement data across multiple initial states and simulation times.

**Key attributes:**
- `Nions`: Number of qubits
- `data`: Dictionary mapping (initial_state, simulation_time) tuples to DataEntry objects

---

## Learning Framework

### hamiltonian_learning

#### `Result`

Stores the results of Hamiltonian/Liouvillian learning algorithms, including learned operators, learning errors, and optimization results.

**Key attributes:**
- `operator_learned`: Learned Hamiltonian operator
- `dissipators_learned`: Learned dissipators
- `learning_error`: Error in the learning process
- `svd_vals`, `svd_vecs`: Singular value decomposition results
- `posterior`: Optional Bayesian posterior distribution
- `gamma_landscape_grid`, `gamma_landscape_vals`: Optimization landscape data

#### `Constraint`

Defines a single constraint for learning ansatz parameters. Each constraint corresponds to a row in the constraint matrix used for solving the learning problem.

**Key attributes:**
- `initial_state`: QuantumState for the constraint
- `simulation_times`: List of times at which constraint is evaluated
- `constraint_operator`: QuantumOperator to be measured
- `nshots_ratio_integrand`: Ratio of shots for intermediate times vs. endpoints

#### `Ansatz`

Defines the ansatz (parameterized form) for the Hamiltonian or Liouvillian to be learned. Manages the learning process, including constraint evaluation and parameter optimization.

**Key attributes:**
- `Nions`: Number of qubits
- `ansatz_operator`: QuantumOperator ansatz for the Hamiltonian
- `ansatz_dissipators`: List of Dissipator ansatz objects
- `parametrization`: Parametrization object(s) for the ansatz
- `data_set`: DataSet containing measurement data
- `constraints`: List of Constraint objects
- `constraint_tensors`: Precomputed constraint tensors
- `result`: Dictionary of Result objects from learning

**Key methods:**
- `get_constraint_tensors(constraints, method, nshots=-1, required_terms=None, evaluate_variance=False, Gaussian_noise=False, use_exact_initial_values=False, min_nshots_per_term=1, label=None, print_timers=False)`: Calculate constraint tensors for learning. Adds constraint tensor and vector to the ansatz object. If n_resampling is set, saves constraint_tensors_samples instead of constraint_tensors.
  
  **Parameters:**
  - `constraints` (list): List of Constraint objects for which to generate constraint tensors.
  - `method` (str): Method for which to generate the constraint tensor and vector. Options are `"generalized-energy-conservation"`, `"random-time-traces"`, or `"short-time-evolution"`.
  - `nshots` (int or list, optional): Number of shots used to estimate expectation values. For `short-time-evolution`, each expectation value is estimated using `nshots` number of measurements. For integration in `generalized-energy-conservation` and `random-time-traces`, the number of measurements at each time step equals `nshots/ntimes`. If `nshots=0`, exact expectation values are used. If `nshots=-1`, all available measurements are used. If `nshots` is a list, `get_constraint_tensors()` is called recursively for each value. Note: For `short-time-evolution` method, `nshots` is set to `0` at `t=0`. Default is `-1`.
  - `required_terms` (tuple, optional): Required terms to be measured for the given states and times. `required_terms` is a tuple of dicts of `(state,time):qop` pairs, where `qop` is the QuantumOperator to be measured at the given state and time. `required_terms[0]` is the required terms for the endpoints. `required_terms[1]` is the required terms for the integrand. If `None`, the required terms are determined automatically. Default is `None`.
  - `evaluate_variance` (bool, optional): If `True`, evaluates the variance of the constraint tensors. Default is `False`.
  - `Gaussian_noise` (bool, optional): If `True`, the exact expectation values with added Gaussian noise with variance `var/nshots` are used for the constraint tensors, instead of using the sampled expectation values. Default is `False`.
  - `use_exact_initial_values` (bool, optional): If `True`, expectation values at `t=0` are evaluated exactly. Default is `False`.
  - `min_nshots_per_term` (int, optional): Minimum number of shots required for estimating expectation values. If the number of shots is smaller than `min_nshots_per_term`, then the corresponding expectation value and variance are set to `np.nan`. Default is `1`.
  - `label` (str, optional): Label added to the key under which the constraint tensor and vector are saved. The key is set to `(label, method, nshots[nsinx])`. Default is `method`.
  - `print_timers` (bool, optional): If `True`, print timers for different steps in the function. Default is `False`.
  
  **Returns:** `None` (constraint tensors are stored in `ansatz.constraint_tensors` dictionary, keyed by `(label, method, nshots)`)

- `learn(learn_method, parametrizations=None, learn_label="learn", nshots=-1, scale_method=None, scale_label="scale", nshots_scale=None, scale_factor=1, diss_method=None, diss_label="diss", nshots_diss=None, normalize_constraints=False, MHexact=False, MDexact=False, MSexact=False, num_cpus=1)`: Learn the coefficients of the ansatz operator from constraint tensors and add the solution to `ansatz.result`. 
  
  **Parameters:**
  - `learn_method` (str): Choice of learning method. Options are `"generalized-energy-conservation"`, `"random-time-traces"`, or `"short-time-evolution"`.
  - `parametrizations` (list, optional): List of Parametrization objects used for learning. `None` is equivalent to a free parametrization.
  - `learn_label` (str, optional): Label of the constraint tensors used for learning. Default is `"learn"`.
  - `nshots` (int): Number of shots used to estimate each constraint tensor element. For `short-time-evolution`, each expectation value is estimated with `nshots` shots. For integration in `generalized-energy-conservation` and `random-time-traces`, the number of shots at each time step equals `nshots/ntimes`. Default is `-1`.
  - `scale_method` (str, optional): If set, the scale is reconstructed from the data set using the given method. Options are `"random-time-traces"` or `"short-time-evolution"`. Default is `None`.
  - `scale_label` (str, optional): Label of the constraint tensors used for scale reconstruction. Default is `"scale"`.
  - `nshots_scale` (int, optional): Number of measurement shots used for scale reconstruction. Default is `nshots`.
  - `scale_factor` (float): Factor by which the scale constraints are multiplied. Only used if `scale_method` is not `None`. Default is `1`.
  - `diss_method` (str, optional): If set, the dissipation is reconstructed from the data set using the given method. Options are `"random-time-traces"` or `"short-time-evolution"`. Default is `None`.
  - `diss_label` (str, optional): Label of the constraint tensors used for dissipation learning. Default is `"diss"`.
  - `nshots_diss` (int, optional): Number of measurement shots used for dissipation learning. Default is `nshots`.
  - `normalize_constraints` (bool, optional): If `True`, each row in (M,b) is normalized by the maximum norm of the row of M. Note: This option only applies to the constraint tensors used for learning, not to the scale reconstruction and the dissipation learning. Default is `False`.
  - `MHexact` (bool, optional): If `True`, the exact tensor is used for learning (no shot noise on MH). Only used for `generalized-energy-conservation` method. Default is `False`.
  - `MDexact` (bool, optional): If `True`, the dissipation correction is evaluated from exact expectation values (no shot noise on MD). Only used for `generalized-energy-conservation` method. Default is `False`.
  - `MSexact` (bool, optional): If `True`, the scale correction is evaluated from exact expectation values (no shot noise on Mscale). Only used for `generalized-energy-conservation` method. Default is `False`.
  - `num_cpus` (int): Number of CPUs used for parallelization. Default is `1`.
  
  **Returns:** `None` (results are stored in `ansatz.result` dictionary, keyed by `(learn_method, nshots)`)

### ansatz_parametrization

#### `ParametrizationFunction`

Defines how a selected subset of terms in the Hamiltonian or Liouvillian ansatz should be parametrized. Specifies criteria for operator selection and the type of parametrization (e.g., free, decay, variation, symmetry).

**Key attributes:**
- `coherent`: Boolean indicating if for Hamiltonian (True) or Liouvillian (False)
- `criterion`: Selection criterion for operators (e.g., "X", "ZZ")
- `type`: Type of parametrization (e.g., "free", "algebraic_decay")
- `cutoff`, `range`: Distance-based filtering parameters
- `subsystems`, `not_subsystems`: Spatial filtering parameters
- `parameters`: Nonlinear parameters for the parametrization

#### `Parametrization`

Collection of ParametrizationFunction objects that together define the complete parametrization (or regularization) of an ansatz Hamiltonian or Liouvillian.

**Key attributes:**
- `name`: Name of the parametrization
- `functions`: List of ParametrizationFunction objects
- `regularizations`: List of regularization functions
- `regularization_factor`: Strength of regularization
- `nonlinear_parameters_optimized`: Flag for optimization status
