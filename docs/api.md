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
