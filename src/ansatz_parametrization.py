"""
This module contains functions to parametrize an ansatz for Hamiltonian learning.
It contains the following classes:
    - ParametrizationFunction
        A class for parametrizing a selected subset terms in the Hamiltonian and Liouvillian ansatz.
    - Parametrization
        A class containing multiple different parametrization functions.

"""
from __future__ import annotations
import numpy as np
import scipy as sp
from src.pauli_algebra import PauliOperator, QuantumOperator, Dissipator
import src.utility_functions as uf

class ParametrizationFunction:
    """ 
    ParametrizationFunction for defining a Parametrization.

    A ParametrizationFunction defines one or more columns of the parametrization matrix G,
    used for parametrizing the ansatz Hamiltonian (or Liouvillian).
    The parametrization matrix G is used to transform the constraint matrix M
    according to M_new = G M.

    Attributes
    ----------
    coherent : bool
        If True, the parametrization function is for coherent terms (Hamiltonian).
        If False, the parametrization function is for dissipative terms (Liouvillian).
        Default is True.
    criterion : list of str
        Criterion for which operators to choose (e.g. "X", "ZZ").
        If coherent==True, criterion is a string or list of strings made from Pauli chars ("X", "Y", "Z") or None.
        An operator is chosen if the type of the operator is equal to the criterion,
        or if the type of the operator contains the same number of each chars ("X","Y","Z") as the criterion.
        If criterion is None, no operators are chosen (Tikhonov regularization).
        For example the criterion "XX" includes "IXX", "XX", "XIX" but does not include "X", "XXZ", "XIXX".
        If coherent==False, criterion is a tuple of strings or a list of tuples of strings.
        A dissipator is chosen if dissipator[0] fulfills the criterion[0] and dissipator[1] fulfills the criterion[1].
        Default is None.
    type : string
        Type of parametrization (e.g. "free", "*_decay", "*_variation", "*_symmetry").
        If type == "free", G becomes the identity whereever the criterion is fulfilled, and 0 otherwise.
        If type == "*_decay", entries of a column of G decay with distance, 
        where * can be "linear", "algebraic" or "exponential".
        If type == "*_variation", entries of a column of G vary along the axial coordinate ("positions"),
        where * can be any of "algebraicX" for X=1,2,3,4,5,6,7,8,9 (X is the order of the polynomial).
        If type == "*_symmetry", entries of a column of G are set to 1 whereever the criterion is fulfilled, and 0 otherwise,
        where * can only be "translational".
        For example algebraic2_variation means 2nd-order polynomial,
        and algebraic_decay means 1/distance decay.
        Default is None
    operator : QuantumOperator, optional
        Quantum operator used if param_type=="operator".
        Default is None.
    dissipators : list of Dissipator objects, optional
        List of Dissipator objects used if param_type=="dissipator".
        Default is None.
    cutoff : int, optional
        Set cutoff for the parametrization (e.g. 3).
        If cutoff is 0, no cutoff is applied.
        If cutoff is positive, coefficients are set to 0 for distances larger than cutoff.
        If cutoff is negative, coefficients are set to 0 for distances smaller than -cutoff.
        Default is None.
    range : non-negative int, optional
        Set range for the parametrization (e.g. 3).
        If range is not None, coefficients are set to 0 for distances not equal to range.
        Default is None.
    subsystems : list of lists of ints, optional
        List of subsystems for which the parametrization is applied.
        Each subsystem is a list of integers that represent the qubits of the subsystem.
        A PauliOperator is chosen if its support lies in any of the subsystems.
        Default is None.
    not_subsystems : list of lists of ints, optional
        List of subsystems for which the parametrization is applied.
        Each subsystem is a list of integers that represent the qubits of the subsystem.
        A PauliOperator is chosen if its support does not lie in any of the subsystems.
        Default is None.
    parameters : list, optional
        nonlinear parameters for the parametrization function (e.g. algebraic decay exponent)
        Default is None.
    exact_parameters : list, optional
        exact nonlinear parameters for the parametrization function (e.g. algebraic decay exponent)
        Default is None.
    bounds : list of tuples, optional
        bounds for the parameters of the parametrization (e.g. [(0, 1), (0, 2), (0, 3)])
        Default is None.
    """
    def __init__(self, 
                coherent: bool = True, 
                criterion: str | list[str] | None = None, 
                param_type: str | None = None, 
                operator: QuantumOperator | None = None, 
                dissipators: list[Dissipator] | None = None, 
                cutoff: int | None = None, 
                range: int | None = None, 
                subsystems: list[list[int]] | None = None, 
                not_subsystems: list[list[int]] | None = None, 
                parameters: list | None = None, 
                exact_parameters: list | None = None, 
                bounds: list[tuple] | None = None
                ):
        """
        Initialize a Parametrization function.

        Parameters
        ----------
        coherent : bool
            If True, the parametrization function is for coherent terms (Hamiltonian).
            If False, the parametrization function is for dissipative terms (Liouvillian).
            Default is True.
        criterion : list of str
            Criterion for which operators to choose (e.g. "X", "ZZ").
            If coherent==True, criterion is a string or list of strings made from Pauli chars ("X", "Y", "Z") or None.
            An operator is chosen if the type of the operator is equal to the criterion,
            or if the type of the operator contains the same number of each chars ("X","Y","Z") as the criterion.
            If criterion is None, no operators are chosen (Tikhonov regularization).
            For example the criterion "XX" includes "IXX", "XX", "XIX" but does not include "X", "XXZ", "XIXX".
            If coherent==False, criterion is a tuple of strings or a list of tuples of strings.
            A dissipator is chosen if dissipator[0] fulfills the criterion[0] and dissipator[1] fulfills the criterion[1].
            Default is None.
        param_        param_type : string
            Type of parametrization (e.g. "free", "*_decay", "*_variation", "*_symmetry").
            If param_type == "free", G becomes the identity wherever the criterion is fulfilled, and 0 otherwise.
            If param_type == "*_decay", entries of a column of G decay with distance, 
            where * can be "linear", "algebraic" or "exponential".
            If param_type == "*_variation", entries of a column of G vary along the axial coordinate ("positions"),
            where * can be any of "algebraicX" for X=1,2,3,4,5,6,7,8,9 (X is the order of the polynomial).
            If param_type == "*_symmetry", entries of a column of G are set to 1 wherever the criterion is fulfilled, and 0 otherwise,
            where * can only be "translational".
            For example algebraic2_variation means 2nd-order polynomial,
            and algebraic_decay means 1/distance decay.
            Default is None
        operator : QuantumOperator, optional
            Quantum operator used if param_type=="operator".
            Default is None.
        dissipators : list of Dissipator objects, optional
            List of Dissipator objects used if param_type=="dissipator".
            Default is None.
        cutoff : int, optional
            Set cutoff for the parametrization (e.g. 3).
            If cutoff is 0, no cutoff is applied.
            If cutoff is positive, coefficients are set to 0 for distances larger than cutoff.
            If cutoff is negative, coefficients are set to 0 for distances smaller than -cutoff.
            Default is None.
        range : non-negative int, optional
            Set range for the parametrization (e.g. 3).
            If range is not None, coefficients are set to 0 for distances not equal to range.
            Default is None.
        subsystems : list of lists of ints, optional
            List of subsystems for which the parametrization is applied.
            Each subsystem is a list of integers that represent the qubits of the subsystem.
            A PauliOperator is chosen if its support lies in any of the subsystems.
            Default is None.
        not_subsystems : list of lists of ints, optional
            List of subsystems for which the parametrization is applied.
            Each subsystem is a list of integers that represent the qubits of the subsystem.
            A PauliOperator is chosen if its support does not lie in any of the subsystems.
            Default is None.
        parameters : list, optional
            nonlinear parameters for the parametrization function (e.g. algebraic decay exponent)
            Default is None.
        exact_parameters : list, optional
            exact nonlinear parameters for the parametrization function (e.g. algebraic decay exponent)
            Default is None.
        bounds : list of tuples, optional
            bounds for the parameters of the parametrization (e.g. [(0, 1), (0, 2), (0, 3)])
            Default is None.
        """
        self.coherent = coherent
        self.criterion = criterion
        self.param_type = param_type
        self.operator = operator
        self.dissipators = dissipators
        self.cutoff = cutoff
        self.range = range
        self.subsystems = subsystems
        self.not_subsystems = not_subsystems
        self.parameters = parameters
        self.exact_parameters = exact_parameters
        self.bounds = bounds
    ### ----------------------------------- ###
    ### ParametrizationFunction attributes ###
    ### ----------------------------------- ###
    @property
    def coherent(self):
        return self._coherent
    @coherent.setter
    def coherent(self, value):
        # check if value is a boolean
        if not isinstance(value, bool):
            raise TypeError("coherent must be a boolean")
        self._coherent = value
    @property
    def criterion(self):
        return self._criterion
    @criterion.setter
    def criterion(self, value):
        if isinstance(value, str):
            if not all([x in ["X", "Y", "Z", "M", "P", "I"] for x in value]):
                raise ValueError("criterion is {}, but must be an uppercase string of X, Y, Z, M, P or I.".format(value))
        elif isinstance(value, list):
            if not all([isinstance(x, str) for x in value]):
                raise TypeError("criterion is {}, but must be a list of strings, but is".format(type(value)))
            if not all([all([y in ["X", "Y", "Z", "M", "P", "I"] for y in x]) for x in value]):
                raise ValueError("criterion is {}, but must be a list of strings of X, Y, Z, M, P or I.".format(value))
        elif value is not None:
            raise TypeError("criterion is {}, but must be a string or a list of strings".format(type(value)))
        if self.coherent and isinstance(value, str):
            value = [value]
        if not self.coherent and isinstance(value, list) and all(isinstance(item, str) for item in value):
            value = [value]
        self._criterion = value
    @property
    def param_type(self):
        return self._param_type
    @param_type.setter
    def param_type(self, value):
        # None is allowed (sets all coefficients to 0)
        if value is None:
            self._param_type = None
            return
        # check if value is a string
        elif not isinstance(value, str):
            raise TypeError("type must be a string")
        # check if value is is among the allowed values
        allowed_values = ["operator", "dissipators", "free", "translational_symmetry", "linear_decay", "algebraic_decay", "exponential_decay", "free_variation", "algebraic", "cos", "sin"]
        if not any([value.startswith(x) for x in allowed_values]):
            raise ValueError("param_type = {}, but must start with one of the allowed values: {}".format(value, allowed_values))
        self._param_type = value
    @property
    def operator(self):
        return self._operator
    @operator.setter
    def operator(self, value):
        # check if value is a QuantumOperator object
        if value is not None:
            if not isinstance(value, QuantumOperator):
                raise TypeError("operator must be a QuantumOperator object")
        self._operator = value
    @property
    def dissipators(self):
        return self._dissipators
    @dissipators.setter
    def dissipators(self, value):
        # check if value is a list of Dissipator objects
        if value is not None:
            if not isinstance(value, list):
                raise TypeError("dissipators must be a list")
            for x in value:
                if not isinstance(x, Dissipator):
                    raise TypeError("dissipators must be a list of Dissipator objects")
        self._dissipators = value
    @property
    def cutoff(self):
        return self._cutoff
    @cutoff.setter
    def cutoff(self, value):
        # check if value is a positive integer
        if value is not None:
            if not isinstance(value, int):
                raise TypeError("cutoff must be an integer")
        self._cutoff = value
    @property
    def range(self):
        return self._range
    @range.setter
    def range(self, value):
        # check if value is a non-negative integer
        if value is not None:
            if not isinstance(value, int):
                raise TypeError("range must be an integer")
            if value < 0:
                raise ValueError("range must be non-negative")
        self._range = value
    @property
    def subsystems(self):
        return self._subsystems
    @subsystems.setter
    def subsystems(self, value):
        # check if value is a list of lists of integers
        if value is not None:
            if not isinstance(value, list):
                raise TypeError("subsystems must be a list")
            for x in value:
                if not isinstance(x, list):
                    raise TypeError("subsystems must be a list of lists")
                if not all([isinstance(y, int) for y in x]):
                    raise TypeError("subsystems must be a list of lists of integers")
        self._subsystems = value
    @property
    def not_subsystems(self):
        return self._not_subsystems
    @not_subsystems.setter
    def not_subsystems(self, value):
        # check if value is a list of lists of integers
        if value is not None:
            if not isinstance(value, list):
                raise TypeError("not_subsystems must be a list")
            for x in value:
                if not isinstance(x, list):
                    raise TypeError("not_subsystems must be a list of lists")
                if not all([isinstance(y, int) for y in x]):
                    raise TypeError("not_subsystems must be a list of lists of integers")
        self._not_subsystems = value
    # TODO: parameters, exact_parameters, bounds
    ### ---------------------------- ###
    ### Parametrize_function methods ###
    ### ---------------------------- ###
    def copy(self):
        """
        Return a copy of the ParametrizationFunction.
        """
        return ParametrizationFunction(coherent=self.coherent, criterion=self.criterion, param_type=self.param_type, operator=self.operator, dissipators=self.dissipators, cutoff=self.cutoff, range=self.range, subsystems=self.subsystems, not_subsystems=self.not_subsystems, parameters=self.parameters, exact_parameters=self.exact_parameters, bounds=self.bounds)

    def __eq__(self, 
            other: ParametrizationFunction
            ) -> bool:
        """
        Check if two ParametrizationFunction objects are equal.

        Two Parametrization functions are equal if they have the same attributes.

        Parameters
        ----------
        other : ParametrizationFunction
            Parametriztion function to compare to

        Returns
        -------
        bool
            whether self and other are equal.
        """
        for attr in ["coherent", "criterion", "type", "operator", "dissipators", "cutoff", "range", "parameters", "exact_parameters", "bounds", "subsystems", "not_subsystems"]:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True
    
    def __ne__(self, 
            other: ParametrizationFunction
            ) -> bool:
        """
        Check if two ParametrizationFunction objects are not equal.

        Parameters
        ----------
        other : ParametrizationFunction
            Parametrization function to compare to

        Returns
        -------
        bool
            whether self and other are not equal.
        """
        return not self.__eq__(other)
    
    def __str__(self):
        return self.str()
    def str(self) -> str:
        """
        Return the string representation of the parametrization function.

        Returns
        -------
        str
            string representation of the parametrization function.
        """
        criterion_str = self.criterion
        param_type_str = self.param_type
        if self.operator is not None:
            param_type_str = "operator"
        if self.dissipators is not None:
            param_type_str = "dissipators"
        if self.cutoff is not None:
            cutoff_str = "cutoff={}".format(self.cutoff)
        else:
            cutoff_str = ""
        if self.range is not None:
            range_str = "range={}".format(self.range)
        else:
            range_str = ""
        if self.parameters is not None:
            parameters_str = "parameters={}".format(self.parameters)
        else:
            parameters_str = ""
        if self.exact_parameters is not None:
            exact_parameters_str = "exact_parameters={}".format(self.exact_parameters)
        else:
            exact_parameters_str = ""
        if self.bounds is not None:
            bounds_str = "bounds={}".format(self.bounds)
        else:
            bounds_str = ""
        if self.subsystems is not None:
            subsystems_str = "subsystems={}".format(self.subsystems)
        else:
            subsystems_str = ""
        if self.not_subsystems is not None:
            not_subsystems_str = "not_subsystems={}".format(self.not_subsystems)
        else:
            not_subsystems_str = ""
        ## get combined string
        msetting_str = "criterion={}, param_type={}, {}, {}, {}, {}, {}, {}, {}".format(criterion_str,param_type_str,cutoff_str,range_str,parameters_str,exact_parameters_str,bounds_str,subsystems_str,not_subsystems_str) 
        return msetting_str

    def __call__(self, 
                operator: PauliOperator | Dissipator
                ) -> float:
        """
        Call the ParametrizationFunction on an operator.

        If the criterion is fulfilled, returns coefficient according to the type of the ParametrizationFunction.
        If the criterion is not fulfilled, returns coefficient ``0``.

        Parameters
        ----------
        operator : PauliOperator or Dissipator
            Operator that is parametrized.
        
        Returns
        -------
        coeff : float
            Coefficient of the ParametrizationFunction.
        """
        coeff = 0
        # check if criterion is fulfilled for operator
        if self.check_criterion(operator) == True:
            # create coefficients according to type
            coeff = self.get_coefficient(operator)
        return coeff

    def check_criterion(self, 
                        operator: PauliOperator | Dissipator, 
                        ) -> bool:
        """
        Check if the criterion of the ParametrizationFunction is fulfilled for an operator.

        Returns False if criterion is None (Tikhonov regularization).
        Returns True if either the type of the operator is equal to the any criterion,
        or if the type of the operator contains the same number of each char ("X","Y","Z") as any criterion.

        Parameters
        ----------
        operator : PauliOperator or Dissipator
            Operator for which the criterion is checked.

        Returns
        -------
            bool
                True if criterion is fulfilled, False otherwise.
        """
        if self.criterion is None:
            return False
        #-------------------------------------
        ### STEP 1 ### coherent ParametrizationFunction
        if isinstance(operator, PauliOperator):
            # check if criterion is for operator
            if self.param_type == "operator":
                if operator.pauli_type in self.operator.terms:
                    return True
            # check if coherent
            if not self.coherent:
                return False
            # check list of criteria
            for criterion in self.criterion:
                # check if criterion is fulfilled
                if criterion == operator.pauli_type:
                    return True
                if "I" not in criterion:
                    # get number of each char in self.criterion
                    criterion_counts = [criterion.count(x) for x in ["X", "Y", "Z", "M", "P"]]
                    # get number of each char in operator.pauli_type
                    op_counts = [operator.pauli_type.count(x) for x in ["X", "Y", "Z", "M", "P"]]
                    # check if criterion is fulfilled
                    if criterion_counts == op_counts:
                        return True
        #-------------------------------------
        ### STEP 2 ### dissipative ParametrizationFunction
        if isinstance(operator, Dissipator):
            # check if dissipator is given
            if self.param_type == "dissipators":
                diss_types = [diss.diss_type for diss in self.dissipators]
                if operator.diss_type in diss_types:
                    return True
            # check if coherent
            if self.coherent:
                return False
            # check list of criteria
            for criterion in self.criterion:
                # check if criterion is fulfilled
                if criterion == operator.diss_type:
                    return True
                if "I" not in criterion[0] and "I" not in criterion[1]:
                    # get number of each char in self.criterion
                    criterion_counts1 = [criterion[0].count(x) for x in ["X", "Y", "Z", "M", "P"]]
                    criterion_counts2 = [criterion[1].count(x) for x in ["X", "Y", "Z", "M", "P"]]
                    # get number of each char in operator.diss_type
                    op_counts1 = [operator.diss_type[0].count(x) for x in ["X", "Y", "Z", "M", "P"]]
                    op_counts2 = [operator.diss_type[1].count(x) for x in ["X", "Y", "Z", "M", "P"]]
                    # check if criterion is fulfilled
                    if criterion_counts1 == op_counts1 and criterion_counts2 == op_counts2:
                        return True
        #-------------------------------------
        # return False if no criterion is fulfilled
        return False

    def check_cutoff(self, 
                    operator: PauliOperator | Dissipator
                    ) -> bool:
        """
        Check if operator is beyond the cutoff of the ParametrizationFunction.

        Returns True if operator is within cutoff, False otherwise.
        Cutoff combines range, cutoff, subsystems and not_subsystems attributes of the ParametrizationFunction.
        Range can be any non-negative integer.
        If range is not None, coefficient of operator is set to 0 if its range is not equal to range.
        Cutoff can be any integer.
        If cutoff is non-negative, coefficient of operator is set to 0 if its range is larger than cutoff.
        If cutoff is negative, coefficient of operator is set to 0 if its range is smaller than -cutoff.
        Subsystems can be a list of lists of integers.
        If subsystems is not None, coefficient of operator is set to 0 its support is not within any of the subsystems.
        Not_subsystems can be a list of lists of integers.
        If not_subsystems is not None, coefficient of operator is set to 0 its support is within any of the not_subsystems.

        Parameters
        ----------
        operator : PauliOperator or Dissipator
            Operator for which the cutoff is checked.

        Returns
        -------
        bool
            True if operator is within cutoff, False otherwise.
        """
        ### STEP 0 ### check range
        if self.range is not None:
            if operator.range() != self.range:
                return False
        #---------------------
        ### STEP 1 ### check cutoff
        if self.cutoff is not None:
            if self.cutoff >= 0:
                if operator.range() > self.cutoff:
                    return False
            if self.cutoff < 0:
                if operator.range() < -self.cutoff:
                    return False
        #---------------------
        ### STEP 2 ### check subsystems
        if self.subsystems is not None:
            if not any([set(operator.support()).issubset(set(subsystem)) for subsystem in self.subsystems]):
                return False
        #---------------------
        ### STEP 3 ### check complementary subsystems
        if self.not_subsystems is not None:
            if any([set(operator.support()).issubset(set(subsystem)) for subsystem in self.not_subsystems]):
                return False
        #---------------------
        return True

    def get_coefficient(self, 
                        operator: PauliOperator | Dissipator
                        ) -> float:
        """
        Returns coefficient for a given operator according to the type of the ParametrizationFunction.

        Parameters
        ----------
        operator : PauliOperator or Dissipator
            Operator for which the coefficient is calculated.

        Returns
        -------
        coeff : float
            Coefficient for the parametrized operator.
        """
        ### STEP 1 ### check if PauliOperator is beyond cutoff
        if not self.check_cutoff(operator):
            return 0
        #---------------------
        ### STEP 2 ### create coefficients according to type
        coeff = None
        if self.param_type == "operator":
            coeff = 0
            if operator.pauli_type in self.operator.terms:
                coeff = self.operator.terms[operator.pauli_type].coeff
        elif self.param_type == "dissipators":   
            diss_types = [diss.diss_type for diss in self.dissipators]
            coeff = 0
            if operator.diss_type in diss_types:
                coeff = self.dissipators[diss_types.index(operator.diss_type)].coeff
        elif self.param_type == "free":
            coeff = 1
        elif self.param_type.endswith("_variation"):
            # get center of operator
            center = operator.center()
            # create coefficient
            if self.param_type.startswith("algebraic"):
                order = int(self.param_type[9])
                coeff = center**order
            else:
                raise ValueError("param_type={} not recognized".format(self.param_type))
        elif self.param_type.endswith("_decay"):
            # get range of operator (maximum distance between acting terms)
            distance = operator.range()
            # create coefficient
            if self.param_type == "algebraic_decay":
                coeff = 1/distance**self.parameters[0]
            elif self.param_type == "exponential_decay":
                coeff = np.exp(-self.parameters[0]*distance)
            elif self.param_type == "linear_decay":
                coeff = distance
            if not self.check_cutoff(operator):
                coeff = 0
        elif self.param_type.endswith("_symmetry"):
            if self.param_type == "translational_symmetry":
                coeff = 1
        ### check if coeff is associated with a value
        if coeff is None:
            raise ValueError("param_type={} not recognized".format(self.param_type))
        #---------------------
        return coeff



class Parametrization:
    """
    Parametrization for a given ansatz operator.

    A Parametrization is a set of ParametrizationFunction objects,
    that define the parametrization (or regularization) of the ansatz Hamiltonian (or Liouvillian).

    Attributes
    ----------
    name : string
        Name of the parametrization.
    functions : list of ParametrizationFunction objects, optional
        ParametrizationFunctions that define the Parametrization.
        Default is None.
    regularizations : list of ParametrizationFunction objects, optional
        ParametrizationFunction objects that define the regularization.
        Default is None
    regularization_factor : float, optional
        The regularization_factor governs the strength of the regularization and is a positive float,
        that is multiplied with the regularization matrix.
        Default is None.
    nonlinear_parameters_optimized : bool, optional
        Indicates whether the nonlinear parameters of the parametrization have already been optimized.
        Default is None.
    """
    def __init__(self, 
                name: str | None = None, 
                functions: list[ParametrizationFunction] | None = None, 
                regularizations: list[ParametrizationFunction] | None = None, 
                regularization_factor: float | None = None, 
                nonlinear_parameters_optimized: bool | None = None
                ) -> None:
        """
        Initialize a Parametrization.

        Parameters
        ----------
        name : string
            Name of the parametrization.
        functions : list of ParametrizationFunctions, optional
            ParametrizationFunctions that define the Parametrization.
            Default is None.
        regularizations : list of ParametrizationFunctions, optional
            ParametrizationFunctions that define the regularization.
            Default is None
        regularization_factor : float, optional
            The regularization_factor governs the strength of the regularization and is a positive float,
            that is multiplied with the regularization matrix.
            Default is None.
        nonlinear_parameters_optimized : bool, optional
            Indicates whether the nonlinear parameters of the parametrization have already been optimized.
            Default is None.
        """
        self.name = name
        self.functions = functions
        self.regularizations = regularizations
        self.regularization_factor = regularization_factor
        self.nonlinear_parameters_optimized = nonlinear_parameters_optimized
    ### -------------------------- ###
    ### Parametrization attributes ###
    ### -------------------------- ###
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        # check if value is a string
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("name must be a string")
        self._name = value
    @property
    def functions(self):
        return self._functions
    @functions.setter
    def functions(self, value):
        if value is not None:
            # check if value is a list
            if type(value) != list:
                raise TypeError("functions must be a list")
            # check if value is a list of ParametrizationFunction objects
            for x in value:
                if not isinstance(x, ParametrizationFunction):
                    raise TypeError("functions must be a list of ParametrizationFunction objects")
        self._functions = value
    @property
    def regularizations(self):
        return self._regularizations
    @regularizations.setter
    def regularizations(self, value):
        if value is not None:
            # check if value is a list
            if not isinstance(value, list):
                raise TypeError("regularizations must be a list")
            # check if value is a list of ParametrizationFunction objects
            for x in value:
                if not isinstance(x, ParametrizationFunction):
                    raise TypeError("regularizations must be a list of ParametrizationFunction objects")
        self._regularizations = value
    @property
    def regularization_factor(self):
        return self._regularization_factor
    @regularization_factor.setter
    def regularization_factor(self, value):
        # check if value is a non-negative float
        if value is not None:
            if not np.isreal(value):
                raise ValueError("regularization_factor is {}, but must be real".format(value))
            if value < 0:
                raise ValueError("regularization_factor is {}, but must be non-negative".format(value))
        self._regularization_factor = value
    ### ----------------------- ###
    ### Parametrization methods ###
    ### ----------------------- ###
    def __eq__(self, 
            other: Parametrization,
            ) -> bool:
        """
        Check if two Parametrization objects are equal.

        Compares each ParametrizationFunction object in the list of functions
        using the __eq__ method of the ParametrizationFunction class.
        Also compares the list of regularizations.
        Two parametrizations functions are equal if they have the same attributes.

        Parameters
        ----------
        other : Parametrization
            parametrization to compare to

        Returns
        -------
        bool
            whether the parametrizations are equal
        """
        ## check parametrization functions
        if self.functions is not None:
            for fct1, fct2 in zip(self.functions, other.functions):
                if fct1 != fct2:
                    return False
        #---------------------
        ## check regularization functions
        if self.regularizations is not None:
            for reg1, reg2 in zip(self.regularizations, other.regularizations):
                if reg1 != reg2:
                    return False
        #---------------------
        ## check regularization factor
        if self.regularization_factor != other.regularization_factor:
            return False
        #---------------------
        return True
    
    def __ne__(self, 
            other : Parametrization,
            ) -> bool:
        """ 
        Check if two Parametrization objects are not equal.

        Parameters
        ----------
        other : Parametrization
            parametrization to compare to

        Returns
        -------
        bool
            whether the parametrizations are not equal
        """
        return not self.__eq__(other)
    
    def __str__(self):
        return self.str()
    def str(self, 
            verbose: bool = False, 
            name_only: bool = False,
            ) -> str:
        """
        Return the string representation of the Parametrization.

        Parameters
        ----------
        verbose : bool, optional
            if True, returns a verbose string
            Default is False.
        name_only : bool, optional
            if True, only returns the name of the Parametrization
            Default is False.

        Returns
        -------
        str
            string representation of the Parametrization
        """
        #---------------------
        ### STEP 1 ### get functions and regularizations
        functions = self.functions
        nfunctions = 0
        regularizations = self.regularizations
        nregularizations = 0
        if functions is not None:
            functions = [fct.str() for fct in functions]
            nfunctions = len(functions)
        if regularizations is not None:
            regularizations = [reg.str() for reg in regularizations]
            nregularizations = len(regularizations)
        #---------------------
        ### STEP 2 ### get nonlinear parameters
        nonlin_pars = self.get_nonlinear_parameters()
        n_nonlin_pars = 0
        if nonlin_pars is not None:
            n_nonlin_pars = len(nonlin_pars)
        ### STEP 3 ### get combined string
        if verbose:
            parstr = "par:{} [nfct={},nreg={},n_nonlin={}], parametrization functions: {}, regularizations: {}".format(self.name, nfunctions, nregularizations, n_nonlin_pars, functions, regularizations)
        elif name_only:
            parstr = self.name
        else:
            parstr = "par:{} [nfct={},nreg={},n_nonlin={}]".format(self.name, nfunctions, nregularizations, n_nonlin_pars)
        #---------------------
        return parstr

    def copy(self) -> Parametrization:
        """
        Return a copy of the Parametrization.

        Returns
        -------
        Parametrization
            A new instance with the same attributes as the parametrization.
        """
        return Parametrization(name=self.name, functions=self.functions, regularizations=self.regularizations, regularization_factor=self.regularization_factor, nonlinear_parameters_optimized=self.nonlinear_parameters_optimized)

    def add_functions(self, 
                    functions: list[ParametrizationFunction], 
                    par_type: str = "parametrization",
                    ) -> None:
        """
        Add parametrization functions or regularizations to the Parametrization.

        Parameters
        ----------
        functions : list of ParametrizationFunction objects
            List of parametrization functions that should be added to the Parametrization.
        par_type : str, optional
            Type of the parametrization function.
            If "parametrization", the function is added to the parametrization.
            If "regularization", the function is added to the regularization.
            Default is ``"parametrization"``.
        """
        #---------------------
        ### STEP 1 ### check input
        # check if functions is a list
        if not isinstance(functions, list):
            raise TypeError("functions must be a list")
        # check if functions is a list of ParametrizationFunction objects
        for x in functions:
            if not isinstance(x, ParametrizationFunction):
                raise TypeError("functions must be a list of ParametrizationFunction objects")
        #---------------------
        ### STEP 2 ### add functions
        if par_type == "parametrization":
            if self.functions is None:
                self.functions = functions
            else:
                self.functions.extend(functions)
        elif par_type == "regularization":
            if self.regularizations is None:
                self.regularizations = functions
            else:
                self.regularizations.extend(functions)
        else:
            raise ValueError("par_type = {}, but must be either 'parametrization' or 'regularization'".format(par_type))

    def add_functions_for_free_terms(self, 
                                    operators: list[PauliOperator] | list[Dissipator] | QuantumOperator, 
                                    par_type: str = "parametrization", 
                                    symmetry: dict | None = None, 
                                    cutoff: int | None = None, 
                                    range: int | None = None
                                    ) -> None:
        """
        Adds operators to Parametrization.

        Adds a ParametrizationFunction of type "free" to the parametrization or regularization,
        for each term in given operators that is not in any other ParametrizationFunction yet.

        Parameters
        ----------
        operators : list of PauliOperators or Dissipators or single QuantumOperator
            List of operators for which the ParametrizationFunctions should be added.
            If a QuantumOperator is given, the parametrization functions are added for all terms in the QuantumOperator.
        par_type : str, optional
            Type of the parametrization function.
            If "parametrization", the function is added to the parametrization.
            If "regularization", the function is added to the regularization.
            Default is ``"parametrization"``.
        symmetry : dict, optional
            Dictionary of replacements for str.translate.
            If given, the criterion of the ParametrizationFunctions
            will be a list of all terms that are equivalent under the given symmetry.
            For example, for a given term "XY", a symmetry of {"X":"Y"}
            creates a parametrization function with criterion ["XY","YX"].
            Default is None.
        cutoff : int, optional
            If given, the parametrization functions will be added only for terms
            that are within the given cutoff distance.
            Default is None.
        range : int
            If given, the parametrization functions will be added only for terms
            that have the given range.
            Default is None.
        """
        # check if operators is a QuantumOperator object
        if isinstance(operators, QuantumOperator):
            operators = list(operators.terms.values())
        ### add terms for each operator
        for op in operators:
            ## check if op is a PauliOperator or a Dissipator
            if isinstance(op, PauliOperator):
                coherent = True
            elif isinstance(op, Dissipator):
                coherent = False
            else:
                raise TypeError("operator must be a PauliOperator or a Dissipator")
            ## ignore terms if they are too far apart (cutoff)
            if cutoff is not None or range is not None:
                if coherent:
                    tmp_parfct = ParametrizationFunction(criterion=op.pauli_type, param_type="free", cutoff=cutoff, range=range, coherent=coherent)
                else:
                    tmp_parfct = ParametrizationFunction(criterion=op.diss_type, param_type="free", cutoff=cutoff, range=range, coherent=coherent)
                if not tmp_parfct.check_cutoff(op):
                    continue
            ## check if term is already in the Parametrization
            found = False
            if par_type == "parametrization" and self.functions is not None:
                for parfct in self.functions:
                    if parfct.check_criterion(op) == True:
                        if parfct.check_cutoff(op) == True:
                            found = True
                            break
            if par_type == "regularization" and self.regularizations is not None:
                for reg in self.regularizations:
                    if reg.check_criterion(op) == True:
                        if reg.check_cutoff(op) == True:
                            found = True
                            break
            ## add parametrization function for missing operator
            if not found:
                ## get criterion
                if isinstance(op, PauliOperator):
                    criterion = op.pauli_type
                elif isinstance(op, Dissipator):
                    criterion = op.diss_type
                else:
                    raise TypeError("operator must be a PauliOperator or a Dissipator")
                ## apply symmetry
                if symmetry is not None:
                    trans = str.maketrans(symmetry)
                    # coherent terms
                    if isinstance(op, PauliOperator):
                        criterion = [op.pauli_type]
                        if op.pauli_type.translate(trans)!=op.pauli_type:
                            criterion.append(op.pauli_type.translate(trans))
                    # dissipative terms
                    if isinstance(op, Dissipator):
                        raise NotImplementedError("Symmetry translation for Dissipators is not implemented yet.")  
                ## add parametrization function 
                if par_type == "parametrization":
                    if self.functions is None:
                        self.functions = [ParametrizationFunction(criterion=criterion, param_type="free", coherent=coherent)]
                    else:
                        self.functions.append(ParametrizationFunction(criterion=criterion, param_type="free", coherent=coherent))       
                elif par_type == "regularization":
                    if self.regularizations is None:
                        self.regularizations = [ParametrizationFunction(criterion=criterion, param_type="free", coherent=coherent)]
                    else:
                        self.regularizations.append(ParametrizationFunction(criterion=criterion, param_type="free", coherent=coherent))
                else:
                    raise ValueError("par_type = {}, but must be either 'parametrization' or 'regularization'".format(par_type)) 

    def fix_parameters(self, 
                    fixed_hamiltonian: QuantumOperator | None = None, 
                    fixed_dissipators: list[Dissipator] | None = None,
                    ) -> None:
        """
        Fix parameters of the Parametrization.

        Fix parameters of the parametrization to given QuantumOperator and/or Dissipators.
        If fixed_Hamiltonian is given, all coherent functions are replaced by a single ParametrizationFunction
        of type "operator" and with operator = fixed_Hamiltonian.
        If fixed_dissipators is given, all dissipative functions are replaced by a single ParametrizationFunction
        of type "dissipators" and with dissipators = fixed_dissipators. [not implemented yet]

        Parameters
        ----------
        fixed_hamiltonian : QuantumOperator, optional
            Hamiltonian to which the parametrization should be fixed.
            Default is None.
        fixed_dissipators : list of Dissipators, optional
            Dissipators to which the parametrization should be fixed.
            Default is None.
        """
        ### set fixed Hamiltonian
        if fixed_hamiltonian is not None:
            # delete all coherent functions
            fcts_new = []
            for fct in self.functions:
                if not fct.coherent:
                    fcts_new.append(fct)
            # add fixed Hamiltonian
            Hfct = ParametrizationFunction(param_type="operator", coherent=True, operator=fixed_hamiltonian)
            fcts_new.append(Hfct)
            self.functions = fcts_new
        ### set fixed dissipators
        if fixed_dissipators is not None:
            raise NotImplementedError("fix_parameters for dissipators is not implemented yet")

    def get_nonlinear_parameters(self) -> list:
        """
        Return list of the nonlinear parameters of the parametrization.

        Returns
        -------
        list
            list of the nonlinear parameters of the Parametrization
        """
        parameters = []
        # parametrization functions
        if self.functions is not None:
            for fct in self.functions:
                if fct.parameters is not None:
                    parameters.extend(fct.parameters)
        # regularization functions
        if self.regularizations is not None:
            for reg in self.regularizations:
                if reg.parameters is not None:
                    parameters.extend(reg.parameters)
        return parameters
    
    def get_exact_nonlinear_parameters(self) -> list:
        """
        Return list of the exact nonlinear parameters of the parametrization.

        Returns
        -------
        list
            list of the exact nonlinar parameters of the Parametrization
        """
        parameters = []
        # parametrization functions
        if self.functions is not None:
            for fct in self.functions:
                if fct.exact_parameters is not None:
                    parameters.extend(fct.exact_parameters)
        # regularization functions
        if self.regularizations is not None:
            for reg in self.regularizations:
                if reg.exact_parameters is not None:
                    parameters.extend(reg.exact_parameters)
        return parameters

    def get_nonlinear_bounds(self) -> list:
        """
        Return list of the bounds of the nonlinear parameters of the parametrization.

        Returns
        -------
        list
            list of the bounds of the nonlinar parameters of the Parametrization
        """
        bounds = []
        # parametrization functions
        if self.functions is not None:
            for fct in self.functions:
                if fct.bounds is not None:
                    bounds.extend(fct.bounds)
        # regularization functions
        if self.regularizations is not None:
            for reg in self.regularizations:
                if reg.bounds is not None:
                    bounds.extend(reg.bounds)
        if bounds == []:
            bounds = None
        return bounds

    def set_nonlinear_parameters(self, 
                                parameters: list[float],
                                ) -> None:
        """
        Set the nonlinear parameters of the Parametrization to parameters.

        Parameters
        ----------
        parameters : list of floats
            List of the new values of the nonlinear parameters.
        """
        ### check if total number of nonlinear parameters is correct
        npar_tot = len(self.get_nonlinear_parameters())
        if npar_tot != len(parameters):
            raise ValueError("number of parameters for parametrization {} does not match: par:{} != set:{}".format(self.name, npar_tot, len(parameters)))
        ### set nonlinear parameters
        inx=0
        # parametrization functions
        if self.functions is not None:
            for fct in self.functions:
                if fct.parameters is not None:
                    npar = len(fct.parameters)
                    fct.parameters = parameters[inx:inx+npar]
                    inx+=npar
        # regularization functions
        if self.regularizations is not None:
            for reg in self.regularizations:
                if reg.parameters is not None:
                    npar = len(reg.parameters)
                    reg.parameters = parameters[inx:inx+npar]
                    inx+=npar

    def get_parametrization_matrices(self, 
                                    ansatz_operator: QuantumOperator | None = None, 
                                    ansatz_dissipators: list[Dissipator] | None = None,
                                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return parametrization matrix for given ansatz_operator and/or ansatz_dissipators.

        The parametrization matrix can be used to transform 
        the cost function for learning the ansatz coefficients.

        Parameters
        ----------
        ansatz_operator : QuantumOperator, optional
            Ansatz for the Hamiltonian that should be parametrized.
            Default is None.
        ansatz_dissipators : list of Dissipator objects, optional
            Ansatz for the Dissipators that should be parametrized.
            Default is None.

        Returns
        -------
        tuple of np.arrays
            parametrization_matrices[0] is the (mA,nA)-dimensional parametrization matrix for the ansatz_operator.
            parametrization_matrices[1] is the (mD,nD)-dimensional parametrization matrix for the ansatz_dissipators.
            Here mA/mD is the number of terms in the 
            ansatz_operator/ansatz_dissipators respectively,
            and nA/nD is the number of coherent/incoherent functions in the parametrization.
        """
        ### STEP 1 ### coherent terms
        parametrization_matrix_H = None
        if ansatz_operator is not None:
            pops = list(ansatz_operator.terms.values())
            parametrization_matrix_H = np.zeros((len(pops),len(self.functions)),dtype=complex)
            for parinx, parfct in enumerate(self.functions):
                if parfct.coherent:
                    for popinx, pop in enumerate(pops):
                        parametrization_matrix_H[popinx,parinx] = parfct(pop)
            parametrization_matrix_H = sp.linalg.orth(parametrization_matrix_H)
        #----------------------------------------
        ### STEP 2 ### dissipative terms
        parametrization_matrix_D = None
        if ansatz_dissipators is not None:
            dops = ansatz_dissipators
            parametrization_matrix_D = np.zeros((len(dops),len(self.functions)),dtype=complex)
            for parinx, parfct in enumerate(self.functions):
                if not parfct.coherent:
                    for dopinx, dop in enumerate(dops):
                        parametrization_matrix_D[dopinx,parinx] = parfct(dop)
            parametrization_matrix_D = sp.linalg.orth(parametrization_matrix_D)
        #----------------------------------------
        return [parametrization_matrix_H, parametrization_matrix_D]

    def get_regularization_matrices(self, 
                                    ansatz_operator: QuantumOperator | None = None, 
                                    ansatz_dissipators: list[Dissipator] | None = None,
                                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns regularization matrix for given ansatz_operator and/or ansatz_dissipators.

        The regularization matrix can be used to transform 
        the cost function for learning the ansatz coefficients.

        Parameters
        ----------
        ansatz_operator : QuantumOperator, optional
            Ansatz for the Hamiltonian that should be parametrized.
            Default is None.
        ansatz_dissipators : list of Dissipators, optional
            Ansatz for the Dissipators that should be parametrized.
            Default is None.

        Returns
        -------
        tuple of np.arrays
            regularization_matrices[0] is beta*(I-RH*RH^T) 
            where RH is the (mA,nA)-dimensional parametrization matrix for the ansatz_operator
            and beta is the regularization_factor.
            regularization_matrices[1] is beta*(I-RD*RD^T)
            where RD is the (mD,nD)-dimensional parametrization matrix for the ansatz_dissipators
            and beta is the regularization_factor.
            Here mA/mD is the number of terms in the 
            ansatz_operator/ansatz_dissipators respectively,
            and nA/nD is the number of coherent/incoherent functions in the regularization.
        """
        if self.regularizations is None:
            return None
        if self.regularization_factor is None:
            raise ValueError("regularization_factor is None, but must be a positive float for par={}".format(self.str()))
        #----------------------------------------
        ### STEP 1 ### coherent terms
        regularization_matrix_H = None
        if ansatz_operator is not None:
            pops = list(ansatz_operator.terms.values())
            R = np.zeros((len(pops),len(self.regularizations)),dtype=complex)
            for reginx, regfct in enumerate(self.regularizations):
                if regfct.coherent:
                    for popinx, pop in enumerate(pops):
                        R[popinx,reginx] = regfct(pop)
            R = sp.linalg.orth(R)
            regularization_matrix_H = np.multiply(self.regularization_factor, np.subtract(np.eye(len(R)), np.einsum("ij,kj",R,R)))
        #----------------------------------------
        ### STEP 2 ### incoherent terms
        regularization_matrix_D = None
        if ansatz_dissipators is not None:
            dops = ansatz_dissipators
            R = np.zeros((len(dops),len(self.regularizations)),dtype=complex)
            for reginx, regfct in enumerate(self.regularizations):
                if not regfct.coherent:
                    for dopinx, dop in enumerate(dops):
                        R[dopinx,reginx] = regfct(dop)
            R = sp.linalg.orth(R)
            regularization_matrix_D = np.multiply(self.regularization_factor, np.subtract(np.eye(len(R)), np.einsum("ij,kj",R,R)))
        #----------------------------------------
        return [regularization_matrix_H, regularization_matrix_D]

    @staticmethod
    def combine_parametrization_matrices(parametrization_matrices) -> np.ndarray:
        """
        Combines parametrization matrices for coherent and incoherent terms.
        
        The matrices are combined into one parametrization matrix
        by stacking them diagonally into a block-diagonal matrix.

        Parameters
        ----------
        parametrization_matrices : tuple of np.arrays
            parametrization_matrices[0] is the (mA,nA)-dimensional parametrization matrix for the ansatz_operator.
            parametrization_matrices[1] is the (mD,nD)-dimensional parametrization matrix for the ansatz_dissipators.
            Here mA/mD is the number of terms in the
            ansatz_operator/ansatz_dissipators respectively,
            and nA/nD is the number of coherent/incoherent functions in the parametrization.
        
        Returns
        -------
        2D np.array
            Combined parametrization matrix.
        """
        ## only parametrize dissipators
        if parametrization_matrices[0] is None:
            combined_parametrization_matrix = parametrization_matrices[1]
        ## only parametrize Hamiltonian
        elif parametrization_matrices[1] is None:
            combined_parametrization_matrix = parametrization_matrices[0]
        ## parametrize both Hamiltonian and dissipators
        else:
            A00 = parametrization_matrices[0]
            A11 = parametrization_matrices[1]
            A01 = np.zeros((A00.shape[0], A11.shape[1]))
            A10 = np.zeros((A11.shape[0], A00.shape[1]))
            combined_parametrization_matrix = np.block([[A00,A01],[A10,A11]])
        # ----------------------------------------
        return combined_parametrization_matrix

    @staticmethod
    def combine_regularization_matrices(regularization_matrices) -> np.ndarray:
        """
        Combines regularization matrices for coherent and incoherent terms.

        The matrices are combined into one regularization matrix
        by stacking them diagonally into a block-diagonal matrix.

        Parameters
        ----------
        regularization_matrices : tuple of np.arrays
            regularization_matrices[0] is the (mA,nA)-dimensional regularization matrix for the ansatz_operator.
            regularization_matrices[1] is the (mD,nD)-dimensional regularization matrix for the ansatz_dissipators.
            Here mA/mD is the number of terms in the
            ansatz_operator/ansatz_dissipators respectively,
            and nA/nD is the number of coherent/incoherent functions in the regularization.

        Returns
        -------
        2D np.array
            Combined regularization matrix.
        """
        if regularization_matrices[0] is None:
            combined_regularization_matrix = regularization_matrices[1]
        elif regularization_matrices[1] is None:
            combined_regularization_matrix = regularization_matrices[0]
        else:
            A00 = regularization_matrices[0]
            A11 = regularization_matrices[1]
            A01 = np.zeros((A00.shape[0], A11.shape[1]))
            A10 = np.zeros((A11.shape[0], A00.shape[1]))
            combined_regularization_matrix = np.block([[A00,A01],[A10,A11]])
        return combined_regularization_matrix

    def get_free_coeffs_from_parametrized_coeffs(self, 
                                                coeffs_G, 
                                                qop: QuantumOperator | None = None, 
                                                dissipators: list[Dissipator] | None = None, 
                                                G : np.ndarray | None = None
                                                ):
        """
        Return ansatz coefficients from parametrized coefficients.

        The free coefficients are calculated via coeffs = G @ coeffs_G,
        where coeffs_G are the coefficients of the of the parametrized ansatz 
        and G is the parametrization matrix defined by the Parametrization.
        Requires either a parametrization matrix or a ansatz in the form of
        a QuantumOperator and/or a list of Dissipator objects as additional argument.

        Parameters
        ----------
        coeffs_G : list of floats
            Free coefficients of the parametrized QuantumOperator.
        qop : QuantumOperator, optional
            ansatz for the Hamiltonian that has been parametrized
            Default is None.
        dissipators : list of Dissipators, optional
            ansatz for the dissipators that have been parametrized
            Default is None.
        G : parametrization matrix, optional
            parametrization matrix of the Parametrization
            Default is None.
        """
        if qop is None and G is None:
            raise ValueError("Either qop or G must be given.")
        if G is None:
            parametrization_matrices = self.get_parametrization_matrices(ansatz_operator=qop, ansatz_dissipators=dissipators)
            G = Parametrization.combine_parametrization_matrices(parametrization_matrices)
        tot_coeffs = list(np.einsum("ij,j",G,coeffs_G)) #G @ par_lin
        return tot_coeffs

    def get_parametrized_coeffs_from_free_coeffs(self, 
                                                coeffs = None, 
                                                qop: QuantumOperator | None = None, 
                                                dissipators: list[Dissipator] | None = None, 
                                                G = None
                                                ):
        """
        Return parametrized coefficients from ansatz coefficients.

        The parametrized coefficents are calculated via coeffs_G = G.T @ coeffs,
        where coeffs are the coefficients of the unparametrized ansatz
        and G is the parametrization matrix defined by the Parametrization.
        Requires either a parametrization matrix or a ansatz in the form of
        a QuantumOperator and/or a list of Dissipator objects as additional argument.

        Parameters
        ----------
        coeffs : list of coefficients of quantum operator, optional
            unparametrized coefficients of the ansatz
            Default is None
        qop : QuantumOperator, optional
            ansatz operator for the Hamiltonian
            Default is None.
        dissipators : list of QuantumOperators
            ansatz dissipators
            Default is None.
        G : 2D np.array
            parametrization matrix
            Default is None.
        """
        if coeffs is None:
            coeffs = qop.coeffs()
        if G is None:
            parametrization_matrices = self.get_parametrization_matrices(ansatz_operator=qop, ansatz_dissipators=dissipators)
            G = Parametrization.combine_parametrization_matrices(parametrization_matrices)
        return list(np.einsum("ji,j",G,coeffs))  #G.T @ coeffs                

    def get_missing_terms(self, 
                        qop: QuantumOperator,
                        ) -> QuantumOperator:
        """
        Given a quantum operator qop,
        return all terms of qop that are not in the parametrization.

        Parameters
        ----------
        qop : QuantumOperator
            QuantumOperator that should be parametrized

        Returns
        -------
        QuantumOperator
            terms in qop that are not in the parametrization
        """
        #---------------------
        missing_terms = qop.copy()
        for parfct in self.functions:
            if parfct.coherent:
                for term in qop.terms.values():
                    if parfct.check_criterion(term):
                        qop_term = term.to_quantum_operator()
                        missing_terms.remove_terms(qop_term)
        #---------------------
        return missing_terms

    def fit_parameters(self, 
                    qop: QuantumOperator, 
                    **kwargs
                    ):
        raise NotImplementedError("fit_parameters is not implemented yet.")
        """
        Fit parameters of the parametrization to a given QuantumOperator.
        Returns the fitted QuantumOperator.
        Note that the method changes the parameters of the Parametrization object.
        args:
            - qop (QuantumOperator)
                Quantum operator that should be parametrized.
        kwargs:
            - max_nfev (int) [default: 11]
                Maximum number of function evaluations for the brute optimization.
        returns:
            - qop_fit (QuantumOperator)
                Parametrized QuantumOperator that is fitted to qop.
        """
        ### get kwargs
        max_nfev = kwargs.get("max_nfev", 11)
        ### set up costfunction
        par = self
        def costfct(par_nonlin):
            par.set_nonlinear_parameters(par_nonlin)
            parametrization_matrices = self.get_parametrization_matrices(ansatz_operator=qop)
            G = Parametrization.combine_parametrization_matrices(parametrization_matrices)
            par_lin, err, svd_vals, svd_vecs, cond_num, err_bnd = solve_linear_eq(M=G, b=qop.coeffs())
            # sol, err, svd_vals, svd_vecs, sol_lsq, err_lsq, condition_number, error_bound = solve_linear_eq(M=M)
            coeffs_fit = np.einsum("ij,j",G,par_lin)
            diff = coeffs_fit-qop.coeffs()
            return np.real(diff)
        ### minimize costfunction using least squares
        x0 = np.zeros(len(self.get_nonlinear_parameters()))
        # nonlinear parameters
        if len(x0)!=0:
            bounds = self.get_nonlinear_bounds()
            par_nonlin_fit, err_nonlin, landscape_grid, landscape_vals = uf.minimize(costfct, x0, bounds=bounds, method="brute", max_nfev=max_nfev)
            par.set_nonlinear_parameters(par_nonlin_fit)
        # linear parameters
        parametrization_matrices = self.get_parametrization_matrices(ansatz_operator=qop)
        G = Parametrization.combine_parametrization_matrices(parametrization_matrices)
        par_lin_fit, err_lin, svd_vals, svd_vecs, cond_num, err_bnd = solve_linear_eq(M=G, b=qop.coeffs())
        # sol, err, svd_vals, svd_vecs, sol_lsq, err_lsq, condition_number, error_bound = solve_linear_eq(M=M)
        coeffs_fit = np.einsum("ij,j",G,par_lin_fit)
        ### create fitted quantum operator
        qop_fit = qop.copy()
        qop_fit.set_coeffs(coeffs_fit) 
        return qop_fit