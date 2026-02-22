"""
This module contains classes and methods for symbolic calculations
with Pauli matrices, quantum operators and dissipators (superoperators).
Classes:
    - PauliOperator
        A class that defines a Pauli operator.
    - QuantumOperator
        A class that defines a quantum operator as linear combination of PauliOperators.
    - Dissipator: 
        A class that defines a dissipator (superoperator) for the master equation in the Lindblad form.
    - CollectiveSpinOperator
        A class that defines a collective spin operator.
"""
from __future__ import annotations
import numpy as np
import itertools as it
import networkx as nx
import matplotlib.pyplot as plt
from icecream import ic

class PauliOperator:
    """
    Pauli operator for symbolic calculations.

    A PauliOperator represents a tensor product of Pauli matrices
    acting on a fixed number of qubits.
    The type of a PauliOperator is a string of pauli chars, 
    e.g. "X1X2X3...XN" where X1...XN are pauli chars.
    A pauli_char is a char in ["I","X","Y","Z","M","P"] 
    representing the identity, sigmax, sigmay, sigmaz, sigmam, sigmap.

    Attributes
    ----------
    N : int
        Number of qubits.
    type : str
        Pauli string representation of the operator (e.g. ``"ZZI"``).
    coeff : complex
        Coefficient multiplying the PauliOperator. Default is ``1.0``.
    """
    def __init__(self, 
                N: int, 
                type: str | None = None,
                pauli_type: str | None = None,
                coeff: complex = 1.0
                ):
        """
        Initialize a Pauli operator.

        Parameters
        ----------
        N : int
            Number of qubits.
        pauli_type : str, optional
            Pauli string specifying the operator.
        coeff : complex, optional
            Coefficient of the Pauli operator. Default is ``1.0``.
        """
        self._N = N
        self.pauli_type = pauli_type
        self.coeff = coeff 
    ### ----------------- ###
    ### custom attributes ###
    ### ----------------- ###
    @property
    def N(self):
        return self._N
    @property
    def type(self):
        return self._type
    @type.setter
    def type(self, pauli_type):
        # check if pauli_type is a string of length Nions
        if not isinstance(pauli_type, str) or len(pauli_type) != self.N:
            raise ValueError("type is {}, but must be a string of length N={}".format(pauli_type, self.N))
        # check if pauli_type is a valid pauli string
        if not all([c in ["I", "X", "Y", "Z", "M", "P"] for c in pauli_type]):
            raise ValueError("type is {}, but must be a valid pauli string".format(pauli_type))
        # set type
        self._type = pauli_type
    @property
    def coeff(self):
        return self._coeff
    @coeff.setter
    def coeff(self, coeff):
        # check if coeff is a number or None
        if not np.isscalar(coeff) and coeff is not None:
            raise ValueError("coeff must be a number or None")
        # set coeff
        self._coeff = coeff
    ## ----------------- ##
    ## custom functions  ##
    ## ----------------- ##
    def copy(self) -> PauliOperator:
        """ 
        Return a copy of the PauliOperator. 

        Returns
        -------
        PauliOperator
            A new instance with the same attributes as this operator.
        """
        return PauliOperator(N=self.N, coeff=self.coeff, pauli_type=self.pauli_type)
    
    def to_quantum_operator(self) -> QuantumOperator:
        """
        Return the QuantumOperator representation of the PauliOperator.

        Returns
        -------
        QuantumOperator
            A new instance with the same attributes as this operator.
        """
        qop = QuantumOperator(self.N, terms={self.pauli_type: self.coeff})
        return qop
    
    def __str__(self):
        return self.str()
    def str(self, 
            index: int | None = None, 
            sign: bool = False, 
            ndigits: int | None = None, 
            nonzero: bool = False, 
            no_coeffs: bool = False
            ) -> str:
        """
        Return the string representation of the PauliOperator.
        
        Parameters
        ----------
        index : int, optional
            If not None, use c(index) instead of coeff. Default is ``None``
        sign : bool, optional
            If True, add + or - sign to string. Default is ``False``
        ndigits : int, optional
            If not None, round coeff to ndigits decimals. Default is ``None``
        nonzero : bool, optional
            If True, only return string if coeff is not 0. Default is ``False``
        no_coeffs : bool, optional
            If True, do not include the coefficient in the string. Default is ``False``

        Returns
        -------
        str
            string representation of the PauliOperator
        """
        #----------------------------------------
        ### STEP 1 ### get string representation of the PauliOperator
        opstr = ""
        coeff = None
        if not no_coeffs:
            coeff = self.coeff
        # round coeff to ndigits decimals
        if ndigits is not None and coeff is not None:
            coeff = np.around(coeff, decimals=ndigits)
        # check if coeff is 0
        if nonzero and coeff is not None and np.isclose(coeff, 0):
            return ""
        if sign:
            if coeff is None:
                opstr += "+ "
            elif np.sign(coeff) == 1 or np.sign(coeff) == 0:
                opstr += "+ "
            elif np.sign(coeff) == -1:
                opstr += "- "
                coeff = -coeff
            else:
                raise ValueError("pauli_operator.__str__: sign must be + or -")
        if coeff is None:
            if index is None:
                opstr += "c(i)"
            else:
                opstr += "c({})".format(index)
        else:
            opstr += "{}*".format(coeff)
        opstr += self.pauli_type       
        #----------------------------------------     
        return opstr

    def __eq__(self, 
            other: PauliOperator,
            ) -> bool:
        """
        Check if two pauli operators have the same type and coefficient.

        Parameters
        ----------
        other : PauliOperator
            PauliOperator object

        Returns
        -------
        bool
            True if the two pauli operators have the same type and coefficient, False otherwise
        """
        # check if other is a pauli operator
        if not isinstance(other, PauliOperator):
            raise TypeError("pauli_operator.__eq__: other must be a pauli_operator")
        # check type
        typeval = self.pauli_type==other.pauli_type
        # check coefficient
        if self.coeff is None or other.coeff is None:
            coeffval = True
        else:
            coeffval = np.isclose(self.coeff, other.coeff)
        return typeval and coeffval

    def __ne__(self, 
            other: PauliOperator,
            ) -> bool:
        """ 
        Inverse of pauli_operator.__eq__
        """
        return not self.__eq__(other)

    def __neg__(self) -> PauliOperator:
        """
        Return the PauliOperator with coeff=-self.coeff

        -------
        PauliOperator
            PauliOperator with coeff=-self.coeff
        """
        if self.coeff is None:
            return PauliOperator(N=self.N, coeff=None, pauli_type=self.pauli_type)
        else:
            return PauliOperator(N=self.N, coeff=-self.coeff, pauli_type=self.pauli_type)

    def __abs__(self) -> PauliOperator:
        """
        Return the PauliOperator with coeff=abs(self.coeff)

        -------
        PauliOperator
            PauliOperator with coeff=abs(self.coeff)
        """
        return PauliOperator(N=self.N, coeff=abs(self.coeff), pauli_type=self.pauli_type)

    def dagger(self) -> PauliOperator:
        """
        Return the transpose and complex conjugation ("dagger") of the PauliOperator

        -------
        PauliOperator
            PauliOperator with coeff=conj(self.coeff) and pauli_type=transpose(self.pauli_type)
        """
        # replace "M" by "P" in self.pauli_type
        type_dag = self.pauli_type.replace("M","P")
        pop_dag = PauliOperator(N=self.N, coeff=complex(np.conj(self.coeff)), pauli_type=type_dag)
        return pop_dag

    def is_identity(self) -> bool:
        """
        Return True if pauli operator type is proportional to the identity

        Returns
        -------
        bool
            True if type is ``"II...I"``, False otherwise
        """        
        return self.pauli_type == "I"*self.N

    def __add__(self, 
            other: PauliOperator,
            ) -> PauliOperator:
        """
        Return sum of 2 EQUAL PauliOperator objects

        Parameters
        ----------
        other : PauliOperator
            Pauli operator to be added

        Returns
        -------
        PauliOperator  
            Pauli operator with coeff=self.coeff+other.coeff
        """
        # check if other is a pauli operator
        if not isinstance(other, PauliOperator):
            raise TypeError("pauli_operator.__add__: other must be a pauli_operator")
        # check if pauli operators have same type
        if self.pauli_type != other.pauli_type:
            raise ValueError("pauli_operator.__add__: pauli operators must have same type")
        # get type of sum
        sum_str = self.pauli_type
        # get coeff of sum
        if self.coeff is None or other.coeff is None:
            sum_coeff = None
        else:
            sum_coeff = self.coeff + other.coeff
        return PauliOperator(N=self.N, coeff=sum_coeff, pauli_type="".join(sum_str))

    def __sub__(self, 
                other: PauliOperator
                ) -> PauliOperator:
        """
        Return difference of 2 EQUAL PauliOperator objects

        Parameters
        ----------
        other : PauliOperator
            Pauli operator to be subtracted

        Returns
        -------
        PauliOperator  
            Pauli operator with coeff=self.coeff-other.coeff
        """
        return self + (-other)

    def __mul__(self, other) -> PauliOperator:
        """
        Return product of two pauli operators or a pauli operator and a complex number.

        Parameters
        ----------
        other : PauliOperator or complex number
            pauli operator or complex number to be multiplied with self

        Returns
        -------
        PauliOperator  
            product of self and other
        """
        # check if other is either a pauli operator or a complex number
        if not isinstance(other, PauliOperator) and not isinstance(other, (int,float,complex)):
            raise TypeError("pauli_operator.__mul__: other must be a pauli_operator or a complex number")
        # scalar multiplication
        if isinstance(other,(int,float,complex)):
            return PauliOperator(N=self.N, coeff=self.coeff * other, pauli_type=self.pauli_type)
        # pauli operator multiplication
        elif isinstance(other, PauliOperator):
            # check if "P" or "M" in self.pauli_type or other.pauli_type
            if "P" in self.pauli_type or "M" in self.pauli_type or "P" in other.pauli_type or "M" in other.pauli_type:
                raise ValueError("pauli_operator.__mul__: cannot multiply pauli operators {}x{} due to P or M entries".format(self.pauli_type, other.pauli_type))
            # initialize product operator type
            prod_type = ["I"]*self.N
            # get coeff of product
            if self.coeff is None or other.coeff is None:
                prod_coeff = None
            else:
                prod_coeff = self.coeff * other.coeff
            # get type of product
            for inx in range(self.N):
                if self.pauli_type[inx] == other.pauli_type[inx]:
                    prod_type[inx] = "I"
                elif self.pauli_type[inx] == "I" or other.pauli_type[inx] == "I":
                    if self.pauli_type[inx] != "I":
                        prod_type[inx] = self.pauli_type[inx]
                    else:
                        prod_type[inx] = other.pauli_type[inx]
                else:
                    for pchar in ["X","Y","Z"]:
                        if pchar not in [self.pauli_type[inx], other.pauli_type[inx]]:
                            prod_type[inx] = pchar
                            break
                    def cycle(pchar):
                        if pchar == "X":
                            return "Y"
                        elif pchar == "Y":
                            return "Z"
                        elif pchar == "Z":
                            return "X"
                    # if cycle(ord(self.pauli_type[i])) == ord(other.pauli_type[i]):
                    # get prefactor of coeff of product
                    if prod_coeff is not None:
                        if cycle(self.pauli_type[inx]) == other.pauli_type[inx]:
                            prod_coeff *= 1j
                        else:
                            prod_coeff *= -1j
            return PauliOperator(N=self.N, coeff=prod_coeff, pauli_type="".join(prod_type))

    def __rmul__(self, other) -> PauliOperator:
        """
        Return right-multiplication of a pauli operator and a complex number.

        Parameters
        ----------
        other : complex number
            complex number to be multiplied with self

        Returns
        -------
        PauliOperator  
            product of other and self
        """
        return self.__mul__(other)

    def __truediv__(self, other) -> PauliOperator:
        """
        Return division of a pauli operator and a complex number.

        Parameters
        ----------
        other : complex number
            complex number to be divided with self

        Returns
        -------
        PauliOperator  
            pauli operator with coeff=self.coeff/other
        """
        if not isinstance(other, (int,float,complex)):
            raise TypeError("pauli_operator.__truediv__: other must be a complex number")
        return self * (1/other)

    def __pow__(self, 
                power: int
                ) -> PauliOperator:
        """
        Return power of a pauli operator

        Parameters
        ----------
        power : int
            power to which the pauli operator is raised

        Returns
        -------
        PauliOperator  
            power of the pauli operator
        """
        # check if power is a positive int
        if not (isinstance(power, int) and power > 0):
            raise TypeError("pauli_operator.__pow__: power must be a positive int")
        # check if "P" or "M" in self.pauli_type or other.pauli_type
        if "P" in self.pauli_type or "M" in self.pauli_type:
            raise ValueError("pauli_operator.__mul__: cannot take power of pauli operators {} due to P or M entries".format(self.pauli_type))
        # get coeff of power
        pow_coeff = self.coeff**power
        # get type of power
        if power % 2 == 0:
            pow_type = "I"*self.N
        else:
            pow_type = self.pauli_type
        return PauliOperator(N=self.N, coeff=pow_coeff, pauli_type=pow_type)

    def norm(self) -> float:
        """
        Return norm of a PauliOperator.

        Returns
        -------
        float  
            norm of the PauliOperator
        """
        return np.abs(self.coeff)

    def commutator(self, 
                other: PauliOperator,
                ) -> PauliOperator:
        """
        Return commutator of two Pauli operators.

        Parameters
        ----------
        other : PauliOperator
            Pauli operator to be commutated with self

        Returns
        -------
        PauliOperator  
            commutator of self and other
        """
        # check if other is a pauli operator
        if not isinstance(other, PauliOperator):
            raise TypeError("pauli_operator.commutator: other must be a pauli_operator")
        return self * other - other * self

    def range(self) -> int:
        """
        Return the range of the PauliOperator.

        Range is the maximum distance between two qubits on which the PauliOperator acts.

        Returns
        -------
        range : int
            range of the PauliOperator
        """
        # get non I entries of self.pauli_type
        non_I = [inx for inx in range(self.N) if self.pauli_type[inx] != "I"]
        # return the maximal distance between two non I entries
        maxran = 0
        if len(non_I)>0:
            maxran = max(non_I) - min(non_I)
        return maxran
    
    def support(self) -> list:
        """
        Return the support of the PauliOperator.

        Support is the set of qubits on which the PauliOperator acts.

        Returns
        -------
        support : list of int
            support of the PauliOperator
        """
        support = [inx for inx in range(self.N) if self.pauli_type[inx] != "I"]
        return support

    def center(self) -> float:
        """
        Return the center of a PauliOperator.

        The center is the axial coordinate of the PauliOperator
        given by (pos1+pos2)/2 - (N-1)/2.

        Returns
        -------
        center : float
            center of the PauliOperator
        """
        # find index with non-I term
        non_I_index = [i for i, x in enumerate(self.pauli_type) if x != "I"]
        if len(non_I_index) not in  [1,2]:
            raise ValueError("pop.pauli_type must contain exactly one or two non-I terms")
        if len(non_I_index) == 1:
            center = non_I_index[0]-(len(self.pauli_type)-1)/2   # [-(N-1)/2, (N-1)/2]
        elif len(non_I_index) == 2:
            center = (non_I_index[0]+non_I_index[1])/2-(len(self.pauli_type)-1)/2   # [-(N-1)/2, (N-1)/2]
        return center

    def split_MP(self) -> list:
        """
        Return a list of Pauli operators that is equal to the original Pauli operator
        when summed up, but without ``"M"`` or ``"P"`` terms.
        Here ``M = X+iY`` and ``P = X-iY``.

        Returns
        -------
        list of PauliOperators
        """
        pop_list = [self]
        # check if "M" or "P" are in self.pauli_type
        if "M" in self.pauli_type or "P" in self.pauli_type:
            # get all indices of "M" and "P" in self.pauli_type
            MP_inx = [inx for inx in range(len(self.pauli_type)) if self.pauli_type[inx] == "M" or self.pauli_type[inx] == "P"]
            # split into Pauli operators without "M" or "P"
            pop_list_new = pop_list.copy()
            while len(MP_inx) > 0:
                pop_list = pop_list_new.copy()
                for pop in pop_list:
                    if pop.pauli_type[MP_inx[0]] == "M":
                        pop_list_new.remove(pop)
                        pop_list_new.append(PauliOperator(N=pop.N, coeff=pop.coeff/2, pauli_type=pop.pauli_type[:MP_inx[0]]+"X"+pop.pauli_type[MP_inx[0]+1:]))
                        pop_list_new.append(PauliOperator(N=pop.N, coeff=-1j*pop.coeff/2, pauli_type=pop.pauli_type[:MP_inx[0]]+"Y"+pop.pauli_type[MP_inx[0]+1:]))
                    elif pop.pauli_type[MP_inx[0]] == "P":
                        pop_list_new.remove(pop)
                        pop_list_new.append(PauliOperator(N=pop.N, coeff=pop.coeff/2, pauli_type=pop.pauli_type[:MP_inx[0]]+"X"+pop.pauli_type[MP_inx[0]+1:]))
                        pop_list_new.append(PauliOperator(N=pop.N, coeff=1j*pop.coeff/2, pauli_type=pop.pauli_type[:MP_inx[0]]+"Y"+pop.pauli_type[MP_inx[0]+1:]))
                MP_inx.pop(0)
            pop_list = pop_list_new.copy()
        return pop_list

    def can_be_measured(
                    self, 
                    bases: list, 
                    return_required_bases: bool = False,
                    ):
        """ 
        Return True if the PauliOperator can be measured in the given bases.

        The PauliOperator can be measured in the given bases if
        all non-I terms in the PauliOperator match the corresponding
        non-I terms in the basis.

        Parameters
        ----------
        bases : list of strings
            list of measurement bases (e.g. ``["XXY","YZZ"]``)
        return_required_bases : bool, optional
            if True, return the minimum required bases to measure the PauliOperator
            default is ``False``

        Returns
        -------
        can_be_measured : bool
            True if the PauliOperator can be measured in the given bases
        required_bases : list of strings, optional
            minimum list of bases required to measure the PauliOperator
            only returned if return_required_bases=True
        """
        #----------------------------------------
        ### STEP 1 ### check if term can be measured
        can_be_measured = False
        required_bases = []
        for base in bases:
            # check if all non-I chars in term match base
            if all([self.pauli_type[inx] == base[inx] for inx in range(self.N) if self.pauli_type[inx] != "I"]):
                can_be_measured = True
                required_bases.append(base)
                # if not return_required_bases:
                break
        #----------------------------------------
        if return_required_bases:
            return can_be_measured, required_bases
        return can_be_measured



class QuantumOperator:
    """
    Quantum operator for symbolic calculations.

    A QuantumOperator represents a linear combination of Pauli matrices.
    It is given as a hashmap of pauli_strings to coefficients, 
    e.g. {"X1X2X3...XN": 1.0, "Y1Y2Y3...YN": 2.0, ...}.

    Attributes
    ----------
    N : int
        number of qubits
    terms : dict
        dictionary of type:coeff pairs e.g. ``{"ZZI": 1.0, "XII": 0.5}``
    """
    def __init__(self, 
                N: int, 
                terms: dict = {}):
        """
        Initialize a quantum operator.

        Parameters
        ----------
        N : int
            number of qubits
        terms : dict
            dictionary of type:coeff pairs e.g. ``{"ZZI": 1.0, "XII": 0.5}``
        
        """
        self._N = N  
        # check if terms is a dictionary
        if not isinstance(terms, dict):
            raise TypeError("QuantumOperator.__init__: terms must be a dictionary")
        # create list of pauli operators
        pop_list = []
        for key in terms:
            # check if key is a string of length N
            if not isinstance(key, str) or len(key) != N:
                raise TypeError("QuantumOperator.__init__: key={}, but must be string of length N={}".format(key,N))
            # check if terms[key] is a complex number
            if not np.isscalar(terms[key]):
                raise TypeError("QuantumOperator.__init__: value={}, but must be complex number".format(terms[key]))
            # create pauli operator
            pop_list.extend(PauliOperator(N=self.N, coeff=terms[key], pauli_type=key).split_MP())
        # create dictionary of pauli terms
        terms = {}
        for pop in pop_list:
            if pop.pauli_type in terms:
                terms[pop.pauli_type] += pop
            else:
                terms[pop.pauli_type] = pop
        self.terms = terms
    ## ----------------- ##
    ## custom properties ##
    ## ----------------- ##
    # make N a read-only property
    @property
    def N(self):
        return self._N
    ## ----------------- ##
    ## custom functions  ##
    ## ----------------- ##
    def copy(self) -> QuantumOperator:
        """
        Return a copy of the quantum operator.

        Returns
        -------
        QuantumOperator
            A new instance with the same attributes as this operator.
        """
        terms = {}
        for key in self.terms:
            terms[key] = self.terms[key].coeff
        return QuantumOperator(N=self.N, terms=terms)        

    def __str__(self):
        return self.str()
    def str(self, 
            ndigits: int | None = None, 
            nonzero: bool = False, 
            no_coeffs: bool = False,
            ) -> str:
        """
        Return the string representation of the quantum operator.

        Parameters
        ----------
        ndigits : int, optional
            If not None, round coeff to ndigits decimals. Default is ``None``
        nonzero : bool, optional
            If True, only return string if coeff is not 0. Default is ``False``
        no_coeffs : bool, optional
            If True, do not include the coefficients in the string. Default is ``False``
        """
        #----------------------------------------
        ### STEP 1 ### get string representation of QuantumOperator
        str_terms = []
        # get string representation of each PauliOperator
        for key in self.terms:
            pop = self.terms[key]  
            if len(str_terms) == 0:
                str_terms.append(pop.str(sign=False, index=len(str_terms), ndigits=ndigits, nonzero=nonzero, no_coeffs=no_coeffs))
            else:
                str_terms.append(pop.str(sign=True, index=len(str_terms), ndigits=ndigits, nonzero=nonzero, no_coeffs=no_coeffs))
            # new line
            str_terms.append("\n")
        qop_str = " ".join(str_terms)
        #----------------------------------------
        return qop_str
    
    def labels(self) -> list:
        """
        Return list of all pauli operator labels in the quantum operator

        Returns
        -------
        list
            list of operator labels
        """
        labels = [] 
        for term in self.terms:
            labels.append(term)
        return labels
    
    def is_zero(self) -> bool:
        """
        Return True if the quantum operator is the zero operator, 
        i.e. if all coefficients are 0.

        Returns
        -------
        bool
            True if the quantum operator is the zero operator, False otherwise
        """
        flag = all([np.isclose(self.terms[key].coeff, 0) for key in self.terms])
        return flag

    def is_identity(self) -> bool:
        """ 
        Return True if the quantum operator is the identity operator,
        i.e. if the only term with nonzero coefficient is the identity operator.

        Returns
        -------
        bool
            True if the quantum operator is the identity operator, False otherwise
        """
        # get all terms with nonzero coefficients
        nonzero_terms = [key for key in self.terms if self.terms[key].coeff != 0]
        # return True if there is only one term and it is the identity operator
        flag = len(nonzero_terms) == 1 and nonzero_terms[0] == "I"*self.N
        return flag
    
    def norm(self) -> float:
        """ 
        Return the 2-norm of the coefficient-vector of the quantum operator. 

        Returns
        -------
        float
            2-norm of the coefficient-vector
        """
        return np.linalg.norm(self.coeffs())

    def normalize(self) -> QuantumOperator:
        """ 
        Return the normalized quantum operator
        with coefficients scaled such that the norm is 1.

        Returns
        -------
        QuantumOperator
            normalized quantum operator
        """
        return self / self.norm()

    def __abs__(self) -> QuantumOperator:
        """ 
        Returns the quantum operator with its coefficients 
        replaced with their absolute values.

        Returns
        -------
        QuantumOperator
            quantum operator with coefficients replaced with their absolute values
        """
        qop = self.copy()
        for key in qop.terms:
            qop.terms[key] = abs(qop.terms[key])
        return qop

    def plot(self, 
            other: QuantumOperator | None = None, 
            normalize: bool = False, 
            title: str | None = None, 
            text: str | None = None, 
            var: QuantumOperator | None = None, 
            add_labels: bool = False, 
            linestyle: str = "",
            figsize: tuple = (6,4),
            dissrates = None,
            var_dissrates = None,
            other_dissrates = None,
            ) -> None:
        """
        Create a 2D-scatter plot of the coefficients of the quantum operator.

        Parameters
        ----------
        other : QuantumOperator, optional
            other quantum operator to be plotted as well. Default is ``None``
        normalize : bool, optional
            if True, normalize the coefficients. Default is ``False``
        title : string, optional
            title of the plot. Default is ``None``
        text : string, optional
            text to be added to the plot. Default is ``None``
        var : QuantumOperator, optional
            variance of the coefficients for error bars. Default is ``None``    
        add_labels : bool, optional
            if True, add labels of the terms to the plot. Default is ``False``
        linestyle : string, optional
            linestyle of the plot. Default is ``""``
        figsize : tuple, optional
            size of the figure. Default is ``(6,4)``
        dissrates : np.array, optional
            list of dissipation rates to add to the figure
            Default is ``None``
        var_dissrates : np.array, optional
            list of variances of dissipation rates to add to the figure
            Default is ``None``
        other_dissrates : np.array, optional
            other dissipation rates to be plotted as well.
            Default is ``None``
        """
        if dissrates is not None and var_dissrates is None:
            raise ValueError("if dissrates are given, var_dissrates must be given")
        # get the coefficients
        coeffs = self.coeffs()
        if var is not None:
            var_coeffs = var.to_vector(ansatz=self, add_identity=False)
        else:
            var_coeffs = np.zeros(len(coeffs), dtype=complex)
        if normalize:
            var_coeffs = var_coeffs / np.linalg.norm(coeffs)**2
            coeffs = coeffs / np.linalg.norm(coeffs)
        if dissrates is not None:
            coeffs = np.concatenate((coeffs, dissrates))
            var_coeffs = np.concatenate((var_coeffs, var_dissrates))
        # plot the coefficients
        plt.figure(figsize=figsize)
        plt.xlabel("term")
        plt.ylabel("coeff")
        if other is not None:
            other_coeffs = other.to_vector(ansatz=self, add_identity=False)
            if normalize:
                other_coeffs = other_coeffs / np.linalg.norm(other_coeffs)
            if other_dissrates is not None:
                other_coeffs = np.concatenate((other_coeffs, other_dissrates))
            plt.plot(range(len(other_coeffs)), np.real(other_coeffs), "x", label="other", color="k", linestyle=linestyle)     
        if var is None:
            plt.plot(range(len(coeffs)), coeffs, "o", label="self", color="tab:red", linestyle=linestyle)
            if add_labels:
                # replace xlabels with term names
                labels = list(self.terms.keys()) + ["diss"]*(len(coeffs)-len(self.terms.keys()))
                plt.xticks(range(len(coeffs)), labels, rotation=90)
        else:
            # add errorbars to plot
            plt.errorbar(range(len(coeffs)), np.real(coeffs), yerr=np.sqrt(np.real(var_coeffs)), fmt="o", label="self", color="tab:red", linestyle=linestyle)
            if add_labels:
                labels = list(self.terms.keys()) + ["diss"]*(len(coeffs)-len(self.terms.keys()))
                plt.xticks(range(len(coeffs)), labels, rotation=90)
        if title is not None:
            plt.title(title)
        if text is not None:
            plt.text(0, 0, text, horizontalalignment="left", verticalalignment="top", transform=plt.gca().transAxes)
        plt.legend()

    def to_quantum_operator(self) -> QuantumOperator:
        """ 
        Returns a copy of the quantum operator. 

        Returns
        -------
        QuantumOperator
            copy of the quantum operator
        """
        return self.copy()
    
    def to_pauli_operator(self) -> PauliOperator:
        """
        Returns the Pauli operator representation of the quantum operator, 
        if possible.

        Returns
        -------
        PauliOperator
            Pauli operator representation of the quantum operator
        """
        # check if there is only 1 term in the quantum operator
        if len(self.terms) != 1:
            raise ValueError("QuantumOperator.to_pauli_operator: quantum operator must have exactly 1 term")
        # get the only term
        key = list(self.terms.keys())[0]
        pauliop = PauliOperator(N=self.N, coeff=self.terms[key].coeff, pauli_type=key)
        return pauliop

    def to_vector(self, 
                ansatz: QuantumOperator | None = None, 
                add_identity: bool = True
                ):
        """
        Returns the vector representation of the quantum operator.

        Parameters
        ----------
        ansatz : QuantumOperator, optional
            If not None, return the vector representation of the quantum operator
            with respect to the ansatz. Default is ``None``
        add_identity : bool, optional
            If True, coefficient of I*N is added as first element to vector. Default is ``True``
        
        Returns
        -------
        numpy array
            vector representation of the quantum operator
        """
        pauli_chars = ["I", "X", "Y", "Z"]
        if ansatz is None:
            N = len(list(self.terms.keys())[0])
            pstr_list = list(it.product(pauli_chars, repeat=N))
        else:
            N = len(list(ansatz.terms.keys())[0])
            if add_identity:
                pstr_list = ["I"*N] + list(ansatz.terms.keys())
            else:
                pstr_list = list(ansatz.terms.keys())
        vector = np.zeros(len(pstr_list), dtype=complex)
        pinx = 0
        for pstr_prod in pstr_list: 
            pstr = "".join(pstr_prod)
            if pstr in self.terms:
                vector[pinx] = self.terms[pstr].coeff
            pinx += 1
        return vector

    def from_vector(self, 
                    vector, 
                    ansatz: QuantumOperator | None = None
                    ) -> None:
        """
        Sets the coefficients of the quantum operator to the given vector.

        Parameters
        ----------
        vector : numpy array
            vector representation of the quantum operator
        ansatz : QuantumOperator, optional
            If not None, set the coefficients of the quantum operator
            with respect to the ansatz. Default is ``None``
        """
        pauli_chars = ["I", "X", "Y", "Z"]
        if ansatz is None:
            N = int(np.log(len(vector))/np.log(4))
            pstr_list = it.product(pauli_chars, repeat=N)
        else:
            N = len(list(ansatz.terms.keys())[0])
            pstr_list = ["I"*N] + list(ansatz.terms.keys())
        pinx = 0
        for pstr_prod in pstr_list: 
            if vector[pinx] != 0:
                pstr = "".join(pstr_prod)
                self.terms[pstr] = PauliOperator(N=N, coeff=vector[pinx], pauli_type=pstr)
            pinx += 1

    def coeffs(self, 
            indexed: bool = False
            ):
        """
        Returns the coefficients vector of the quantum operator as a numpy array.

        Parameters
        ----------
        indexed : bool, optional
            If True, return the coefficients together with 
            the indices of the qubits the pauli operators acts on. Default is ``False``

        Returns
        -------
        coeffs : numpy array
            coefficients of the quantum operator
            if indexed is True, coeffs is a list of tuples (coeff, index)
        """
        coeffs = np.array([self.terms[key].coeff for key in self.terms])
        if indexed:
            indices = []
            for term in self.terms:
                inx = []
                for cinx in range(len(term)):
                    if term[cinx] != "I":
                        inx.append(cinx)
                indices.append(inx)
            return coeffs, indices
        return coeffs
    
    def set_coeffs(self, coeffs) -> None:
        """ 
        Set the coefficients of the quantum operator to coeffs. 

        Parameters
        ----------
        coeffs : numpy array
            coefficients of the quantum operator
        """
        if len(coeffs) != len(self.terms):
            raise ValueError("QuantumOperator.set_coeffs: coeffs must have same length as number of terms")
        for key, coeff in zip(self.terms, coeffs):
            self.terms[key].coeff = coeff
            
    def set_coeffs_none(self) -> None:
        """ Set all coefficients of the quantum operator to None. """
        for key in self.terms:
            self.terms[key].coeff = None
    def set_coeffs_one(self) -> None:
        """ Set all coefficients of the quantum operator to 1. """
        for key in self.terms:
            self.terms[key].coeff = 1
    def set_coeffs_zero(self) -> None:
        """ Set all coefficients of the quantum operator to 0. """
        for key in self.terms:
            self.terms[key].coeff = 0

    def remove_zero_coeffs(self) -> QuantumOperator:
        """
        Remove all terms with coefficient 0 from the quantum operator.

        Returns
        -------
        QuantumOperator
            Quantum operator without terms with coefficient 0
        """
        for key in list(self.terms.keys()):
            if np.isclose(self.terms[key].coeff,0):
                del self.terms[key]
        # check if all coefficients are zero
        if len(self.terms) == 0:
            raise ValueError("QuantumOperator.remove_zero_coeffs: all coefficients are zero")
        return self
    
    def remove_zero_coeffs_and_set_coeffs_1(self) -> None:
        """
        Remove all terms with coefficient 0 from the quantum operator and set all coefficients to 1.
        """
        for key in list(self.terms.keys()):
            if np.isclose(self.terms[key].coeff,0):
                del self.terms[key]
            else:
                self.terms[key].coeff = 1

    def remove_identity(self) -> QuantumOperator:
        """
        Remove the identity term from the quantum operator.

        Returns
        -------
        QuantumOperator
            Quantum operator without the identity term
        """
        if "I"*self.N in self.terms:
            del self.terms["I"*self.N]
        return self

    def dagger(self) -> QuantumOperator:
        """
        Return the transpose and complex conjugation ("dagger") of the quantum operator.

        Returns
        -------
        QuantumOperator
            Quantum operator with each term transposed and complex conjugated ("daggered")
        """
        qop_dag = self.copy()
        for key in self.terms:
            qop_dag.terms[key] = self.terms[key].dagger()
        return qop_dag

    def __eq__(self, 
            other : QuantumOperator
            ) -> bool:
        """
        Return True if self is equal to other.

        Parameters
        ----------
        other : QuantumOperator
            Quantum operator to compare to self

        Returns
        -------
        bool
            True if all nonzero terms in self are equal to all nonzero terms in other
        """
        self_copy = self.copy()
        other_copy = other.copy()
        self_copy.remove_zero_coeffs()
        if isinstance(other, QuantumOperator):
            other_copy.remove_zero_coeffs()
            # check if keys are the same
            if set(self_copy.terms.keys()) != set(other_copy.terms.keys()):
                return False
            for key in self_copy.terms:
                if self_copy.terms[key] != other_copy.terms[key]:
                    return False
            return True
        else:
            raise TypeError("quantum_operator.__eq__: other must be a quantum_operator")

    def __ne__(self, 
            other: QuantumOperator
            ) -> bool:
        """
        Return True if self is not equal to other.

        Parameters
        ----------
        other : QuantumOperator
            Quantum operator to compare to self

        Returns
        -------
        bool
            True if self is not equal to other
        """
        return not self.__eq__(other)

    def add_term(self, 
                pauli_operator: PauliOperator
                ) -> None:
        """
        Add a pauli operator to the quantum operator.

        Parameters
        ----------
        pauli_operator : PauliOperator
            Pauli operator to add to the quantum operator
        """
        if isinstance(pauli_operator, PauliOperator):
            pauli_list = pauli_operator.split_MP()
            for pop in pauli_list:
                # check if pauli operator is already in the dictionary
                if pop.pauli_type in self.terms:
                    self.terms[pop.pauli_type] += pop
                # if pauli operator is not in the quantum operator, add it
                else:
                    self.terms[pop.pauli_type] = pop
        else:
            raise TypeError("quantum_operator.add_term: pauli_operator must be a pauli_operator")

    def add_terms(self, 
                pauli_operators: list
                ) -> None:
        """
        Add a list of pauli operators to the quantum operator.

        Parameters
        ----------
        pauli_operators : list
            List of pauli operators to add to the quantum operator
        """
        if isinstance(pauli_operators, list):
            for pauli_operator in pauli_operators:
                self.add_term(pauli_operator)
        else:
            raise TypeError("quantum_operator.add_terms: pauli_operators must be a list of pauli_operators, use add_term instead")

    def __add__(self, 
                other: QuantumOperator
                ) -> QuantumOperator:
        """
        Return the sum of two quantum operators.

        Parameters
        ----------
        other : QuantumOperator
            Quantum operator to add to self

        Returns
        -------
        QuantumOperator
            Sum of self and other
        """
        # create a copy of the first quantum operator
        sum_op = self.copy()
        # add the terms of the other quantum operator
        if isinstance(other, QuantumOperator):
            for key in other.terms:
                pauliop = other.terms[key]
                sum_op.add_term(pauliop)
            return sum_op
        else:
            raise TypeError("quantum_operator.__add__: other must be a quantum_operator")

    def __neg__(self) -> QuantumOperator:
        """
        Return the negation of the quantum operator.

        Returns
        -------
        QuantumOperator    
            Quantum operator with all coefficients negated
        """
        neg_op = QuantumOperator(N=self.N)
        for key in self.terms:
            pauliop = self.terms[key] 
            neg_op.add_term(-pauliop)
        return neg_op

    def __sub__(self, 
                other: QuantumOperator
                ) -> QuantumOperator:
        """
        Return the difference of two quantum operators.

        Parameters
        ----------
        other : QuantumOperator
            Quantum operator to subtract from self

        Returns
        -------
        QuantumOperator  
            Difference self + (-other)
        """

        return self + (-other)
    
    def __mul__(self, other) -> QuantumOperator:
        """
        Product of a QuantumOperator with a complex number or another QuantumOperator.

        Parameters
        ----------
        other : QuantumOperator or complex number
            Quantum operator or complex number to be multiplied with self

        Returns
        -------
        QuantumOperator
            product self * other
        """
        # multiply quantum operator with scalar
        if isinstance(other, (int,float,complex)):
            prod_op = self.copy()
            for key in prod_op.terms:
                prod_op.terms[key] *= other
            return prod_op
        # multiply quantum operator with another quantum operator
        elif isinstance(other, QuantumOperator):
            prod_op = QuantumOperator(N=self.N)
            for key1 in self.terms:
                pauliop1 = self.terms[key1]
                for key2 in other.terms:
                    pauliop2 = other.terms[key2]
                    prod_pauliop = pauliop1 * pauliop2
                    prod_op.add_term(prod_pauliop)
            return prod_op
        else:
            raise TypeError("quantum_operator.__mul__: other must be a quantum_operator or a scalar")

    def __rmul__(self, other) -> QuantumOperator:
        """
        Return right-multiplication of a quantum operator and a complex number.

        Parameters
        ----------
        other : complex number
            complex number to be multiplied with self

        Returns
        -------
        QuantumOperator  
            product other * self
        """
        return self.__mul__(other)

    def __truediv__(self, other) -> QuantumOperator:
        """
        Return division of a quantum operator and a complex number.

        Parameters
        ----------
        other : complex number
            complex number to be divided with self

        Returns
        -------
        QuantumOperator  
            quantum operator with coeff=self.coeff/other
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return self * (1/other)
        else:
            raise TypeError("quantum_operator.__truediv__: other must be a scalar")

    def __pow__(self, 
                power: int
                ) -> QuantumOperator:
        """
        Return power of a quantum operator

        Parameters
        ----------
        power : int
            power to which the quantum operator is raised

        Returns
        -------
        QuantumOperator  
            power of the quantum operator
        """
        if np.isclose(power, int(power)) and power > 0:
            if power == 1:
                return self
            else:
                return self * self**(power-1)
        else:
            raise TypeError("quantum_operator.__pow__: power must be a positive int")

    def commutator(self, 
                other: QuantumOperator
                ) -> QuantumOperator:
        """
        Return commutator of two quantum operators.

        Parameters
        ----------
        other : QuantumOperator
            Quantum operator to be commutated with self

        Returns
        -------
        QuantumOperator  
            commutator of self and other
        """
        if isinstance(other, PauliOperator):
            #convert to quantum operator
            other = QuantumOperator(N=other.N, terms={other.pauli_type: other.coeff})
        if isinstance(other, QuantumOperator):
            return self * other - other * self
        else:
            raise TypeError("quantum_operator.commutator: other must be a quantum_operator")

    def remove_terms(self, 
                    other: QuantumOperator | list
                    ) -> QuantumOperator:
        """
        Remove all terms of other from self.

        Parameters
        ----------
        other : QuantumOperator or list of QuantumOperators

        Returns
        -------
        QuantumOperator
            self with all terms of other removed
        """
        if other is None:
            return self
        if isinstance(other, QuantumOperator):
            other = [other]
        for qop in other:
            for term in qop.terms:
                if term in self.terms:
                    del self.terms[term]
        return self

    def project_out(self, 
                    other: QuantumOperator
                    ) -> QuantumOperator:
        """ 
        Projects other out of self (removes other from self).
        
        A projection finds beta that minimizes ||self-beta*other||.
        beta is given by beta = <other|self>/||other||^2

        Parameters
        ----------
        other : QuantumOperator
            Quantum operator to be projected out from self

        Returns
        -------
        QuantumOperator
            projected quantum operator self-beta*other
        """
        # convert other to quantum operator
        if isinstance(other, PauliOperator):
            other = QuantumOperator(N=other.N, terms={other.pauli_type: other.coeff})
        if not isinstance(other, QuantumOperator):
            raise TypeError("QuantumOperator.project: other must be a QuantumOperator or PauliOperator")
        # set ansatz (to get coefficient vectors)
        tmpop = 0*(self+other)
        # setup cost function
        self_vec = self.to_vector(ansatz=tmpop)
        other_vec = other.to_vector(ansatz=tmpop)
        beta = np.dot(np.conj(other_vec),self_vec)/np.linalg.norm(other_vec)**2
        proj = self - beta*other
        return proj

    def project_onto(self, 
                    other: QuantumOperator, 
                    projectors: list | None = None, 
                    fit_scale: bool = True
                    ) -> QuantumOperator:
        """
        Projects self onto other.

        Find operator as close as possible to other,
        that is a linear combination of self and (optional) projectors.
        Finds real numbers scale [optional] and beta that minimize
        ||scale*self - beta*projectors - other||.

        Parameters
        ----------
        other : QuantumOperator object
            quantum operator to be projected onto
        projectors : list of QuantumOperator objects, optional
            quantum operators to be projected out. Default is None.
        fit_scale : bool, optional
            if True, fits scale even if projectors are not None. Default is True.

        Returns
        -------
        QuantumOperator
            projection scale*self - beta*projectors
        """
        if projectors is None:
            # get ansatz (to get coefficient vectors)
            tmpop = 0*(self+other)
            v = self.to_vector(ansatz=tmpop)
            w = other.to_vector(ansatz=tmpop)
            scale = np.dot(np.conj(w),v)/np.linalg.norm(v)**2
            # scale = other.norm()/self.norm() * np.sign(other.coeffs()[0])*np.sign(self.coeffs()[0])
            return scale*self
        # set ansatz (to get coefficient vectors)
        tmpop = 0*(self+other+qop_sum(projectors))
        # setup cost function
        projcoeffs = [proj.to_vector(ansatz=tmpop) for proj in projectors]
        M = projcoeffs
        if fit_scale:
            M.append((-self.to_vector(ansatz=tmpop)))
        M = np.transpose(np.array(M))
        b = -other.to_vector(ansatz=tmpop)
        if not fit_scale:
            b = np.add(b,self.to_vector(ansatz=tmpop))
        # solve linear system
        beta_opt = np.linalg.lstsq(M,b,rcond=None)[0]
        # calculate projection
        proj_list = [beta_opt[inx]*projectors[inx] for inx in range(len(projectors))]
        proj = qop_sum(proj_list)
        scale = 1
        if fit_scale:
            scale = beta_opt[-1]
        projection = scale*self - proj
        return projection

    def range(self) -> int:
        """
        Return the maximal distance between qubits
        on which the QuantumOperator acts.

        Returns
        -------
        int 
            range of the QuantumOperator
        """
        max_dist = 0
        for key in self.terms:
            range = self.terms[key].range()
            if range > max_dist:
                max_dist = range
        return max_dist
    
    def support(self) -> list:
        """
        Return the indices of the qubits on which the QuantumOperator acts.

        Returns
        -------
        list 
            support of the QuantumOperator
        """
        support = []
        for key in self.terms:
            support.extend(self.terms[key].support())
        return list(set(support))

    def can_be_measured(
                    self, 
                    bases: list, 
                    no_output: bool = False, 
                    return_required_bases: bool = False
                    ) -> bool:
        """ 
        Return True, if the QuantumOperator can be measured in the given bases.

        Parameters
        ----------
        bases : list of strings
            list of measurement bases
        no_output : bool, optional
            If True, do not print output for terms that 
            cannot be measured in any given bases. Default is False.
        return_required_bases : bool, optional
            If True, return required_bases as well. Default is False.

        Returns
        -------
        can_be_measured : bool
            True if the QuantumOperator can be measured in the given bases
        required_bases : list of strings, optional
            list of bases required to measure the QuantumOperator
            only returned if return_required_bases=True
        """
        #--------------------------------#
        ### STEP 1 ### check if all terms can be measured in bases
        can_be_measured = True
        required_bases = []
        for term in self.terms.values():
            ## only check if term can be measured
            if not return_required_bases:
                can_be_measured_term = term.can_be_measured(bases, return_required_bases=False) #,no_output=no_output)
            ## save required bases
            else:
                can_be_measured_term, required_bases_term = term.can_be_measured(bases, return_required_bases=True) #,no_output=no_output)
                for basis in required_bases_term:
                    if basis not in required_bases: # and required_basis_term is not None:
                        required_bases.append(basis)
            ## update can_be_measured
            can_be_measured = can_be_measured and can_be_measured_term
            ## print terms that cannot be measured and break if return_required_bases=False
            if can_be_measured_term==False:
                if no_output==False:
                    print("term {} cannot be measured in bases {}".format(term.pauli_type,bases))
                if not return_required_bases:
                    break
        #--------------------------------#
        if return_required_bases:
            return can_be_measured, required_bases
        return can_be_measured

    def split_terms_into_bases(self, 
                            bases: list
                            ):
        """
        Split QuantumOperator into measurable and non-measurable terms.

        Return a dictionary of basis:terms pairs 
        where terms is a list of terms that can be measured in that basis.
        Also return the list of terms that cannot be measured in any given basis.

        Parameters
        ----------
        bases : list of strings
            list of measurement bases

        Returns
        -------
        terms_per_basis (dict)
            dictionary of basis:terms pairs where terms are the terms that can be measured in that basis
        non_measurable_terms (list of PauliOperator objects)
            list of terms that cannot be measured in any basis
        """
        terms_nonI = [term for term in self.terms.values() if term.pauli_type!="I"*self.N]
        all_bases = [basis for basis in bases if basis!="I"*self.N]
        #--------------------------------#
        ### STEP 1 ### split terms into measurable and non-measurable
        terms_per_basis = {basis:[] for basis in all_bases}
        non_measurable_terms = []
        for term in terms_nonI:
            measurable = False
            ## check if term can be measured in any basis
            measurable, bases = term.can_be_measured(all_bases, return_required_bases=True)
            ## save non-measurable terms
            if not measurable:
                non_measurable_terms.append(term)
            ## save measurable terms
            else:
                for basis in bases:
                    terms_per_basis[basis].append(term)
        #--------------------------------#
        return terms_per_basis, non_measurable_terms

    def get_measurement_bases(self, 
                            bases_given: list | None = None, 
                            no_output: bool = True
                            ):
        """
        Return a minimal set of measurement bases to measure the QuantumOperator.

        This function uses the graph-coloring algorithm proposed in https://arxiv.org/pdf/1907.03358.pdf.

        Parameters
        ----------
        bases_given : list of strings, optional
            list of measurement bases to be used.
            if sufficient, chooses minimal subset.
            if insufficient, extends to minimal sufficient set.
        no_output : bool, optional
            If True, no output is printed. Default is True.

        Returns
        -------
        bases : list of strings
            list of measurement bases to measure all terms in the QuantumOperator
        clique_cover : list of list of strings
            list of cliques that cover all terms in the QuantumOperator
        """
        qop = self.copy()
        required_terms = qop.terms.keys()
        ### take out all terms that can be measured in bases_given
        if bases_given is not None:
            required_terms_new = []
            for term in required_terms:
                can_be_measured = qop.terms[term].can_be_measured(bases_given)
                if not can_be_measured:
                # if not qop.terms[term].can_be_measured(bases_given):
                    required_terms_new.append(term)
            required_terms = required_terms_new
        if not no_output:
            if len(required_terms)!=0:
                print("Provided bases {} are not sufficient to measure all terms in the required operator {}.".format(bases_given, self.str()))
                print("Missing terms are: {}".format(required_terms))
                print("Already measured terms are: {}".format([term for term in qop.terms.keys() if term not in required_terms]))
            else:
                print("Provided bases {} are sufficient to measure all terms in the required operator {}.".format(bases_given, self.str()))
        ### create commutativity graph
        graph = nx.Graph()
        graph.add_nodes_from(required_terms)
        # add edges for pointwise-commuting terms  (only pointwise commuting can be measured in product basis)
        for op1, op2 in it.combinations(required_terms, 2):
            if pointwise_pauli_commutator(op1, op2) == 1:
                graph.add_edge(op1, op2)
        # invert graph
        inverted_graph = nx.complement(graph)
        # color inverted graph using greedy coloring
        # https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.coloring.greedy_color.html
        coloring = nx.algorithms.coloring.greedy_color(inverted_graph)
        # get clique covering from coloring
        clique_cover = []
        for color in np.unique(list(coloring.values())):
            clique_cover.append([op for op in coloring.keys() if coloring[op]==color])
        # get bases from clique cover
        bases = []
        for clique in clique_cover:
            base = ["I"]*len(clique[0])
            for pstr in clique:
                for cinx in range(len(pstr)):
                    if base[cinx] == "I" and pstr[cinx] != "I":
                        base[cinx] = pstr[cinx]
            bases.append("".join(base))
        if bases_given is not None:
            bases = bases_given + bases
        if not no_output:
            print("Minimal set of bases is {}, to measure all terms in the required operator {}".format(bases, self.str()))
        return bases, clique_cover

    def extend_to_larger_system(self, 
                                extension_factor: int
                                ) -> list:
        """
        Return the QuantumOperator extended to a larger system.

        This function returns a quantum operator acting on 
        a larger system, that consists of extension_factor many non-interacting 
        subsystems, each of the size of the original system.

        Parameters
        ----------
        extension_factor : int
            number of non-interacting subsystems

        Returns
        -------
        list of QuantumOperators
            quantum operators extended to larger system
        """
        qops_extended = []
        for sys_inx in range(extension_factor):
            qop =  QuantumOperator(N=self.N*extension_factor)
            for term in self.terms:
                term_extended_key = sys_inx*"I"*self.N + term + (extension_factor-sys_inx-1)*"I"*self.N
                term_extended = PauliOperator(N=self.N*extension_factor, coeff=self.terms[term].coeff, pauli_type=term_extended_key)
                qop.add_term(term_extended)
            qops_extended.append(qop)
        #-----------------------------------
        return qops_extended

    def flip_left_to_right(self) -> QuantumOperator:
        """
        Return the quantum operator with all indices flipped,
        such that the leftmost qubit becomes the rightmost qubit.

        Returns
        -------
        QuantumOperator
            Quantum operator with all indices flipped
        """
        qop_flipped = QuantumOperator(N=self.N)
        for term in self.terms:
            term_flipped = term[::-1]
            qop_flipped.terms[term_flipped] = self.terms[term]
        return qop_flipped



class Dissipator:
    """
    Dissipator (superoperator) for symbolic calculations.

    This class defines a dissipator (superoperator) as it appears in the 
    master equation in the Lindblad form.
    A dissipator is a sum of pairs of PauliOperators.

    Attributes
    ----------
    N : int
        number of qubits
    type : tuple
        type of the dissipator, e.g. ``("ZII","ZII")``
    coeff : complex number
        dissipation rate of the dissipator, e.g. ``1.0``
    """
    def __init__(self, 
                N: int, 
                diss_type: tuple, 
                coeff: complex = 1.0
                ):
        """
        Initialize a dissipator.

        Parameters
        ----------
        N : int
            number of qubits
        diss_type : tuple
            type of the dissipator, e.g. ``("ZII","ZII")``
        coeff : complex number, optional
            dissipation rate of the dissipator. Default is ``1.0``.
        """
        self._N = N
        self.diss_type = diss_type
        self.coeff = coeff
    ### ----------------- ###
    ### custom attributes ###
    ### ----------------- ###
    # read only N property
    @property
    def N(self):
        return self._N
    @property
    def type(self):
        return self._type
    @type.setter
    def type(self, diss_type):
        # check if diss_type is a tuple of length 2
        if len(diss_type) != 2:
            raise ValueError("type must be a tuple of length 2, but has length {}".format(len(diss_type)))
        # check if each string in diss_type has correct length
        if len(diss_type[0]) != self.N or len(diss_type[1]) != self.N:
            raise ValueError("type = {}, but must be a tuple of two strings of length Nions".format(diss_type))
        # check if each string in diss_type is a valid pauli string
        if not all([c in ["I", "X", "Y", "Z", "M", "P"] for c in diss_type[0]]) or not all([c in ["I", "X", "Y", "Z", "M", "P"] for c in diss_type[1]]):
            raise ValueError("type must be a valid pauli string")
        # set type
        self._type = diss_type
    @property
    def coeff(self):
        return self._coeff
    @coeff.setter
    def coeff(self, coeff):
        # if coeff is not None and coeff < 0:
        #     raise ValueError("coeff must be non-negativ or None, but is {}".format(coeff))
        # set coeff
        self._coeff = coeff
    ## --------------- ##
    ## custom methods  ##
    ## --------------- ##
    def copy(self) -> Dissipator:
        """
        Return a copy of the dissipator.

        Returns
        -------
        Dissipator
            A new instance with the same attributes as this dissipator.
        """
        return Dissipator(N=self.N, coeff=self.coeff, diss_type=self.diss_type)
    
    def __str__(self):
        return self.str()
    def str(self, 
            index: int | None = None, 
            sign: bool = False, 
            ndigits: int | None = None, 
            nonzero: bool = False
            ) -> str:
        """
        Return the string representation of the Dissipator.

        Parameters
        ----------
        index : int, optional
            If not None, use c(index) instead of coeff. Default is ``None``
        sign : bool, optional
            If True, add + or - sign to string. Default is ``False``
        ndigits : int, optional
            If not None, round coeff to ndigits decimals. Default is ``None``
        nonzero : bool, optional
            If True, only return string if coeff is not 0. Default is ``False``

        Returns
        -------
        str
            String representation of the dissipator
        """
        opstr = ""
        coeff = self.coeff
        # round coeff to ndigits decimals
        if ndigits is not None and coeff is not None:
            coeff = np.around(coeff, decimals=ndigits)
        # check if coeff is 0
        if nonzero and coeff is not None and np.isclose(coeff, 0):
            return ""
        if sign:
            if coeff is None:
                opstr += "+ "
            elif np.sign(coeff) == 1 or np.sign(coeff) == 0:
                opstr += "+ "
            elif np.sign(coeff) == -1:
                opstr += "- "
                coeff = -coeff
            else:
                raise ValueError("pauli_operator.__str__: sign must be + or -")
        if coeff is None:
            if index is None:
                opstr += "c(i)"
            else:
                opstr += "c({})".format(index)
        else:
            opstr += "{}*".format(coeff)
        opstr += self.diss_type[0]+","+self.diss_type[1]          
        return opstr
    
    def __eq__(self, 
            other: Dissipator
            ) -> bool:
        """
        Check if two dissipators have the same type and coefficient.

        Parameters
        ----------
        other : Dissipator
            dissipator object

        Returns
        -------
        bool
            True if the dissipators are equal, False otherwise.
        """
        # check if other is a dissipator
        if not isinstance(other, Dissipator):
            raise TypeError("dissipator.__eq__: other must be a dissipator")
        # check type
        typeval = self.diss_type==other.diss_type
        # check coefficient
        if self.coeff is None or other.coeff is None:
            coeffval = True
        else:
            coeffval = np.isclose(self.coeff, other.coeff)
        return typeval and coeffval

    def __ne__(self, 
            other: Dissipator
            ) -> bool:
        """
        Check if two dissipators are not equal.

        Parameters
        ----------
        other : Dissipator
            dissipator object

        Returns
        -------
        bool
            True if the dissipators are not equal, False otherwise
        """
        return not self.__eq__(other)

    def __call__(self, 
                operator: PauliOperator | QuantumOperator
                ) -> QuantumOperator:
        """
        Return the Dissipator acting on an operator.

        The dissipator acting on the operator results in the expression
        expr = l2dag * op * l1 - 1/2 * (op * l1dag * l2 + l1dag * l2 * op) + h.c.

        Parameters
        ----------
        operator : PauliOperator or QuantumOperator
            operator on which the dissipator acts

        Returns
        -------
        QuantumOperator
            Expression of the dissipator acting on operator
        """
        # transform PauliOperator to QuantumOperator
        if isinstance(operator, PauliOperator):
            operator = operator.to_quantum_operator()
        # get required operator
        pop1 = PauliOperator(N=self.N, coeff=1.0, pauli_type=self.diss_type[0]).to_quantum_operator()
        pop2 = PauliOperator(N=self.N, coeff=1.0, pauli_type=self.diss_type[1]).to_quantum_operator()
        pop1_dag = pop1.dagger()
        pop2_dag = pop2.dagger()
        # get terms
        term1 = pop2_dag * operator * pop1
        term2 = -1/2 * operator * pop1_dag * pop2
        term3 = -1/2 * pop1_dag * pop2 * operator
        term = term1 + term2 + term3
        # get expression
        expr = self.coeff * (term + term.dagger())
        return expr

    def __mul__(self, other) -> Dissipator:
        """
        Return the product of the dissipator with a real float.

        Parameters
        ----------
        other : real float
            real float to be multiplied with dissipator

        Returns
        -------
        Dissipator
            Product of Dissipator with float.
        """
        # check if other is real
        if not np.isclose(np.imag(other), 0):
            raise ValueError("dissipator.__mul__: other={}, but must be real".format(other))
        other = np.real(other)
        # scalar multiplication
        return Dissipator(N=self.N, coeff=self.coeff * other, diss_type=self.diss_type)

    def __rmul__(self, other) -> Dissipator:
        """
        Return the right-product of the dissipator with a real float.

        This is equivalent the standard left-product.

        Parameters
        ----------
        other : real float
            real float to be multiplied with dissipator

        Returns
        -------
        Dissipator
            Product of Dissipator with float.
        """
        return self.__mul__(other)

    def range(self) -> int:
        """
        Return the range of the Dissipator.

        The range of the dissipator is the maximum distance between 
        two qubits on which the Dissipator acts.

        Returns
        -------
        int 
            range of the dissipator
        """
        non_I_indices1 = [i for i, x in enumerate(self.diss_type[0]) if x != "I"]
        non_I_indices2 = [i for i, x in enumerate(self.diss_type[1]) if x != "I"]
        non_I_indices = non_I_indices1 + non_I_indices2
        # return the maximal distance between two non I entries
        maxran = 0
        if len(non_I_indices)>0:
            maxran = max(non_I_indices) - min(non_I_indices)
        return maxran
    
    def support(self) -> list:
        """
        Return the support of the Dissipator.

        The support of the dissipator is the set of qubits 
        on which the Dissipator acts.

        Returns
        -------
        list of int
            support of the dissipator
        """
        non_I_indices1 = [i for i, x in enumerate(self.diss_type[0]) if x != "I"]
        non_I_indices2 = [i for i, x in enumerate(self.diss_type[1]) if x != "I"]
        non_I_indices = non_I_indices1 + non_I_indices2
        return non_I_indices
    
    def center(self) -> float:
        """
        Return the center of a Dissipator.

        The center of the dissipator is the axial coordinate of the Dissipator
        given by center = (pos1+pos2)/2 - (N-1)/2

        Returns
        -------
        float
            center of the dissipator
        """
        # find index with non-I term
        non_I_indices1 = [i for i, x in enumerate(self.diss_type[0]) if x != "I"]
        non_I_indices2 = [i for i, x in enumerate(self.diss_type[1]) if x != "I"]
        non_I_indices = non_I_indices1 + non_I_indices2
        if len(non_I_indices) not in [1,2]:
            raise ValueError("dissipator.diss_type must contain exactly one or two non-I terms")
        if len(non_I_indices) == 1:
            center = non_I_indices[0]-(len(self.diss_type[0])-1)/2
        elif len(non_I_indices) == 2:
            center = (non_I_indices[0]+non_I_indices[1])/2-(len(self.diss_type[0])-1)/2
        return center

    def extend_to_larger_system(self, 
                                extension_factor: int
                                ) -> list:
        """
        Return the Dissipator extended to a larger system.

        This function returns a dissipator acting on 
        a larger system that consists of extension_factor many non-interacting 
        subsystems, each of the size of the original system.

        Parameters
        ----------
        extension_factor : int
            number of non-interacting subsystems

        Returns
        -------
        list of Dissipators
            list of dissipators for the larger system
        """
        ### STEP 1 ### extend individual terms
        dissipators_extended = []
        for sys_inx in range(extension_factor):
            type_extended = (sys_inx*"I"*self.N+self.diss_type[0]+(extension_factor-sys_inx-1)*"I"*self.N, sys_inx*"I"*self.N+self.diss_type[1]+(extension_factor-sys_inx-1)*"I"*self.N)
            diss_extended = Dissipator(N=self.N*extension_factor, coeff=self.coeff, diss_type=type_extended)
            dissipators_extended.append(diss_extended)
        #-----------------------------------
        return dissipators_extended



class CollectiveSpinOperator:
    """ 
    Collective spin operator for symbolic calculations.

    A collective spin operator is a sum of Pauli matrices.
    
    Attributes
    ----------
    N : int
        Number of qubits.
    type : string
        Type of the collective spin operator, among [``"X"``,``"Y"``,``"Z"``]
    coeff : complex number
        Coefficient of the collective spin operator, e.g. ``1.0``
    """
    def __init__(self,
                N: int, 
                spin_type: str, 
                coeff: complex = 1.0
                ):
        """
        Parameters
        ----------
        N : int
            Number of qubits.
        spin_type : string
            Type of the collective spin operator, among [``"X"``,``"Y"``,``"Z"``]
        coeff : complex number, optional
            Coefficient of the collective spin operator. Default is ``1.0``.
        """
        self._N = N 
        self.spin_type = spin_type 
        self.coeff = coeff
    ### ----------------- ###
    ### custom attributes ###
    ### ----------------- ###
    # read only N property
    @property
    def N(self):
        return self._N
    @property
    def spin_type(self):
        return self._spin_type
    @spin_type.setter
    def spin_type(self, spin_type):
        # check if spin_type is a valid collective string
        if spin_type not in ["X","Y","Z"]:
            raise ValueError("type is {}, but must be a valid string among X, Y, Z".format(spin_type))
        # set type
        self._spin_type = spin_type
    @property
    def coeff(self):
        return self._coeff
    @coeff.setter
    def coeff(self, coeff):
        # check if coeff is a number or None
        if not np.isscalar(coeff) and coeff is not None:
        # if not isinstance(coeff, (int, float, complex)) and coeff is not None:
            raise ValueError("coeff must be a number or None")
        self._coeff = coeff
    ## ----------------- ##
    ## custom functions  ##
    ## ----------------- ##
    def copy(self) -> CollectiveSpinOperator:
        """
        Return a copy of the CollectiveSpinOperator.

        Returns
        -------
        CollectiveSpinOperator
            A new instance with the same attributes as this operator.
        """
        return CollectiveSpinOperator(N=self.N, coeff=self.coeff, spin_type=self.spin_type)
    
    def __str__(self):
        return self.str()
    def str(self) -> str:
        """
        Return a string representation of the CollectiveSpinOperator.

        Returns
        -------
        str
            String representation of the CollectiveSpinOperator
        """
        return "{}*collective{}".format(self.coeff, self.spin_type)
    
    def to_quantum_operator(self) -> QuantumOperator:
        """
        Return a QuantumOperator representation of the CollectiveSpinOperator.

        Returns
        -------
        QuantumOperator
            QuantumOperator representation of the CollectiveSpinOperator
        """
        qop = QuantumOperator(self.N, terms={"I"*inx+self.spin_type+"I"*(self.N-inx-1): self.coeff for inx in range(self.N)})
        return qop

    def __mul__(self, other) -> CollectiveSpinOperator:
        """
        Return product of a CollectiveSpinOperator and a complex number.

        Parameters
        ----------
        other : complex number
            complex number to be multiplied with operator

        Returns
        -------
        CollectiveSpinOperator
            CollectiveSpinOperator with coeff=self.coeff*other
        """
        # check if other is a complex number
        if not isinstance(other, (int,float,complex)):
            raise TypeError("CollectiveSpinOperator.__mul__: other must be a complex number")
        # scalar multiplication
        return CollectiveSpinOperator(N=self.N, coeff=self.coeff * other, spin_type=self.spin_type)

    def __rmul__(self, other) -> CollectiveSpinOperator:
        """
        Return right-product of a complex number and a CollectiveSpinOperator.

        This is the same as the standard left-product.

        Parameters
        ----------
        other : complex number
            complex number to be multiplied with operator

        Returns
        -------
        CollectiveSpinOperator
            CollectiveSpinOperator with coeff=self.coeff*other
        """
        return self.__mul__(other)






#----------------------------#
### other helper functions ###
#----------------------------#
def pointwise_pauli_commutator(
                        pstr1: str,
                        pstr2: str
                        ) -> int:
    """
    Return the point-wise commutator of two pauli strings.

    This helper function returns ``1`` if the two Pauli strings
    commute, and 0 otherwise.

    Parameters
    ----------
    pstr1 : str
        first Pauli string
    pstr2 : str
        second Pauli string

    Returns
    -------
    int
        1 if the Pauli strings commute, 0 otherwise
    """
    # check if pauli strings are of same length
    if len(pstr1) != len(pstr2):
        raise ValueError("pauli strings are of different lengths.")
    # check pauli strings char by char
    for cinx in range(len(pstr1)):
        if pstr1[cinx] != pstr2[cinx] and pstr1[cinx] != "I" and pstr2[cinx] != "I":
            return 0
    return 1

def pops_to_qop(pops) -> QuantumOperator:
    """
    Returns a QuantumOperator as a sum of Pauli operators from a list.

    Parameters
    ----------
    pops : list of PauliOperator objects

    Returns
    -------
    QuantumOperator
        QuantumOperator object
    """
    # check if pops is a Pauli operator
    if isinstance(pops,PauliOperator):
        pops = [pops]
    # check if pops is a list of Pauli operators
    if not isinstance(pops,list):
        raise TypeError("pops_to_qop: pops must be a list of PauliOperator objects")
    for pop in pops:
        if not isinstance(pop,PauliOperator):
            raise TypeError("pops_to_qop: pops must be a list of PauliOperator objects")
    # create quantum operator
    qop = QuantumOperator(N=pops[0].N)
    # add terms to quantum operator
    for pop in pops:
        qop.add_term(pop)
    return qop

def get_expval_from_Zproduct_state(qop, 
                                state: str
                                ) -> float:
    """
    Return the expectation value of a quantum operator in a product state in the Z-basis.

    Parameters
    ----------
    qop : PauliOperator or QuantumOperator
        quantum operator to take the expectation value of
    state : str
        excitations of product state in the Z-basis

    Returns
    -------
    float
        expectation value of qop in state
    """
    # check if qop is a quantum operator
    if isinstance(qop, PauliOperator):
        qop = QuantumOperator(N=qop.N, terms={qop.pauli_type: qop.coeff})
    elif not isinstance(qop, QuantumOperator):
        raise TypeError("get_expval_from_product_state: qop must be either a PauliOperator or a quantum_operator")
    expval_total = 0
    for pstr, pop in qop.terms.items():
        # check if term has a non-z index
        if "X" in pstr or "Y" in pstr:
            expval = 0
        else:
            expval = 1
            for inx in range(len(pstr)):
                if pstr[inx] == "Z":
                    expval *= -(-1)**int(state[inx])
        expval_total += pop.coeff * expval
    return expval_total

def qop_sum(qoplist, 
            axis: int | None = None
            ) -> QuantumOperator:
    """
    Calculate the sum of a list of QuantumOperators.

    This function takes a list of QuantumOperators and returns a QuantumOperator
    that is the sum of all QuantumOperators in qoplist.

    Parameters
    ----------
    qoplist : list of QuantumOperators
        list of QuantumOperators to be summed
    axis : int, optional
        axis along which the sum is calculated. Default is None.

    Returns
    -------
    QuantumOperator
        QuantumOperator that is the sum of all QuantumOperators in qoplist
    """
    #-----------------------------------
    ### STEP 1 ### calculate sum
    if axis is None:
        sum = QuantumOperator(N=qoplist[0].N)
        for qop in qoplist:
            sum += qop
    else:
        # check if qoplist is a list of lists
        if not isinstance(qoplist[0], list):
            raise TypeError("qop_sum: axis == 1 but qoplist is not a list of lists")
        # check if all lists in qoplist have the same length
        for qop in qoplist:
            if len(qop) != len(qoplist[0]):
                raise ValueError("qop_sum: axis == 1 but not all lists in qoplist have the same length")
        # calculate sum
        sum = []
        if axis == 0:
            for inx in range(len(qoplist[0])):
                sum.append(qop_sum([qop[inx] for qop in qoplist], axis=None))
        elif axis == 1:
            for inx in range(len(qoplist)):
                sum.append(qop_sum(qoplist[inx], axis=None))
        else:
            raise ValueError("qop_sum: axis must be 0, 1 or None")
    return sum

def qop_mean(qoplist, 
            axis: int | None = None
            ) -> QuantumOperator:
    """
    Calculate the mean of a list of quantum operators.

    Parameters
    ----------
    qoplist : list of Quantum operator objects
        operators to be averaged
    axis : int, optional
        axis along which the mean is calculated. Default is None.

    Returns
    -------
    QuantumOperator
        mean of the quantum operators
    """
    if axis is None:
        mean = QuantumOperator(N=qoplist[0].N)
        for qop in qoplist:
            mean += qop
        mean /= len(qoplist)
    else:
        # check if qoplist is a list of lists
        if not isinstance(qoplist[0], list):
            raise TypeError("qop_mean: axis == 1 but qoplist is not a list of lists")
        # check if all lists in qoplist have the same length
        for qop in qoplist:
            if len(qop) != len(qoplist[0]):
                raise ValueError("qop_mean: axis == 1 but qoplist contains lists of different length")
        # calculate mean
        mean = []
        if axis == 0:
            for inx in range(len(qoplist[0])):
                mean.append(qop_mean([qop[inx] for qop in qoplist]))
        elif axis == 1:
            for inx in range(len(qoplist)):
                mean.append(qop_mean(qoplist[inx]))
        else:
            raise ValueError("qop_mean: axis must be 0 or 1")
    return mean

# TODO: axis option does not work yet
def qop_var(qoplist, 
            mean: QuantumOperator | None = None, 
            axis: int | None = None,
            ) -> QuantumOperator:
    """
    Calculate the variance in the coefficients of a list of quantum operators.

    Parameters
    ----------
    qoplist : list of QuantumOperator objects
        quantum operators to be taken the variance of
    mean : QuantumOperator, optional
        If not given, calculates the mean of the quantum operators. Default is None.
    axis : int, optional
        axis along which the variance is calculated. Default is None.
    
    Returns
    -------
    QuantumOperator
        variance of the quantum operators
    """
    # calculate mean if not given
    if mean is None:
        mean = qop_mean(qoplist, axis=axis)
    var = mean.copy()
    if axis is None:
        var.set_coeffs_zero()
        for qop in qoplist:
            tmp_coeffs = np.power(qop.coeffs()-mean.coeffs(),2)
            var.set_coeffs(var.coeffs()+tmp_coeffs)
        var /= len(qoplist)
    else:
        # check if qoplist is a list of lists
        if not isinstance(qoplist[0], list):
            raise TypeError("qop_var: axis == 1 but qoplist is not a list of lists")
        # check if all lists in qoplist have the same length
        for qop in qoplist:
            if len(qop) != len(qoplist[0]):
                raise ValueError("qop_var: axis == 1 but qoplist contains lists of different length")
        # calculate var
        var = []
        if axis == 0:
            for inx in range(len(qoplist[0])):
                var.append(qop_var([qop[inx] for qop in qoplist], mean=mean[inx]))            
        elif axis == 1:
            for inx in range(len(qoplist)):
                var.append(qop_var(qoplist[inx], mean=mean[inx]))
        else:
            raise ValueError("qop_var: axis must be 0 or 1")
    return var

def qop_relvar(qoplist) -> QuantumOperator:
    """
    Calculate the relative variance in the coefficients of a list of quantum operators.

    The relative variance is defined as the variance divided by the mean**2.

    Parameters
    ----------
    qoplist : list of Quantum operator objects
        list of operators to be taken the relative variance of
        
    Returns
    -------
    QuantumOperator
        relative variance of the operators
    """
    # calculate mean and variance
    mean = qop_mean(qoplist)
    var = qop_var(qoplist, mean=mean)
    # calculate relative variance
    relvar = var.copy()
    relvar.set_coeffs_zero()
    for termkey in var.terms.keys():
        if var.terms[termkey].coeff != 0:
            relvar.terms[termkey].coeff = var.terms[termkey].coeff/mean.terms[termkey].coeff**2
        else:
            del relvar.terms[termkey]
    return relvar   

# slow but good
def get_nullspace(A):
    """ 
    Calculate the nullspace of a matrix A.

    Parameters
    ----------
    A : np-array

    Results
    -------
    np-array
        the nullspace of the matrix A
    """
    import sympy as symp
    matrix = symp.Matrix(A)
    nullspace_sympy = matrix.nullspace()
    # transform to numpy arrays
    nullspace = []
    for vec_sympy in nullspace_sympy:
        # print("vec_sympy", vec_sympy)
        vec = np.array(vec_sympy).astype(np.float64)
        vec = vec.transpose()[0]
        # print("vec", vec)
        nullspace.append(vec)
    # nullspace = np.array(nullspace)
    # print("nullspace", nullspace)
    # nullspace = nullspace.transpose()
    return nullspace

def find_conserved_quantities(qops_list, 
                            ansatz: QuantumOperator | None = None
                            ) -> list[QuantumOperator]:
    """
    Return the conserved quantities of a list of quantum operators.

    Parameters
    ----------
    qop_list : list of QuantumOperator objects
    ansatz : Quantum Operator

    Returns
    -------
    list of QuantumOperator objects
    """
    # check if qops_list is a list of quantum operators
    if not all([isinstance(qop, QuantumOperator) for qop in qops_list]):
        raise TypeError("find_conserved_quantities: qops_list must be a list of quantum operators")
    # check if all quantum operators have the same number of ions
    if not all([len(list(qop.terms.keys())[0]) == len(list(qops_list[0].terms.keys())[0]) for qop in qops_list]):
        raise ValueError("find_conserved_quantities: all quantum operators must have the same number of ions")
    # get number of ions
    N = len(list(qops_list[0].terms.keys())[0])
    # create a list of all pauli operators for Nions ions
    pauli_chars = ["I", "X", "Y", "Z"]
    if ansatz is None:
        pstr_list = ["".join(pstr_prod) for pstr_prod in it.product(pauli_chars, repeat=N)]
    else:
        pstr_list = ["I"*N] + list(ansatz.terms.keys())
    # get vector of commutators of all pauli operators in pstr_list with qop (for each qop)
    comm_matrix_parts = []
    for qop in qops_list:
        commutators = []
        for pstr in pstr_list:
            commutators.append(qop.commutator(PauliOperator(N=N, coeff=1.0, pauli_type=pstr)))
        comm_matrix_parts.append(np.array([comm.to_vector() for comm in commutators]).transpose())
    comm_matrix = np.concatenate(comm_matrix_parts, axis=0)
    # comm_matrix = np.array([comm.to_vector() for comm in commutators]).transpose()
    # get rank of comm_matrix
    rank = np.linalg.matrix_rank(comm_matrix)
    # print("rank", rank)
    # print("shape comm_matrix", comm_matrix.shape)
    ##NULLSPACE1 via LU decomposition
    # print("shape comm_matrix", comm_matrix.shape)
    null = get_nullspace(comm_matrix)
    conserved = []
    for sol in null:
        # igonore if fist element of sol is 1
        if np.isclose(sol[0], 1):
            continue
        # print("sol", sol)
        qop = QuantumOperator(N=N)
        qop.from_vector(sol, ansatz=ansatz)
        conserved.append(qop)
    return conserved
