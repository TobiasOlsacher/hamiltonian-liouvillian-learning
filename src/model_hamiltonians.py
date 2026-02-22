"""
This file contains functions to generate model Hamiltonians
as QuantumOperator objects.
functions:
    - get_hamiltonian
        Get a Hamiltonian with arbitrary single- and two- and three-qubit interactions
    - get_dissipators
        Get a list of dissipators with jump operators of type sigma_alpha where alpha can be any of x, y, z, p, m.
"""
from src.pauli_algebra import PauliOperator, QuantumOperator, Dissipator
import itertools as it
import numpy as np

def get_hamiltonian(Nions: int, 
                    terms: dict,
                    cutoff: int | None = None, 
                    add_noise: float = 0, 
                    flip_left_to_right: bool = False
                    ) -> QuantumOperator:
    """
    Returns a Hamiltonian with arbitrary 
    single- and two- and three-qubit interactions
    as a QuantumOperator object.

    Parameters
    ----------
    Nions : int
        Number of qubits in the system.
    terms : dict
        Dictionary with coupling strengths for different terms in the Hamiltonian.
        Keys can be "Bx", "By", "Bz", "Jxx", "Jyy", "Jzz", "Jxy", "Jxz", "Jyz", "Jxxx", "Jyyy", "Jzzz", "Jxyz", "Jxxy", "Jyyz", "Jxzz", "Jyzz".
        Values must be vectors of length Nions for fields (Bx,By,Bz),
        square matrices of size (Nions,Nions) for 2-qubit couplings (Jxx,Jyy,Jzz,Jxy,Jxz,Jyz) and
        3D tensors of size (Nions,Nions,Nions) for 3-qubit couplings (Jxxx,Jyyy,Jzzz,Jxyz,Jxxy,Jyyz,Jxzz,Jyzz).
    cutoff : int, optional
        Cutoff for the range of terms in the Hamiltonian
        All coefficients beyond cutoff are set to 0 and excluded from the Hamiltonian
        Set cutoff for the parametrization (e.g. 3).
        If cutoff is 0, no cutoff is applied.
        If cutoff is positive, coefficients are set to 0 for distances larger than cutoff.
        If cutoff is negative, coefficients are set to 0 for distances smaller or equal to -cutoff.
        Default is None.
    add_noise : float, optional
        Add Gaussian noise to the Hamiltonian coefficients with a standard deviation of add_noise.
        Default is ``0``.
    flip_left_to_right : bool, optional
        If True, flips the order of the terms in the Hamiltonian, 
        i.e., first becomes last and last becomes first.
        Default is False.

    Returns
    -------
    QuantumOperator
        The Hamiltonian as a QuantumOperator object 
        with terms given by a dict in Hamiltonian.terms.
    """
    if cutoff is None:
        cutoff = Nions
    fields = ["Bx", "By", "Bz"]
    couplings2 = ["Jxx", "Jyy", "Jzz", "Jxy", "Jxz", "Jyz"]
    couplings3 = ["J{}{}{}".format(a,b,c) for a,b,c in it.product(["x","y","z"], repeat=3)]
    # -----------------------------------
    ### STEP 1 ### verify input
    for fieldstr in fields:
        if fieldstr in terms.keys():
            field = terms[fieldstr]
            # check dimension of field
            if len(field) != Nions:
                raise ValueError("field must be a vector of length Nions")
    for couplingstr in couplings2:
        if couplingstr in terms.keys():
            coupling = terms[couplingstr]
            # check dimension of coupling
            if coupling.shape != (Nions, Nions):
                raise ValueError("coupling must be a square matrix of size (Nions,Nions)")
            # check if diagonal elements of coupling are zero
            if not np.allclose(np.diag(coupling), np.zeros(Nions)):
                raise ValueError("coupling must have zero diagonal elements")
            # check if coupling is symmetric or upper triangular (i<j)
            if not np.allclose(coupling, coupling.T) and not np.allclose(coupling, np.triu(coupling)):
                raise ValueError("coupling must be symmetric or upper triangular")
    for couplingstr in couplings3:
        if couplingstr in terms.keys():
            coupling = terms[couplingstr]
            # check dimension of coupling
            if coupling.shape != (Nions, Nions, Nions):
                raise ValueError("coupling must be a 3D tensor of size (Nions,Nions,Nions)")
    # -----------------------------------
    # ### STEP 2 ### add noise to coeffs
    if add_noise != 0:
        terms = {k: np.array(v, copy=True) for k, v in terms.items()}
        for fieldstr in fields:
            if fieldstr in terms.keys():
                terms[fieldstr] = terms[fieldstr] + np.multiply(add_noise*terms[fieldstr], np.random.randn(Nions))
        for couplingstr in couplings2:
            if couplingstr in terms.keys():
                terms[couplingstr] = terms[couplingstr] + np.multiply(add_noise*terms[couplingstr], np.random.randn(Nions, Nions))
        for couplingstr in couplings3:
            if couplingstr in terms.keys():
                terms[couplingstr] = terms[couplingstr] + np.multiply(add_noise*terms[couplingstr], np.random.randn(Nions, Nions, Nions))
    # -----------------------------------
    ### STEP 3 ### construct Hamiltonian
    Ham = QuantumOperator(N=Nions)
    ### fields
    for fieldstr in fields:
        if fieldstr in terms.keys():
            field = terms[fieldstr]
            for i in range(Nions):
                paulistr = ["I"]*Nions
                paulistr[i] = fieldstr[1].upper()
                if flip_left_to_right:
                    paulistr = paulistr[::-1]
                pauliop = PauliOperator(N=Nions, coeff=complex(field[i]), pauli_type="".join(paulistr))
                Ham.add_term(pauliop)
    ### 2-qubit couplings
    for couplingstr in couplings2:
        if couplingstr in terms.keys():
            coupling = terms[couplingstr]
            for i in range(Nions):
                for j in range(Nions):
                    if i!=j:
                        if (cutoff>0 and np.abs(i-j)>cutoff) or (cutoff<0 and np.abs(i-j)<=-cutoff):
                            continue
                        paulistr = ["I"]*Nions
                        paulistr[i] = couplingstr[1].upper()
                        paulistr[j] = couplingstr[2].upper()
                        if flip_left_to_right:
                            paulistr = paulistr[::-1]
                        pauliop = PauliOperator(N=Nions, coeff=complex(coupling[i,j]), pauli_type="".join(paulistr))
                        Ham.add_term(pauliop)
    ### 3-qubit couplings
    for couplingstr in couplings3:
        if couplingstr in terms.keys():
            coupling = terms[couplingstr]
            for i in range(Nions):
                for j in range(Nions):
                    for k in range(Nions):
                        if i<j and j<k:  #and np.abs(i-k)<=cutoff:
                            if (cutoff>0 and np.abs(i-k)>cutoff) or (cutoff<0 and np.abs(i-k)<=-cutoff):
                                continue
                            paulistr = ["I"]*Nions
                            paulistr[i] = couplingstr[1].upper()
                            paulistr[j] = couplingstr[2].upper()
                            paulistr[k] = couplingstr[3].upper()
                            if flip_left_to_right:
                                paulistr = paulistr[::-1]
                            pauliop = PauliOperator(N=Nions, coeff=complex(coupling[i,j,k]), pauli_type="".join(paulistr))
                            Ham.add_term(pauliop)
    # -----------------------------------
    if len(Ham.terms) == 0:
        return None
    return Ham

def get_dissipators(Nions: int, 
                    dissipation_rates: dict, 
                    cutoff: int = 1,
                    ) -> list[Dissipator]:
    """
    Returns a list of Dissipator objects for given dissipation rates.

    Only uses upper triangular part of the dissipation rates.

    Parameters
    ----------
    Nions : int
        Number of qubits in the system.
    dissipation_rates : dict
        Dictionary with dissipation rates for different Lindblad operators.
        Keys can be ``"Gx", "Gy", "Gz", "Gp", "Gm"``.
        Values must be square matrices of size (Nions,Nions).
    cutoff : int, optional
        Cutoff for the range of terms in the Hamiltonian
        All coefficients beyond cutoff are set to ``0`` and excluded from the Hamiltonian
        Set cutoff for the parametrization (e.g. ``3``).
        If cutoff is ``0``, no cutoff is applied.
        If cutoff is positive, coefficients are set to ``0`` for distances larger than cutoff.
        If cutoff is negative, coefficients are set to ``0`` for distances smaller or equal to -cutoff.
        Default is ``1``.

    Returns
    -------
    list
        List of ``Dissipator`` objects.
    """
    # -----------------------------------
    ### STEP 1 ### verify input
    allowed_disstypes = ["Gx","Gy","Gz","Gp","Gm"]
    for disstype, Gdiss in dissipation_rates.items():
        if disstype not in allowed_disstypes:
            raise ValueError("dissipation_rate {} not allowed, must be one of {}".format(disstype, allowed_disstypes))
        if Gdiss.shape != (Nions, Nions):
            raise ValueError("{} must be a square matrix of size (Nions,Nions)".format(disstype))
        if not np.allclose(Gdiss, Gdiss.T) and not np.allclose(Gdiss, np.triu(Gdiss)):
            raise ValueError("{} must be symmetric or upper triangular".format(disstype))
    # -----------------------------------
    ### STEP 2 ### construct dissipators
    Dsim = []
    for disstype, Gdiss in dissipation_rates.items():
        disschar = disstype[1].upper()
        if np.allclose(Gdiss, np.zeros((Nions,Nions))):
            print(f"Warning: {disstype} is zero matrix. Skipping...")
            continue
        for inx,iny in it.combinations_with_replacement(range(Nions),2):
            if Gdiss[inx,iny] == 0 or Gdiss[inx,iny] is None:
                continue
            if (cutoff>0 and np.abs(inx-iny)>cutoff) or (cutoff<0 and np.abs(inx-iny)<=-cutoff):
                continue
            dtype = ["I"*inx + disschar + "I"*(Nions-inx-1), "I"*iny + disschar + "I"*(Nions-iny-1)]
            diss = Dissipator(N=Nions, diss_type=dtype, coeff=Gdiss[inx,iny])
            Dsim.append(diss)
    # -----------------------------------
    if len(Dsim) == 0:
        return None
    return Dsim