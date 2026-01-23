"""
This module contains methods for quantum states.
classes:
    - QuantumState
        A class to represent the initial state of the quantum simulator.
"""
from __future__ import annotations
import numpy as np
import qutip as qt
from icecream import ic
from src.pauli_algebra import PauliOperator, QuantumOperator


class QuantumState:
    """
    Initial state for the quantum simulator.

    Attributes
    ----------
    N : int
        number of qubits in the system
    excitations : str
        bitstring representation of the state in the given basis
        Here, "0" is the down state and "1" is the up state.
    basis : str, optional
        basis in which the state is defined
        Default is the computational basis "z"*N.
    state_preparation : list of str, optional
        After initialization, a product of single-qubit unitary operators
        defined by a list of tuples is applied to the initial state.
        Each tuple in the list is of the form 
        [gatestr, float] where gatestr is one of ["Rx{}", "Ry{}", "Rz{}"] 
        with {} the ion index, and float is the angle,
        or ["qop", qop] where qop is a QuantumOperator object.
        Default is None.
    state_preparation_label : str, optional 
        label for the state preparation used for hashing
        Default is None.
    state_preparation_error : float, optional
        Probability of a bit-flip error in the initial state.
        Default is None.
    qutip_state : QuTip quantum state, optional
        QuTip quantum state object
        Default is None.
    label : str
        label for the QuantumState object used for str representation and hashing
        Default is None.
    """
    def __init__(
            self, 
            N: int, 
            excitations: str | None = None, 
            basis: str | None = None,
            state_preparation: list | None = None,
            state_preparation_label: str | None = None,
            state_preparation_error: float | None = None,
            qutip_state: qt.Qobj | None = None,
            label: str | None = None,
            ):
        """
        Parameters
        ----------
        N : int
            number of qubits in the system
        excitations : str
            bitstring representation of the state in the given basis
            Here, "0" is the down state and "1" is the up state.
        basis : str, optional
            basis in which the state is defined
            Default is the computational basis "z"*N.
        state_preparation : list of str, optional
            After initialization, a product of single-qubit unitary operators
            defined by a list of tuples is applied to the initial state.
            Each tuple in the list is of the form 
            [gatestr, float] where gatestr is one of ["Rx{}", "Ry{}", "Rz{}"] 
            with {} the ion index, and float is the angle,
            or ["qop", qop] where qop is a QuantumOperator object.
            Default is None.
        state_preparation_label : str, optional 
            label for the state preparation used for hashing
            Default is None.
        state_preparation_error : float, optional
            Probability of a bit-flip error in the initial state.
            Default is None.
        qutip_state : QuTip quantum state, optional
            QuTip quantum state object
            Default is None.
        label : str
            label for the QuantumState object used for str representation and hashing
            Default is None.
        """
        self._N = N
        self.excitations = excitations
        self.basis = basis
        if basis is None:
            self.basis = "z"*N
        self.state_preparation = state_preparation
        self.state_preparation_label = state_preparation_label
        self.state_preparation_error = state_preparation_error
        self.qutip_state = qutip_state
        self.label = label
    ################################
    ### QuantumState attributes ###
    ################################
    @property
    def N(self):
        return self._N
    @N.setter
    def N(self, value):
        raise AttributeError("N is a read-only property.")
    @property
    def excitations(self):
        return self._excitations
    @excitations.setter
    def excitations(self, value):
        if value is not None:
            # check if value is a string of length Nions
            if not isinstance(value, str) or len(value) != self.N:
                raise TypeError("excitations = {}, but must be a string of length N={}.".format(value, self.N))
            # check if value contains only "0", "1"
            if not all([x in ["0", "1"] for x in value]):
                raise ValueError("excitations = {}, but must contain only '0', '1'.".format(value))
        self._excitations = value
    @property
    def basis(self):
        return self._basis
    @basis.setter 
    def basis(self, value):
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("basis is not a string.")
            # check if basis is of length Nions
            if len(value) != self.N:
                raise ValueError("basis = {}, but must be a string of length N={}.".format(value, self.N))
            # check if basis contains only "x", "y", "z"
            if not all([x in ["x", "y", "z"] for x in value]):
                raise ValueError("basis = {}, but must contain only 'x', 'y', 'z'.".format(value))
        self._basis = value
    @property
    def state_preparation(self):
        return self._state_preparation
    @state_preparation.setter
    def state_preparation(self, value):
        if value is not None:
            ## check if value is a list
            if not isinstance(value, list):
                raise TypeError("state_preparation is not a list.")
            ## check individual gates
            for gate in value:
                if gate[0][0:2] in ["Rx", "Ry", "Rz"]:
                    ic(gate)
                    ioninx = int(gate[0][2:])
                    if ioninx >= self.N:
                        raise ValueError("ion index {} in state_preparation is larger than N={}.".format(ioninx, self.N))
        self._state_preparation = value
    @property
    def state_preparation_label(self):
        return self._state_preparation_label
    @state_preparation_label.setter
    def state_preparation_label(self, value):
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("state_preparation_label is {}, but must be a string.".format(value))
            # check if lowercase
            if not value.islower():
                raise ValueError("state_preparation_label is {}, but must be lowercase.".format(value))
        self._state_preparation_label = value
    @property
    def state_preparation_error(self):
        return self._state_preparation_error
    @state_preparation_error.setter
    def state_preparation_error(self, value):
        if value is not None:
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                raise TypeError("state_preparation_error is not a number between 0 and 1.")
        self._state_preparation_error = value
    ### --------------------- ###
    ### QuantumState methods ###
    ### --------------------- ###
    def copy(self) -> QuantumState:
        """ 
        Return a copy of the quantum state.

        Returns
        -------
        QuantumState
            A new instance with the same attributes as this quantum state.
        """
        return QuantumState(self.N, self.excitations, self.basis, self.state_preparation, self.state_preparation_label, self.state_preparation_error, self.qutip_state, self.label)
    
    def __eq__(self, 
            other: QuantumState
            ) -> bool:
        """
        Equality relation of two quantum states.

        Two quantum states are equal if they have the same attributes.
        TODO: state_preparation_error not included in eq 

        Parameters
        ----------
        other : QuantumState
            quantum state to compare to

        Returns
        -------
        bool
            whether the two quantum states have the same attributes
        """
        if not isinstance(other, QuantumState):
            return False
        return self.N == other.N and self.excitations == other.excitations and self.basis == other.basis and self.state_preparation == other.state_preparation and self.state_preparation_label == other.state_preparation_label # and self.state_preparation_error == other.state_preparation_error

    def __hash__(self) -> int:
        """
        Hash of the quantum state.
        
        TODO: state_preparation_error not included in hash

        Returns
        -------
        int
            hash of the quantum state
        """
        return hash((self.N, self.excitations, self.basis, self.state_preparation_label, self.label))

    def __str__(self):
        return self.str()
    def str(self) -> str:
        """
        Return the string representation of the quantum state

        Returns
        -------
        str
            string representation of the quantum state
        """
        exc = self.excitations
        base = self.basis
        state_str = None
        try:
            if self.qutip_state is not None:
                if self.label is not None:
                    state_str = "|{}⟩".format(self.label)
                else:
                    state_str = "|QUTIP STATE⟩"
        except:
            pass
        if state_str is None:
            # get string that is exc1base1,exc2base2,...
                state_str = "|" + "".join([base[inx]+exc[inx] + "," for inx in range(self.N)])[:-1] + "⟩"
                if self.state_preparation_label is not None:
                    state_str = "R({})*{}".format(self.state_preparation_label,state_str)
                elif self.state_preparation is not None:
                    stprep_str = "".join([stprep[0]+"*" for stprep in self.state_preparation[::-1] if stprep[1] != 0])
                    state_str = "{}{}".format(stprep_str,state_str)
        #--------------------------------
        return state_str

    # TODO add state preparation error
    def to_QuTip(self,
                apply_state_preparation: bool = True,
                apply_state_preparation_error: bool = True,
                ):
        """
        Return QuTip quantum state object from QuantumState,
        including state preparation and state preparation errors.

        Parameters
        ----------
        apply_state_preparation : bool
            if True, applies state preparation to the initial state
            Default is True.
        apply_state_preparation_error : bool
            if True, applies state preparation error to the initial state
            Default is True.

        Returns
        -------
        Qutip QuantumState or Qutip QuantumOperator
            Qutip quantum state or density matrix
        """
        ### STEP 1 ### get QuTip state
        if self.qutip_state is not None:
            psi0 = self.qutip_state
        else:
            psi0 = getQuantumState_QuTip(self.excitations, basis=self.basis, state_preparation_error=self.state_preparation_error)
        # -----------------------------------------------
        ### STEP 2 ### apply state preparation
        if apply_state_preparation and self.state_preparation is not None:
            state_preparation_QuTip = getQuantumGate_QuTip(self.state_preparation, self.N)
            # check if psi0 is state or density matrix
            if psi0.type == "ket":
                psi0 = state_preparation_QuTip*psi0
            else:
                psi0 = state_preparation_QuTip*psi0*state_preparation_QuTip.dag()
            # check if psi0 contains nan
            if np.isnan(np.array(psi0.full())).any():
                raise ValueError("qutip quantum state after state preparation contains nan.")
        # --------------------------------
        ### STEP 3 ### apply state preparation error
        # TODO: already set in getQuantumState_QuTip above
        if apply_state_preparation_error and self.state_preparation_error is not None:
            raise NotImplementedError("state_preparation_error is not implemented yet.")
        #--------------------------------
        return psi0

    def is_eigenstate(self, 
                    observable: QuantumOperator
                    ) -> bool:
        """
        Returns True if the state is an eigenstate of the given observable.

        Parameters
        ----------
        observable : QuantumOperator
            observable for which to check if the state is an eigenstate

        Returns
        -------
        bool
            whether the state is an eigenstate of the given observable
        """
        # create qutip quantum state object
        psi0 = self.to_QuTip()
        # create qutip observable object
        qop = to_QuTip(observable)
        # check if variance of observable is zero in psi0
        variance = qt.expect(qop**2, psi0) - qt.expect(qop, psi0)**2
        if np.abs(variance) < 1e-3:
            return True
        return False

    def evaluate_exact_expvals(self, 
                            qop: QuantumOperator | PauliOperator
                            ) -> dict:
        """
        Returns exact expectation values of the QuantumOperator qop, 
        calculated for the QuantumState object.

        Parameters
        ----------
        qop : QuantumOperator or PauliOperator
            quantum operator for which the expectation values are calculated

        Returns
        -------
        exact_expvals : dict
            dictionary of exact expectation values of qop
        """
        ##  check if qop is a QuantumOperator or a PauliOperator
        if not isinstance(qop, QuantumOperator) and not isinstance(qop, PauliOperator):
            raise TypeError("qop = {}, but must be a QuantumOperator or a PauliOperator.".format(qop))
        ## get list of Pauli operators
        pops = [qop]
        if isinstance(qop, QuantumOperator):
            pops = [PauliOperator(N=qop.N, pauli_type=pop.pauli_type) for pop in qop.terms.values()] # if pop.pauli_type!="I"*qop.N]
        # ---------------------------------------------------
        ### STEP 1 ### basis state without state preparation
        if self.qutip_state is None and self.state_preparation is None and self.state_preparation_error is None:
            basis = self.basis
            excitations = self.excitations
            pops_vals = []
            for pop in pops:
                pop_type = pop.pauli_type
                val = 1
                for inx in range(len(pop_type)):
                    if pop_type[inx] != "I":
                        if pop_type[inx] != basis[inx].upper():
                            val = 0
                            # break
                        else:
                            if excitations[inx] == "0":
                                val *= -1
                pops_vals.append(pop.coeff*val)
            exact_expvals = {pops[inx].pauli_type : [pops_vals[inx]] for inx in range(len(pops))}
        # ---------------------------------------------------
        ### STEP 2 ### basis state with state preparation or general qutip state
        else:
            # transform to QuTip quantum state
            state_qt = self.to_QuTip()
            ## calculate the exact expectation values
            pops_vals = get_expvals_QuTip(state_qt, pops)
            exact_expvals = {pops[inx].pauli_type : [pops_vals[inx]] for inx in range(len(pops))}
        # ---------------------------------------------------
        return exact_expvals

    def extend_to_larger_system(self, 
                                extension_factor: int
                                ) -> QuantumState:
        """
        Returns a QuantumState object that is the same as self, 
        but with the excitations extended to a larger system.

        Parameters
        ----------
        extension_factor : int
            factor by which the system is extended

        Returns
        -------
        QuantumState
            extended QuantumState object
        """
        # check if extension_factor is an integer
        if not isinstance(extension_factor, int) or extension_factor <= 0:
            raise TypeError("extension_factor is not a positive integer.")
        # extend excitations
        excitations = self.excitations * extension_factor
        basis = self.basis * extension_factor
        state_extended = QuantumState(N=self.N*extension_factor, excitations=excitations, basis=basis, state_preparation=self.state_preparation, state_preparation_label=self.state_preparation_label, state_preparation_error=self.state_preparation_error)
        return state_extended

    def split_into_basis(self, 
                        basis: list
                        ) -> list:
        """
        Expands the QuantumState in a given basis.
        Returns the corresponding expansion coefficients.

        Parameters
        ----------
        basis : list of QuantumStates
            basis in which the state is expanded

        Returns
        -------
        coeffs : list of floats
            expansion coefficients of the state in the given basis
        """
        ### STEP 0 ### validate input
        # check if basis is a list of QuantumState objects
        if not isinstance(basis, list) or not all([isinstance(x, QuantumState) for x in basis]):
            raise TypeError("basis is not a list of QuantumState objects.")
        #--------------------------------
        ### STEP 1 ### get coefficients via inner products (QuTip)
        psi0 = self.to_QuTip()
        coeffs = []
        for state in basis:
            psi1 = state.to_QuTip()
            coeff = qt.expect(psi1.dag(), psi0)
            coeffs.append(coeff)
        #--------------------------------
        return coeffs

    @staticmethod
    def get_state_preparation_from_data(
                data_entry,
                nshots: int = -1, 
                ) -> list:
        """
        Return a state preparation that maps |0...0⟩ to the state 
        that produced the data_entry.

        Estimates the state preparation required to map
        the computational basis state |0...0⟩ to the state 
        that produced the data_entry.
        This assumes that the data_entry is the result of a measurement
        of a product state. 
        In this case, the state preparation is given by the 
        single-qubit Pauli expectation values of the data_entry 
        via |psi> = R_z(phi) R_y(theta) R_x(-pi) |0...0⟩,
        with theta = 2*arccos(<Z>) and phi = arctan(<Y>/<X>).
        The R_x(-pi) gate is added because in this code |0⟩ is the down state with <Z> = -1,
        and not the up state as in the standard convention.

        Parameters
        ----------
        data_entry : DataEntry
            data_entry object
        nshots : int 
            number of shots used for estimating the expectation values
            If nshots is ``-1``, all available shots are used.
            If nshots is ``0``, the exact expectation values are used.
            Default is ``-1``.

        Returns
        -------
        state_preparation : list of str
            state preparation
        """
        ### STEP 1 ### evaluate all single-qubit Pauli expectation values (X1,Y1,Z1,X2,Y2,Z2,...)
        obs_terms_all1loc = {inx*"I"+var+"I"*(data_entry.Nions-inx-1): 1.0 for inx in range(data_entry.Nions) for var in "XYZ"}
        qop_all1loc = QuantumOperator(N=data_entry.Nions, terms=obs_terms_all1loc)
        qop_list = list(qop_all1loc.terms.values())
        # get exact expectation values
        expvals, var_expvals = data_entry.evaluate_observable(qop_list=qop_list, nshots=nshots)
        expvals = np.squeeze(expvals)
        # expvals_str = "".join(["{}={:.4f}, ".format(qop_list[inx].type, expvals[inx]) for inx in range(len(qop_list))])
        # ic(expvals_str)
        # --------------------------------
        ### STEP 2 ### get theta and phi for each ion
        theta_list = []
        phi_list = []
        for inx in range(data_entry.Nions):
            # Calculate theta and phi from expectation values
            theta = np.arccos(expvals[3*inx+2])
            phi = np.arctan2(expvals[3*inx+1], expvals[3*inx])
            # phi = np.arctan(expvals[3*inx+1]/expvals[3*inx])
            theta_list.append(theta)
            phi_list.append(phi)
        # --------------------------------
        ### STEP 3 ### get state preparation  Rz(phi)*Ry(theta)*Rx(-pi) (in reverse order)
        state_preparation = []
        for inx in range(data_entry.Nions):
            state_preparation.append(["Rx{}".format(inx), -np.pi])
            state_preparation.append(["Ry{}".format(inx), theta_list[inx]])
            state_preparation.append(["Rz{}".format(inx), phi_list[inx]])
        # --------------------------------
        return state_preparation

    @staticmethod
    def get_mixed_state_from_data(data_entry, 
                                nshots: int = -1,
                                label: str | None = None,
                                ) -> QuantumState:
        """
        Estimates the the mixed state that produced the data_entry.
        This assumes that the data_entry is the result of a measurement
        of a product state. 
        In this case the mixed state is given by the
        single-qubit Pauli expectation values of the data_entry 
        via |rho> = 1/2*(I + <X> X + <Y> Y + <Z> Z).

        Returns
        -------
        data_entry : DataEntry
            data_entry object
        nshots : int
            number of shots used for estimating the expectation values
            If nshots is ``-1``, all available shots are used.
            If nshots is ``0``, the exact expectation values are used.
            Default is ``-1``.
        label : str 
            label for the QuantumState object
            Default is None.

        Returns
        -------
        QuantumState
            Mixed product state of the system as a QuantumState object
            where the state is stored in rho_state.qutip_state
        """
        ### STEP 1 ### evaluate all single-qubit Pauli expectation values (X1,Y1,Z1,X2,Y2,Z2,...)
        obs_terms_all1loc = {inx*"I"+var+"I"*(data_entry.Nions-inx-1): 1.0 for inx in range(data_entry.Nions) for var in "XYZ"}
        qop_all1loc = QuantumOperator(N=data_entry.Nions, terms=obs_terms_all1loc)
        qop_list = list(qop_all1loc.terms.values())
        # get exact expectation values
        expvals, var_expvals = data_entry.evaluate_observable(qop_list=qop_list, nshots=nshots)
        expvals = np.squeeze(expvals)
        # expvals_str = "".join(["{}={:.4f}, ".format(qop_list[inx].type, expvals[inx]) for inx in range(len(qop_list))])
        # ic(expvals_str)
        # --------------------------------
        ### STEP 2 ### get the mixed state for each ion
        rho_list = []
        for inx in range(data_entry.Nions):
            # get qutip density matrix
            xval = expvals[3*inx]
            yval = expvals[3*inx+1]
            zval = expvals[3*inx+2]
            rho = 0.5 * (qt.qeye(2) + xval * qt.sigmax() + yval * qt.sigmay() + zval * qt.sigmaz())
            rho_list.append(rho)
        # --------------------------------
        ### STEP 3 ### get the mixed state for the whole system
        rho_tot = qt.tensor(rho_list)
        # --------------------------------
        ### STEP 4 ### prepare quantum state object
        rho_state = QuantumState(N=data_entry.Nions, qutip_state=rho_tot, label=label)
        # --------------------------------
        return rho_state


def getQuantumState_QuTip(
        excitations: str, 
        basis: str | None = None,
        state_preparation_error: float | None = None,
        ):
    """
    Returns a QuTip quantum state from excitations in a given basis.

    Returns QuTip quantum product state from basis and excitations.
    basis is a string of "x", "y" or "z".
    excitations is an array of 0s and 1s, where 0 means down and 1 means up.
    state_preparation_error is the probability of a spin flip error for each qubit.

    Parameters
    ----------
    excitations : str
        excitations in the given basis
    basis : str 
        basis of the quantum state ("x","y","z")
        Default is "z"*len(excitations).
    state_preparation_error (float) 
        probability of a bit-flip error in the initial state
        Default is None.

    Returns
    -------
    Qutip QuantumState or QuantumOperator
        Qutip quantum state or density matrix
    """
    if basis is None:
        basis = "z"*len(excitations)
    #--------------------------------
    ### STEP 1 ### create individual qubit states
    qstate_list = []
    for sinx in range(len(excitations)):
        base = basis[sinx]
        exc = int(excitations[sinx])
        if base == "x":
            state = (qt.basis(2,0)+(-1+2*exc)*qt.basis(2,1))/np.sqrt(2)
        elif base == "y":
            state = (qt.basis(2,0)+(-1+2*exc)*1j*qt.basis(2,1))/np.sqrt(2)
        elif base == "z":
            state = qt.basis(2,1-exc)
        else:
            raise ValueError("Basis {} not recognized".format(base))
        qstate_list.append(state)
    #--------------------------------
    ### STEP 2 ### create product quantum state
    qstate = qt.tensor(qstate_list)
    #--------------------------------
    ### STEP 3 ### apply state preparation error #TODO this can be done above???
    # check norm of qstate
    if np.abs(1-qstate.norm()) > 1e-6:
        raise ValueError("norm of qstate is {} instead of 1.".format(qstate.norm()))
    # apply state preparation error
    if state_preparation_error is not None:
        raise NotImplementedError("state_preparation_error is not implemented yet.")
        N = len(excitations)
        p = state_preparation_error
        preal = (1-p)**N
        # threshold for cutoff of excitations (cutoff if ptot > 1-pthresh)
        pthresh = 0
        # loop over number of flips
        qstate = preal * qstate*qstate.dag()
        ptot = preal
        for Nflips in range(1,N+1):
            # get all possible indices of flips
            flips_list = list(it.combinations(range(N), Nflips))
            for flips in flips_list:
                qstate_error = qstate_list.copy()
                for inx in flips:
                    if qstate_error[inx] == qt.basis(2,0):
                        qstate_error[inx] = qt.basis(2,1)
                    else:
                        qstate_error[inx] = qt.basis(2,0)
                qstate_error = qt.tensor(qstate_error)
                qstate += p**Nflips*(1-p)**(N-Nflips) * qstate_error*qstate_error.dag()
                ptot += p**Nflips*(1-p)**(N-Nflips)
            # check if ptot is beyond threshold
            if ptot > 1-pthresh:
                print("cutoff at Nflips = {} with ptot = {}".format(Nflips, ptot))
                break
        # check if ptot is equal to trace of qstate
        if np.abs(ptot-np.trace(qstate)) > 1e-6:
            raise ValueError("ptot is {} instead of {}.".format(ptot, np.trace(qstate)))
        # renormalize qstate
        if np.abs(ptot-1) > 1e-6:
            print("renormalizing qstate, ptot = {}".format(ptot))
            qstate = qstate/np.trace(qstate)
        # check if trace of qstate is 1
        if np.abs(1-np.trace(qstate)) > 1e-6:
            raise ValueError("trace of qstate is {} instead of 1.".format(np.trace(qstate)))
    #--------------------------------
    return qstate