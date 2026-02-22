"""
Tests for ansatz_parametrization module.
"""
import pytest
import numpy as np
from src.ansatz_parametrization import ParametrizationFunction, Parametrization
from src.pauli_algebra import QuantumOperator, Dissipator


class TestParametrizationFunction:
    """Tests for ParametrizationFunction class."""

    def test_init(self):
        """Test ParametrizationFunction initialization."""
        pf = ParametrizationFunction(coherent=True, criterion="XX")
        assert pf.coherent is True
        assert pf.criterion == ["XX"]

    def test_init_with_cutoff(self):
        """Test ParametrizationFunction with cutoff."""
        pf = ParametrizationFunction(coherent=True, criterion="XX", cutoff=2)
        assert pf.cutoff == 2

    def test_init_with_range(self):
        """Test ParametrizationFunction with range."""
        pf = ParametrizationFunction(coherent=True, criterion="XX", range=1)
        assert pf.range == 1

    def test_init_dissipative(self):
        """Test ParametrizationFunction for dissipative terms."""
        pf = ParametrizationFunction(coherent=False, criterion=["XX", "YY"])
        assert pf.coherent is False
        assert pf.criterion == [["XX", "YY"]]

    def test_init_with_operator(self):
        """Test ParametrizationFunction with operator."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        pf = ParametrizationFunction(operator=qop)
        assert pf.operator == qop

    def test_init_with_dissipators(self):
        """Test ParametrizationFunction with dissipators."""
        diss = Dissipator(N=2, diss_type=("XX", "YY"), coeff=0.1)
        pf = ParametrizationFunction(dissipators=[diss])
        assert pf.dissipators == [diss]

    def test_init_with_parameters(self):
        """Test ParametrizationFunction with parameters."""
        pf = ParametrizationFunction(parameters=[1.0, 2.0])
        assert pf.parameters == [1.0, 2.0]

    def test_init_with_bounds(self):
        """Test ParametrizationFunction with bounds."""
        bounds = [(0, 1), (0, 2)]
        pf = ParametrizationFunction(bounds=bounds)
        assert pf.bounds == bounds

    def test_copy(self):
        """Test copy method."""
        pf = ParametrizationFunction(coherent=True, criterion="XX")
        pf_copy = pf.copy()
        assert pf_copy.coherent == pf.coherent
        assert pf_copy.criterion == pf.criterion
        assert pf_copy is not pf

    def test_check_criterion_pauli(self):
        """Test check_criterion for PauliOperator."""
        pf = ParametrizationFunction(coherent=True, criterion="XX")
        from src.pauli_algebra import PauliOperator
        pop_xx = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        pop_yy = PauliOperator(N=2, pauli_type="YY", coeff=1.0)
        assert pf.check_criterion(pop_xx) is True
        assert pf.check_criterion(pop_yy) is False

    def test_check_criterion_dissipator(self):
        """Test check_criterion for Dissipator."""
        pf = ParametrizationFunction(coherent=False, criterion=["XX", "YY"])
        diss = Dissipator(N=2, diss_type=("XX", "YY"), coeff=0.1)
        assert pf.check_criterion(diss) is True

    def test_check_cutoff(self):
        """Test check_cutoff with range."""
        pf = ParametrizationFunction(coherent=True, criterion="XX", range=1)
        from src.pauli_algebra import PauliOperator
        pop = PauliOperator(N=3, pauli_type="XIX", coeff=1.0)
        assert pf.check_cutoff(pop) is True

    def test_get_coefficient_free(self):
        """Test get_coefficient for free parametrization."""
        pf = ParametrizationFunction(coherent=True, criterion="XX", param_type="free")
        from src.pauli_algebra import PauliOperator
        pop = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        coeff = pf.get_coefficient(pop)
        assert coeff == 1.0


class TestParametrization:
    """Tests for Parametrization class."""

    def test_init(self):
        """Test Parametrization initialization."""
        param = Parametrization()
        assert param.functions is None

    def test_init_with_functions(self):
        """Test Parametrization with parametrization functions."""
        pf1 = ParametrizationFunction(coherent=True, criterion="XX")
        pf2 = ParametrizationFunction(coherent=True, criterion="YY")
        param = Parametrization(functions=[pf1, pf2])
        assert len(param.functions) == 2

    def test_add_functions(self):
        """Test adding parametrization function."""
        param = Parametrization()
        pf = ParametrizationFunction(coherent=True, criterion="XX")
        param.add_functions([pf])
        assert len(param.functions) == 1

    def test_copy(self):
        """Test copy method."""
        pf = ParametrizationFunction(coherent=True, criterion="XX")
        param = Parametrization(functions=[pf])
        param_copy = param.copy()
        assert len(param_copy.functions) == len(param.functions)
        assert param_copy is not param
