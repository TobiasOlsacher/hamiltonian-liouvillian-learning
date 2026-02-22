"""
Tests for pauli_algebra module.
"""
import pytest
import numpy as np
from src.pauli_algebra import (
    PauliOperator,
    QuantumOperator,
    Dissipator,
    CollectiveSpinOperator,
    pointwise_pauli_commutator,
    pops_to_qop,
    get_expval_from_Zproduct_state,
    qop_sum,
    qop_mean,
    qop_var,
    qop_relvar,
    find_conserved_quantities,
)


class TestPauliOperator:
    """Tests for PauliOperator class."""

    def test_init(self):
        """Test PauliOperator initialization."""
        pop = PauliOperator(N=3, pauli_type="XZI", coeff=1.0)
        assert pop.N == 3
        assert pop.pauli_type == "XZI"
        assert pop.coeff == 1.0

    def test_type_setter(self):
        """Test type property setter."""
        pop = PauliOperator(N=2, pauli_type="XX")
        pop.type = "YY"
        assert pop.type == "YY"

    def test_type_setter_invalid(self):
        """Test type setter with invalid input."""
        pop = PauliOperator(N=2, pauli_type="XX")
        with pytest.raises(ValueError):
            pop.type = "X"  # Wrong length
        with pytest.raises(ValueError):
            pop.type = "AB"  # Invalid pauli char

    def test_coeff_setter(self):
        """Test coeff property setter."""
        pop = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        pop.coeff = 2.0
        assert pop.coeff == 2.0

    def test_copy(self):
        """Test copy method."""
        pop = PauliOperator(N=2, pauli_type="XX", coeff=1.5)
        pop_copy = pop.copy()
        assert pop_copy.N == pop.N
        assert pop_copy.pauli_type == pop.pauli_type
        assert pop_copy.coeff == pop.coeff
        assert pop_copy is not pop

    def test_to_quantum_operator(self):
        """Test conversion to QuantumOperator."""
        pop = PauliOperator(N=2, pauli_type="XX", coeff=2.0)
        qop = pop.to_quantum_operator()
        assert isinstance(qop, QuantumOperator)
        assert qop.N == 2
        assert "XX" in qop.terms
        assert qop.terms["XX"].coeff == 2.0

    def test_str(self):
        """Test string representation."""
        pop = PauliOperator(N=2, pauli_type="XX", coeff=1.5)
        s = pop.str()
        assert "XX" in s
        assert "1.5" in s

    def test_eq(self):
        """Test equality operator."""
        pop1 = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        pop2 = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        pop3 = PauliOperator(N=2, pauli_type="YY", coeff=1.0)
        assert pop1 == pop2
        assert pop1 != pop3

    def test_neg(self):
        """Test negation."""
        pop = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        neg_pop = -pop
        assert neg_pop.coeff == -1.0
        assert neg_pop.pauli_type == "XX"

    def test_abs(self):
        """Test absolute value."""
        pop = PauliOperator(N=2, pauli_type="XX", coeff=-1.5)
        abs_pop = abs(pop)
        assert abs_pop.coeff == 1.5

    def test_dagger(self):
        """Test dagger operation."""
        pop = PauliOperator(N=2, pauli_type="XM", coeff=1.0 + 1j)
        pop_dag = pop.dagger()
        assert pop_dag.pauli_type == "XP"
        assert np.isclose(pop_dag.coeff, 1.0 - 1j)

    def test_is_identity(self):
        """Test identity check."""
        pop1 = PauliOperator(N=2, pauli_type="II")
        pop2 = PauliOperator(N=2, pauli_type="XX")
        assert pop1.is_identity()
        assert not pop2.is_identity()

    def test_add(self):
        """Test addition."""
        pop1 = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        pop2 = PauliOperator(N=2, pauli_type="XX", coeff=2.0)
        pop_sum = pop1 + pop2
        assert pop_sum.coeff == 3.0
        assert pop_sum.pauli_type == "XX"

    def test_add_different_types(self):
        """Test addition with different types raises error."""
        pop1 = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        pop2 = PauliOperator(N=2, pauli_type="YY", coeff=2.0)
        with pytest.raises(ValueError):
            pop1 + pop2

    def test_sub(self):
        """Test subtraction."""
        pop1 = PauliOperator(N=2, pauli_type="XX", coeff=3.0)
        pop2 = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        pop_diff = pop1 - pop2
        assert pop_diff.coeff == 2.0

    def test_mul_scalar(self):
        """Test scalar multiplication."""
        pop = PauliOperator(N=2, pauli_type="XX", coeff=2.0)
        pop_mul = pop * 3.0
        assert pop_mul.coeff == 6.0

    def test_mul_pauli_operators(self):
        """Test Pauli operator multiplication."""
        pop1 = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        pop2 = PauliOperator(N=2, pauli_type="YY", coeff=1.0)
        pop_prod = pop1 * pop2
        assert pop_prod.pauli_type == "ZZ"
        assert np.isclose(pop_prod.coeff, -1.0)

    def test_rmul(self):
        """Test right multiplication."""
        pop = PauliOperator(N=2, pauli_type="XX", coeff=2.0)
        pop_rmul = 3.0 * pop
        assert pop_rmul.coeff == 6.0

    def test_truediv(self):
        """Test division."""
        pop = PauliOperator(N=2, pauli_type="XX", coeff=6.0)
        pop_div = pop / 3.0
        assert pop_div.coeff == 2.0

    def test_pow(self):
        """Test power operation."""
        pop = PauliOperator(N=2, pauli_type="XX", coeff=2.0)
        pop_pow = pop ** 2
        assert pop_pow.pauli_type == "II"
        assert pop_pow.coeff == 4.0

    def test_norm(self):
        """Test norm calculation."""
        pop = PauliOperator(N=2, pauli_type="XX", coeff=3.0 + 4j)
        assert pop.norm() == 5.0

    def test_commutator(self):
        """Test commutator."""
        pop1 = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        pop2 = PauliOperator(N=2, pauli_type="YY", coeff=1.0)
        comm = pop1.commutator(pop2)
        # XX and YY commute, so commutator should be zero
        assert isinstance(comm, PauliOperator)

    def test_range(self):
        """Test range calculation."""
        pop = PauliOperator(N=3, pauli_type="XIZ")
        assert pop.range() == 2

    def test_support(self):
        """Test support calculation."""
        pop = PauliOperator(N=3, pauli_type="XIZ")
        support = pop.support()
        assert 0 in support
        assert 2 in support
        assert 1 not in support

    def test_center(self):
        """Test center calculation."""
        pop = PauliOperator(N=3, pauli_type="XII")
        center = pop.center()
        assert isinstance(center, float)

    def test_split_MP(self):
        """Test M/P splitting."""
        pop = PauliOperator(N=2, pauli_type="XM", coeff=1.0)
        pop_list = pop.split_MP()
        assert len(pop_list) > 1
        assert all(isinstance(p, PauliOperator) for p in pop_list)

    def test_can_be_measured(self):
        """Test measurement check."""
        pop = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        bases = ["XX", "YY", "ZZ"]
        can_measure = pop.can_be_measured(bases)
        assert can_measure is True


class TestQuantumOperator:
    """Tests for QuantumOperator class."""

    def test_init(self):
        """Test QuantumOperator initialization."""
        terms = {"XX": 1.0, "YY": 2.0}
        qop = QuantumOperator(N=2, terms=terms)
        assert qop.N == 2
        assert "XX" in qop.terms
        assert "YY" in qop.terms

    def test_init_invalid_terms(self):
        """Test initialization with invalid terms."""
        with pytest.raises(TypeError):
            QuantumOperator(N=2, terms="invalid")

    def test_copy(self):
        """Test copy method."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        qop_copy = qop.copy()
        assert qop_copy.N == qop.N
        assert qop_copy is not qop

    def test_str(self):
        """Test string representation."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        s = qop.str()
        assert isinstance(s, str)

    def test_labels(self):
        """Test labels method."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0, "YY": 2.0})
        labels = qop.labels()
        assert "XX" in labels
        assert "YY" in labels

    def test_is_zero(self):
        """Test zero check."""
        qop1 = QuantumOperator(N=2, terms={"XX": 0.0})
        qop2 = QuantumOperator(N=2, terms={"XX": 1.0})
        assert qop1.is_zero()
        assert not qop2.is_zero()

    def test_is_identity(self):
        """Test identity check."""
        qop1 = QuantumOperator(N=2, terms={"II": 1.0})
        qop2 = QuantumOperator(N=2, terms={"XX": 1.0})
        assert qop1.is_identity()
        assert not qop2.is_identity()

    def test_norm(self):
        """Test norm calculation."""
        qop = QuantumOperator(N=2, terms={"XX": 3.0, "YY": 4.0})
        norm = qop.norm()
        assert np.isclose(norm, 5.0)

    def test_normalize(self):
        """Test normalization."""
        qop = QuantumOperator(N=2, terms={"XX": 3.0, "YY": 4.0})
        qop_norm = qop.normalize()
        assert np.isclose(qop_norm.norm(), 1.0)

    def test_add(self):
        """Test addition."""
        qop1 = QuantumOperator(N=2, terms={"XX": 1.0})
        qop2 = QuantumOperator(N=2, terms={"YY": 2.0})
        qop_sum = qop1 + qop2
        assert "XX" in qop_sum.terms
        assert "YY" in qop_sum.terms

    def test_sub(self):
        """Test subtraction."""
        qop1 = QuantumOperator(N=2, terms={"XX": 3.0})
        qop2 = QuantumOperator(N=2, terms={"XX": 1.0})
        qop_diff = qop1 - qop2
        assert np.isclose(qop_diff.terms["XX"].coeff, 2.0)

    def test_split_terms_into_bases(self):
        """Regression: split_terms_into_bases uses self.N, returns correct structure."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0, "YY": 0.5, "ZZ": 0.3})
        bases = ["XX", "YY", "ZZ"]
        terms_per_basis, non_measurable = qop.split_terms_into_bases(bases)
        assert isinstance(terms_per_basis, dict)
        assert isinstance(non_measurable, list)
        assert "XX" in terms_per_basis and len(terms_per_basis["XX"]) >= 1
        assert "YY" in terms_per_basis and len(terms_per_basis["YY"]) >= 1
        assert "ZZ" in terms_per_basis and len(terms_per_basis["ZZ"]) >= 1

    def test_mul_scalar(self):
        """Test scalar multiplication."""
        qop = QuantumOperator(N=2, terms={"XX": 2.0})
        qop_mul = qop * 3.0
        assert qop_mul.terms["XX"].coeff == 6.0

    def test_mul_quantum_operators(self):
        """Test quantum operator multiplication."""
        qop1 = QuantumOperator(N=2, terms={"XX": 1.0})
        qop2 = QuantumOperator(N=2, terms={"YY": 1.0})
        qop_prod = qop1 * qop2
        assert isinstance(qop_prod, QuantumOperator)

    def test_commutator(self):
        """Test commutator."""
        qop1 = QuantumOperator(N=2, terms={"XX": 1.0})
        qop2 = QuantumOperator(N=2, terms={"YY": 1.0})
        comm = qop1.commutator(qop2)
        assert isinstance(comm, QuantumOperator)

    def test_dagger(self):
        """Test dagger operation."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0 + 1j})
        qop_dag = qop.dagger()
        assert isinstance(qop_dag, QuantumOperator)

    def test_to_vector(self):
        """Test vector conversion."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0})
        vec = qop.to_vector()
        assert isinstance(vec, np.ndarray)

    def test_coeffs(self):
        """Test coefficients extraction."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0, "YY": 2.0})
        coeffs = qop.coeffs()
        assert len(coeffs) == 2

    def test_remove_zero_coeffs(self):
        """Test zero coefficient removal."""
        qop = QuantumOperator(N=2, terms={"XX": 1.0, "YY": 0.0})
        qop_clean = qop.remove_zero_coeffs()
        assert "YY" not in qop_clean.terms

    def test_range(self):
        """Test range calculation."""
        qop = QuantumOperator(N=3, terms={"XIZ": 1.0})
        range_val = qop.range()
        assert isinstance(range_val, int)

    def test_support(self):
        """Test support calculation."""
        qop = QuantumOperator(N=3, terms={"XIZ": 1.0})
        support = qop.support()
        assert isinstance(support, list)


class TestDissipator:
    """Tests for Dissipator class."""

    def test_init(self):
        """Test Dissipator initialization."""
        diss = Dissipator(N=2, diss_type=("XX", "YY"), coeff=1.0)
        assert diss.N == 2
        assert diss.diss_type == ("XX", "YY")
        assert diss.coeff == 1.0

    def test_copy(self):
        """Test copy method."""
        diss = Dissipator(N=2, diss_type=("XX", "YY"), coeff=1.0)
        diss_copy = diss.copy()
        assert diss_copy.N == diss.N
        assert diss_copy.diss_type == diss.diss_type
        assert diss_copy.coeff == diss.coeff

    def test_str(self):
        """Test string representation."""
        diss = Dissipator(N=2, diss_type=("XX", "YY"), coeff=1.5)
        s = diss.str()
        assert isinstance(s, str)

    def test_eq(self):
        """Test equality."""
        diss1 = Dissipator(N=2, diss_type=("XX", "YY"), coeff=1.0)
        diss2 = Dissipator(N=2, diss_type=("XX", "YY"), coeff=1.0)
        diss3 = Dissipator(N=2, diss_type=("XX", "ZZ"), coeff=1.0)
        assert diss1 == diss2
        assert diss1 != diss3

    def test_mul(self):
        """Test multiplication."""
        diss = Dissipator(N=2, diss_type=("XX", "YY"), coeff=2.0)
        diss_mul = diss * 3.0
        assert diss_mul.coeff == 6.0

    def test_range(self):
        """Test range calculation."""
        diss = Dissipator(N=3, diss_type=("XIZ", "IIZ"))
        range_val = diss.range()
        assert isinstance(range_val, int)

    def test_support(self):
        """Test support calculation."""
        diss = Dissipator(N=3, diss_type=("XIZ", "IIZ"))
        support = diss.support()
        assert isinstance(support, list)


class TestCollectiveSpinOperator:
    """Tests for CollectiveSpinOperator class."""

    def test_init(self):
        """Test CollectiveSpinOperator initialization."""
        cso = CollectiveSpinOperator(N=3, spin_type="X", coeff=1.0)
        assert cso.N == 3
        assert cso.spin_type == "X"
        assert cso.coeff == 1.0

    def test_copy(self):
        """Test copy method."""
        cso = CollectiveSpinOperator(N=3, spin_type="X", coeff=1.0)
        cso_copy = cso.copy()
        assert cso_copy.N == cso.N
        assert cso_copy.spin_type == cso.spin_type

    def test_to_quantum_operator(self):
        """Test conversion to QuantumOperator."""
        cso = CollectiveSpinOperator(N=2, spin_type="X", coeff=1.0)
        qop = cso.to_quantum_operator()
        assert isinstance(qop, QuantumOperator)
        assert qop.N == 2

    def test_mul(self):
        """Test multiplication."""
        cso = CollectiveSpinOperator(N=3, spin_type="X", coeff=2.0)
        cso_mul = cso * 3.0
        assert cso_mul.coeff == 6.0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_pointwise_pauli_commutator(self):
        """Test pointwise Pauli commutator."""
        assert pointwise_pauli_commutator("XX", "XX") == 1
        assert pointwise_pauli_commutator("XX", "YY") == 0

    def test_pops_to_qop(self):
        """Test conversion of Pauli operators to QuantumOperator."""
        pop1 = PauliOperator(N=2, pauli_type="XX", coeff=1.0)
        pop2 = PauliOperator(N=2, pauli_type="YY", coeff=2.0)
        qop = pops_to_qop([pop1, pop2])
        assert isinstance(qop, QuantumOperator)
        assert qop.N == 2

    def test_get_expval_from_Zproduct_state(self):
        """Test expectation value from Z product state."""
        qop = QuantumOperator(N=2, terms={"ZZ": 1.0})
        state = "00"
        expval = get_expval_from_Zproduct_state(qop, state)
        assert isinstance(expval, (int, float, complex))

    def test_get_expval_from_Zproduct_state_multi_term(self):
        """Regression: multi-term qop sums all term contributions."""
        # ZZ on "00" gives +1, ZZ on "11" gives +1; ZZ on "01" or "10" gives -1
        qop = QuantumOperator(N=2, terms={"ZZ": 1.0, "II": 0.5})
        assert np.isclose(get_expval_from_Zproduct_state(qop, "00"), 1.0 + 0.5)
        assert np.isclose(get_expval_from_Zproduct_state(qop, "11"), 1.0 + 0.5)
        assert np.isclose(get_expval_from_Zproduct_state(qop, "01"), -1.0 + 0.5)

    def test_qop_sum(self):
        """Test quantum operator sum."""
        qop1 = QuantumOperator(N=2, terms={"XX": 1.0})
        qop2 = QuantumOperator(N=2, terms={"YY": 2.0})
        qop_sum_result = qop_sum([qop1, qop2])
        assert isinstance(qop_sum_result, QuantumOperator)

    def test_qop_mean(self):
        """Test quantum operator mean."""
        qop1 = QuantumOperator(N=2, terms={"XX": 1.0})
        qop2 = QuantumOperator(N=2, terms={"XX": 3.0})
        qop_mean_result = qop_mean([qop1, qop2])
        assert isinstance(qop_mean_result, QuantumOperator)
        assert np.isclose(qop_mean_result.terms["XX"].coeff, 2.0)

    def test_qop_var(self):
        """Test quantum operator variance."""
        qop1 = QuantumOperator(N=2, terms={"XX": 1.0})
        qop2 = QuantumOperator(N=2, terms={"XX": 3.0})
        qop_var_result = qop_var([qop1, qop2])
        assert isinstance(qop_var_result, QuantumOperator)
