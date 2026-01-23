"""
Tests for utility_functions module.
"""
import pytest
import numpy as np
from src.utility_functions import (
    get_total_size,
    deep_get_size,
    get_size,
    minimize,
    integrate,
    simpson38,
    estimate_derivative,
    Richardson_extrapolation,
)


class TestMemoryFunctions:
    """Tests for memory-related utility functions."""

    def test_get_total_size(self):
        """Test get_total_size function."""
        obj = [1, 2, 3]
        size = get_total_size(obj)
        assert size > 0

    def test_deep_get_size(self):
        """Test deep_get_size function."""
        obj = {"a": [1, 2, 3], "b": "test"}
        size = deep_get_size(obj)
        assert size > 0

    def test_get_size(self):
        """Test get_size function."""
        obj = [1, 2, 3]
        size = get_size(obj)
        assert size > 0

    def test_get_size_invalid_type(self):
        """Test get_size with invalid type."""
        with pytest.raises(TypeError):
            get_size(type)


class TestMinimize:
    """Tests for minimize function."""

    def test_minimize_lsq(self):
        """Test minimize with least squares method."""
        def cost(x):
            return np.array([x[0]**2 + x[1]**2 - 1])
        
        x0 = np.array([0.5, 0.5])
        solution, optimal_cost, _, _ = minimize(cost, x0, method='lsq')
        assert isinstance(solution, np.ndarray)
        assert isinstance(optimal_cost, np.ndarray)

    def test_minimize_nelder(self):
        """Test minimize with Nelder-Mead method."""
        def cost(x):
            return np.array([x[0]**2 + x[1]**2])
        
        x0 = np.array([1.0, 1.0])
        solution, optimal_cost, _, _ = minimize(cost, x0, method='nelder')
        assert isinstance(solution, np.ndarray)

    def test_minimize_brute(self):
        """Test minimize with brute force method."""
        def cost(x):
            return np.array([x[0]**2 + x[1]**2])
        
        x0 = np.array([1.0, 1.0])
        bounds = [(-2, 2), (-2, 2)]
        solution, optimal_cost, landscape_grid, landscape_vals = minimize(
            cost, x0, method='brute', bounds=bounds, max_nfev=100
        )
        assert isinstance(solution, np.ndarray)
        assert landscape_grid is not None
        assert landscape_vals is not None

    def test_minimize_with_bounds(self):
        """Test minimize with bounds."""
        def cost(x):
            return np.array([x[0]**2])
        
        x0 = np.array([1.0])
        bounds = [(-2, 2)]
        solution, _, _, _ = minimize(cost, x0, method='lsq', bounds=bounds)
        assert isinstance(solution, np.ndarray)


class TestIntegrate:
    """Tests for integrate function."""

    def test_integrate_trapz(self):
        """Test integration with trapezoidal method."""
        x = np.linspace(0, 1, 100)
        y = x**2
        integral, neval = integrate(x, y, method='trapz')
        assert isinstance(integral, (int, float))
        assert neval == len(y)

    def test_integrate_simpson(self):
        """Test integration with Simpson method."""
        x = np.linspace(0, 1, 100)
        y = x**2
        integral, neval = integrate(x, y, method='simpson')
        assert isinstance(integral, (int, float))

    def test_integrate_simpson38(self):
        """Test integration with Simpson 3/8 method."""
        x = np.linspace(0, 1, 100)
        y = x**2
        integral, neval = integrate(x, y, method='simpson38')
        assert isinstance(integral, (int, float))

    def test_integrate_romb(self):
        """Test integration with Romberg method."""
        x = np.linspace(0, 1, 1+2**10)
        y = x**2
        integral, neval = integrate(x, y, method='romb')
        assert isinstance(integral, (int, float))

    def test_integrate_invalid_length(self):
        """Test integration with mismatched lengths."""
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 50)
        with pytest.raises(ValueError):
            integrate(x, y)

    def test_integrate_unsorted(self):
        """Test integration with unsorted x."""
        x = np.array([1, 0, 2])
        y = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            integrate(x, y)


class TestSimpson38:
    """Tests for simpson38 function."""

    def test_simpson38(self):
        """Test Simpson 3/8 integration."""
        x = np.linspace(0, 1, 10)
        y = x**2
        integral = simpson38(x, y)
        assert isinstance(integral, (int, float))


class TestEstimateDerivative:
    """Tests for estimate_derivative function."""

    def test_estimate_derivative_two_points(self):
        """Test derivative estimation with two points."""
        times = np.array([0.0, 1.0])
        values = np.array([0.0, 1.0])
        derivative = estimate_derivative(values, times)
        assert isinstance(derivative, (int, float, complex))
        assert np.isclose(derivative, 1.0)

    def test_estimate_derivative_three_points(self):
        """Test derivative estimation with three points."""
        times = np.array([0.0, 0.5, 1.0])
        values = np.array([0.0, 0.25, 1.0])
        derivative = estimate_derivative(values, times)
        assert isinstance(derivative, (int, float, complex))

    def test_estimate_derivative_invalid_length(self):
        """Test derivative estimation with mismatched lengths."""
        times = np.array([0.0, 1.0])
        values = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError):
            estimate_derivative(values, times)

    def test_estimate_derivative_duplicate_times(self):
        """Test derivative estimation with duplicate times."""
        times = np.array([0.0, 0.0])
        values = np.array([0.0, 1.0])
        with pytest.raises(ValueError):
            estimate_derivative(values, times)


class TestRichardsonExtrapolation:
    """Tests for Richardson_extrapolation function."""

    def test_richardson_extrapolation(self):
        """Test Richardson extrapolation."""
        deriv1 = 1.0
        deriv2 = 1.1
        dt1 = 0.1
        dt2 = 0.2
        order = 1
        result = Richardson_extrapolation(deriv1, deriv2, dt1, dt2, order)
        assert isinstance(result, (int, float, complex))
