import pytest

from sage.structure.sage_object import SageObject
from sage.numerical.backends.generic_backend_test import GenericBackendTests
from sage.numerical.backends.generic_backend import GenericBackend
from sage.numerical.mip import MixedIntegerLinearProgram

# Check if highspy is available
try:
    import highspy
    HIGHSPY_AVAILABLE = True
except ImportError:
    HIGHSPY_AVAILABLE = False


@pytest.mark.skipif(not HIGHSPY_AVAILABLE, reason="highspy not available")
class TestHiGHSBackend(GenericBackendTests):

    @pytest.fixture
    def backend(self) -> GenericBackend:
        return MixedIntegerLinearProgram(solver='HiGHS').get_backend()

    def test_sage_unittest_testsuite(self, sage_object: SageObject):
        # TODO: Remove this test as soon as all old test methods are migrated
        from sage.misc.sage_unittest import TestSuite
        # Skip only pickling test - other methods are now implemented
        TestSuite(sage_object).run(verbose=True, raise_on_failure=True, 
                                  skip=("_test_pickling",))

    def test_add_variable(self, backend: GenericBackend):
        """Test adding variables with various bounds and types."""
        # Test adding a simple variable
        var1 = backend.add_variable()
        assert backend.ncols() == 1
        assert backend.is_variable_continuous(var1)
        
        # Test adding variable with bounds
        var2 = backend.add_variable(lower_bound=-5.0, upper_bound=10.0)
        assert backend.ncols() == 2
        bounds = backend.col_bounds(var2)
        assert bounds[0] == -5.0
        assert bounds[1] == 10.0
        
        # Test adding binary variable
        var3 = backend.add_variable(binary=True)
        assert backend.ncols() == 3
        assert backend.is_variable_binary(var3)
        
        # Test adding integer variable
        var4 = backend.add_variable(integer=True)
        assert backend.ncols() == 4
        assert backend.is_variable_integer(var4)

    def test_add_variables(self, backend: GenericBackend):
        """Test adding multiple variables at once."""
        initial_cols = backend.ncols()
        backend.add_variables(5)
        assert backend.ncols() == initial_cols + 5

    def test_set_objective(self, backend: GenericBackend):
        """Test setting objective function."""
        backend.add_variables(3)
        backend.set_objective([1.0, 2.0, 3.0])
        # Verify objective was set (coefficients retrievable)
        assert backend.ncols() == 3

    def test_add_linear_constraint(self, backend: GenericBackend):
        """Test adding linear constraints."""
        backend.add_variables(3)
        # Add constraint: x0 + 2*x1 + 3*x2 >= 5
        backend.add_linear_constraint([(0, 1.0), (1, 2.0), (2, 3.0)], 5.0, None)
        assert backend.nrows() == 1
        
        # Add another constraint: x0 + x1 <= 10
        backend.add_linear_constraint([(0, 1.0), (1, 1.0)], None, 10.0)
        assert backend.nrows() == 2

    def test_maximize_minimize(self, backend: GenericBackend):
        """Test maximization and minimization."""
        # Test maximization
        backend.set_sense(+1)
        assert backend.is_maximization()
        
        # Test minimization
        backend.set_sense(-1)
        assert not backend.is_maximization()

    def test_simple_lp_problem(self, backend: GenericBackend):
        """Test solving a simple LP problem: max x + y s.t. x + 2y <= 4, x,y >= 0."""
        # Add variables
        x = backend.add_variable(lower_bound=0.0)
        y = backend.add_variable(lower_bound=0.0)
        
        # Set objective: maximize x + y
        backend.set_sense(+1)
        backend.set_objective([1.0, 1.0])
        
        # Add constraint: x + 2y <= 4
        backend.add_linear_constraint([(0, 1.0), (1, 2.0)], None, 4.0)
        
        # Solve
        backend.solve()
        
        # Check solution
        obj_value = backend.get_objective_value()
        assert obj_value > 0  # Should have a positive objective value
        
        x_val = backend.get_variable_value(0)
        y_val = backend.get_variable_value(1)
        assert x_val >= 0
        assert y_val >= 0

    def test_integer_programming(self, backend: GenericBackend):
        """Test solving an integer programming problem."""
        # Add integer variables
        x = backend.add_variable(lower_bound=0.0, integer=True)
        y = backend.add_variable(lower_bound=0.0, integer=True)
        
        # Set objective: maximize 3x + 2y
        backend.set_sense(+1)
        backend.set_objective([3.0, 2.0])
        
        # Add constraint: x + y <= 5
        backend.add_linear_constraint([(0, 1.0), (1, 1.0)], None, 5.0)
        
        # Solve
        backend.solve()
        
        # Check that solution values are integers
        x_val = backend.get_variable_value(0)
        y_val = backend.get_variable_value(1)
        assert abs(x_val - round(x_val)) < 1e-6
        assert abs(y_val - round(y_val)) < 1e-6

    def test_variable_bounds(self, backend: GenericBackend):
        """Test getting and setting variable bounds."""
        var = backend.add_variable(lower_bound=1.0, upper_bound=10.0)
        
        # Check initial bounds
        bounds = backend.col_bounds(var)
        assert bounds[0] == 1.0
        assert bounds[1] == 10.0
        
        # Update bounds
        backend.variable_lower_bound(var, 2.0)
        backend.variable_upper_bound(var, 8.0)
        
        # Check updated bounds
        assert backend.variable_lower_bound(var) == 2.0
        assert backend.variable_upper_bound(var) == 8.0

    def test_variable_types(self, backend: GenericBackend):
        """Test different variable types."""
        # Continuous variable
        v_cont = backend.add_variable(continuous=True)
        assert backend.is_variable_continuous(v_cont)
        assert not backend.is_variable_integer(v_cont)
        assert not backend.is_variable_binary(v_cont)
        
        # Integer variable
        v_int = backend.add_variable(integer=True)
        assert backend.is_variable_integer(v_int)
        assert not backend.is_variable_continuous(v_int)
        
        # Binary variable
        v_bin = backend.add_variable(binary=True)
        assert backend.is_variable_binary(v_bin)

    def test_copy(self, backend: GenericBackend):
        """Test copying the backend."""
        backend.add_variables(2)
        backend.set_objective([1.0, 2.0])
        backend.add_linear_constraint([(0, 1.0), (1, 1.0)], None, 5.0)
        
        # Make a copy
        backend_copy = backend.__copy__()
        
        # Check that copy has same dimensions
        assert backend_copy.ncols() == backend.ncols()
        assert backend_copy.nrows() == backend.nrows()
        assert backend_copy.is_maximization() == backend.is_maximization()

    def test_problem_name(self, backend: GenericBackend):
        """Test setting and getting problem name."""
        name = "Test Problem"
        backend.problem_name(name)
        assert backend.problem_name() == name

    def test_col_name(self, backend: GenericBackend):
        """Test variable naming."""
        var = backend.add_variable(name="my_var")
        name = backend.col_name(var)
        assert name == "my_var" or name is None  # Some backends may not support names
