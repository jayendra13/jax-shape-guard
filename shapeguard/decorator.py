"""
Contract decorators for shape validation.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

from shapeguard.core import UnificationContext
from shapeguard.spec import ShapeSpec, match_shape
from shapeguard.errors import ShapeGuardError
from shapeguard._compat import get_shape, is_array


F = TypeVar("F", bound=Callable[..., Any])


def expects(**shape_specs: ShapeSpec) -> Callable[[F], F]:
    """
    Decorator to validate input shapes on function entry.

    Args:
        **shape_specs: Mapping from argument names to shape specifications

    Returns:
        Decorator function

    Example:
        from shapeguard import Dim, expects

        n, m, k = Dim("n"), Dim("m"), Dim("k")

        @expects(a=(n, m), b=(m, k))
        def matmul(a, b):
            return a @ b

    Shape specifications can contain:
        - int: Exact dimension match (e.g., 128)
        - Dim: Symbolic dimension that unifies across arguments
        - None: Wildcard that accepts any value

    Example with mixed specs:
        @expects(x=(n, 128), y=(None, 128))  # x has symbolic first dim,
        def f(x, y):                          # both have concrete second dim
            ...
    """

    def decorator(fn: F) -> F:
        # Get function signature for argument binding
        sig = inspect.signature(fn)
        fn_name = fn.__qualname__

        # Validate that all spec keys are valid argument names
        param_names = set(sig.parameters.keys())
        for arg_name in shape_specs:
            if arg_name not in param_names:
                raise ValueError(
                    f"@expects: '{arg_name}' is not a parameter of {fn_name}. "
                    f"Valid parameters: {sorted(param_names)}"
                )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Bind arguments to parameter names
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
            except TypeError as e:
                # Let the original function raise its own error
                return fn(*args, **kwargs)

            # Create unification context for this call
            ctx = UnificationContext()

            # Check each specified argument
            for arg_name, spec in shape_specs.items():
                if arg_name not in bound.arguments:
                    continue

                value = bound.arguments[arg_name]

                # Skip non-array arguments (allows optional array args)
                if not is_array(value):
                    continue

                actual = get_shape(value)

                try:
                    match_shape(actual, spec, ctx, arg_name)
                except ShapeGuardError as e:
                    # Enrich error with function context
                    e.function = fn_name
                    e.argument = arg_name
                    e.expected = spec
                    e.actual = actual
                    e.bindings = ctx.format_bindings()
                    raise

            return fn(*args, **kwargs)

        # Attach metadata for introspection
        wrapper.__shapeguard_specs__ = shape_specs  # type: ignore

        return wrapper  # type: ignore

    return decorator
