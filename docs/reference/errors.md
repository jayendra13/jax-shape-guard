# Errors

All exception classes raised by ShapeGuard.

## ShapeGuardError

Base exception for shape contract violations.

::: shapeguard.errors.ShapeGuardError

## UnificationError

Raised when a symbolic dimension cannot unify with a concrete value.

::: shapeguard.errors.UnificationError

## RankMismatchError

Raised when array rank doesn't match the spec.

::: shapeguard.errors.RankMismatchError

## DimensionMismatchError

Raised when a specific dimension doesn't match the expected value.

::: shapeguard.errors.DimensionMismatchError

## OutputShapeError

Raised when a function's return value has the wrong shape.

::: shapeguard.errors.OutputShapeError

## BroadcastError

Raised when shapes cannot be broadcast together.

::: shapeguard.errors.BroadcastError
