"""Operator sample-space definitions for symbolic model construction."""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: int
    fn: Callable
    ast_node: type[ast.AST] | None = None


BINARY_OPERATORS: Dict[str, OperatorSpec] = {
    "add": OperatorSpec("add", 2, lambda a, b: a + b, ast.Add),
    "sub": OperatorSpec("sub", 2, lambda a, b: a - b, ast.Sub),
    "mul": OperatorSpec("mul", 2, lambda a, b: a * b, ast.Mult),
    "truediv": OperatorSpec("truediv", 2, lambda a, b: a / b if b != 0 else 0.0, ast.Div),
    "floordiv": OperatorSpec("floordiv", 2, lambda a, b: a // b if b != 0 else 0.0, ast.FloorDiv),
    "mod": OperatorSpec("mod", 2, lambda a, b: a % b if b != 0 else 0.0, ast.Mod),
    "pow": OperatorSpec("pow", 2, lambda a, b: a ** max(min(b, 4), -4), ast.Pow),
    "gt": OperatorSpec("gt", 2, lambda a, b: float(a > b)),
    "lt": OperatorSpec("lt", 2, lambda a, b: float(a < b)),
    "ge": OperatorSpec("ge", 2, lambda a, b: float(a >= b)),
    "le": OperatorSpec("le", 2, lambda a, b: float(a <= b)),
    "eq": OperatorSpec("eq", 2, lambda a, b: float(a == b)),
    "ne": OperatorSpec("ne", 2, lambda a, b: float(a != b)),
}

UNARY_OPERATORS: Dict[str, OperatorSpec] = {
    "uadd": OperatorSpec("uadd", 1, lambda a: +a, ast.UAdd),
    "usub": OperatorSpec("usub", 1, lambda a: -a, ast.USub),
    "abs": OperatorSpec("abs", 1, abs),
    "greatest_integer": OperatorSpec("greatest_integer", 1, math.floor),
    "least_integer": OperatorSpec("least_integer", 1, math.ceil),
}


def sample_space_names() -> List[str]:
    """Return the visible sample space (all operators + GIF/LIF)."""
    return sorted(list(BINARY_OPERATORS.keys()) + list(UNARY_OPERATORS.keys()))
