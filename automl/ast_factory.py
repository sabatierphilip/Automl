"""AST synthesis for tokenizer/model/training code generation."""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Tuple

from automl.operators import BINARY_OPERATORS, UNARY_OPERATORS


class ASTFactory:
    """Builds executable Python functions from symbolic blueprints."""

    def build_tokenizer_factory(self):
        src = (
            "def generated_tokenizer(vocabulary):\n"
            "    def encode(text):\n"
            "        tokens = text.replace('(', ' ( ').replace(')', ' ) ').split()\n"
            "        return [vocabulary.index(t) if t in vocabulary else -1 for t in tokens]\n"
            "    return encode\n"
        )
        return self._compile_from_source(src, "generated_tokenizer")

    def build_model_forward(self, expression: Dict[str, Any]):
        fn_def = ast.FunctionDef(
            name="generated_forward",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="x")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[ast.Return(value=self._build_expr(expression))],
            decorator_list=[],
        )
        module = ast.Module(body=[fn_def], type_ignores=[])
        ast.fix_missing_locations(module)
        env: Dict[str, Any] = {"floor": __import__("math").floor, "ceil": __import__("math").ceil, "abs": abs}
        exec(compile(module, "<ast-model>", "exec"), env)
        return env["generated_forward"]

    def build_training_step(self):
        src = (
            "def generated_training_step(weights, grads, lr):\n"
            "    updated = []\n"
            "    for w, g in zip(weights, grads):\n"
            "        updated.append(w - lr * g)\n"
            "    return updated\n"
        )
        return self._compile_from_source(src, "generated_training_step")

    def _build_expr(self, node: Dict[str, Any]) -> ast.expr:
        if node["type"] == "feature":
            return ast.Subscript(
                value=ast.Name(id="x", ctx=ast.Load()),
                slice=ast.Constant(value=node["index"]),
                ctx=ast.Load(),
            )
        if node["type"] == "const":
            return ast.Constant(value=node["value"])

        op_name = node["op"]
        args = [self._build_expr(arg) for arg in node["args"]]

        if op_name in BINARY_OPERATORS and BINARY_OPERATORS[op_name].ast_node is not None:
            left, right = args
            if op_name in {"gt", "lt", "ge", "le", "eq", "ne"}:
                op_map = {
                    "gt": ast.Gt(),
                    "lt": ast.Lt(),
                    "ge": ast.GtE(),
                    "le": ast.LtE(),
                    "eq": ast.Eq(),
                    "ne": ast.NotEq(),
                }
                return ast.Call(
                    func=ast.Name(id="float", ctx=ast.Load()),
                    args=[ast.Compare(left=left, ops=[op_map[op_name]], comparators=[right])],
                    keywords=[],
                )
            return ast.BinOp(left=left, op=BINARY_OPERATORS[op_name].ast_node(), right=right)

        if op_name in UNARY_OPERATORS:
            if op_name in {"uadd", "usub"}:
                return ast.UnaryOp(op=UNARY_OPERATORS[op_name].ast_node(), operand=args[0])
            fn = "floor" if op_name == "greatest_integer" else "ceil" if op_name == "least_integer" else "abs"
            return ast.Call(func=ast.Name(id=fn, ctx=ast.Load()), args=[args[0]], keywords=[])

        raise ValueError(f"Unsupported op {op_name}")

    @staticmethod
    def _compile_from_source(source: str, fn_name: str):
        module = ast.parse(source)
        env: Dict[str, Any] = {}
        exec(compile(module, "<ast-generated>", "exec"), env)
        return env[fn_name]


DEFAULT_EXPRESSION: Dict[str, Any] = {
    "type": "op",
    "op": "add",
    "args": [
        {"type": "op", "op": "mul", "args": [{"type": "feature", "index": 0}, {"type": "const", "value": 0.65}]},
        {
            "type": "op",
            "op": "sub",
            "args": [
                {"type": "op", "op": "greatest_integer", "args": [{"type": "feature", "index": 1}]},
                {"type": "op", "op": "least_integer", "args": [{"type": "feature", "index": 2}]},
            ],
        },
    ],
}
