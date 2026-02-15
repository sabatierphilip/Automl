from automl.ast_factory import ASTFactory, DEFAULT_EXPRESSION
from automl.operators import sample_space_names
from automl.trainer import run_experiment


def test_operator_sample_space_contains_required_ops():
    ops = sample_space_names()
    assert "greatest_integer" in ops
    assert "least_integer" in ops
    for op in ["add", "sub", "mul", "truediv", "pow", "gt", "lt", "ge", "le", "eq", "ne"]:
        assert op in ops


def test_ast_model_executes():
    factory = ASTFactory()
    fn = factory.build_model_forward(DEFAULT_EXPRESSION)
    out = fn([1.2, -0.4, 0.7])
    assert isinstance(out, float)


def test_end_to_end_experiment_runs():
    result = run_experiment(output_dir="results")
    assert result.test_mse < 5.0
    assert result.invariants["positive_definite"]
