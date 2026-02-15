"""Training orchestration across tokenizer/model/output design phases."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from automl.ast_factory import ASTFactory, DEFAULT_EXPRESSION
from automl.manifold_model import ManifoldAutoMLModel
from automl.operators import sample_space_names
from automl.tokenizer import TokenizerDesign


REAL_SERIES_CO2 = np.array([
    316.43, 316.97, 317.58, 318.22, 318.87, 319.57, 320.26, 320.94, 321.46, 321.83,
    322.18, 322.40, 322.57, 322.82, 323.16, 323.59, 324.04, 324.42, 324.68, 324.88,
    325.00, 325.16, 325.40, 325.83, 326.35, 326.89, 327.38, 327.70, 327.88, 327.95,
    327.98, 328.11, 328.43, 328.89, 329.37, 329.78, 330.04, 330.19, 330.25, 330.38,
    330.68, 331.14, 331.70, 332.18, 332.52, 332.72, 332.78, 332.89, 333.17, 333.63,
    334.16, 334.58, 334.87, 335.02, 335.09, 335.26, 335.59, 336.05, 336.52, 336.86,
])


@dataclass
class ExperimentResult:
    train_mse: float
    test_mse: float
    test_mae: float
    phases: List[str]
    sample_space: List[str]
    invariants: Dict[str, float]


def make_features(series: np.ndarray, lookback: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx in range(lookback, len(series)):
        X.append(series[idx - lookback:idx])
        y.append(series[idx])
    return np.asarray(X), np.asarray(y)


def run_experiment(output_dir: str = "results") -> ExperimentResult:
    phases = [
        "1) Tokenizer Design",
        "2) Model Design",
        "3) Output Design",
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ast_factory = ASTFactory()

    # Phase 1: Tokenizer design
    vocabulary = ["(", ")", "add", "sub", "mul", "div", "floor", "ceil", "x0", "x1", "x2"]
    tokenizer = TokenizerDesign(vocabulary=vocabulary)
    generated_tokenizer = ast_factory.build_tokenizer_factory()(vocabulary)
    _ = tokenizer.encode("( add x0 x1 )")
    _ = generated_tokenizer("( add x0 x1 )")

    # Phase 2: Model design
    symbolic_forward = ast_factory.build_model_forward(DEFAULT_EXPRESSION)
    model = ManifoldAutoMLModel(start_dimension=2)
    training_step = ast_factory.build_training_step()

    series = REAL_SERIES_CO2
    X, y = make_features(series, lookback=3)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()

    split = int(0.75 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # warm-start with symbolic AST model
    symbolic_component = np.array([symbolic_forward(row) for row in X_train])
    y_train_adjusted = y_train - 0.05 * symbolic_component

    weights = [0.1, 0.1, 0.1]
    for _ in range(60):
        loss = model.search_update(X_train, y_train_adjusted, lr=0.03)
        grads = [loss * w for w in weights]
        weights = training_step(weights, grads, 0.01)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = float(np.mean((train_pred - y_train_adjusted) ** 2))
    test_mse = float(np.mean((test_pred - y_test) ** 2))
    test_mae = float(np.mean(np.abs(test_pred - y_test)))

    # Phase 3: Output design (graphs + summary)
    plt.figure(figsize=(8, 4))
    plt.plot(model.history, label="Training Loss")
    plt.title("Manifold Riemann Search Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(y_test, label="Actual (standardized)")
    plt.plot(test_pred, label="Predicted", linestyle="--")
    plt.title("Test Series: Actual vs Predicted")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "prediction_curve.png", dpi=150)
    plt.close()

    return ExperimentResult(
        train_mse=train_mse,
        test_mse=test_mse,
        test_mae=test_mae,
        phases=phases,
        sample_space=sample_space_names(),
        invariants=model.state.invariants,
    )
