from __future__ import annotations

import json

from automl import run_experiment


if __name__ == "__main__":
    result = run_experiment(output_dir="results")
    print(json.dumps({
        "train_mse": result.train_mse,
        "test_mse": result.test_mse,
        "test_mae": result.test_mae,
        "phases": result.phases,
        "sample_space_size": len(result.sample_space),
        "invariants": result.invariants,
    }, indent=2))
