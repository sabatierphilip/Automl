"""Model-design phase: n-dimensional manifold with Riemann search."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class ManifoldState:
    center: np.ndarray
    metric: np.ndarray
    dimension: int
    invariants: dict = field(default_factory=dict)


class ManifoldAutoMLModel:
    def __init__(self, start_dimension: int = 2):
        self.state = ManifoldState(
            center=np.zeros(start_dimension, dtype=float),
            metric=np.eye(start_dimension, dtype=float),
            dimension=start_dimension,
        )
        self.history: List[float] = []

    def ensure_dimension(self, n: int) -> None:
        if n <= self.state.dimension:
            return
        old_d = self.state.dimension
        new_center = np.zeros(n)
        new_center[:old_d] = self.state.center
        new_metric = np.eye(n)
        new_metric[:old_d, :old_d] = self.state.metric
        self.state.center = new_center
        self.state.metric = new_metric
        self.state.dimension = n

    def _anti_entropy(self, v: np.ndarray) -> np.ndarray:
        # phi(t)=t/(1+t^2); primitive Phi(t)=0.5*log(1+t^2)
        return v / (1.0 + v**2)

    def _primitive(self, v: np.ndarray) -> np.ndarray:
        return 0.5 * np.log1p(v**2)

    def _riemann_distance(self, x: np.ndarray) -> float:
        d = x - self.state.center
        return float(np.sqrt(d.T @ self.state.metric @ d))

    def search_update(self, X: np.ndarray, y: np.ndarray, lr: float = 0.02) -> float:
        self.ensure_dimension(X.shape[1])
        grad_center = np.zeros_like(self.state.center)
        losses: List[float] = []

        for xi, yi in zip(X, y):
            d = xi - self.state.center
            anti = self._anti_entropy(d)
            prim = self._primitive(d)
            zero_mask = np.isclose(prim, 0.0, atol=1e-5)
            runge_boost = 1.0 / (np.abs(prim) + 1e-3)
            runge_boost = np.clip(runge_boost, 0.0, 50.0)
            anti = anti * runge_boost

            # stable integral proxy using bounded tanh transform
            stable_integral = np.trapezoid(np.tanh(anti), dx=1.0 / max(len(anti), 1))
            pred = stable_integral + 0.1 * self._riemann_distance(xi)
            loss = (pred - yi) ** 2
            losses.append(loss)

            grad_center += 2.0 * (pred - yi) * (-anti)
            grad_center[zero_mask] *= 0.5

        grad_center /= len(X)
        self.state.center -= lr * grad_center

        # update metric with invariant: positive-definite + bounded trace
        g = np.outer(grad_center, grad_center)
        self.state.metric += lr * g
        eigvals, eigvecs = np.linalg.eigh(self.state.metric)
        eigvals = np.clip(eigvals, 1e-3, 10.0)
        self.state.metric = eigvecs @ np.diag(eigvals) @ eigvecs.T

        self.state.invariants = {
            "trace_metric": float(np.trace(self.state.metric)),
            "center_norm": float(np.linalg.norm(self.state.center)),
            "positive_definite": bool(np.all(np.linalg.eigvals(self.state.metric) > 0)),
        }
        epoch_loss = float(np.mean(losses))
        self.history.append(epoch_loss)
        return epoch_loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for xi in X:
            self.ensure_dimension(xi.shape[0])
            anti = self._anti_entropy(xi - self.state.center)
            prim = self._primitive(xi - self.state.center)
            runge_boost = np.clip(1.0 / (np.abs(prim) + 1e-3), 0.0, 50.0)
            anti = anti * runge_boost
            stable_integral = np.trapezoid(np.tanh(anti), dx=1.0 / max(len(anti), 1))
            preds.append(stable_integral + 0.1 * self._riemann_distance(xi))
        return np.asarray(preds)
