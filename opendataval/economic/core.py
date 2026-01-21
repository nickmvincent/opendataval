from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ContextSplit:
    name: str
    indices: np.ndarray
    phi: float = 1.0

    def size(self) -> int:
        return int(self.indices.size)


def _to_numpy(array) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _subset(array, indices: np.ndarray):
    if isinstance(array, torch.Tensor):
        idx = torch.as_tensor(indices, dtype=torch.long, device=array.device)
        return array.index_select(0, idx)
    return array[indices]


def normalize_context_probs(probs: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(probs.values()))
    if total <= 0.0:
        raise ValueError("Context probabilities must sum to a positive value.")
    return {name: float(value) / total for name, value in probs.items()}


def label_contexts(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    phi_by_label: Mapping[int, float],
    prefix: str = "label",
) -> list[ContextSplit]:
    label_array = _to_numpy(labels)
    if label_array.ndim > 1:
        label_array = label_array.argmax(axis=1)

    contexts = []
    for label, phi in phi_by_label.items():
        indices = np.where(label_array == label)[0]
        if indices.size == 0:
            continue
        contexts.append(
            ContextSplit(
                name=f"{prefix}_{label}",
                indices=indices.astype(int, copy=False),
                phi=float(phi),
            )
        )
    return contexts


def threshold_contexts(
    covariates: np.ndarray | torch.Tensor,
    feature_index: int = 0,
    threshold: float = 0.0,
    phi_low: float = 1.0,
    phi_high: float = 1.0,
    names: tuple[str, str] = ("low", "high"),
) -> list[ContextSplit]:
    covar = _to_numpy(covariates)
    if covar.ndim == 1:
        covar = covar.reshape(-1, 1)
    if feature_index >= covar.shape[1]:
        raise ValueError("feature_index exceeds covariate dimension.")

    low_mask = covar[:, feature_index] <= threshold
    high_mask = ~low_mask

    low_indices = np.where(low_mask)[0]
    high_indices = np.where(high_mask)[0]

    contexts = []
    if low_indices.size:
        contexts.append(
            ContextSplit(
                name=names[0],
                indices=low_indices.astype(int, copy=False),
                phi=float(phi_low),
            )
        )
    if high_indices.size:
        contexts.append(
            ContextSplit(
                name=names[1],
                indices=high_indices.astype(int, copy=False),
                phi=float(phi_high),
            )
        )
    return contexts


def value_by_context(
    context_values: Mapping[str, np.ndarray],
    contexts: Sequence[ContextSplit],
) -> dict[str, np.ndarray]:
    values = {}
    for context in contexts:
        if context.name not in context_values:
            raise KeyError(f"Missing context value for {context.name}.")
        values[context.name] = context.phi * context_values[context.name]
    return values


def aggregate_value_by_context(
    context_values: Mapping[str, np.ndarray],
    context_probs: Mapping[str, float],
) -> np.ndarray:
    probs = normalize_context_probs(context_probs)
    values = None
    for name, value in context_values.items():
        if name not in probs:
            continue
        weighted = probs[name] * value
        values = weighted if values is None else values + weighted
    if values is None:
        raise ValueError("No overlapping context probabilities provided.")
    return values


def aggregate_values(
    context_values: Mapping[str, np.ndarray],
    contexts: Sequence[ContextSplit],
    context_probs: Mapping[str, float],
) -> np.ndarray:
    return aggregate_value_by_context(value_by_context(context_values, contexts), context_probs)


def sample_ex_post_context_values(
    context_values: Mapping[str, np.ndarray],
    contexts: Sequence[ContextSplit],
    counts: Mapping[str, int],
    sigma_eta: float,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng() if rng is None else rng
    noisy_values = {}
    for context in contexts:
        if context.name not in context_values:
            raise KeyError(f"Missing context value for {context.name}.")
        count = int(counts.get(context.name, 0))
        if count <= 0:
            raise ValueError(f"Count for context {context.name} must be positive.")
        base_value = context.phi * context_values[context.name]
        noise_scale = sigma_eta / np.sqrt(count)
        noise = rng.normal(scale=noise_scale, size=base_value.shape)
        noisy_values[context.name] = base_value + noise
    return noisy_values


class EconomicInfluenceEstimator:
    def __init__(
        self,
        dataval_factory: Callable[[], object],
        pred_model_factory: Optional[Callable[[], object]] = None,
        metric: Optional[Callable[..., float]] = None,
        train_kwargs: Optional[dict[str, object]] = None,
    ):
        self.dataval_factory = dataval_factory
        self.pred_model_factory = pred_model_factory
        self.metric = metric
        self.train_kwargs = {} if train_kwargs is None else dict(train_kwargs)

    def evaluate_contexts(
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        contexts: Sequence[ContextSplit],
    ) -> dict[str, np.ndarray]:
        context_values = {}
        for context in contexts:
            x_ctx = _subset(x_valid, context.indices)
            y_ctx = _subset(y_valid, context.indices)

            dataval = self.dataval_factory()
            if not hasattr(dataval, "input_data"):
                raise TypeError("dataval_factory must return a DataEvaluator-like object.")

            dataval.input_data(x_train, y_train, x_ctx, y_ctx)
            if self.pred_model_factory is not None and hasattr(dataval, "input_model"):
                dataval.input_model(self.pred_model_factory())
            if self.metric is not None and hasattr(dataval, "input_metric"):
                dataval.input_metric(self.metric)

            dataval.train_data_values(**self.train_kwargs)
            context_values[context.name] = np.asarray(dataval.evaluate_data_values())
        return context_values
