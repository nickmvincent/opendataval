from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from opendataval.dataloader import DataFetcher
from opendataval.dataloader.register import one_hot_encode
from opendataval.dataval import InfluenceFunction, LeaveOneOut
from opendataval.economic import (
    EconomicInfluenceEstimator,
    aggregate_values,
    label_contexts,
    sample_ex_post_context_values,
)
from opendataval.metrics import accuracy
from opendataval.model import LogisticRegression


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata(a)
    rb = rankdata(b)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def make_synthetic_gaussians(
    total_points: int,
    rng: np.random.Generator,
    mean_offset: float = 2.0,
    cov_scale: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    n0 = total_points // 2
    n1 = total_points - n0

    mean0 = np.array([-mean_offset / 2.0, 0.0])
    mean1 = np.array([mean_offset / 2.0, 0.0])
    cov = np.eye(2) * cov_scale

    x0 = rng.multivariate_normal(mean0, cov, size=n0)
    x1 = rng.multivariate_normal(mean1, cov, size=n1)
    y0 = np.zeros(n0, dtype=int)
    y1 = np.ones(n1, dtype=int)

    x = np.vstack([x0, x1])
    y = np.concatenate([y0, y1])
    indices = rng.permutation(total_points)
    return x[indices], y[indices]


def build_fetcher(
    dataset_name: str,
    seed: int,
    train_count: int,
    valid_count: int,
    test_count: int,
    synthetic_total: int,
) -> DataFetcher:
    if dataset_name == "synthetic":
        rng = np.random.default_rng(seed)
        x, y = make_synthetic_gaussians(synthetic_total, rng)
        y_one_hot = one_hot_encode(y)
        fetcher = DataFetcher.from_data(x, y_one_hot, one_hot=True, random_state=seed)
        return fetcher.split_dataset_by_count(train_count, valid_count, test_count)

    fetcher = DataFetcher(dataset_name, random_state=seed)
    return fetcher.split_dataset_by_count(train_count, valid_count, test_count)


def to_label_indices(labels: torch.Tensor | np.ndarray) -> np.ndarray:
    label_array = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    if label_array.ndim > 1:
        return label_array.argmax(axis=1)
    return label_array.reshape(-1)


def plot_demand_shift(values_low: np.ndarray, values_high: np.ndarray, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.scatter(values_low, values_high, s=18, alpha=0.6)

    min_val = min(values_low.min(), values_high.min())
    max_val = max(values_low.max(), values_high.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)

    ax.set_xlabel("value under low-demand")
    ax.set_ylabel("value under high-demand")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_forecast_mismatch(
    probs: np.ndarray,
    errors: np.ndarray,
    corrs: np.ndarray,
    path: Path,
    title: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    axes[0].plot(probs, errors, marker="o")
    axes[0].set_xlabel("forecast share for high-demand")
    axes[0].set_ylabel("mean abs error")

    axes[1].plot(probs, corrs, marker="o")
    axes[1].set_xlabel("forecast share for high-demand")
    axes[1].set_ylabel("rank correlation")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ex_post_noise(context_names: list[str], stds: list[float], path: Path, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.bar(context_names, stds, color="#4C78A8")
    ax.set_ylabel("avg std of noisy value")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def context_noise_stats(
    context_values: dict[str, np.ndarray],
    contexts,
    counts: dict[str, int],
    sigma_eta: float,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    samples = {context.name: [] for context in contexts}
    for _ in range(draws):
        noisy = sample_ex_post_context_values(context_values, contexts, counts, sigma_eta, rng)
        for name, values in noisy.items():
            samples[name].append(values)

    stats = {}
    for name, values in samples.items():
        stack = np.stack(values, axis=0)
        stats[name] = float(np.mean(np.std(stack, axis=0)))
    return stats


def run_method(
    method_name: str,
    dataval_factory,
    fetcher: DataFetcher,
    output_dir: Path,
    seed: int,
    train_kwargs: dict[str, float],
    deploy_total: int,
    sigma_eta: float,
    phi_high: float,
):
    x_train, y_train, x_valid, y_valid, *_ = fetcher.datapoints
    labels_valid = to_label_indices(y_valid)

    unique_labels = sorted(set(labels_valid.tolist()))
    if len(unique_labels) != 2:
        raise ValueError("This script expects a binary dataset with two contexts.")

    phi_by_label = {unique_labels[0]: 1.0, unique_labels[1]: phi_high}
    contexts = label_contexts(labels_valid, phi_by_label, prefix="label")

    if len(contexts) != 2:
        raise ValueError("Expected exactly two contexts.")

    num_classes = y_train.shape[1] if y_train.ndim > 1 else int(np.max(labels_valid)) + 1
    input_dim = x_train.shape[1]
    pred_model_factory = lambda: LogisticRegression(input_dim, num_classes)

    estimator = EconomicInfluenceEstimator(
        dataval_factory=dataval_factory,
        pred_model_factory=pred_model_factory,
        metric=accuracy,
        train_kwargs=train_kwargs,
    )

    context_values = estimator.evaluate_contexts(x_train, y_train, x_valid, y_valid, contexts)

    low_demand = {contexts[0].name: 0.8, contexts[1].name: 0.2}
    high_demand = {contexts[0].name: 0.2, contexts[1].name: 0.8}

    values_low = aggregate_values(context_values, contexts, low_demand)
    values_high = aggregate_values(context_values, contexts, high_demand)

    plot_demand_shift(
        values_low,
        values_high,
        output_dir / f"{method_name}_demand_shift.png",
        f"{method_name}: demand shift",
    )

    probs = np.linspace(0.05, 0.95, 19)
    base_values = aggregate_values(context_values, contexts, low_demand)

    errors = []
    corrs = []
    for prob in probs:
        forecast = {contexts[0].name: 1.0 - prob, contexts[1].name: prob}
        forecast_values = aggregate_values(context_values, contexts, forecast)
        errors.append(float(np.mean(np.abs(forecast_values - base_values))))
        corrs.append(spearman_corr(forecast_values, base_values))

    plot_forecast_mismatch(
        probs,
        np.array(errors),
        np.array(corrs),
        output_dir / f"{method_name}_forecast_mismatch.png",
        f"{method_name}: forecast mismatch",
    )

    counts = {
        contexts[0].name: max(1, int(deploy_total * low_demand[contexts[0].name])),
        contexts[1].name: max(1, deploy_total - int(deploy_total * low_demand[contexts[0].name])),
    }

    rng = np.random.default_rng(seed)
    stats = context_noise_stats(context_values, contexts, counts, sigma_eta, draws=200, rng=rng)

    plot_ex_post_noise(
        list(stats.keys()),
        [stats[name] for name in stats],
        output_dir / f"{method_name}_ex_post_noise.png",
        f"{method_name}: ex-post noise",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Economic influence toy experiments")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--output-dir", default="experiments/economic_influence/output")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--methods", default="influence,loo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_count = 60
    valid_count = 40
    test_count = 40
    synthetic_total = train_count + valid_count + test_count

    fetcher = build_fetcher(
        args.dataset,
        args.seed,
        train_count,
        valid_count,
        test_count,
        synthetic_total,
    )

    train_kwargs = {"epochs": 20, "batch_size": 32, "lr": 0.05}
    deploy_total = 200
    sigma_eta = 0.5
    phi_high = 1.5

    method_map = {
        "influence": lambda: InfluenceFunction(),
        "loo": lambda: LeaveOneOut(random_state=np.random.RandomState(args.seed)),
    }

    requested = [method.strip() for method in args.methods.split(",") if method.strip()]
    for method_name in requested:
        if method_name not in method_map:
            raise ValueError(f"Unknown method: {method_name}")

    for method_name in requested:
        run_method(
            method_name,
            method_map[method_name],
            fetcher,
            output_dir,
            args.seed,
            train_kwargs,
            deploy_total,
            sigma_eta,
            phi_high,
        )


if __name__ == "__main__":
    main()
