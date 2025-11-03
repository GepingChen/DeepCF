from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from IV_stage2_9 import (
    Stage2_9Config,
    FullDataStructuralFunctionModel,
    ConditionalCDFEstimator,
    sample_tabpfn_y_given_x,
)
from IV_stage1_4 import Stage1Config, load_dgp_data, CondCDFModel

try:
    from tabpfn.regressor import TabPFNRegressor
except Exception:
    from tabpfn import TabPFNRegressor  # type: ignore[import]


def _prepare_components_from_train(
    cfg: Stage2_9Config,
    stage1_cfg: Stage1Config,
) -> Dict[str, object]:
    """
    Build Stage 2/3 components by computing V_hat from raw training data.
    """
    codes = f"{cfg.first_stage_code}_{cfg.second_stage_code}"
    train_df, _ = load_dgp_data(
        cfg.first_stage_code,
        cfg.second_stage_code,
        train_sample_size=cfg.n_train_samples,
        base_dir=cfg.dgp_base_dir,
    )

    observed_train_samples = len(train_df)
    if cfg.n_train_samples is not None and observed_train_samples != cfg.n_train_samples:
        raise ValueError(
            f"Training CSV has {observed_train_samples} rows; expected {cfg.n_train_samples}."
        )

    Z_train = train_df["Z"].to_numpy(dtype=float)
    X_train = train_df["X"].to_numpy(dtype=float)
    Y_train = train_df["Y"].to_numpy(dtype=float)

    stage1_model = CondCDFModel(quantiles=stage1_cfg.quantiles)
    stage1_model.fit(Z_train, X_train)
    V_train_hat = stage1_model.predict(Z_train, X_train)

    train_data = {
        "Z": Z_train,
        "X": X_train,
        "Y": Y_train,
        "V_hat": V_train_hat,
        "V_true": train_df["V_true"].to_numpy(dtype=float) if "V_true" in train_df else None,
        "eps": train_df["eps"].to_numpy(dtype=float) if "eps" in train_df else None,
        "eta": train_df["eta"].to_numpy(dtype=float) if "eta" in train_df else None,
    }

    m_model = FullDataStructuralFunctionModel(use_tabpfn=cfg.use_tabpfn)
    _ = m_model.fit_full(train_data["X"], train_data["V_hat"], train_data["Y"])

    cdf_model = ConditionalCDFEstimator(use_tabpfn=cfg.use_tabpfn)
    _ = cdf_model.fit_full(train_data["X"], train_data["V_hat"], train_data["Y"])

    return {
        "codes": codes,
        "train_data": train_data,
        "train_df": train_df,
        "m_model": m_model,
        "cdf_model": cdf_model,
        "n_train_samples": observed_train_samples,
        "stage1_model": stage1_model,
    }


@dataclass
class FunctionalSpec:
    """Specification of a causal functional θ(x) = F(P_x)."""

    statistic: Callable[[np.ndarray], float]
    name: str = "theta"
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class FunctionalCurveResult:
    """Container for evaluated functional values on a grid of x."""

    x_grid: np.ndarray
    estimates: np.ndarray
    functional_name: str
    raw_samples: List[np.ndarray]
    metadata: Dict[str, object] = field(default_factory=dict)
    smoothed_x_grid: Optional[np.ndarray] = None
    smoothed_estimates: Optional[np.ndarray] = None

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy DataFrame with x and the plug-in estimates."""
        return pd.DataFrame(
            {
                "x": self.x_grid.astype(float),
                self.functional_name: self.estimates.astype(float),
            }
        )

    def to_smoothed_frame(self) -> pd.DataFrame:
        """Return a DataFrame for the smoothed curve if available."""
        if self.smoothed_x_grid is None or self.smoothed_estimates is None:
            raise ValueError("Smoothed curve is not available.")
        return pd.DataFrame(
            {
                "x": self.smoothed_x_grid.astype(float),
                self.functional_name: self.smoothed_estimates.astype(float),
            }
        )

    def plot(
        self,
        ax=None,
        *,
        xlabel: str = "x",
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        line_kwargs: Optional[Dict[str, object]] = None,
    ):
        """
        Plot the functional curve using matplotlib.

        Args:
            ax: Optional matplotlib Axes to plot on. Creates one if omitted.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis. Defaults to functional_name.
            title: Optional plot title.
            save_path: If provided, save figure to this path.
            show: If True, call plt.show() before returning.
            line_kwargs: Optional style dictionary for the line plot.
        """
        import matplotlib.pyplot as plt

        owns_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            owns_fig = True
        else:
            fig = ax.figure

        base_line_style = {"color": "#2c7fb8", "linewidth": 2.0}
        if line_kwargs:
            base_line_style.update(line_kwargs)

        if self.smoothed_x_grid is not None and self.smoothed_estimates is not None:
            line_style = dict(base_line_style)
            line_style.setdefault("label", "Smoothed curve")
            ax.plot(self.smoothed_x_grid, self.smoothed_estimates, **line_style)
            ax.scatter(
                self.x_grid,
                self.estimates,
                color="#1b9e77",
                s=20,
                alpha=0.6,
                label="Grid estimates",
            )
            ax.legend(loc="best", frameon=False)
        else:
            line_style = dict(base_line_style)
            line_style.setdefault("label", "Grid estimates")
            ax.plot(self.x_grid, self.estimates, **line_style)
            ax.legend(loc="best", frameon=False)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or self.functional_name)
        if title:
            ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        if owns_fig and not show:
            plt.close(fig)


class FunctionalCurveEngine:
    """
    Lightweight engine for scalar functional estimation using Y|do(X=x) samples.

    This first iteration ignores smoothing and uncertainty quantification.
    It focuses on:
        1. Building a scalar x-grid over a user-specified range.
        2. Sampling Y|do(X=x) via the Stage 2 TabPFN models.
        3. Evaluating a plug-in functional on the Monte Carlo samples.
    """

    def __init__(
        self,
        stage2_config: Stage2_9Config,
        *,
        stage1_config: Optional[Stage1Config] = None,
        y_support: Optional[np.ndarray] = None,
        y_support_points: int = 401,
        y_support_padding: float = 0.05,
    ) -> None:
        self.cfg = stage2_config
        self.stage1_cfg = stage1_config or Stage1Config()
        self.components = _prepare_components_from_train(stage2_config, self.stage1_cfg)
        self.cdf_model = self.components["cdf_model"]
        self.train_data = self.components["train_data"]

        self.y_support = (
            np.asarray(y_support, dtype=float)
            if y_support is not None
            else self._build_default_y_support(
                y_support_points=y_support_points, padding=y_support_padding
            )
        )

        if self.y_support.ndim != 1:
            raise ValueError("y_support must be a one-dimensional grid.")
        if len(self.y_support) < 10:
            raise ValueError("y_support grid is too coarse; need at least 10 points.")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def infer_x_range(
        self, lower_quantile: float = 0.01, upper_quantile: float = 0.99
    ) -> tuple[float, float]:
        """
        Suggest an x-range based on the observed support in the training data.

        Quantiles are taken from the Stage-1 training X samples to avoid
        extrapolation while remaining robust to outliers.
        """
        x_train = np.asarray(self.train_data["X"], dtype=float)
        stacked = x_train.astype(float, copy=False)
        lower = float(np.quantile(stacked, lower_quantile))
        upper = float(np.quantile(stacked, upper_quantile))
        if lower >= upper:
            raise ValueError(
                "Inferred x-range is degenerate. Adjust quantiles or provide a range."
            )
        return lower, upper

    def build_x_grid(
        self,
        x_min: float,
        x_max: float,
        num_points: int,
        *,
        method: str = "linspace",
    ) -> np.ndarray:
        """
        Construct a scalar grid between x_min and x_max.

        Currently only supports equally spaced grids, aligning with the MVP.
        """
        if num_points < 2:
            raise ValueError("num_points must be at least 2 to form a grid.")
        if method != "linspace":
            raise NotImplementedError(f"Grid method '{method}' is not implemented yet.")
        return np.linspace(float(x_min), float(x_max), int(num_points))

    def estimate_functional_curve(
        self,
        functional: FunctionalSpec,
        x_grid: Sequence[float],
        *,
        mc_samples: int,
        seed: Optional[int] = None,
        store_samples: bool = True,
        fit_smoother: bool = True,
        smoother_grid_points: int = 2000,
    ) -> FunctionalCurveResult:
        """
        Evaluate θ(x) on a provided grid using Monte Carlo plug-in estimates.

        Args:
            functional: Specification of the plug-in functional.
            x_grid: Iterable of scalar x values.
            mc_samples: Number of Y draws per grid point (m in the plan).
            seed: Optional seed for reproducibility.
            store_samples: Whether to keep Monte Carlo draws in the result bundle.
        """
        if mc_samples <= 0:
            raise ValueError("mc_samples must be a positive integer.")

        x_array = np.asarray(list(x_grid), dtype=float)
        rng = np.random.default_rng(seed)

        estimates = np.empty_like(x_array, dtype=float)
        all_samples: List[np.ndarray] = []

        # Evaluate the functional at each grid point.
        for idx, x_value in enumerate(x_array):
            y_samples = self._draw_interventional_samples(x_value, mc_samples, rng=rng)
            estimates[idx] = float(functional.statistic(y_samples))
            if store_samples:
                all_samples.append(y_samples)

        if not store_samples:
            all_samples = []

        metadata = {
            "mc_samples": mc_samples,
            "seed": seed,
            "stage2_config": self.cfg,
            "stage1_config": self.stage1_cfg,
            "y_support_min": float(self.y_support.min()),
            "y_support_max": float(self.y_support.max()),
            "y_support_points": int(len(self.y_support)),
            "fit_smoother": bool(fit_smoother),
            "smoother_grid_points": int(smoother_grid_points) if fit_smoother else None,
        }
        metadata.update(functional.metadata)

        smoothed_x: Optional[np.ndarray] = None
        smoothed_estimates: Optional[np.ndarray] = None
        if fit_smoother:
            smoothed_x, smoothed_estimates = self._fit_stage3_curve(
                x_array,
                estimates,
                smoother_grid_points=smoother_grid_points,
                seed=seed,
            )

        return FunctionalCurveResult(
            x_grid=x_array,
            estimates=estimates,
            functional_name=functional.name,
            raw_samples=all_samples,
            metadata=metadata,
            smoothed_x_grid=smoothed_x,
            smoothed_estimates=smoothed_estimates,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_default_y_support(
        self, *, y_support_points: int, padding: float
    ) -> np.ndarray:
        """
        Construct a Y grid from observed outcomes with optional padding.
        """
        if y_support_points <= 10:
            raise ValueError("y_support_points should exceed 10 for stable sampling.")

        y_train = np.asarray(self.train_data["Y"], dtype=float)
        stacked = y_train.astype(float, copy=False)
        y_min = stacked.min()
        y_max = stacked.max()
        span = y_max - y_min

        if span <= 0:
            span = max(1.0, abs(y_min) * 0.1)

        pad = span * float(max(padding, 0.0))
        low = y_min - pad
        high = y_max + pad

        return np.linspace(low, high, int(y_support_points))

    def _draw_interventional_samples(
        self, x_value: float, n_samples: int, *, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample Y|do(X=x) using the Stage 2 interventional CDF estimates.
        """
        y_samples, _, _ = sample_tabpfn_y_given_x(
            self.cdf_model,
            float(x_value),
            int(n_samples),
            self.y_support,
            rng,
        )
        return np.asarray(y_samples, dtype=float)

    def _fit_stage3_curve(
        self,
        x_obs: np.ndarray,
        theta_obs: np.ndarray,
        *,
        smoother_grid_points: int,
        seed: Optional[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit TabPFNRegressor on (x, θ̂(x)) pairs and predict on a dense grid.
        """
        if smoother_grid_points <= 1:
            raise ValueError("smoother_grid_points must be greater than 1.")

        reg = TabPFNRegressor(
            random_state=seed if seed is not None else 1,
            ignore_pretraining_limits=True,
        )
        features = np.asarray(x_obs, dtype=float).reshape(-1, 1)
        targets = np.asarray(theta_obs, dtype=float)
        reg.fit(features, targets)

        x_min = float(np.min(features))
        x_max = float(np.max(features))
        dense_grid = np.linspace(x_min, x_max, int(smoother_grid_points))
        preds = reg.predict(dense_grid.reshape(-1, 1))
        return dense_grid.astype(float), np.asarray(preds, dtype=float)
