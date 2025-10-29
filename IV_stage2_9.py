"""
IV Stage 2.9 Implementation: Density Diagnostics
================================================

Stage 2.9 extends Stage 2.8's full-distribution integration with diagnostics
targeted at the interventional density f(y | do(X = x)).

New in Stage 2.9:
- Align μ_c(x) evaluation points with the held-out test samples and export the
  paired `y_test` column for the μ_c curves CSV.
- Integrate criterion-based PDFs to recover f(y|do(X=x)) and compare with
  analytical ground-truth curves on the same grid.
- Produce a Figure-3-style kernel density visualization for representative test
  points, sourced from configurable X-quantiles.

Stage 2.8 foundations retained:
- μ_c(x) via Simpson integration over TabPFN structural predictions.
- F(y|do(X=x)) without quantile inversion, using criterion.cdf directly.
- Deterministic integration grids and reproducible output artefacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
from scipy.stats import norm, gaussian_kde
from scipy.integrate import simpson
import os
from datetime import datetime
import torch

# Import from Stage 1
from DGP import DGPConfig, set_seed, sigmaY_of_X

V_EPSILON = 1e-6  # Avoid norm.ppf endpoints
B4_SOFTPLUS_EPS = 1e-8  # Match numerical guard in DGP B4 branch

# --- TabPFN (with fallbacks) ---
try:
    from tabpfn.regressor import TabPFNRegressor
    _HAVE_TABPFN = True
    print("✅ TabPFNRegressor imported successfully")
except Exception as e:
    raise ImportError(f"TabPFNRegressor is required for Stage 2.9 but could not be imported: {e}")

# =========================================================
# Stage 1 data loading helper
# =========================================================


# 1) Extended Configuration and I/O helpers
# =========================================================


def _latest_matching_file(directory: str, prefix: str):
    """Return the most recently modified CSV in directory that starts with prefix."""
    import os
    if not os.path.isdir(directory):
        return None
    cands = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".csv")]
    if not cands:
        return None
    cands.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
    return os.path.join(directory, cands[0])

def load_stage1_data(csv_path: str) -> Dict[str, np.ndarray]:
    """Load Stage 1 outputs from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Stage 1 CSV file not found: {csv_path}\n"
            "Please run IV_stage1_1.py first to generate Stage 1 outputs."
        )

    print(f"Loading Stage 1 data from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = ["X", "Y", "V_hat", "V_true"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in CSV: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    data = {
        "Z": df["Z"].to_numpy() if "Z" in df.columns else None,
        "X": df["X"].to_numpy(),
        "Y": df["Y"].to_numpy(),
        "V_hat": df["V_hat"].to_numpy(),
        "V_true": df["V_true"].to_numpy(),
        "eps": df["eps"].to_numpy() if "eps" in df.columns else None,
        "eta": df["eta"].to_numpy() if "eta" in df.columns else None,
    }

    V_hat = data["V_hat"]
    if np.any(V_hat < 0) or np.any(V_hat > 1):
        print("⚠️  Warning: V_hat has values outside [0,1]. Clipping to [0,1].")
        print(f"   Original range: [{V_hat.min():.4f}, {V_hat.max():.4f}]")
        data["V_hat"] = np.clip(V_hat, 0.0, 1.0)

    n = len(data["X"])
    print(f"✅ Loaded {n} samples from Stage 1")
    print(f"   X range: [{data['X'].min():.3f}, {data['X'].max():.3f}]")
    print(f"   Y range: [{data['Y'].min():.3f}, {data['Y'].max():.3f}]")
    print(f"   V_hat range: [{data['V_hat'].min():.3f}, {data['V_hat'].max():.3f}]")
    print(f"   V_true range: [{data['V_true'].min():.3f}, {data['V_true'].max():.3f}]")

    return data


def load_dgp_test_data(first_stage: str, second_stage: str, base_dir: str = "IV_datasets") -> Tuple[str, Dict[str, Optional[np.ndarray]]]:
    """
    Load immutable DGP test split from IV_datasets directory.

    Returns the resolved CSV path along with column arrays needed downstream.
    """
    test_file = os.path.join(base_dir, "test", f"test_data_{first_stage}_{second_stage}.csv")
    if not os.path.exists(test_file):
        raise FileNotFoundError(
            f"DGP test data not found: {test_file}\n"
            "Please ensure the pre-generated datasets are available under IV_datasets/test."
        )

    print(f"Loading DGP test data from: {test_file}")
    df = pd.read_csv(test_file)

    required_cols = ["Z", "X", "Y", "V_true"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"DGP test CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    data: Dict[str, Optional[np.ndarray]] = {
        "Z": df["Z"].to_numpy(),
        "X": df["X"].to_numpy(),
        "Y": df["Y"].to_numpy(),
        "V_true": df["V_true"].to_numpy(),
        "V_hat": None,
        "eps": df["eps"].to_numpy() if "eps" in df.columns else None,
        "eta": df["eta"].to_numpy() if "eta" in df.columns else None,
    }

    return test_file, data

# =========================================================
# Structural function model from Stage 2.1
# =========================================================
class StructuralFunctionModel:
    """Estimate structural function m(x,v) = E[Y|X=x, V=v]."""

    def __init__(self, use_tabpfn: bool = True):
        self.use_tabpfn = use_tabpfn and _HAVE_TABPFN
        self.model = None
        self._using_tabpfn = False

    def _new_regressor(self):
        """Create new regressor instance (TabPFN or fallback)."""
        if self.use_tabpfn:
            try:
                self._using_tabpfn = True
                return TabPFNRegressor(random_state=42, ignore_pretraining_limits=True)
            except Exception as e:
                print(f"⚠️  TabPFN initialization failed: {e}")
                raise RuntimeError('Stage 2.9 requires TabPFNRegressor; initialization or use_tabpfn failed')

    def predict(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Predict m(x, v) for given arrays."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit_full() first.")

        x = np.asarray(x).reshape(-1, 1)
        v = np.asarray(v).reshape(-1, 1)
        features = np.hstack([x, v])
        return self.model.predict(features)

    def __call__(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.predict(x, v)


class FullDataStructuralFunctionModel(StructuralFunctionModel):
    """Structural function model trained once on the full dataset.

    This augments the Stage 2.1 implementation by exposing a helper that
    fits TabPFN (or the fallback) without cross validation.
    """

    def fit_full(self, X: np.ndarray, V: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.asarray(X).reshape(-1, 1)
        V = np.asarray(V).reshape(-1, 1)
        Y = np.asarray(Y)
        features = np.hstack([X, V])

        print("Training structural function on full dataset (no CV)...")
        reg = self._new_regressor()
        reg.fit(features, Y)
        self.model = reg

        preds = self.model.predict(features)
        method = "TabPFN" if getattr(self, '_using_tabpfn', False) else "RandomForest"
        print(f"✅ {method} structural function fitted on {len(Y)} samples")
        return preds


@dataclass
class Stage2_9Config:
    """Configuration for Stage 2.9 diagnostics with numerical integration"""
    # Input/output
    input_dir: str = "IV_datasets/stage1_output"
    output_dir: str = "IV_datasets/stage2_output"
    dgp_base_dir: str = "IV_datasets"
    
    random_state: int = 1
    
    # Test grids (uniform grids)
    n_x_test: int = 50      # Number of test X points (uniform grid)
    n_y_grid: int = 100     # Number of Y points for CDF evaluation
    
    # V integration grid (NEW: numerical integration)
    n_v_integration_points: int = 100  # Number of V points for numerical integration
    
    # Model settings
    use_tabpfn: bool = True

    # Density diagnostics
    kde_quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75)
    kde_sample_size: int = 1000

    # DGP identifiers (for file selection)
    first_stage_code: str = "A1"
    second_stage_code: str = "B1"


def prepare_stage2_components(cfg: Stage2_9Config):
    """
    Load Stage 1 outputs, train Stage 2 models, and return reusable components.
    """
    codes = f"{cfg.first_stage_code}_{cfg.second_stage_code}"
    train_prefix = f"iv_stage1_train_{codes}_"
    train_csv = _latest_matching_file(cfg.input_dir, train_prefix)
    if train_csv is None:
        raise FileNotFoundError(f"No Stage-1 training CSV found in {cfg.input_dir} matching prefix {train_prefix}")

    print(f"Resolved Stage-1 training CSV: {train_csv}", flush=True)
    test_csv, test_data = load_dgp_test_data(
        cfg.first_stage_code,
        cfg.second_stage_code,
        base_dir=cfg.dgp_base_dir,
    )
    print(f"Resolved DGP test CSV: {test_csv}", flush=True)

    train_data = load_stage1_data(train_csv)

    dgp_cfg = DGPConfig(first_stage=cfg.first_stage_code, second_stage=cfg.second_stage_code)

    m_model = FullDataStructuralFunctionModel(use_tabpfn=cfg.use_tabpfn)
    _ = m_model.fit_full(train_data["X"], train_data["V_hat"], train_data["Y"])

    cdf_model = ConditionalCDFEstimator(use_tabpfn=cfg.use_tabpfn)
    _ = cdf_model.fit_full(train_data["X"], train_data["V_hat"], train_data["Y"])

    return {
        "codes": codes,
        "train_csv": train_csv,
        "test_csv": test_csv,
        "train_data": train_data,
        "test_data": test_data,
        "dgp_config": dgp_cfg,
        "m_model": m_model,
        "cdf_model": cdf_model,
    }

# =========================================================
# V Integration Grid Helper
# =========================================================
def create_v_integration_grid(n_points: int = 100) -> np.ndarray:
    """Create uniform grid over (0,1) for V integration.
    
    Uses Simpson's rule compatible spacing (odd number of points).
    
    Args:
        n_points: Number of integration points (will be adjusted to odd if needed)
        
    Returns:
        v_grid: (n_points,) array of V values in (0,1)
    """
    if n_points % 2 == 0:
        n_points += 1  # Simpson needs odd number
    return np.linspace(V_EPSILON, 1.0 - V_EPSILON, n_points)


def build_test_grid(
    X_test: np.ndarray,
    Y_test: np.ndarray,
    desired_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align the evaluation grid with observed test samples and record indices.

    Returns the X grid sorted in ascending order alongside the matching Y
    observations and the corresponding indices back into the original test
    arrays. If the test split has more points than requested, we pick evenly
    spaced indices to respect the requested resolution.
    """
    X_test = np.asarray(X_test, dtype=float)
    Y_test = np.asarray(Y_test, dtype=float)
    sort_idx = np.argsort(X_test)
    X_sorted = X_test[sort_idx]
    Y_sorted = Y_test[sort_idx]

    n_available = len(X_sorted)
    if desired_points is None or desired_points <= 0 or desired_points >= n_available:
        return X_sorted, Y_sorted, sort_idx

    pick_idx = np.linspace(0, n_available - 1, desired_points).round().astype(int)
    pick_idx = np.clip(pick_idx, 0, n_available - 1)
    return X_sorted[pick_idx], Y_sorted[pick_idx], sort_idx[pick_idx]


def select_kde_indices(n_points: int, quantiles: Tuple[float, ...]) -> np.ndarray:
    """
    Convert quantile requests into integer indices on the test grid.
    """
    if n_points <= 0:
        raise ValueError("Number of test points must be positive to select KDE indices.")
    q = np.asarray(quantiles, dtype=float)
    q = np.clip(q, 0.0, 1.0)
    idx = np.round(q * (n_points - 1)).astype(int)
    idx = np.clip(idx, 0, n_points - 1)
    ordered: list[int] = []
    seen = set()
    for val in idx:
        if int(val) not in seen:
            ordered.append(int(val))
            seen.add(int(val))
    if not ordered:
        ordered = [int((n_points - 1) // 2)]
    return np.asarray(ordered, dtype=int)

# =========================================================
# 2) Integrated Structural Function on Test Grid: μ_c(x)
# =========================================================
def compute_mu_c_on_grid(m_model: StructuralFunctionModel,
                         x_grid: np.ndarray,
                         n_v_points: int,
                         max_points_per_batch: int | None = None) -> np.ndarray:
    """
    Compute μ_c(x) = ∫₀¹ m̂(x,v) dv using numerical integration.
    
    This integrates the structural function over the V distribution using
    Simpson's rule, giving the causal effect E[Y|do(X=x)] at each test point x.
    
    Args:
        m_model: Fitted structural function model m̂(x,v)
        x_grid: (k,) array of NEW test x values
        n_v_points: Number of V integration points
        
    Returns:
        mu_c_grid: (k,) array of μ_c(x_j) for each test point
    """
    k = len(x_grid)
    v_grid = create_v_integration_grid(n_v_points)
    n_v = len(v_grid)

    print(
        "Computing μ_c(x) on test grid: "
        f"{k} x values × {n_v} V points (vectorized TabPFN inference)..."
    )

    if max_points_per_batch is None:
        target_products = 20000  # heuristic limit to keep GPU memory manageable
        if k * n_v <= target_products:
            max_points_per_batch = k
        else:
            max_points_per_batch = max(1, target_products // n_v)
    else:
        max_points_per_batch = max(1, int(max_points_per_batch))

    mu_c_grid = np.empty(k, dtype=float)

    for start in range(0, k, max_points_per_batch):
        end = min(start + max_points_per_batch, k)
        x_chunk = np.asarray(x_grid[start:end], dtype=float)
        repeat_count = len(x_chunk)
        x_col = np.repeat(x_chunk, n_v).astype(float)
        v_col = np.tile(v_grid, repeat_count).astype(float)

        m_vals = np.asarray(m_model.predict(x_col, v_col), dtype=float)
        if m_vals.size != repeat_count * n_v:
            raise ValueError(
                f"Unexpected structural predictions shape {m_vals.shape}; "
                f"expected {repeat_count * n_v} entries for the X×V grid."
            )

        m_matrix = m_vals.reshape(repeat_count, n_v)
        mu_c_grid[start:end] = simpson(m_matrix, x=v_grid, axis=1)

    print(f"✅ μ_c computed on test grid: mean={np.mean(mu_c_grid):.4f}, std={np.std(mu_c_grid):.4f}")
    return mu_c_grid

# =========================================================
# 3) Conditional CDF Estimation: F(y|X,V)
# =========================================================

def _broadcast_y_for_logits(y_values: np.ndarray, n_samples: int) -> np.ndarray:
    """Expand y inputs to align with the batch of logits."""
    y_arr = np.asarray(y_values, dtype=float)
    if y_arr.ndim == 0:
        return np.full((n_samples, 1), float(y_arr))
    if y_arr.ndim == 1:
        if y_arr.size == n_samples:
            return y_arr.reshape(n_samples, 1)
        return np.tile(y_arr.reshape(1, -1), (n_samples, 1))
    if y_arr.ndim == 2:
        if y_arr.shape[0] != n_samples:
            raise ValueError(
                f"Incompatible y_values shape {y_arr.shape}; expected first dimension {n_samples}."
            )
        return y_arr
    raise ValueError("y_values must be scalar, 1-D, or 2-D.")


def cdf_from_full_output(
    full_output: Dict[str, object],
    y_values: np.ndarray,
    *,
    squeeze_last: bool = True,
) -> np.ndarray:
    """Evaluate the conditional CDF using TabPFN full-distribution output."""
    if "logits" not in full_output or "criterion" not in full_output:
        raise KeyError("Full output missing required 'logits' or 'criterion' fields.")

    logits = full_output["logits"]
    criterion = full_output["criterion"]

    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)

    device = criterion.borders.device  # type: ignore[attr-defined]
    logits = logits.to(device)

    y_matrix = _broadcast_y_for_logits(y_values, logits.shape[0])
    y_tensor = torch.as_tensor(y_matrix, dtype=criterion.borders.dtype, device=device)  # type: ignore[attr-defined]

    with torch.no_grad():
        cdf_tensor = criterion.cdf(logits, y_tensor)  # type: ignore[attr-defined]

    cdf_np = cdf_tensor.cpu().numpy()
    if squeeze_last:
        if cdf_np.ndim == 2 and cdf_np.shape[1] == 1:
            cdf_np = cdf_np[:, 0]
        if isinstance(cdf_np, np.ndarray) and cdf_np.ndim > 1 and cdf_np.shape[0] == 1:
            cdf_np = cdf_np[0]
    return cdf_np


def pdf_from_full_output(
    full_output: Dict[str, object],
    y_values: np.ndarray,
    *,
    squeeze_last: bool = True,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Approximate the conditional PDF via finite differences of the CDF."""
    if "logits" not in full_output or "criterion" not in full_output:
        raise KeyError("Full output missing required 'logits' or 'criterion' fields.")

    logits = full_output["logits"]
    criterion = full_output["criterion"]

    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)

    device = criterion.borders.device  # type: ignore[attr-defined]
    logits = logits.to(device)

    y_matrix = _broadcast_y_for_logits(y_values, logits.shape[0])
    y_tensor = torch.as_tensor(y_matrix, dtype=criterion.borders.dtype, device=device)  # type: ignore[attr-defined]

    with torch.no_grad():
        delta = torch.full_like(y_tensor, float(epsilon))
        cdf_hi = criterion.cdf(logits, y_tensor + delta)  # type: ignore[attr-defined]
        cdf_lo = criterion.cdf(logits, y_tensor - delta)  # type: ignore[attr-defined]
        pdf_tensor = (cdf_hi - cdf_lo) / (2.0 * epsilon)
        pdf_tensor = torch.clamp(pdf_tensor, min=0.0)

    pdf_np = pdf_tensor.cpu().numpy()
    if squeeze_last:
        if pdf_np.ndim == 2 and pdf_np.shape[1] == 1:
            pdf_np = pdf_np[:, 0]
        if isinstance(pdf_np, np.ndarray) and pdf_np.ndim > 1 and pdf_np.shape[0] == 1:
            pdf_np = pdf_np[0]
    return pdf_np


class ConditionalCDFEstimator:
    """
    Estimate conditional CDF F_{Y|X,V}(y | x, v) using TabPFN full distribution output.

    Stage 2.9 requests the model's logits and bar-distribution criterion directly,
    enabling exact CDF evaluation via `criterion.cdf` without quantile inversion.
    """

    def __init__(self, use_tabpfn: bool = True):
        self.use_tabpfn = use_tabpfn and _HAVE_TABPFN
        self.model = None
        self._using_tabpfn = False
        self.criterion_ = None  # type: ignore[assignment]

    def _new_regressor(self):
        """Create new regressor instance."""
        if self.use_tabpfn:
            try:
                self._using_tabpfn = True
                return TabPFNRegressor(random_state=42, ignore_pretraining_limits=True)
            except Exception as e:
                print(f"⚠️  TabPFN initialization failed: {e}")
                raise RuntimeError("Stage 2.9 requires TabPFNRegressor; initialization or use_tabpfn failed")
        raise RuntimeError("TabPFNRegressor usage disabled unexpectedly.")

    def _predict_full_output(self, features: np.ndarray) -> Dict[str, object]:
        """Helper to grab the full distribution output for downstream evaluation."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit_full() first.")

        full_output = self.model.predict(features, output_type="full")
        if not isinstance(full_output, dict):
            raise RuntimeError("TabPFN full output was not a mapping structure as expected.")
        if "criterion" not in full_output or "logits" not in full_output:
            raise RuntimeError("TabPFN full output missing distribution components.")

        self.criterion_ = full_output["criterion"]  # type: ignore[assignment]
        return full_output

    def fit_full(self, X: np.ndarray, V: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fit conditional CDF model on the full dataset (no cross-validation)."""
        X = np.asarray(X).reshape(-1, 1)
        V = np.asarray(V).reshape(-1, 1)
        Y = np.asarray(Y)
        features = np.hstack([X, V])

        model_type = "TabPFNRegressor" if self.use_tabpfn else "RandomForestRegressor"
        print(f"Training full-data CDF model with {model_type} (no CV)...", flush=True)

        if not self.use_tabpfn:
            raise RuntimeError("use_tabpfn=False is not supported for CDF estimation in Stage 2.9")

        reg = self._new_regressor()
        reg.fit(features, Y)
        self.model = reg

        full_output = self._predict_full_output(features)
        train_cdf = cdf_from_full_output(full_output, Y, squeeze_last=True)
        print("✅ TabPFN full-distribution CDF model fitted on full dataset", flush=True)
        return train_cdf

    def predict_full_distribution(self, x: np.ndarray, v: np.ndarray) -> Dict[str, object]:
        """Return TabPFN full-distribution output for provided (x, v) pairs."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit_full() first.")

        x = np.asarray(x, dtype=float).reshape(-1, 1)
        v = np.asarray(v, dtype=float).reshape(-1, 1)
        features = np.hstack([x, v])
        return self._predict_full_output(features)

    def predict_cdf(self, x: np.ndarray, v: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Predict F(y | x, v) for given (x, v, y) triples via criterion-based evaluation.

        Args:
            x: (n,) x values
            v: (n,) v values
            y: Array of values to evaluate the CDF at. Supports shapes:
               - (n,) matching each (x_i, v_i)
               - (m,) common evaluation grid shared across samples
               - (n, m) sample-specific evaluation grids

        Returns:
            Array of CDF evaluations aligned with the shape of `y`.
        """
        y = np.asarray(y)
        full_output = self.predict_full_distribution(x, v)
        return cdf_from_full_output(full_output, y, squeeze_last=True)




# =========================================================
# 4) Interventional CDF: F(y|do(X=x))
# =========================================================
def compute_interventional_cdf(cdf_model: ConditionalCDFEstimator,
                               x_grid: np.ndarray,
                               y_grid: np.ndarray,
                               n_v_points: int) -> Dict[str, np.ndarray]:
    """
    Compute interventional CDF F(y|do(X=x)) by numerical integration over V.
    
    For each (x, y) pair on the grid:
    F(y|do(X=x)) = ∫₀¹ F(y|X=x,v) dv
    
    Args:
        cdf_model: Fitted conditional CDF model F(y|X,V)
        x_grid: (k_x,) array of x values to evaluate
        y_grid: (k_y,) array of y values to evaluate
        n_v_points: Number of V integration points
        
    Returns:
        Dictionary with:
            - x_grid: (k_x,) x values
            - y_grid: (k_y,) y values
            - F_interventional: (k_x, k_y) matrix of CDF values
    """
    k_x = len(x_grid)
    k_y = len(y_grid)
    v_grid = create_v_integration_grid(n_v_points)
    
    F_interventional = np.zeros((k_x, k_y), dtype=float)
    
    print(f"\nComputing interventional CDF F(y|do(X=x)) on {k_x} × {k_y} grid...")
    print(f"  X grid: {k_x} points from {x_grid.min():.3f} to {x_grid.max():.3f}")
    print(f"  Y grid: {k_y} points from {y_grid.min():.3f} to {y_grid.max():.3f}")
    print(f"  Numerical integration over {len(v_grid)} V points...")
    
    # For each x on the grid reuse logits-based distribution across all y values
    for i, x0 in enumerate(x_grid):
        print(f"  Processing x = {x0:.3f} ({i+1}/{k_x})...")
        x_vec = np.full(len(v_grid), x0, dtype=float)
        full_output = cdf_model.predict_full_distribution(x_vec, v_grid)
        
        # get the full cdf from TabPFN full-distribution output
        cdf_vals = np.asarray(
            cdf_from_full_output(full_output, y_grid, squeeze_last=False),
            dtype=float,
        )
        if cdf_vals.ndim == 1:
            integrated = np.atleast_1d(simpson(cdf_vals, x=v_grid))
        else:
            integrated = simpson(cdf_vals, x=v_grid, axis=0)
        F_interventional[i, :] = integrated
    
    print(f"✅ Interventional CDF computed")
    print(f"   Range: [{F_interventional.min():.4f}, {F_interventional.max():.4f}]")
    
    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "F_interventional": F_interventional
    }


def compute_interventional_pdf(cdf_model: ConditionalCDFEstimator,
                               x_grid: np.ndarray,
                               y_grid: np.ndarray,
                               n_v_points: int) -> Dict[str, np.ndarray]:
    """
    Compute interventional density f(y|do(X=x)) by integrating criterion-based PDFs.
    """
    k_x = len(x_grid)
    k_y = len(y_grid)
    v_grid = create_v_integration_grid(n_v_points)

    pdf_interventional = np.zeros((k_x, k_y), dtype=float)

    print(f"\nComputing interventional PDF f(y|do(X=x)) on {k_x} × {k_y} grid...")
    print(f"  Numerical integration over {len(v_grid)} V points...")

    for i, x0 in enumerate(x_grid):
        print(f"  Processing x = {x0:.3f} ({i+1}/{k_x}) for PDF...")
        x_vec = np.full(len(v_grid), x0, dtype=float)
        full_output = cdf_model.predict_full_distribution(x_vec, v_grid)
        pdf_vals = np.asarray(
            pdf_from_full_output(full_output, y_grid, squeeze_last=False),
            dtype=float,
        )

        if pdf_vals.ndim == 1:
            integrated = np.atleast_1d(simpson(pdf_vals, x=v_grid))
        else:
            integrated = simpson(pdf_vals, x=v_grid, axis=0)

        pdf_interventional[i, :] = np.maximum(integrated, 0.0)

    print("✅ Interventional PDF computed")
    rowsums = np.trapz(pdf_interventional, x=y_grid, axis=1)
    print(f"   Normalization check (∫ f(y) dy): min={rowsums.min():.4f}, max={rowsums.max():.4f}")

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "pdf": pdf_interventional
    }

def sample_eps_marginal(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample ε from its marginal distribution.

    Under the joint normal construction used in the DGP, ε ~ N(0, 1).
    We keep this helper in case future DGP variants change the marginal.
    """
    return rng.standard_normal(size=n_samples)


def simulate_y_given_x_eps(cfg: DGPConfig,
                           x_value: float,
                           eps_draws: np.ndarray,
                           *,
                           h_draws: np.ndarray | None = None) -> np.ndarray:
    """
    Evaluate Y given X=x and sampled ε draws.

    CRITICAL: eps_draws must come from marginal N(0,1), not ε|V.
    """
    eps_arr = np.asarray(eps_draws, dtype=float)
    x_arr = np.full_like(eps_arr, float(x_value), dtype=float)

    if cfg.second_stage == "B1":
        sigma_y = sigmaY_of_X(x_arr, cfg)
        m1 = cfg.beta1 * x_arr + cfg.beta2 * (x_arr ** 2)
        return m1 + sigma_y * eps_arr
    elif cfg.second_stage == "B2":
        # B2 uses bimodal mixture - generate Y using marginal ε
        n = len(eps_arr)
        mixture_indicators = np.random.binomial(1, cfg.b2_mixture_weight, size=n)

        mu1 = np.sin(x_arr) + 0.3 * x_arr * eps_arr
        mu2 = np.sin(x_arr + cfg.b2_beta_offset) + cfg.b2_peak_separation + 0.3 * x_arr * eps_arr

        sigma1 = cfg.b2_sigma1 * (1.0 + 0.2 * np.abs(x_arr))
        sigma2 = cfg.b2_sigma2 * (1.0 + 0.2 * np.abs(x_arr))

        y1 = mu1 + sigma1 * np.random.randn(n)
        y2 = mu2 + sigma2 * np.random.randn(n)

        return mixture_indicators * y1 + (1 - mixture_indicators) * y2
    elif cfg.second_stage == "B3":
        if h_draws is None:
            raise ValueError("B3 simulation requires latent H draws.")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B3.")
        return x_arr - 3.0 * h_arr + eps_arr
    elif cfg.second_stage == "B4":
        if h_draws is None:
            raise ValueError("B4 simulation requires latent H draws.")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B4.")
        linear_branch = 0.2 * (5.5 + 2.0 * x_arr + 3.0 * h_arr + eps_arr)
        softplus_arg = (2.0 * x_arr + h_arr) ** 2 + eps_arr
        safe_arg = np.maximum(softplus_arg, B4_SOFTPLUS_EPS)
        softplus_branch = np.log(safe_arg)
        return np.where(x_arr <= 1.0, linear_branch, softplus_branch)
    else:
        raise ValueError(f"Unknown second_stage: {cfg.second_stage}")


def monte_carlo_y_given_x(cfg: DGPConfig,
                          x_value: float,
                          n_samples: int,
                          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo sampler for Y|do(X=x):
      1. Draw ε from its marginal distribution (N(0,1) under current DGP)
      2. Evaluate Y with the DGP second-stage equation.

    Previous versions sampled ε | η; that path is retained in comments for reference.
    """
    if cfg.second_stage in {"B3", "B4"}:
        h_samples = rng.standard_normal(size=n_samples)
        eps_y_samples = sample_eps_marginal(n_samples, rng)
        y_samples = simulate_y_given_x_eps(cfg, x_value, eps_y_samples, h_draws=h_samples)
        return y_samples, eps_y_samples

    eps_samples = sample_eps_marginal(n_samples, rng)
    y_samples = simulate_y_given_x_eps(cfg, x_value, eps_samples)
    return y_samples, eps_samples


def sample_tabpfn_y_given_x(cdf_model: ConditionalCDFEstimator,
                            x_value: float,
                            n_samples: int,
                            y_support: np.ndarray,
                            rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw samples from the TabPFN-estimated interventional distribution Y|do(X=x).

    We rely on inverse transform sampling using the criterion-based CDF evaluated
    on a shared y-support grid to stay aligned with the KDE diagnostics.
    """
    y_support = np.asarray(y_support, dtype=float)
    if y_support.ndim != 1:
        raise ValueError("y_support must be a 1-D array for inverse transform sampling.")

    v_samples = rng.uniform(low=V_EPSILON, high=1.0 - V_EPSILON, size=n_samples)
    u_samples = rng.uniform(low=0.0, high=1.0, size=n_samples)
    x_vec = np.full(n_samples, float(x_value), dtype=float)

    full_output = cdf_model.predict_full_distribution(x_vec, v_samples)
    cdf_vals = np.asarray(
        cdf_from_full_output(full_output, y_support, squeeze_last=False),
        dtype=float,
    )
    if cdf_vals.ndim == 1:
        cdf_vals = cdf_vals.reshape(1, -1)

    y_samples = np.empty(n_samples, dtype=float)
    for idx in range(n_samples):
        row = np.asarray(cdf_vals[idx], dtype=float)
        row = np.clip(row, 0.0, 1.0)
        row = np.maximum.accumulate(row)
        row[-1] = 1.0
        row[0] = max(row[0], 0.0)
        y_samples[idx] = np.interp(u_samples[idx], row, y_support)

    return y_samples, v_samples, u_samples


def compute_y_clean(cfg: DGPConfig, x: np.ndarray) -> np.ndarray:
    """Deterministic component of Y under the DGP (noise removed)."""
    x_arr = np.asarray(x, dtype=float)
    if cfg.second_stage == "B1":
        return cfg.beta1 * x_arr + cfg.beta2 * (x_arr ** 2)
    elif cfg.second_stage == "B2":
        # B2 uses bimodal mixture: weighted mean with E[ε]=0
        w = cfg.b2_mixture_weight
        mu1 = np.sin(x_arr)
        mu2 = np.sin(x_arr + cfg.b2_beta_offset) + cfg.b2_peak_separation
        return w * mu1 + (1 - w) * mu2
    elif cfg.second_stage == "B3":
        return x_arr
    elif cfg.second_stage == "B4":
        linear_branch = 0.2 * (5.5 + 2.0 * x_arr)
        softplus_branch = np.log((2.0 * x_arr) ** 2)
        return np.where(x_arr <= 1.0, linear_branch, softplus_branch)
    else:
        raise ValueError(f"Unknown second_stage: {cfg.second_stage}")


def create_kernel_density_plot(y_grid: np.ndarray,
                               pdf_estimated: np.ndarray,
                               pdf_true: Optional[np.ndarray],
                               x_values: np.ndarray,
                               y_observed: Optional[np.ndarray],
                               quantile_levels: Optional[np.ndarray],
                               output_path: str) -> None:
    """Render Figure-3-style KDE panels comparing estimated vs. true densities."""
    import matplotlib.pyplot as plt

    n_panels = len(x_values)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8.5, 8.5), sharex=True)
    if n_panels == 1:
        axes = [axes]

    fill_est_color = (231/255, 186/255, 82/255, 0.75)  # soft amber with alpha
    fill_true_color = (66/255, 133/255, 244/255, 0.35)  # muted blue with alpha
    line_est_color = (196/255, 139/255, 30/255)
    line_true_color = (38/255, 90/255, 136/255)

    for idx, ax in enumerate(axes):
        est_curve = pdf_estimated[idx]
        true_curve = pdf_true[idx] if pdf_true is not None else None

        ax.plot(y_grid, est_curve, color=line_est_color, linewidth=1.8)
        ax.fill_between(y_grid, 0.0, est_curve, color=fill_est_color, label="Estimated" if idx == 0 else None)

        if true_curve is not None:
            ax.plot(y_grid, true_curve, color=line_true_color, linewidth=1.5)
            ax.fill_between(y_grid, 0.0, true_curve, color=fill_true_color, label="True" if idx == 0 else None)

        if y_observed is not None:
            ax.axvline(y_observed[idx], color=line_true_color, linestyle=":", linewidth=1.0, alpha=0.7)

        quantile_text = ""
        if quantile_levels is not None:
            quantile_text = f" (q={quantile_levels[idx]:.2f})"

        ax.set_ylabel(r"$P_Y^{do(X=x)}$")
        ax.text(0.98, 0.85, f"x = {x_values[idx]:.2f}{quantile_text}", transform=ax.transAxes,
                ha="right", va="center", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    axes[-1].set_xlabel("Y")

    handles = []
    labels = []
    if pdf_estimated is not None:
        handles.append(plt.Line2D([0], [0], color=line_est_color, linewidth=1.8))
        labels.append("Estimated")
    if pdf_true is not None:
        handles.append(plt.Line2D([0], [0], color=line_true_color, linewidth=1.5))
        labels.append("True")
    if handles:
        axes[0].legend(handles, labels, loc="upper right", frameon=False)

    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.98))
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)



# =========================================================
# 6) Main execution pipeline
# =========================================================
def run_stage2_9_experiment(cfg: Stage2_9Config) -> Dict[str, object]:
    """Execute Stage 2.9 pipeline with numerical integration optimization.

    Steps:
      1. Load Stage 1 outputs
      2. Fit structural model m(x, v) on full data
      3. Integrate out V to obtain μ_c(x) on grids and at each test observation
      4. Fit conditional CDF models on full data
      5. Compute interventional CDF/PDF diagnostics (estimated vs. analytical ground truth)
    """
    import sys

    print("Starting IV Stage 2.9 Experiment (Numerical Integration Optimization)...", flush=True)
    print(f"Configuration: {asdict(cfg)}", flush=True)
    sys.stdout.flush()

    set_seed(cfg.random_state)
    rng = np.random.default_rng(cfg.random_state)

    components = prepare_stage2_components(cfg)
    train_data = components["train_data"]
    test_data = components["test_data"]
    dgp_cfg = components["dgp_config"]
    m_model = components["m_model"]
    cdf_model = components["cdf_model"]
    X_test = test_data["X"]
    Y_test = test_data["Y"]
    V_true_test = test_data["V_true"]
    n_train = len(train_data["X"])
    n_test = len(X_test)

    print(f"\n[1/5] Components prepared: train={n_train} samples, test={n_test} samples", flush=True)

    print("\n[2/5] Aligning evaluation grid with held-out test samples...", flush=True)
    x_test_grid, y_test_grid, selected_idx = build_test_grid(X_test, Y_test, cfg.n_x_test)
    selected_idx = np.asarray(selected_idx, dtype=int)
    k_x = len(x_test_grid)
    x_min, x_max = float(np.min(x_test_grid)), float(np.max(x_test_grid))
    y_min, y_max = float(np.min(Y_test)), float(np.max(Y_test))
    y_grid = np.linspace(y_min, y_max, cfg.n_y_grid)
    print(f"  X grid: {k_x} test points from {x_min:.3f} to {x_max:.3f}", flush=True)
    print(f"  Y grid: {cfg.n_y_grid} points from {y_min:.3f} to {y_max:.3f}", flush=True)

    mu_c_estimated = compute_mu_c_on_grid(m_model, x_test_grid, cfg.n_v_integration_points)

    Z_subset = None if test_data["Z"] is None else test_data["Z"][selected_idx]
    X_test_subset = X_test[selected_idx]
    Y_test_subset = Y_test[selected_idx]
    Y_clean_full = compute_y_clean(dgp_cfg, X_test)
    Y_clean_subset = Y_clean_full[selected_idx]
    V_hat_subset = test_data["V_hat"][selected_idx] if test_data["V_hat"] is not None else None
    V_true_subset = V_true_test[selected_idx]
    eps_subset = None if test_data["eps"] is None else test_data["eps"][selected_idx]
    eta_subset = None if test_data["eta"] is None else test_data["eta"][selected_idx]
    print("\n[3/5] Integrating structural function for selected test observations...", flush=True)
    # Full-test integration used for global diagnostics (e.g., MSE over all test points)
    mu_c_all_test = compute_mu_c_on_grid(m_model, X_test, cfg.n_v_integration_points)
    mu_c_test_estimated = mu_c_all_test[selected_idx]

    print("\n[4/5] Preparing outputs...", flush=True)

    kde_indices = select_kde_indices(len(x_test_grid), cfg.kde_quantiles)
    kde_x_values = x_test_grid[kde_indices]
    kde_y_observed = y_test_grid[kde_indices]

    kde_y_samples_true: list[np.ndarray] = []
    kde_eps_samples_true: list[np.ndarray] = []
    kde_y_samples_est: list[np.ndarray] = []
    kde_v_samples_est: list[np.ndarray] = []
    kde_u_samples_est: list[np.ndarray] = []
    kde_pdf_true_rows: list[np.ndarray] = []
    kde_pdf_est_rows: list[np.ndarray] = []

    for x0 in kde_x_values:
        y_draws_true, eps_draws_true = monte_carlo_y_given_x(
            dgp_cfg,
            float(x0),
            cfg.kde_sample_size,
            rng,
        )
        y_draws_est, v_draws_est, u_draws_est = sample_tabpfn_y_given_x(
            cdf_model,
            float(x0),
            cfg.kde_sample_size,
            y_grid,
            rng,
        )

        kde_y_samples_true.append(y_draws_true)
        kde_eps_samples_true.append(eps_draws_true)
        kde_y_samples_est.append(y_draws_est)
        kde_v_samples_est.append(v_draws_est)
        kde_u_samples_est.append(u_draws_est)

        try:
            density_true = gaussian_kde(y_draws_true)
            pdf_true_row = density_true(y_grid)
        except np.linalg.LinAlgError:
            jitter = 1e-6 * rng.standard_normal(size=y_draws_true.shape)
            density_true = gaussian_kde(y_draws_true + jitter)
            pdf_true_row = density_true(y_grid)

        try:
            density_est = gaussian_kde(y_draws_est)
            pdf_est_row = density_est(y_grid)
        except np.linalg.LinAlgError:
            jitter_est = 1e-6 * rng.standard_normal(size=y_draws_est.shape)
            density_est = gaussian_kde(y_draws_est + jitter_est)
            pdf_est_row = density_est(y_grid)

        kde_pdf_true_rows.append(np.maximum(pdf_true_row, 0.0))
        kde_pdf_est_rows.append(np.maximum(pdf_est_row, 0.0))

    kde_pdf_true = np.vstack(kde_pdf_true_rows)
    kde_pdf_est = np.vstack(kde_pdf_est_rows)

    iae_per_x = np.trapz(np.abs(kde_pdf_est - kde_pdf_true), x=y_grid, axis=1)
    metrics = {
        "iae_mean": float(np.mean(iae_per_x)),
        "iae_max": float(np.max(iae_per_x)),
        "iae_per_x": iae_per_x,
    }

    mse_do_pred = float(np.mean((mu_c_all_test - Y_clean_full) ** 2))
    metrics["mse_do_pred_vs_clean"] = mse_do_pred

    return {
        "config": asdict(cfg),
        "data_stats": {
            "n_train_samples": n_train,
            "n_test_samples_raw": n_test,
            "n_test_samples_selected": k_x,
            "n_x_test": len(x_test_grid),
            "n_y_grid": cfg.n_y_grid,
            "n_v_integration_points": cfg.n_v_integration_points
        },
        "mu_c": {
            "x_test_grid": x_test_grid,
            "y_test_grid": y_test_grid,
            "estimated": mu_c_estimated,
        },
        "predictions": {
            "Z": Z_subset,
            "X": X_test_subset,
            "Y_true": Y_test_subset,
            "Y_do_pred": mu_c_test_estimated,
            "Y_clean": Y_clean_subset,
            "V_hat": V_hat_subset,
            "V_true": V_true_subset,
            "eps": eps_subset,
            "eta": eta_subset
        },
        "kde": {
            "indices": kde_indices,
            "x_values": kde_x_values,
            "y_observed": kde_y_observed,
            "pdf_estimated": kde_pdf_est,
            "pdf_true": kde_pdf_true,
            "y_grid": y_grid,
            "quantiles": np.asarray(cfg.kde_quantiles, dtype=float),
            "y_samples_true": np.asarray(kde_y_samples_true),
            "eps_samples_true": np.asarray(kde_eps_samples_true),
            "y_samples_est": np.asarray(kde_y_samples_est),
            "v_samples_est": np.asarray(kde_v_samples_est),
            "u_samples_est": np.asarray(kde_u_samples_est),
        },
        "metrics": metrics,
        "metadata": {
            "train_csv": components["train_csv"],
            "test_csv": components["test_csv"],
            "codes": components["codes"]
        }
    }




# =========================================================
# 7) Save results
# =========================================================
def save_stage2_9_results(results: Dict[str, object], output_dir: str):
    """Persist Stage 2.9 outputs as CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    codes = results["metadata"]["codes"]
    artifact_names = []

    # Predictions CSV
    pred = results["predictions"]
    pred_columns: list[tuple[str, object]] = []

    if pred.get("Z") is not None:
        pred_columns.append(("Z", pred["Z"]))

    pred_columns.extend([
        ("X", pred["X"]),
        ("Y_true", pred["Y_true"]),
        ("Y_clean", pred["Y_clean"]),
        ("Y_do_pred", pred["Y_do_pred"]),
    ])

    for optional_key in ("V_hat", "V_true", "eps", "eta"):
        value = pred.get(optional_key)
        if value is not None:
            pred_columns.append((optional_key, value))

    predictions_df = pd.DataFrame({name: data for name, data in pred_columns})
    predictions_csv = os.path.join(output_dir, f"iv_stage2_9_{codes}_predictions_{timestamp}.csv")
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"✅ Predictions saved to: {predictions_csv}")
    artifact_names.append(os.path.basename(predictions_csv))

    # Summary CSV
    summary_rows = []
    for key, value in results["data_stats"].items():
        summary_rows.append({"key": key, "value": value})
    summary_rows.append({"key": "train_csv", "value": results["metadata"]["train_csv"]})
    summary_rows.append({"key": "test_csv", "value": results["metadata"]["test_csv"]})
    summary_rows.append({"key": "timestamp", "value": timestamp})
    summary_rows.append({"key": "prediction_mode", "value": "mu_c_integrated"})

    metrics = results.get("metrics")
    if metrics:
        summary_rows.append({"key": "metric_mse_do_pred_vs_clean", "value": f"{metrics['mse_do_pred_vs_clean']:.6f}"})
        summary_rows.append({"key": "metric_iae_mean", "value": f"{metrics['iae_mean']:.6f}"})
        summary_rows.append({"key": "metric_iae_max", "value": f"{metrics['iae_max']:.6f}"})
        iae_series = ";".join(f"{val:.6f}" for val in metrics['iae_per_x'])
        summary_rows.append({"key": "metric_iae_per_x", "value": iae_series})

    kde = results.get("kde")
    if kde is not None:
        x_summary = ";".join(f"{val:.4f}" for val in kde["x_values"])
        summary_rows.append({"key": "kde_x_values", "value": x_summary})
    summary_csv = os.path.join(output_dir, f"iv_stage2_9_{codes}_summary_{timestamp}.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"✅ Summary saved to: {summary_csv}")
    artifact_names.append(os.path.basename(summary_csv))

    # KDE Plot
    if kde is not None:
        kde_plot_path = os.path.join(output_dir, f"iv_stage2_9_{codes}_kde_{timestamp}.png")
        create_kernel_density_plot(
            y_grid=kde["y_grid"],
            pdf_estimated=kde["pdf_estimated"],
            pdf_true=kde["pdf_true"],
            x_values=kde["x_values"],
            y_observed=kde["y_observed"],
            quantile_levels=kde.get("quantiles"),
            output_path=kde_plot_path,
        )
        print(f"✅ Kernel density plot saved to: {kde_plot_path}")
        artifact_names.append(os.path.basename(kde_plot_path))

    print(f"\n📁 CSV outputs stored in: {output_dir}/")
    print(f"🕒 Timestamp: {timestamp}")
    for name in artifact_names:
        print(f"   - {name}")




# =========================================================
# 8) Runner
# =========================================================
if __name__ == "__main__":
    """Command-line entry point for Stage 2.9."""

    import sys

    print("=" * 60, flush=True)
    print("IV STAGE 2.9: STARTING EXECUTION", flush=True)
    print("=" * 60, flush=True)

    cfg = Stage2_9Config(
        input_dir="IV_datasets/stage1_output",
        output_dir="IV_datasets/stage2_output",
        random_state=1,
        n_x_test=50,
        n_y_grid=100,
        n_v_integration_points=100,
        use_tabpfn=True,
        first_stage_code="A3",
        second_stage_code="B3",
        kde_quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        kde_sample_size=1000
    )

    print(f"Configuration: {cfg}", flush=True)
    print(f"TabPFN available: {_HAVE_TABPFN}", flush=True)
    print("", flush=True)

    try:
        results = run_stage2_9_experiment(cfg)
        save_stage2_9_results(results, cfg.output_dir)

        print("\n" + "=" * 60)
        print("STAGE 2.9 COMPLETED SUCCESSFULLY", flush=True)
        print("=" * 60)

        print("\nInterventional density diagnostics saved to output directory.")

    except Exception as exc:
        print("\n❌ Stage 2.9 failed with error:", exc, flush=True)
        raise

#2
