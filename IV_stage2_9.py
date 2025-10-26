"""
IV Stage 2.9 Implementation: Density Diagnostics
================================================

Stage 2.9 extends Stage 2.8's full-distribution integration with diagnostics
targeted at the interventional density f(y | do(X = x)).

New in Stage 2.9:
- Align μ_c(x) evaluation points with the held-out test samples and export the
  paired `y_test` column for the μ_c curves CSV.
- Integrate criterion-based PDFs to recover f(y|do(X=x)) and a DGP oracle
  counterpart on the same grid.
- Produce a Figure-3-style kernel density visualization for three representative
  test points, sourced from configurable X-quantiles.

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
from scipy.stats import norm
from scipy.integrate import simpson
import os
from datetime import datetime
import torch

# Import from Stage 1
from DGP import DGPConfig, set_seed

V_EPSILON = 1e-6  # Avoid norm.ppf endpoints

# --- TabPFN (with fallbacks) ---
try:
    from tabpfn.regressor import TabPFNRegressor
    _HAVE_TABPFN = True
    print("✅ TabPFNRegressor imported successfully")
except Exception as e:
    raise ImportError(f"TabPFNRegressor is required for Stage 2.9 but could not be imported: {e}")




# =========================================================
# 0) True Structural Function from DGP
# =========================================================
def m_true(cfg: DGPConfig, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    True structural function E[Y|X=x, V=v] from DGP specification.
    
    This is the ORACLE function - the analytical form of the true relationship
    between X, V, and Y as specified in the data generating process.
    
    Args:
        cfg: DGP configuration with parameters (beta1, beta2, rho, delta0, delta1, etc.)
        x: (n,) array of X values
        v: (n,) array of V values (control function)
        
    Returns:
        m_true: (n,) array of true conditional means E[Y|X=x,V=v]
    """
    x = np.asarray(x)
    v = np.asarray(v)
    t = norm.ppf(v)  # Φ^{-1}(v)

    if cfg.second_stage == "B1":
        # m(x,v) = β1 x + β2 x^2 + σ_Y(v) * E[ε | V=v]
        e_mean = cfg.rho * t
        sigY = np.exp(cfg.delta0 + cfg.delta1 * v)
        return cfg.beta1 * x + cfg.beta2 * (x ** 2) + sigY * e_mean

    elif cfg.second_stage == "B2":
        # m(x,v) = sin(x) + 0.5 x E[ε|V=v] + 0.5 E[ε^2|V=v]
        e_mean = cfg.rho * t
        e_var = 1.0 - cfg.rho ** 2      # Var(ε | T=t) under joint normal
        e_second = e_var + (e_mean ** 2)
        return np.sin(x) + 0.5 * x * e_mean + 0.5 * e_second

    else:
        raise ValueError("Unknown second_stage")


# =========================================================
# 1) Extended Configuration
# =========================================================
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
    
    random_state: int = 1
    
    # Test grids (uniform grids)
    n_x_test: int = 50      # Number of test X points (uniform grid)
    n_y_grid: int = 100     # Number of Y points for CDF evaluation
    
    # V integration grid (NEW: numerical integration)
    n_v_integration_points: int = 100  # Number of V points for numerical integration
    
    # Model settings
    use_tabpfn: bool = True
    
    # Oracle comparison
    include_oracle: bool = True

    # Density diagnostics
    kde_quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75)
    kde_sample_size: int = 1000

    # DGP identifiers (for file selection and oracle)
    first_stage_code: str = "A1"
    second_stage_code: str = "B1"


def prepare_stage2_components(cfg: Stage2_9Config):
    """
    Load Stage 1 outputs, train Stage 2 models, and return reusable components.
    """
    codes = f"{cfg.first_stage_code}_{cfg.second_stage_code}"
    train_prefix = f"iv_stage1_train_{codes}_"
    test_prefix = f"iv_stage1_test_{codes}_"
    train_csv = _latest_matching_file(cfg.input_dir, train_prefix)
    if train_csv is None:
        raise FileNotFoundError(f"No Stage-1 training CSV found in {cfg.input_dir} matching prefix {train_prefix}")
    test_csv = _latest_matching_file(cfg.input_dir, test_prefix)
    if test_csv is None:
        raise FileNotFoundError(f"No Stage-1 test CSV found in {cfg.input_dir} matching prefix {test_prefix}")

    print(f"Resolved Stage-1 training CSV: {train_csv}", flush=True)
    print(f"Resolved Stage-1 test CSV: {test_csv}", flush=True)

    train_data = load_stage1_data(train_csv)
    test_data = load_stage1_data(test_csv)

    dgp_cfg = DGPConfig(first_stage=cfg.first_stage_code, second_stage=cfg.second_stage_code)

    m_model = FullDataStructuralFunctionModel(use_tabpfn=cfg.use_tabpfn)
    _ = m_model.fit_full(train_data["X"], train_data["V_hat"], train_data["Y"])
    Y_test_pred = m_model.predict(test_data["X"], test_data["V_hat"])

    cdf_model = ConditionalCDFEstimator(use_tabpfn=cfg.use_tabpfn)
    _ = cdf_model.fit_full(train_data["X"], train_data["V_hat"], train_data["Y"])

    cdf_oracle = None
    if cfg.include_oracle:
        cdf_oracle = ConditionalCDFEstimator(use_tabpfn=cfg.use_tabpfn)
        _ = cdf_oracle.fit_full(train_data["X"], train_data["V_true"], train_data["Y"])

    return {
        "codes": codes,
        "train_csv": train_csv,
        "test_csv": test_csv,
        "train_data": train_data,
        "test_data": test_data,
        "dgp_config": dgp_cfg,
        "m_model": m_model,
        "cdf_model": cdf_model,
        "cdf_oracle": cdf_oracle,
        "Y_test_pred": Y_test_pred
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
                         n_v_points: int) -> np.ndarray:
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

    # Build the joint (x, v) feature grid once and reuse TabPFN batching.
    x_col = np.repeat(x_grid, n_v).astype(float)
    v_col = np.tile(v_grid, k).astype(float)

    m_vals = np.asarray(m_model.predict(x_col, v_col), dtype=float)
    if m_vals.size != k * n_v:
        raise ValueError(
            f"Unexpected structural predictions shape {m_vals.shape}; "
            f"expected {k * n_v} entries for the X×V grid."
        )

    m_matrix = m_vals.reshape(k, n_v)
    mu_c_grid = simpson(m_matrix, x=v_grid, axis=1)

    print(f"✅ μ_c computed on test grid: mean={np.mean(mu_c_grid):.4f}, std={np.std(mu_c_grid):.4f}")
    return mu_c_grid


def compute_mu_c_oracle_on_grid(cfg: DGPConfig, 
                               x_grid: np.ndarray, 
                               n_v_points: int) -> np.ndarray:
    """
    Compute TRUE μ_c(x) = ∫₀¹ m_true(x,v) dv using numerical integration.
    
    This is the oracle/ground truth for comparison using analytical integration
    of the true structural function from the DGP.
    
    Args:
        cfg: DGP configuration (for m_true parameters)
        x_grid: (k,) array of NEW test x values
        n_v_points: Number of V integration points
    
    Returns:
        mu_c_true_grid: (k,) array of oracle μ_c values
    """
    k = len(x_grid)
    v_grid = create_v_integration_grid(n_v_points)
    mu_c_true_grid = np.zeros(k, dtype=float)
    
    print(f"Computing ORACLE μ_c(x) using numerical integration of true DGP formula...")
    
    for j, x_val in enumerate(x_grid):
        x_vec = np.full(len(v_grid), x_val, dtype=float)
        m_true_vals = m_true(cfg, x_vec, v_grid)  # Analytical m_true
        
        # Numerical integration using Simpson's rule
        mu_c_true_grid[j] = simpson(m_true_vals, x=v_grid)
        
        if (j + 1) % 10 == 0:
            print(f"  Progress: {j+1}/{k} test points completed")
    
    print(f"✅ Oracle μ_c computed: mean={np.mean(mu_c_true_grid):.4f}, std={np.std(mu_c_true_grid):.4f}")
    return mu_c_true_grid


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

# ---------------------------------------------------------
# TRUE CONDITIONAL CDF UTILITIES
# ---------------------------------------------------------
def F_true_conditional_B1(cfg: DGPConfig, x: np.ndarray, v: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    v = np.asarray(v)
    y = np.asarray(y)
    t = norm.ppf(v)
    e_mean = cfg.rho * t
    sigY = np.exp(cfg.delta0 + cfg.delta1 * v)
    mu_cond = cfg.beta1 * x + cfg.beta2 * (x ** 2) + sigY * e_mean
    sigma_cond = sigY
    return norm.cdf(y, loc=mu_cond, scale=sigma_cond)


def F_true_conditional_B2(cfg: DGPConfig, x: np.ndarray, v: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    v = np.asarray(v)
    y = np.asarray(y)
    t = norm.ppf(v)
    e_mean = cfg.rho * t
    e_var = 1.0 - cfg.rho ** 2
    mu_cond = np.sin(x) + 0.5 * x * e_mean + 0.5 * (e_var + e_mean ** 2)
    var_eps_squared = 2 * e_var ** 2 + 4 * e_mean ** 2 * e_var
    cov_eps_eps_squared = 2 * e_mean * e_var
    sigma_cond_sq = 0.25 * x ** 2 * e_var + 0.25 * var_eps_squared + 0.5 * x * cov_eps_eps_squared
    sigma_cond = np.sqrt(np.maximum(sigma_cond_sq, 1e-10))
    return norm.cdf(y, loc=mu_cond, scale=sigma_cond)


def F_true_conditional(cfg: DGPConfig, x: np.ndarray, v: np.ndarray, y: np.ndarray) -> np.ndarray:
    if cfg.second_stage == "B1":
        return F_true_conditional_B1(cfg, x, v, y)
    elif cfg.second_stage == "B2":
        return F_true_conditional_B2(cfg, x, v, y)
    else:
        raise ValueError(f"Unknown second_stage: {cfg.second_stage}")

def f_true_density_B1(cfg: DGPConfig, x: np.ndarray, v: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    v = np.asarray(v)
    y = np.asarray(y)
    t = norm.ppf(v)
    e_mean = cfg.rho * t
    sigma_y = np.exp(cfg.delta0 + cfg.delta1 * v)
    mu_cond = cfg.beta1 * x + cfg.beta2 * (x ** 2) + sigma_y * e_mean
    sigma_cond = sigma_y
    return norm.pdf(y, loc=mu_cond, scale=sigma_cond)

def f_true_density_B2(cfg: DGPConfig, x: np.ndarray, v: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    v = np.asarray(v)
    y = np.asarray(y)
    t = norm.ppf(v)
    e_mean = cfg.rho * t
    e_var = 1.0 - cfg.rho ** 2
    mu_cond = np.sin(x) + 0.5 * x * e_mean + 0.5 * (e_var + e_mean ** 2)
    var_eps_squared = 2 * e_var ** 2 + 4 * e_mean ** 2 * e_var
    cov_eps_eps_squared = 2 * e_mean * e_var
    sigma_cond_sq = 0.25 * x ** 2 * e_var + 0.25 * var_eps_squared + 0.5 * x * cov_eps_eps_squared
    sigma_cond = np.sqrt(np.maximum(sigma_cond_sq, 1e-10))
    return norm.pdf(y, loc=mu_cond, scale=sigma_cond)


def f_true_density(cfg: DGPConfig, x: np.ndarray, v: np.ndarray, y: np.ndarray) -> np.ndarray:
    if cfg.second_stage == "B1":
        return f_true_density_B1(cfg, x, v, y)
    elif cfg.second_stage == "B2":
        return f_true_density_B2(cfg, x, v, y)
    else:
        raise ValueError(f"Unknown second_stage: {cfg.second_stage}")


def compute_true_interventional_cdf(cfg: DGPConfig,
                                   x_grid: np.ndarray,
                                   y_grid: np.ndarray,
                                   n_v_points: int) -> Dict[str, np.ndarray]:
    k_x = len(x_grid)
    k_y = len(y_grid)
    v_grid = create_v_integration_grid(n_v_points)
    F_true_grid = np.zeros((k_x, k_y), dtype=float)

    print(f"\nComputing TRUE interventional CDF F_true(y|do(X=x)) using numerical integration...")
    for i, x0 in enumerate(x_grid):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing x = {x0:.3f} ({i+1}/{k_x})")
        x_vec = np.full(len(v_grid), x0, dtype=float)
        for j, y0 in enumerate(y_grid):
            y_vec = np.full(len(v_grid), y0, dtype=float)
            F_vals = F_true_conditional(cfg, x_vec, v_grid, y_vec)
            F_true_grid[i, j] = simpson(F_vals, x=v_grid)

    print("✅ TRUE interventional CDF computed")
    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "F_interventional": F_true_grid
    }


def compute_true_interventional_pdf(cfg: DGPConfig,
                                    x_grid: np.ndarray,
                                    y_grid: np.ndarray,
                                    n_v_points: int) -> Dict[str, np.ndarray]:
    k_x = len(x_grid)
    k_y = len(y_grid)
    v_grid = create_v_integration_grid(n_v_points)
    f_true_grid = np.zeros((k_x, k_y), dtype=float)

    print("\nComputing TRUE interventional PDF f_true(y|do(X=x))...")
    for i, x0 in enumerate(x_grid):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing x = {x0:.3f} ({i+1}/{k_x})")
        x_vec = np.full(len(v_grid), x0, dtype=float)
        for j, y0 in enumerate(y_grid):
            y_vec = np.full(len(v_grid), y0, dtype=float)
            f_vals = f_true_density(cfg, x_vec, v_grid, y_vec)
            f_true_grid[i, j] = simpson(f_vals, x=v_grid)

    print("✅ TRUE interventional PDF computed")
    rowsums = np.trapz(f_true_grid, x=y_grid, axis=1)
    print(f"   Normalization check (∫ f_true(y) dy): min={rowsums.min():.4f}, max={rowsums.max():.4f}")

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "pdf": f_true_grid
    }


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
      5. Compute interventional CDF/PDF diagnostics (estimated/oracle/true)
    """
    import sys

    print("Starting IV Stage 2.9 Experiment (Numerical Integration Optimization)...", flush=True)
    print(f"Configuration: {asdict(cfg)}", flush=True)
    sys.stdout.flush()

    set_seed(cfg.random_state)

    components = prepare_stage2_components(cfg)
    train_data = components["train_data"]
    test_data = components["test_data"]
    dgp_cfg = components["dgp_config"]
    m_model = components["m_model"]
    cdf_model = components["cdf_model"]
    cdf_oracle = components["cdf_oracle"]
    Y_test_pred = components["Y_test_pred"]

    X_test = test_data["X"]
    Y_test = test_data["Y"]
    V_hat_test = test_data["V_hat"]
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
    mu_c_oracle = None
    if cfg.include_oracle:
        mu_c_oracle = compute_mu_c_oracle_on_grid(dgp_cfg, x_test_grid, cfg.n_v_integration_points)

    Z_subset = None if test_data["Z"] is None else test_data["Z"][selected_idx]
    X_test_subset = X_test[selected_idx]
    Y_test_subset = Y_test[selected_idx]
    V_hat_subset = V_hat_test[selected_idx]
    V_true_subset = V_true_test[selected_idx]
    eps_subset = None if test_data["eps"] is None else test_data["eps"][selected_idx]
    eta_subset = None if test_data["eta"] is None else test_data["eta"][selected_idx]
    Y_test_pred_subset = Y_test_pred[selected_idx]

    print("\n[3/5] Integrating structural function for selected test observations...", flush=True)
    mu_c_test_estimated = compute_mu_c_on_grid(m_model, X_test_subset, cfg.n_v_integration_points)
    mu_c_test_oracle = None
    if cfg.include_oracle:
        mu_c_test_oracle = compute_mu_c_oracle_on_grid(dgp_cfg, X_test_subset, cfg.n_v_integration_points)

    print("\n[4/5] Computing interventional CDFs on test grid...", flush=True)
    interventional_est = compute_interventional_cdf(cdf_model, x_test_grid, y_grid, cfg.n_v_integration_points)
    interventional_orc = None
    if cfg.include_oracle and cdf_oracle is not None:
        interventional_orc = compute_interventional_cdf(cdf_oracle, x_test_grid, y_grid, cfg.n_v_integration_points)
    interventional_true = compute_true_interventional_cdf(dgp_cfg, x_test_grid, y_grid, cfg.n_v_integration_points)
    interventional_pdf_est = compute_interventional_pdf(cdf_model, x_test_grid, y_grid, cfg.n_v_integration_points)
    interventional_pdf_true = compute_true_interventional_pdf(dgp_cfg, x_test_grid, y_grid, cfg.n_v_integration_points)

    print("\n[5/5] Preparing outputs...", flush=True)

    kde_indices = select_kde_indices(len(x_test_grid), cfg.kde_quantiles)
    kde_x_values = x_test_grid[kde_indices]
    kde_pdf_est = interventional_pdf_est["pdf"][kde_indices, :]
    kde_pdf_true = interventional_pdf_true["pdf"][kde_indices, :]
    kde_y_observed = y_test_grid[kde_indices]

    # --- Metrics ---
    pdf_est_matrix = interventional_pdf_est["pdf"]
    pdf_true_matrix = interventional_pdf_true["pdf"]
    cdf_est_matrix = interventional_est["F_interventional"]
    cdf_true_matrix = interventional_true["F_interventional"]

    iae_per_x = np.trapz(np.abs(pdf_est_matrix - pdf_true_matrix), y_grid, axis=1)
    ks_per_x = np.max(np.abs(cdf_est_matrix - cdf_true_matrix), axis=1)
    cvm_per_x = np.trapz((cdf_est_matrix - cdf_true_matrix) ** 2, y_grid, axis=1)
    mse_do_pred = float(np.mean((mu_c_test_estimated - Y_test_subset) ** 2))

    metrics = {
        "mse_do_pred_vs_true": mse_do_pred,
        "iae_mean": float(np.mean(iae_per_x)),
        "iae_max": float(np.max(iae_per_x)),
        "ks_mean": float(np.mean(ks_per_x)),
        "ks_max": float(np.max(ks_per_x)),
        "cvm_mean": float(np.mean(cvm_per_x)),
        "cvm_max": float(np.max(cvm_per_x)),
        "iae_per_x": iae_per_x,
        "ks_per_x": ks_per_x,
        "cvm_per_x": cvm_per_x
    }

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
            "oracle": mu_c_oracle
        },
        "interventional_cdf": {
            "estimated": interventional_est,
            "oracle": interventional_orc,
            "true": interventional_true
        },
        "interventional_pdf": {
            "estimated": interventional_pdf_est,
            "true": interventional_pdf_true
        },
        "predictions": {
            "Z": Z_subset,
            "X": X_test_subset,
            "Y_true": Y_test_subset,
            "Y_do_pred": mu_c_test_estimated,
            "Y_do_oracle": mu_c_test_oracle,
            "Y_structural_pred": Y_test_pred_subset,
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
            "quantiles": np.asarray(cfg.kde_quantiles, dtype=float)
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
        ("Y_do_pred", pred["Y_do_pred"]),
    ])

    if pred.get("Y_do_oracle") is not None:
        pred_columns.append(("Y_do_oracle", pred["Y_do_oracle"]))
    if pred.get("Y_structural_pred") is not None:
        pred_columns.append(("Y_structural_pred", pred["Y_structural_pred"]))

    for optional_key in ("V_hat", "V_true", "eps", "eta"):
        value = pred.get(optional_key)
        if value is not None:
            pred_columns.append((optional_key, value))

    predictions_df = pd.DataFrame({name: data for name, data in pred_columns})
    predictions_csv = os.path.join(output_dir, f"iv_stage2_9_{codes}_predictions_{timestamp}.csv")
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"✅ Predictions saved to: {predictions_csv}")
    artifact_names.append(os.path.basename(predictions_csv))

    # μ_c curves CSV
    mu_c = results["mu_c"]
    mu_c_data = {
        "x_test": mu_c["x_test_grid"],
        "y_test": mu_c["y_test_grid"],
        "mu_c_estimated": mu_c["estimated"],
    }
    if mu_c["oracle"] is not None:
        mu_c_data["mu_c_oracle"] = mu_c["oracle"]
    else:
        mu_c_data["mu_c_oracle"] = np.nan
    mu_c_df = pd.DataFrame(mu_c_data)
    mu_c_csv = os.path.join(output_dir, f"iv_stage2_9_{codes}_mu_c_curves_{timestamp}.csv")
    mu_c_df.to_csv(mu_c_csv, index=False)
    print(f"✅ μ_c curves saved to: {mu_c_csv}")
    artifact_names.append(os.path.basename(mu_c_csv))

    # Interventional CDF CSV (long format)
    interventional_est = results["interventional_cdf"]["estimated"]
    interventional_orc = results["interventional_cdf"]["oracle"]
    interventional_true = results["interventional_cdf"]["true"]
    x_grid = interventional_est["x_grid"]
    y_grid = interventional_est["y_grid"]
    F_est = interventional_est["F_interventional"]
    F_orc = interventional_orc["F_interventional"] if interventional_orc else None
    F_true = interventional_true["F_interventional"]

    rows = []
    for i, x_val in enumerate(x_grid):
        for j, y_val in enumerate(y_grid):
            rows.append({
                "x_value": x_val,
                "y_value": y_val,
                "F_estimated": F_est[i, j],
                "F_oracle": F_orc[i, j] if F_orc is not None else np.nan,
                "F_true": F_true[i, j]
            })
    interventional_csv = os.path.join(output_dir, f"iv_stage2_9_{codes}_interventional_cdf_{timestamp}.csv")
    pd.DataFrame(rows).to_csv(interventional_csv, index=False)
    print(f"✅ Interventional CDF saved to: {interventional_csv}")
    artifact_names.append(os.path.basename(interventional_csv))

    # Interventional PDF CSV (long format)
    pdf_est = results["interventional_pdf"]["estimated"]
    pdf_true = results["interventional_pdf"]["true"]
    pdf_rows = []
    pdf_y_grid = pdf_est["y_grid"]
    pdf_est_values = pdf_est["pdf"]
    pdf_true_values = pdf_true["pdf"]
    for i, x_val in enumerate(pdf_est["x_grid"]):
        for j, y_val in enumerate(pdf_y_grid):
            pdf_rows.append({
                "x_value": x_val,
                "y_value": y_val,
                "pdf_estimated": pdf_est_values[i, j],
                "pdf_true": pdf_true_values[i, j]
            })
    pdf_csv = os.path.join(output_dir, f"iv_stage2_9_{codes}_interventional_pdf_{timestamp}.csv")
    pd.DataFrame(pdf_rows).to_csv(pdf_csv, index=False)
    print(f"✅ Interventional PDF saved to: {pdf_csv}")
    artifact_names.append(os.path.basename(pdf_csv))

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
        summary_rows.append({"key": "metric_mse_do_pred_vs_true", "value": f"{metrics['mse_do_pred_vs_true']:.6f}"})
        summary_rows.append({"key": "metric_iae_mean", "value": f"{metrics['iae_mean']:.6f}"})
        summary_rows.append({"key": "metric_iae_max", "value": f"{metrics['iae_max']:.6f}"})
        summary_rows.append({"key": "metric_ks_mean", "value": f"{metrics['ks_mean']:.6f}"})
        summary_rows.append({"key": "metric_ks_max", "value": f"{metrics['ks_max']:.6f}"})
        summary_rows.append({"key": "metric_cvm_mean", "value": f"{metrics['cvm_mean']:.6f}"})
        summary_rows.append({"key": "metric_cvm_max", "value": f"{metrics['cvm_max']:.6f}"})
        iae_series = ";".join(f"{val:.6f}" for val in metrics['iae_per_x'])
        ks_series = ";".join(f"{val:.6f}" for val in metrics['ks_per_x'])
        cvm_series = ";".join(f"{val:.6f}" for val in metrics['cvm_per_x'])
        summary_rows.append({"key": "metric_iae_per_x", "value": iae_series})
        summary_rows.append({"key": "metric_ks_per_x", "value": ks_series})
        summary_rows.append({"key": "metric_cvm_per_x", "value": cvm_series})

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
        include_oracle=True,
        first_stage_code="A1",
        second_stage_code="B1",
        kde_quantiles=(0.25, 0.5, 0.75),
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
