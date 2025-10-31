"""
IV Stage 1 Implementation: Pre-generated Data Loading and TabPFN Control Function Estimation (Version 4)
========================================================================================================

This file implements objectives 1 and 2 from the research plan:
1) Load pre-generated triangular DGP data with instrument Z, endogenous regressor X, and outcome Y
2) Estimate control function V = F_{X|Z}(X | Z) using TabPFNRegressor full-distribution output

Key Changes from IV_stage1_3.py:
- Load pre-generated data from IV_datasets/train and IV_datasets/test directories
- Remove dynamic data generation functions and DGPConfig dependency
- Accept first_stage and second_stage parameters to locate correct data files
- Export 7-column training set (Z, X, Y, V_true, eps, eta, V_hat) for Stage 2 consumption
- Save results to IV_datasets/stage1_output with DGP-aware naming

The control function is the conditional CDF of X given Z, estimated via:
- Use TabPFNRegressor to obtain the full conditional distribution of X given Z
- Evaluate criterion-based CDF values F̂(x|z) directly (no interpolation)
- Train on full training set; downstream stages source test data directly from IV_datasets/test
- Error if TabPFN full distribution API is unavailable

Reference: TabPFN_Demo_HPC.ipynb and TabPFN_Demo_Local.ipynb for proper TabPFN usage
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import os
from datetime import datetime
import torch

# --- TabPFN (required, no fallbacks) ---
try:
    from tabpfn.regressor import TabPFNRegressor
    print("✅ TabPFNRegressor imported successfully (tabpfn.regressor)")
except Exception:
    try:
        from tabpfn import TabPFNRegressor
        print("✅ TabPFNRegressor imported successfully (tabpfn root)")
    except Exception as e:
        raise ImportError(f"❌ TabPFN is required but not available: {e}")


# =========================================================
# Helper utilities for TabPFN full distribution output
# =========================================================
def _broadcast_targets_for_logits(targets: np.ndarray, n_samples: int) -> np.ndarray:
    """Expand target evaluations to align with TabPFN logits batch."""
    arr = np.asarray(targets, dtype=float)
    if arr.ndim == 0:
        return np.full((n_samples, 1), float(arr))
    if arr.ndim == 1:
        if arr.size == n_samples:
            return arr.reshape(n_samples, 1)
        return np.tile(arr.reshape(1, -1), (n_samples, 1))
    if arr.ndim == 2 and arr.shape[0] == n_samples:
        return arr
    raise ValueError(
        f"Incompatible target shape {arr.shape}; expected scalar, (n,), or (n, m)."
    )


def cdf_from_full_output(
    full_output: Dict[str, object],
    x_values: np.ndarray,
    *,
    squeeze_last: bool = True,
) -> np.ndarray:
    """Evaluate F̂_{X|Z}(x | z) via TabPFN criterion.cdf."""
    if "logits" not in full_output or "criterion" not in full_output:
        raise KeyError("TabPFN full output missing 'logits' or 'criterion'.")

    logits = full_output["logits"]
    criterion = full_output["criterion"]

    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)

    device = criterion.borders.device  # type: ignore[attr-defined]
    logits = logits.to(device)

    targets_matrix = _broadcast_targets_for_logits(x_values, logits.shape[0])
    targets_tensor = torch.as_tensor(
        targets_matrix,
        dtype=criterion.borders.dtype,  # type: ignore[attr-defined]
        device=device,
    )

    with torch.no_grad():
        cdf_tensor = criterion.cdf(logits, targets_tensor)  # type: ignore[attr-defined]

    cdf_np = cdf_tensor.cpu().numpy()
    if squeeze_last and cdf_np.ndim == 2 and cdf_np.shape[1] == 1:
        cdf_np = cdf_np[:, 0]
    return cdf_np


# =========================================================
# 0) Configuration Extension
# =========================================================
@dataclass
class Stage1Config:
    """Configuration for Stage 1 Control Function Estimation"""
    # Optional quantile levels to request alongside full TabPFN outputs
    quantiles: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
    random_state: int = 1


# =========================================================
# 1) Data Loading Function
# =========================================================
def load_dgp_data(
    first_stage: str,
    second_stage: str,
    *,
    train_sample_size: int | None = None,
    base_dir: str = "IV_datasets",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pre-generated training and test data from IV_datasets directory.
    
    Expected file structure:
        base_dir/train/train_data_{first_stage}_{second_stage}.csv
        base_dir/train/train_data_{first_stage}_{second_stage}_{train_sample_size}.csv  # optional
        base_dir/test/test_data_{first_stage}_{second_stage}.csv
    
    Files must contain columns: Z, X, Y, V_true, eps, eta
    
    Args:
        first_stage: First stage DGP type (e.g., "A1", "A2")
        second_stage: Second stage DGP type (e.g., "B1", "B2")
        train_sample_size: If provided, append "_{train_sample_size}" when locating the training file
        base_dir: Base directory containing train/ and test/ subdirectories
    
    Returns:
        train_df, test_df: DataFrames with columns [Z, X, Y, V_true, eps, eta]
    
    Raises:
        FileNotFoundError: If matching files don't exist
        ValueError: If required columns are missing
    """
    if train_sample_size is None:
        train_filename = f"train_data_{first_stage}_{second_stage}.csv"
    else:
        train_filename = f"train_data_{first_stage}_{second_stage}_{train_sample_size}.csv"

    train_file = os.path.join(base_dir, "train", train_filename)
    test_file = os.path.join(base_dir, "test", f"test_data_{first_stage}_{second_stage}.csv")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test data not found: {test_file}")
    
    print(f"Loading training data from: {train_file}")
    print(f"Loading test data from: {test_file}")
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Validate columns
    required_cols = ["Z", "X", "Y", "V_true", "eps", "eta"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing column '{col}' in {train_file}")
        if col not in test_df.columns:
            raise ValueError(f"Missing column '{col}' in {test_file}")
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    if train_sample_size is not None and len(train_df) != train_sample_size:
        raise ValueError(
            f"Training data in {train_file} has {len(train_df)} rows, expected {train_sample_size}."
        )

    return train_df, test_df


class CondCDFModel:
    """
    Conditional CDF estimator F̂_{X|Z}(x | z) using TabPFNRegressor full distribution output.
    """
    
    def __init__(self, quantiles: Tuple[float, ...]):
        self.quantiles = tuple(quantiles)
        self.reg: TabPFNRegressor | None = None

    def _predict_full_output(self, Z: np.ndarray) -> Dict[str, object]:
        """Return TabPFN full-distribution output for provided Z values."""
        if self.reg is None:
            raise RuntimeError("Model must be fitted before prediction.")

        features = np.asarray(Z, dtype=float).reshape(-1, 1)
        kwargs: Dict[str, object] = {}
        if self.quantiles:
            kwargs["quantiles"] = list(self.quantiles)

        full_output = self.reg.predict(features, output_type="full", **kwargs)  # type: ignore[arg-type]
        if not isinstance(full_output, dict):
            raise RuntimeError("TabPFN full output not returned as a dictionary.")
        if "criterion" not in full_output or "logits" not in full_output:
            raise RuntimeError("TabPFN full output missing distribution components.")
        return full_output

    def predict_quantiles(self, Z: np.ndarray) -> Dict[float, np.ndarray]:
        """Optional helper to extract requested quantiles alongside the full output."""
        if not self.quantiles:
            raise RuntimeError("Quantile levels were not configured for this estimator.")
        full_output = self._predict_full_output(Z)
        q_arrays = full_output.get("quantiles")
        if q_arrays is None:
            raise RuntimeError("TabPFN full output missing 'quantiles' entries.")
        if len(q_arrays) != len(self.quantiles):
            raise RuntimeError(
                f"Quantile mismatch: expected {len(self.quantiles)} arrays, got {len(q_arrays)}."
            )
        return {tau: np.asarray(q_arrays[idx]) for idx, tau in enumerate(self.quantiles)}

    def fit(self, Z_train: np.ndarray, X_train: np.ndarray) -> None:
        """
        Train control function model on full training set.
        
        Args:
            Z_train: (n_train,) instrument values for training
            X_train: (n_train,) endogenous regressor values for training
        """
        Z_train = np.asarray(Z_train, dtype=float).reshape(-1, 1)

        print("Training model with TabPFNRegressor (full distribution output)...")

        self.reg = TabPFNRegressor(random_state=42, ignore_pretraining_limits=True)
        self.reg.fit(Z_train, X_train)

        # Validate that full output (criterion/logits) is available
        sample_size = min(12, len(Z_train))
        _ = self._predict_full_output(Z_train[:sample_size])
        print("  ✅ TabPFN full output verified (criterion + logits available)")

    def predict(self, Z_test: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        Predict control function values for test data.
        
        Args:
            Z_test: (n_test,) instrument values
            X_test: (n_test,) endogenous regressor values
        
        Returns:
            V_hat: (n_test,) estimated control function values, uniformly distributed on [0,1]
        """
        if self.reg is None:
            raise RuntimeError("Model must be fitted before prediction.")

        full_output = self._predict_full_output(Z_test)
        V_hat = cdf_from_full_output(full_output, X_test, squeeze_last=True)
        V_hat = np.clip(np.asarray(V_hat, dtype=float), 0.0, 1.0)
        if V_hat.shape[0] != len(X_test):
            raise ValueError(
                f"Unexpected CDF prediction shape {V_hat.shape}; expected ({len(X_test)},)."
            )
        return V_hat


# =========================================================
# 3) Main execution pipeline
# =========================================================
def run_stage1_experiment(first_stage: str, second_stage: str, stage1_cfg: Stage1Config,
                          train_sample_size: int | None = None,
                          output_dir: str = "IV_datasets/stage1_output") -> Dict[str, pd.DataFrame]:
    """
    Main pipeline implementing objectives 1 and 2:
    1) Load pre-generated triangular DGP data (training and test sets)
    2) Estimate control function using TabPFN
    
    Args:
        first_stage: First stage DGP type (e.g., "A1", "A2")
        second_stage: Second stage DGP type (e.g., "B1", "B2")
        stage1_cfg: Stage 1 estimation configuration
        train_sample_size: Optional training sample size identifier used in the CSV filename
        output_dir: Directory for saving outputs
    
    Returns:
        Dict with a single key "train" containing the training DataFrame
        augmented with V_hat.
    """
    print("Starting IV Stage 1 Experiment (Version 4)")
    print(f"DGP Configuration: first_stage={first_stage}, second_stage={second_stage}")
    print(f"Stage 1 Configuration: {asdict(stage1_cfg)}")
    if train_sample_size is not None:
        print(f"Requested training sample size: {train_sample_size}")
    
    # --- Objective 1: Load Pre-generated Data ---
    print("\n[1/3] Loading pre-generated training data...")
    train_df, test_df = load_dgp_data(first_stage, second_stage, train_sample_size=train_sample_size)
    
    # Extract arrays from DataFrames
    Z_train = train_df["Z"].values
    X_train = train_df["X"].values
    Y_train = train_df["Y"].values
    V_train_true = train_df["V_true"].values
    
    Z_test = test_df["Z"].values
    X_test = test_df["X"].values
    Y_test = test_df["Y"].values
    V_test_true = test_df["V_true"].values  # retained for diagnostics even though no predictions are generated
    
    print(f"Training data summary:")
    print(f"  Sample size: {len(Z_train)}")
    print(f"  X range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"  Y range: [{Y_train.min():.3f}, {Y_train.max():.3f}]")
    print(f"  Z range: [{Z_train.min():.3f}, {Z_train.max():.3f}]")
    
    print(f"Test data summary:")
    print(f"  Sample size: {len(Z_test)}")
    print(f"  X range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    print(f"  Y range: [{Y_test.min():.3f}, {Y_test.max():.3f}]")
    print(f"  Z range: [{Z_test.min():.3f}, {Z_test.max():.3f}]")
    print(f"  V_true range: [{V_test_true.min():.3f}, {V_test_true.max():.3f}]")

    # --- Objective 2: Stage 1 CDF via TabPFN ---
    print(f"\n[2/3] Estimating control function V = F_{{X|Z}}(X | Z)...")
    
    print("Using TabPFN full distribution output for CDF estimation.")
    if stage1_cfg.quantiles:
        print(f"Requested supplemental quantiles ({len(stage1_cfg.quantiles)}): {stage1_cfg.quantiles}")
    print(f"Training X range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"Test X range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    
    # Train conditional CDF model
    cdf_model = CondCDFModel(quantiles=stage1_cfg.quantiles)
    cdf_model.fit(Z_train, X_train)
    
    # Predict on training data
    print(f"\n[3/3] Predicting control function on training data...")
    V_train_hat = cdf_model.predict(Z_train, X_train)
    print(f"V̂_train statistics: mean={np.mean(V_train_hat):.3f}, std={np.std(V_train_hat):.3f}")

    train_result_df = pd.DataFrame({
        "Z": Z_train,
        "X": X_train,
        "Y": Y_train,
        "V_true": V_train_true,
        "eps": train_df["eps"].values,
        "eta": train_df["eta"].values,
        "V_hat": V_train_hat
    })

    return {"train": train_result_df}


# =========================================================
# 4) Runner
# =========================================================
if __name__ == "__main__":
    """
    Main execution: demonstrate Stage 1 implementation with pre-generated data.
    Loads training and test data from IV_datasets directory based on DGP parameters.
    """
    
    # DGP parameters (default values)
    first_stage = "A3"
    second_stage = "B5"

    # Training data size identifier used in the CSV filename; set to None for legacy naming
    train_sample_size = 8000
    
    # Stage 1 estimation configuration
    stage1_cfg = Stage1Config(
        random_state=1,
    )

    # Run experiment
    try:
        results = run_stage1_experiment(first_stage, second_stage, stage1_cfg, train_sample_size=train_sample_size)
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Print basic statistics
        for subset, df in results.items():
            print(f"\nSubset '{subset}' results:")
            print(f"  Sample size: {len(df)}")
            print(f"  X range: [{df['X'].min():.3f}, {df['X'].max():.3f}]")
            print(f"  Y range: [{df['Y'].min():.3f}, {df['Y'].max():.3f}]")
            print(f"  Z range: [{df['Z'].min():.3f}, {df['Z'].max():.3f}]")
            print(f"  V_true range: [{df['V_true'].min():.3f}, {df['V_true'].max():.3f}]")
            print(f"  V_hat range: [{df['V_hat'].min():.3f}, {df['V_hat'].max():.3f}]")
        
        # Save results to output directory
        output_dir = "IV_datasets/stage1_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save DataFrames as CSV files with DGP-aware naming
        for subset, df in results.items():
            sample_tag = f"_{train_sample_size}" if train_sample_size is not None else ""
            csv_file = os.path.join(
                output_dir,
                f"iv_stage1_{subset}_{first_stage}_{second_stage}{sample_tag}_{timestamp}.csv",
            )
            df.to_csv(csv_file, index=False)
            print(f"\n✅ Results saved to: {csv_file}")
        
        print(f"\n📁 Output saved in directory: {output_dir}/")
        print(f"🕒 Timestamp: {timestamp}")
        print(f"📊 DGP Configuration: {first_stage}_{second_stage}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()


#1
