"""
Data Generating Process (DGP) Module
====================================

This module contains all data generation functions for the IV estimation project.
It implements triangular DGPs with instrument Z, endogenous regressor X, and outcome Y.

Key Functions:
- training_data_generation(): Generate dynamic training data (2000 samples)
- testdata_generation(): Generate and cache fixed test data (10000 samples)
- simulate_dataset(): Core data generation function

DGP Specifications:
- First-stage: X = G(Z, η) with options A1 (additive) and A2 (non-additive)
- Second-stage: Y = H(X, ε) with options B1 (polynomial) and B2 (nonlinear)
- Endogeneity: corr(ε, Φ^{-1}(η)) = ρ creates correlation between X and Y errors
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, replace
from typing import Dict, Tuple, Literal
from scipy.stats import norm
from pathlib import Path
import os


# =========================================================
# Configuration
# =========================================================
@dataclass
class DGPConfig:
    """Configuration for Data Generating Process"""
    n: int = 2000
    seed: int = 42
    rho: float = 0.6                        # corr(ε, Φ^{-1}(η))

    # First-stage family: X = G(Z, η)
    first_stage: Literal["A1", "A2", "A3"] = "A1"  # A1: additive location–scale; A2: monotone non-additive; A3: additive with latent H
    alpha0: float = 0.0
    alpha1: float = 1.0
    gamma0: float = -0.2
    gamma1: float = 0.2
    a2_h: Literal["exp", "cubic"] = "exp"   # A2 transform

    # Second-stage family: Y = H(X, ε)
    second_stage: Literal["B1", "B2", "B3", "B4", "B5"] = "B1"
    beta1: float = 1.0
    beta2: float = 0.5
    delta0: float = -0.5                    # σ_Y(V) = exp(δ0 + δ1 V)
    delta1: float = 0.7
    
    # B2 bimodal parameters (B2 always uses bimodal mixture)
    b2_mixture_weight: float = 0.5              # w: mixture weight for bimodal
    b2_peak_separation: float = 2.0             # Δμ: separation between peaks
    b2_sigma1: float = 0.3                      # σ₁: first component std
    b2_sigma2: float = 0.3                      # σ₂: second component std
    b2_beta_offset: float = 1.0                 # offset for second peak


# =========================================================
# DGP Utility Functions
# =========================================================
def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)


def mu_of_Z(z: np.ndarray, cfg: DGPConfig) -> np.ndarray:
    """Location parameter as function of Z"""
    return cfg.alpha0 + cfg.alpha1 * z


def sigma_of_Z(z: np.ndarray, cfg: DGPConfig) -> np.ndarray:
    """Scale parameter as function of Z"""
    return np.exp(cfg.gamma0 + cfg.gamma1 * z)


def sigmaY_of_X(x: np.ndarray, cfg: DGPConfig) -> np.ndarray:
    """Outcome noise scale σ_Y(X) for second-stage B1 specification."""
    x_arr = np.asarray(x, dtype=float)
    if cfg.second_stage != "B1":
        raise ValueError("sigmaY_of_X is only defined for second_stage='B1'.")
    return np.log1p(np.exp(cfg.delta0 + cfg.delta1 * x_arr))


def a2_h_transform(t: np.ndarray, cfg: DGPConfig) -> np.ndarray:
    """Non-linear transformation for A2 first-stage"""
    if cfg.a2_h == "exp":
        return np.exp(t)
    elif cfg.a2_h == "cubic":
        return t + (t ** 3) / 3.0
    else:
        raise ValueError("Unknown A2 transform")


def simulate_eps_eta(n: int, rho: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw (ε, η) with corr(ε, Φ^{-1}(η)) = ρ.
    This creates endogeneity between X and Y through correlated errors.
    
    Args:
        n: Number of samples
        rho: Correlation coefficient between ε and Φ^{-1}(η)
    
    Returns:
        eps: (n,) array of error terms for Y
        eta: (n,) array of uniform(0,1) quantiles for X
    """
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    Z0 = np.random.randn(2, n)
    E, T = L @ Z0
    eps = E
    eta = norm.cdf(T)  # in (0,1)
    return eps, eta


def simulate_bimodal_y(cfg: DGPConfig, X: np.ndarray, V_true: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """
    Generate Y from bimodal mixture of normals for B2-bimodal.
    
    Y follows mixture: w·N(μ₁, σ₁) + (1-w)·N(μ₂, σ₂)
    where each component mean depends on (X, V, ε).
    
    Args:
        cfg: DGP configuration
        X: (n,) endogenous regressor
        V_true: (n,) control function values
        eps: (n,) error terms from MARGINAL N(0,1)
    
    Returns:
        Y: (n,) bimodal outcomes
    """
    n = len(X)
    
    # Draw mixture indicators (which component each obs comes from)
    mixture_indicators = np.random.binomial(1, cfg.b2_mixture_weight, size=n)
    
    # Component means (both depend on X, eps)
    # First peak: similar to original sin-based B2
    mu1 = np.sin(X) + 0.3 * X * eps
    
    # Second peak: offset version
    mu2 = np.sin(X + cfg.b2_beta_offset) + cfg.b2_peak_separation + 0.3 * X * eps
    
    # Component standard deviations (X-dependent heteroskedasticity)
    sigma1 = cfg.b2_sigma1 * (1.0 + 0.2 * np.abs(X))
    sigma2 = cfg.b2_sigma2 * (1.0 + 0.3 * np.abs(X))
    
    # Generate from each component
    y1 = mu1 + sigma1 * np.random.randn(n)
    y2 = mu2 + sigma2 * np.random.randn(n)
    
    # Mix according to indicators
    Y = mixture_indicators * y1 + (1 - mixture_indicators) * y2
    
    return Y


def simulate_first_stage(cfg: DGPConfig, Z: np.ndarray, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Generate X = G(Z, η) according to first-stage specification.
    
    Args:
        cfg: DGP configuration
        Z: (n,) array of instruments
        eta: (n,) array of uniform(0,1) quantiles
    
    Returns:
        X: (n,) array of endogenous regressors
        V_true: (n,) array of true control function values (= η)
    """

    mu = mu_of_Z(Z, cfg)
    sig = sigma_of_Z(Z, cfg)
    t = mu + sig * norm.ppf(eta)  # location-scale index

    if cfg.first_stage == "A1":
        X = t
    elif cfg.first_stage == "A2":
        X = a2_h_transform(t, cfg)
    else:
        raise ValueError("Unknown first_stage")

    V_true = eta  # oracle control = PIT(η)
    return X, V_true


def simulate_second_stage(
    cfg: DGPConfig,
    X: np.ndarray,
    V_true: np.ndarray,
    eps: np.ndarray,
    *,
    latent_h: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate Y = H(X, ε) according to second-stage specification.
    
    Args:
        cfg: DGP configuration
        X: (n,) array of endogenous regressors
        V_true: (n,) array of true control function values
        eps: (n,) array of error terms
    
    Returns:
        Y: (n,) array of outcomes
    """
    
    if cfg.second_stage == "B1":
        m1 = cfg.beta1 * X + cfg.beta2 * (X ** 2)
        sigY = sigmaY_of_X(X, cfg)
        # Nonzero E[ε|V] under joint normal:  E[ε | V=v] = ρ Φ^{-1}(v)
        # We simulate Y directly; the conditional mean is handled in m_true().
        Y = m1 + sigY * eps
        return Y
    elif cfg.second_stage == "B2":
        # B2 always uses bimodal mixture of normals
        return simulate_bimodal_y(cfg, X, V_true, eps)
    elif cfg.second_stage == "B3":
        if latent_h is None:
            raise ValueError("Second-stage B3 requires latent_h draws.")
        return X - 3.0 * latent_h + eps
    elif cfg.second_stage == "B4":
        if latent_h is None:
            raise ValueError("Second-stage B4 requires latent_h draws.")
        h_arr = np.asarray(latent_h, dtype=float)
        if h_arr.shape != X.shape:
            raise ValueError("latent_h must have same shape as X for B4.")
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.shape != X.shape:
            raise ValueError("eps must have same shape as X for B4.")

        linear_branch = 0.2 * (5.5 + 2.0 * X + 3.0 * h_arr + eps_arr)
        # Ensure argument of log stays positive; clamp at a small epsilon.
        softplus_arg = (2.0 * X + h_arr) ** 2 + eps_arr ** 2
        safe_arg = np.maximum(softplus_arg, 1e-8)
        softplus_branch = np.log(safe_arg)
        return np.where(X <= 1.0, linear_branch, softplus_branch)
    elif cfg.second_stage == "B5":
        if latent_h is None:
            raise ValueError("Second-stage B5 requires latent_h draws.")
        h_arr = np.asarray(latent_h, dtype=float)
        if h_arr.shape != X.shape:
            raise ValueError("latent_h must have same shape as X for B5.")
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.shape != X.shape:
            raise ValueError("eps must have same shape as X for B5.")
        return 3.0 * np.sin(2.0 * X) + 2.0 * X - 3.0 * h_arr + eps_arr
    else:
        raise ValueError("Unknown second_stage")


def simulate_dataset(cfg: DGPConfig) -> Dict[str, np.ndarray]:
    """
    Main data generation function.
    
    Args:
        cfg: DGP configuration including sample size and random seed
    
    Returns:
        Dictionary containing:
            - Z: (n,) instrument
            - X: (n,) endogenous regressor
            - Y: (n,) outcome
            - V_true: (n,) true control function
            - eps: (n,) error term for Y
            - eta: (n,) uniform(0,1) quantile for X
    """
    set_seed(cfg.seed)
    n = cfg.n
    #Z = np.random.randn(n)
    Z = np.random.uniform(0, 3, n)
    if cfg.first_stage == "A3":
        if cfg.second_stage not in {"B3", "B4", "B5"}:
            raise ValueError("First-stage 'A3' currently supports second-stage 'B3', 'B4', or 'B5' only.")
        latent_h = np.random.randn(n)
        eps_x = np.random.randn(n)
        eps_y = np.random.randn(n)
        X = Z + latent_h + eps_x
        eta = norm.cdf((X - Z) / np.sqrt(2.0))  # F_{X|Z}(X|Z) under additive normal noise
        V_true = eta
        Y = simulate_second_stage(cfg, X, V_true, eps_y, latent_h=latent_h)
        return {
            "Z": Z,
            "X": X,
            "Y": Y,
            "V_true": V_true,
            "eps": eps_y,
            "eta": eta,
            "H": latent_h,
            "eps_x": eps_x,
            "eps_y": eps_y,
        }

    eps, eta = simulate_eps_eta(n, cfg.rho)
    X, V_true = simulate_first_stage(cfg, Z, eta)
    Y = simulate_second_stage(cfg, X, V_true, eps)
    return {"Z": Z, "X": X, "Y": Y, "V_true": V_true, "eps": eps, "eta": eta}


# =========================================================
# Training and Test Data Generation Functions
# =========================================================
def _resolve_path(path_like: str | os.PathLike[str]) -> Path:
    """Resolve paths relative to the project root (folder containing this file)."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parent
    return project_root / path


def training_data_generation(
    cfg: DGPConfig,
    save_csv: bool = True,
    train_dir: str | os.PathLike[str] = "IV_datasets/train",
) -> Dict[str, np.ndarray]:
    """
    Generate training data dynamically for each run.
    
    This function should be called at the start of each training run to generate
    fresh training data with the specified sample size (typically 2000).
    
    Args:
        cfg: DGP configuration with training sample size and seed
        save_csv: If True, save data to CSV format in IV_datasets/train/
    
    Returns:
        Dictionary with training data arrays (Z, X, Y, V_true, eps, eta)
    """
    print(f"Generating training data: n={cfg.n}, seed={cfg.seed}")
    data = simulate_dataset(cfg)
    
    if save_csv:
        # Create training data directory (resolved relative to repo root by default)
        train_dir = _resolve_path(train_dir)
        os.makedirs(train_dir, exist_ok=True)

        # Create DataFrame and save as CSV
        base_cols = {
            'Z': data['Z'],
            'X': data['X'], 
            'Y': data['Y'],
            'V_true': data['V_true'],
            'eps': data['eps'],
            'eta': data['eta']
        }
        extra_cols = {k: v for k, v in data.items() if k not in base_cols}
        df = pd.DataFrame({**base_cols, **extra_cols})
        
        # Save with DGP configuration
        csv_file = train_dir / f"train_data_{cfg.first_stage}_{cfg.second_stage}.csv"
        df.to_csv(csv_file, index=False)
        print(f"  ✅ Training data saved to: {csv_file}")
    
    print(f"  ✅ Training data generated successfully")
    return data


def testdata_generation(
    cfg: DGPConfig,
    test_dir: str | os.PathLike[str] = "IV_datasets/test",
    test_seed: int = 999,
    force_regenerate: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate or load fixed test dataset.
    
    This function creates a large fixed test set (typically 10000 samples) and caches
    it to disk in CSV format. Subsequent calls will load the cached data unless force_regenerate=True.
    
    Args:
        cfg: DGP configuration (n will be overridden to 10000 for test set)
        test_dir: Directory to save/load test data
        test_seed: Fixed seed for test data reproducibility (default: 999)
        force_regenerate: If True, regenerate test data even if it exists
    
    Returns:
        Dictionary with test data arrays (Z, X, Y, V_true, eps, eta)
    """
    # Create test directory if it doesn't exist
    test_dir = _resolve_path(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    # Test data file path (CSV format)
    test_data_file = test_dir / f"test_data_{cfg.first_stage}_{cfg.second_stage}.csv"

    # Check if test data already exists
    if os.path.exists(test_data_file) and not force_regenerate:
        print(f"Loading existing test data from: {test_data_file}")
        try:
            df = pd.read_csv(test_data_file)
            data = {
                'Z': df['Z'].values,
                'X': df['X'].values,
                'Y': df['Y'].values,
                'V_true': df['V_true'].values,
                'eps': df['eps'].values,
                'eta': df['eta'].values
            }
            print(f"  ✅ Test data loaded successfully (n={len(data['Z'])})")
            return data
        except Exception as e:
            print(f"  ⚠️  Failed to load test data: {e}")
            print(f"  Regenerating test data...")
    
    # Generate new test data
    print(f"Generating fixed test data: n=10000, seed={test_seed}")
    
    # Create test configuration with fixed parameters
    test_cfg = DGPConfig(
        n=10000,
        seed=test_seed,
        rho=cfg.rho,
        first_stage=cfg.first_stage,
        second_stage=cfg.second_stage,
        alpha0=cfg.alpha0,
        alpha1=cfg.alpha1,
        gamma0=cfg.gamma0,
        gamma1=cfg.gamma1,
        a2_h=cfg.a2_h,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
        delta0=cfg.delta0,
        delta1=cfg.delta1
    )
    
    # Generate test data
    test_data = simulate_dataset(test_cfg)
    
    # Save test data to CSV
    try:
        base_cols = {
            'Z': test_data['Z'],
            'X': test_data['X'],
            'Y': test_data['Y'],
            'V_true': test_data['V_true'],
            'eps': test_data['eps'],
            'eta': test_data['eta']
        }
        extra_cols = {k: v for k, v in test_data.items() if k not in base_cols}
        df = pd.DataFrame({**base_cols, **extra_cols})
        df.to_csv(test_data_file, index=False)
        print(f"  ✅ Test data generated and saved to: {test_data_file}")
    except Exception as e:
        print(f"  ⚠️  Failed to save test data: {e}")
    
    return test_data


# =========================================================
# Utility Functions for Data Summary
# =========================================================
def print_data_summary(data: Dict[str, np.ndarray], cfg: DGPConfig, data_type: str = "Data"):
    """
    Print summary statistics of generated data.
    
    Args:
        data: Dictionary containing Z, X, Y, V_true arrays
        cfg: DGP configuration
        data_type: Type of data for display (e.g., "Training", "Test")
    """
    Z, X, Y, V_true = data["Z"], data["X"], data["Y"], data["V_true"]
    
    print("\n" + "="*60)
    print(f"{data_type.upper()} DATA SUMMARY")
    print("="*60)
    print(f"Sample size: {len(Z)}")
    print(f"DGP configuration: {cfg.first_stage}/{cfg.second_stage}")
    print(f"Endogeneity correlation ρ: {cfg.rho}")
    print(f"Random seed: {cfg.seed}")
    
    print(f"\nVariable statistics:")
    print(f"  Z: mean={np.mean(Z):.3f}, std={np.std(Z):.3f}, range=[{np.min(Z):.3f}, {np.max(Z):.3f}]")
    print(f"  X: mean={np.mean(X):.3f}, std={np.std(X):.3f}, range=[{np.min(X):.3f}, {np.max(X):.3f}]")
    print(f"  Y: mean={np.mean(Y):.3f}, std={np.std(Y):.3f}, range=[{np.min(Y):.3f}, {np.max(Y):.3f}]")
    print(f"  V_true: mean={np.mean(V_true):.3f}, std={np.std(V_true):.3f}, range=[{np.min(V_true):.3f}, {np.max(V_true):.3f}]")


# =========================================================
# Main Execution (for testing)
# =========================================================
if __name__ == "__main__":
    """
    DGP CSV Data Generation
    Generates training and test data in CSV format
    """
    print("DGP CSV Data Generation")
    print("="*60)
    
    # Create configuration for training data
    train_cfg = DGPConfig(
        n=8000,
        seed=123,
        rho=0.6,
        first_stage="A3",
        second_stage="B5"
    )
    
    # Create configuration for test data  
    test_cfg = DGPConfig(
        n=10000,
        seed=999,
        rho=0.6,
        first_stage="A3",
        second_stage="B5"
    )
    
    print("Configuration:")
    print(f"  Training: n={train_cfg.n}, seed={train_cfg.seed}")
    print(f"  Test: n={test_cfg.n}, seed={test_cfg.seed}")
    print(f"  DGP: {train_cfg.first_stage}/{train_cfg.second_stage}, ρ={train_cfg.rho}")
    print()
    
    # Generate training data
    print("Generating training data...")
    train_data = training_data_generation(train_cfg, save_csv=True)
    print_data_summary(train_data, train_cfg, "Training")
    print()
    
    # Generate test data
    print("Generating test data...")
    test_data = testdata_generation(test_cfg, force_regenerate=True)
    print_data_summary(test_data, test_cfg, "Test")
    print()
    
    # Verify files were created
    train_dir = _resolve_path("IV_datasets/train")
    test_dir = _resolve_path("IV_datasets/test")
    
    print("Verifying generated files...")
    
    if train_dir.exists():
        train_files = [f for f in train_dir.glob("*.csv")]
        print(f"  Training CSV files: {len(train_files)}")
        for f in train_files:
            size = f.stat().st_size
            print(f"    {f.name}: {size:,} bytes")
    else:
        print("  ❌ Training directory not found")

    if test_dir.exists():
        test_files = [f for f in test_dir.glob("*.csv")]
        print(f"  Test CSV files: {len(test_files)}")
        for f in test_files:
            size = f.stat().st_size
            print(f"    {f.name}: {size:,} bytes")
    else:
        print("  ❌ Test directory not found")
    
    print()
    print("="*60)
    print("DGP CSV Generation Completed Successfully!")
    print("="*60)

#0
