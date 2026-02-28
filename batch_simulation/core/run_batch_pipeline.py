#!/usr/bin/env python3
"""
Batch runner orchestrating DGP → Stage 1 → Stage 2 pipelines across multiple
random seeds, DGP specifications, and training sample sizes.

All inputs and outputs are scoped to the batch_simulation sub-directory.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import pandas as pd

from DGP_batch import DGPConfig, training_data_generation, testdata_generation
from IV_stage1_4_batch import (
    Stage1Config,
    run_stage1_experiment,
    DEFAULT_DATA_DIR as STAGE1_DATA_DIR,
    DEFAULT_STAGE1_OUTPUT_DIR,
)
from IV_stage2_9_batch import (
    Stage2_9Config,
    run_stage2_9_experiment,
    save_stage2_9_results,
    DEFAULT_STAGE2_OUTPUT_DIR,
)


BATCH_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(STAGE1_DATA_DIR)
STAGE1_OUTPUT = Path(DEFAULT_STAGE1_OUTPUT_DIR)
STAGE2_OUTPUT = Path(DEFAULT_STAGE2_OUTPUT_DIR)
# Default to A5 baseline plus new A6 variants; override via --dgp-codes or DGP_CODES_OVERRIDE in SLURM wrapper.
DEFAULT_DGP_CODES = ("A5_B3", "A6_B3", "A6_B4", "A6_B5", "A6_B6")


def parse_dgp_code(code: str) -> Tuple[str, str]:
    parts = code.split("_")
    if len(parts) != 2:
        raise ValueError(f"DGP code '{code}' must be formatted as 'A?_B?'.")
    return parts[0].upper(), parts[1].upper()


def expected_stage1_csv(first_stage: str, second_stage: str, n: int, seed: int) -> Path:
    codes = f"{first_stage}_{second_stage}"
    return STAGE1_OUTPUT / f"iv_stage1_train_{codes}_n{n}_seed{seed}.csv"


def expected_stage2_summary(first_stage: str, second_stage: str, n: int, seed: int) -> Path:
    codes = f"{first_stage}_{second_stage}"
    return STAGE2_OUTPUT / f"s2_{codes}_n{n}_seed{seed}_summary.csv"


def ensure_test_dataset(first_stage: str, second_stage: str, test_seed: int, force: bool) -> None:
    cfg = DGPConfig(
        n=10000,
        seed=test_seed,
        first_stage=first_stage,
        second_stage=second_stage,
    )
    testdata_generation(
        cfg,
        test_dir=DATA_DIR / "test",
        test_seed=test_seed,
        force_regenerate=force,
    )


def extract_mse_from_summary(summary_path: Path) -> float:
    df = pd.read_csv(summary_path)
    metrics = {row["key"]: row["value"] for _, row in df.iterrows()}
    mse_str = metrics.get("metric_mse_do_pred_vs_clean")
    if mse_str is None:
        raise KeyError(f"metric_mse_do_pred_vs_clean not found in {summary_path}")
    return float(mse_str)


def run_pipeline(
    dgp_codes: Iterable[str],
    train_sizes: Iterable[int],
    seeds: Iterable[int],
    *,
    stage1_cfg: Stage1Config,
    stage2_random_state: int,
    test_seed: int,
    force_regenerate_test: bool,
    skip_existing: bool,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    ensured_tests: set[Tuple[str, str]] = set()

    combinations = itertools.product(dgp_codes, train_sizes, seeds)
    for code, n_samples, seed in combinations:
        combo_start = time.time()
        first_stage, second_stage = parse_dgp_code(code)
        key = (first_stage, second_stage)
        print(f"\n=== Running combo: {code}, n={n_samples}, seed={seed} ===")

        if key not in ensured_tests:
            print("Ensuring test dataset is available...")
            ensure_test_dataset(first_stage, second_stage, test_seed, force_regenerate_test)
            ensured_tests.add(key)

        stage1_csv_path = expected_stage1_csv(first_stage, second_stage, n_samples, seed)
        stage2_summary_path = expected_stage2_summary(first_stage, second_stage, n_samples, seed)

        if skip_existing and stage1_csv_path.exists():
            print(f"Stage 1 output already exists at {stage1_csv_path}. Skipping regeneration.")
        else:
            print("Generating training dataset...")
            train_cfg = DGPConfig(
                n=n_samples,
                seed=seed,
                first_stage=first_stage,
                second_stage=second_stage,
            )
            training_data_generation(
                train_cfg,
                save_csv=True,
                train_dir=DATA_DIR / "train",
            )

            print("Running Stage 1...")
            run_stage1_experiment(
                first_stage,
                second_stage,
                stage1_cfg,
                train_sample_size=n_samples,
                seed=seed,
                output_dir=STAGE1_OUTPUT,
                base_dir=DATA_DIR,
                save_outputs=True,
                use_timestamp=False,
            )

        if skip_existing and stage2_summary_path.exists():
            print(f"Stage 2 summary already exists at {stage2_summary_path}. Skipping Stage 2 execution.")
        else:
            print("Running Stage 2...")
            stage2_cfg = Stage2_9Config(
                input_dir=str(STAGE1_OUTPUT),
                output_dir=str(STAGE2_OUTPUT),
                dgp_base_dir=str(DATA_DIR),
                n_train_samples=n_samples,
                train_seed=seed,
                random_state=stage2_random_state,
                first_stage_code=first_stage,
                second_stage_code=second_stage,
            )
            results = run_stage2_9_experiment(stage2_cfg)
            artifacts = save_stage2_9_results(results, STAGE2_OUTPUT, use_timestamp=False)
            stage2_summary_path = artifacts["summary"]

        mse_value = extract_mse_from_summary(stage2_summary_path)
        records.append(
            {
                "first_stage": first_stage,
                "second_stage": second_stage,
                "dgp_code": code,
                "train_sample_size": n_samples,
                "seed": seed,
                "stage1_csv": str(stage1_csv_path),
                "stage2_summary": str(stage2_summary_path),
                "mse_do_pred_vs_clean": mse_value,
            }
        )
        combo_elapsed = time.time() - combo_start
        print(
            f"Recorded MSE for combo {code}, n={n_samples}, seed={seed}: {mse_value:.6f} "
            f"(elapsed {combo_elapsed:.1f}s)"
        )

    return pd.DataFrame.from_records(records)


def summarise_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby(["first_stage", "second_stage", "train_sample_size"], as_index=False)
        .agg(
            mean_mse=("mse_do_pred_vs_clean", "mean"),
            std_mse=("mse_do_pred_vs_clean", "std"),
            min_mse=("mse_do_pred_vs_clean", "min"),
            max_mse=("mse_do_pred_vs_clean", "max"),
            n_runs=("mse_do_pred_vs_clean", "count"),
        )
        .sort_values(["first_stage", "second_stage", "train_sample_size"])
        .reset_index(drop=True)
    )
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Run batch IV simulations over multiple seeds.")
    parser.add_argument(
        "--dgp-codes",
        nargs="+",
        default=DEFAULT_DGP_CODES,
        help=(
            "List of DGP codes formatted as 'A?_B?' (e.g., A3_B2). "
            f"Default: {', '.join(DEFAULT_DGP_CODES)}"
        ),
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        required=True,
        help="Training sample sizes to simulate (e.g., 1000 2000 4000).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Random seeds for DGP → Stage1 → Stage2 pipeline.",
    )
    parser.add_argument(
        "--stage1-random-state",
        type=int,
        default=1,
        help="Random state for Stage 1 TabPFNRegressor.",
    )
    parser.add_argument(
        "--stage2-random-state",
        type=int,
        default=1,
        help="Random state for Stage 2 pipeline components.",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=999,
        help="Seed used when generating fixed test datasets.",
    )
    parser.add_argument(
        "--force-test-regenerate",
        action="store_true",
        help="Force regeneration of test datasets even if cached versions exist.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip Stage 1/Stage 2 if deterministic outputs already exist.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional path to save aggregated MSE summary.",
    )

    args = parser.parse_args()

    STAGE1_OUTPUT.mkdir(parents=True, exist_ok=True)
    STAGE2_OUTPUT.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "train").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "test").mkdir(parents=True, exist_ok=True)

    stage1_cfg = Stage1Config(random_state=args.stage1_random_state)

    results_df = run_pipeline(
        args.dgp_codes,
        args.train_sizes,
        args.seeds,
        stage1_cfg=stage1_cfg,
        stage2_random_state=args.stage2_random_state,
        test_seed=args.test_seed,
        force_regenerate_test=args.force_test_regenerate,
        skip_existing=args.skip_existing,
    )

    if results_df.empty:
        print("No runs executed. Verify configuration or disable --skip-existing.")
        return

    print("\n=== Individual run metrics ===")
    print(results_df.to_string(index=False))

    summary_df = summarise_results(results_df)
    if not summary_df.empty:
        print("\n=== Aggregated MSE summary ===")
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"\nAggregated summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
