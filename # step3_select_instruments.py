# step3_select_instruments_belloni_parallel_progress.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from joblib import Parallel, delayed
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def define_instruments(hist_df):
    return [col for col in hist_df.columns if col.startswith("dev_prcp_") or col.startswith("dev_air_")]


def compute_theoretical_alpha(n, p, c):
    return c * np.sqrt((2 * np.log(p)) / n)


def select_stable_instruments_belloni_parallel(resid_data_path, hist_path,
                                               d_resid_var,
                                               batch_frac,
                                               n_batches,
                                               max_iter,
                                               c,
                                               n_jobs):
    # Set up directories
    base_dir = os.path.dirname(resid_data_path)
    results_dir = os.path.join(base_dir, "results")
    figure_dir = os.path.join(base_dir, "figure")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    # Load data
    data = pd.read_csv(resid_data_path)
    hist = pd.read_csv(hist_path)
    z_vars = define_instruments(hist)
    z_vars = [z for z in z_vars if z in data.columns]

    if not z_vars:
        raise ValueError("No valid instrument variables found in residualized dataset.")

    print(f"\n📊 {len(z_vars)} candidate IVs found.")
    print(f"IVs: {', '.join(z_vars)}\n")

    # Compute theoretical alpha
    np.random.seed(123)
    indices = np.random.permutation(len(data))
    batch_size = int(batch_frac * len(data))
    n_obs = batch_size
    p_iv = len(z_vars)
    alpha = compute_theoretical_alpha(n_obs, p_iv, c=c)
    print(f"\n🔧 Using theoretical alpha = {alpha:.6f} (n={n_obs}, p={p_iv}, c={c})\n")

    # Fit StandardScaler once
    scaler = StandardScaler().fit(data[z_vars])

    # Prepare batches
    batches = [indices[i * batch_size: (i + 1) * batch_size] for i in range(n_batches)]

    # Define batch runner with progress reporting
    def run_batch(i, batch_idx):
        start_time = time.time()
        batch = data.iloc[batch_idx]
        Z_scaled = scaler.transform(batch[z_vars])
        d = batch[d_resid_var].values
        try:
            lasso = Lasso(alpha=alpha, max_iter=max_iter).fit(Z_scaled, d)
            selected = [(var, coef) for var, coef in zip(z_vars, lasso.coef_) if coef != 0]
            elapsed = time.time() - start_time
            iv_names = [var for var, _ in selected]
            print(f"[{i+1:02}/{n_batches}] ✅ Batch {i+1:02}: {len(selected)} IVs selected in {elapsed:.2f}s")
            if iv_names:
                print(f"     → Selected: {', '.join(iv_names)}")
            return {
                "batch": i + 1,
                "selected": selected
            }
        except Exception as e:
            print(f"[{i+1:02}/{n_batches}] ⚠️ Batch {i+1:02} failed: {e}")
            return {
                "batch": i + 1,
                "selected": []
            }

    # Run batches in parallel and track time
    print("🚀 Starting batch Lasso selection...\n")
    start_all = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_batch)(i, batch_idx) for i, batch_idx in enumerate(batches)
    )
    elapsed_all = time.time() - start_all
    print(f"\n⏱️ All {n_batches} batches completed in {elapsed_all:.2f} seconds.\n")

    # Aggregate results
    appearances = defaultdict(int)
    coef_sums = defaultdict(float)
    batch_logs = []

    for result in results:
        selected = result["selected"]
        for var, coef in selected:
            appearances[var] += 1
            coef_sums[var] += abs(coef)
        batch_logs.append({
            "batch": result["batch"],
            "n_selected": len(selected),
            "selected_vars": ";".join([f"{var}:{coef:.4f}" for var, coef in selected])
        })

    # Compute stats
    freq = pd.Series(appearances) / n_batches
    avg_coef = pd.Series(coef_sums) / n_batches
    stats = pd.DataFrame({"frequency": freq, "avg_coef": avg_coef})
    stats.sort_values(by=["frequency", "avg_coef"], ascending=False, inplace=True)

    selected_iv = stats.index.tolist()

    # Save outputs
    stats.to_csv(os.path.join(results_dir, "instrument_selection_frequency.csv"))
    pd.Series(selected_iv).to_csv(os.path.join(results_dir, "selected_instruments.csv"), index=False)
    pd.DataFrame(batch_logs).to_csv(os.path.join(results_dir, "batch_selection_log.csv"), index=False)

    stats.head(10)["frequency"].sort_values().plot(kind="barh", title="Top IVs by Selection Frequency")
    plt.xlabel("Share of Batches Selected")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "top_instruments.png"))
    plt.close()

    print(f"✅ Done. {len(selected_iv)} instruments selected and saved.\n")
    return selected_iv


if __name__ == "__main__":
    # 🔧 Unified config block
    config = {
        "resid_data_path": "",
        "hist_path": "",
        "d_resid_var": "d_resid",
        "batch_frac": 0.05,
        "n_batches": 20,
        "max_iter": 50000,
        "c": 400,
        "n_jobs": 2
    }

    select_stable_instruments_belloni_parallel(**config)

