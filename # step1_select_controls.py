import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import gc
import warnings

# === Define paths ===
base_dir = ""
temp_dir = os.path.join(base_dir, "temp")
results_dir = os.path.join(base_dir, "results")
plots_dir = os.path.join(base_dir, "plots")

# Ensure output folders exist
for folder in [temp_dir, results_dir, plots_dir]:
    os.makedirs(folder, exist_ok=True)

# === Feature selection logic ===
def define_variable_groups(hist):
    hist_cols = hist.columns
    group1 = ["plan_30_31", "plan_31_32", "wheat_pct", "pf_wheat_pct", "cf_wheat_pct", "sf_wheat_pct",  # <- change if needed
              "pf_winter_pct", "cf_winter_pct", "sf_winter_pct", "cf_abs", "cf_per1000",
              "dist_to_rail", "distance_city_20k", "district_area"]
    group2 = ["pop_1927", "rpop_1927", "urbanization_1927", "literacy_rural_1927", "rethnic_frac_1927"]
    group2 += [col for col in hist_cols if col.startswith("rshare_")]
    group3 = ["share_industrial_workers_1930", "industrial_output_pc_1930", "afactories_pc_1930",
              "bfactories_pc_1930", "abfactories_pc_1930", "aworkers_pc_1930",
              "bworkers_pc_1930", "abworkers_pc_1930"]
    return group1 + group2 + group3

def downcast_df(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def select_controls_lasso(baseline_path, hist_path, y_var, d_var, threshold, batch_frac=0.05, n_batches=20):
    baseline = pd.read_csv(baseline_path)
    hist = pd.read_csv(hist_path)

    xvars = define_variable_groups(hist)
    hist = hist[["oblast", d_var] + xvars]
    data = pd.merge(baseline, hist, on="oblast", how="left")
    data = downcast_df(data)

    print("\n📊 Diagnostics after merging:")
    print(f"Number of observations: {len(data)}")
    print(f"Number of covariates (candidate controls): {len(xvars)}")

    missing_vars = [var for var in xvars if var not in data.columns]
    if missing_vars:
        print("❗ Warning: Missing covariates:")
        for var in missing_vars:
            print(f" - {var}")
        xvars = [var for var in xvars if var in data.columns]

    na_counts = data[[y_var, d_var] + xvars].isna().sum()
    print("\n🔎 Missing values (nonzero only):")
    print(na_counts[na_counts > 0])

    data = data.dropna(subset=[y_var, d_var] + xvars)

    np.random.seed(42)
    idx = np.random.permutation(len(data))
    batch_size = int(len(data) * batch_frac)

    batch_results = []
    print("\n🔁 Running Lasso selection over batches...")

    for i in range(n_batches):
        batch = data.iloc[idx[i * batch_size: (i + 1) * batch_size]]
        X = batch[xvars].values
        X = StandardScaler().fit_transform(X)
        y = batch[y_var].values
        d = batch[d_var].values

        try:
            lasso_y = LassoCV(cv=5, max_iter=10000).fit(X, y)
            lasso_d = LassoCV(cv=5, max_iter=10000).fit(X, d)

            selected_idx = np.union1d(
                np.where(lasso_y.coef_ != 0)[0],
                np.where(lasso_d.coef_ != 0)[0]
            )
            selected_vars = np.array(xvars)[selected_idx]
            batch_results.append(set(selected_vars))

            print(f"  ✅ Batch {i+1:02}: {len(selected_vars)} selected -> {list(selected_vars)}")

        except Exception as e:
            warnings.warn(f"⚠️ LassoCV failed on batch {i}: {e}")
            batch_results.append(set())
            continue

        del batch, X, y, d, lasso_y, lasso_d
        gc.collect()

    # === Compute selection frequency across batches ===
    var_counts = pd.Series(0, index=pd.Index(xvars, name="Variable"))

    for selected in batch_results:
        for var in selected:
            if var in var_counts:
                var_counts[var] += 1

    var_freq = var_counts / n_batches
    stable_controls = var_freq[var_freq >= threshold].index.tolist()

    # === Save outputs ===
    var_freq.to_csv(os.path.join(results_dir, "control_selection_frequency.csv"))
    pd.Series(stable_controls).to_csv(os.path.join(results_dir, "selected_controls.csv"), index=False)

    # === Plot top 15 ===
    var_freq.sort_values(ascending=False).head(15).plot(kind='barh', title='Top 15 Controls by Batch Selection Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top_controls.png"))
    plt.close()

    print(f"\n✅ Step 1 Complete. {len(stable_controls)} controls selected based on {threshold*100:.0f}% batch stability.\n")
    return stable_controls

# === Run Step 1 ===
if __name__ == "__main__":
    baseline_path = os.path.join(base_dir, "baseline_import.csv")
    hist_path = ""
    threshold = 0.8  # <- Change this if needed
    select_controls_lasso(
        baseline_path=baseline_path,
        hist_path=hist_path,
        y_var="Import_USD",
        d_var="loss_per1000_33_34",
        threshold=threshold
    )


