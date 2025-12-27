# step2 residualize sample
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# === Define paths ===
base_dir = "C:/Research material/all trade data/import"
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

def downcast_df(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def residualize(y, X):
    model = LinearRegression().fit(X, y)
    return y - model.predict(X)

def run_residualization(baseline_path, hist_path, y_var, d_var, w_var):
    # Load selected controls
    selected_controls_path = os.path.join(results_dir, "selected_controls.csv")
    if not os.path.exists(selected_controls_path):
        raise FileNotFoundError(f"{selected_controls_path} not found. Run variable selection step first.")
    selected_controls = pd.read_csv(selected_controls_path)
    selected_x = selected_controls.iloc[:, 0].tolist()

    # Load datasets
    baseline = pd.read_csv(baseline_path)
    hist = pd.read_csv(hist_path)

    # Identify IVs
    z_vars = [col for col in hist.columns if col.startswith("dev_prcp_") or col.startswith("dev_air_")]

    # Validate columns
    baseline_required = [
        "oblast", "Year", "Month", "HS_Code_Group_2", "origin_country_name",
        "HS_Code_Position_4", "HS_Code_Subposition_6", "HS_Code_Category_8", y_var, w_var
    ]
    hist_required = ["oblast", d_var] + selected_x + z_vars

    for col in baseline_required:
        if col not in baseline.columns:
            raise KeyError(f"'{col}' not found in baseline dataset.")
    for col in hist_required:
        if col not in hist.columns:
            raise KeyError(f"'{col}' not found in historical dataset.")

    # Subset and merge
    baseline = baseline[baseline_required]
    hist = hist[hist_required]
    data = pd.merge(baseline, hist, on="oblast", how="left")

    # Generate time ID and downcast
    data["ym"] = data["Year"].astype(str) + "_" + data["Month"].astype(str)
    data = downcast_df(data)

    # Residualization
    if selected_x:
        categoricals = [col for col in selected_x if data[col].dtype == 'object']
        numerics = [col for col in selected_x if col not in categoricals]

        frames = []
        if numerics:
            frames.append(data[numerics])
        if categoricals:
            dummies = pd.get_dummies(data[categoricals], drop_first=True)
            if not dummies.empty:
                frames.append(dummies)

        if not frames:
            raise ValueError("No valid numeric or categorical variables found for residualization.")

        X = pd.concat(frames, axis=1)

        y_resid = residualize(data[y_var].values, X)
        d_resid = residualize(data[d_var].values, X)
        w_resid = residualize(data[w_var].values, X)

        resid_df = pd.DataFrame({
            "y_resid": y_resid,
            "d_resid": d_resid,
            "w_resid": w_resid
        }, index=data.index)

        data = pd.concat([data, resid_df], axis=1)
    else:
        data["y_resid"] = data[y_var]
        data["d_resid"] = data[d_var]
        data["w_resid"] = data[w_var]

    # === Overwrite d_resid with frequency-grouped and rounded version ===
    rounded_d = data["d_resid"].round(3)
    value_counts = rounded_d.value_counts()

    # Identify minor and major values
    minor_vals = value_counts[value_counts < 100].index
    major_vals = value_counts[value_counts >= 1000].index

    # Build mapping from each minor to closest major
    minor_to_major_map = {}
    for val in minor_vals:
        if len(major_vals) == 0:
            continue  # edge case: no majors available
        closest_major = major_vals[np.argmin(np.abs(major_vals - val))]
        minor_to_major_map[val] = closest_major

    # Replace minor values in d_resid
    d_resid_rounded = rounded_d.replace(minor_to_major_map)
    data["d_resid"] = d_resid_rounded.round(3).astype("float32")

    # Output dataset
    keep_cols = [
        "y_resid", "d_resid", "w_resid", "ym", "HS_Code_Group_2", "oblast",
        "origin_country_name", "HS_Code_Position_4", "HS_Code_Subposition_6", "HS_Code_Category_8"
    ] + z_vars
    keep_cols = [col for col in keep_cols if col in data.columns]
    data_out = data[keep_cols]

    output_file = os.path.join(results_dir, "residualized_dataset.csv")
    data_out.to_csv(output_file, index=False)

    print(f"\n✅ Step 2 Complete. Residualized dataset with grouped 'd_resid' saved to:\n{output_file}\n")
    return output_file

# === Run Step 2 ===
if __name__ == "__main__":
    baseline_path = os.path.join(base_dir, "baseline_import.csv")
    hist_path = "C:/Research material/Historical data/oblast_weighted_averages.csv"

    run_residualization(
        baseline_path=baseline_path,
        hist_path=hist_path,
        y_var="Import_USD",
        w_var="Net_Weight_kg",
        d_var="loss_per1000_33_34"
    )
