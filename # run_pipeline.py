# run_pipeline.py

import os
import subprocess

def run_script(script_path):
    print(f"\n▶ Running: {script_path}")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("⚠️ Error:", result.stderr)

def main():
    base_path = "C:/Research material/all trade data"
    hist_path = "C:/Research material/Historical data/oblast_weighted_averages.csv"

    # Define subfolders
    results_dir = os.path.join(base_path, "results")
    figures_dir = os.path.join(base_path, "figures")
    temp_dir = os.path.join(base_path, "temp")

    # Create required folders if not exist
    for folder in [results_dir, figures_dir, temp_dir]:
        os.makedirs(folder, exist_ok=True)

    # === STEP 1: Select controls ===
    run_script(os.path.join(base_path, "step1_select_controls.py"))

    # === STEP 2: Residualize ===
    run_script(os.path.join(base_path, "step2_residualize_fullsample.py"))

    # === STEP 3: Select IVs ===
    run_script(os.path.join(base_path, "step3_select_instruments.py"))

    # === STEP 4: IV Estimation ===
    run_script(os.path.join(base_path, "step4_iv_estimation.py"))

    print("\n✅ Pipeline execution complete.")

if __name__ == "__main__":
    main()
