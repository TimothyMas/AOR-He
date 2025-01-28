import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import warnings
import multiprocessing
from multiprocessing import Pool
from functools import partial
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
CONFIG_FILE = 'config_benzaldehyde.json'  # (1) A separate config file if you like

# Extinction coefficient (M^-1 cm^-1) and path length (cm)
# (2) Change these if your benzaldehyde system requires different values:
EPSILON = 6220
PATH_LENGTH = 1.0

# File-specific cutoffs: reaction starts or substrate added at these times
# (3) Add your new benzaldehyde data files here:
FILE_SPECIFIC_CUTOFFS = {
    "AOR51_250127_153051": {"time_cutoff": 56.0, "abs_offset": 0.0423},
    "AOR52_250127_154356": {"time_cutoff": 55.0, "abs_offset": 0.0489},
    "AOR53_250127_155403": {"time_cutoff": 55.0, "abs_offset": 0.0694}
}

# ---------------------------------------------------------------------
# PIECEWISE BURST-PHASE MODEL
# ---------------------------------------------------------------------
def piecewise_burst_phase(t, Pi, kb, Vss, t0):
    """
    [P](t) = 0 for t < t0,
             Pi*(1 - exp(-kb*(t - t0))) + Vss*(t - t0) for t >= t0
    """
    out = np.zeros_like(t, dtype=float)
    mask = (t >= t0)
    t_shift = t[mask] - t0
    out[mask] = Pi*(1.0 - np.exp(-kb*t_shift)) + Vss*t_shift
    return out

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'processing.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up.")

# ---------------------------------------------------------------------
# LOAD CONFIG
# ---------------------------------------------------------------------
def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}.")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise

# ---------------------------------------------------------------------
# DATA READING
# ---------------------------------------------------------------------
def convert_abs_to_conc(abs_value, epsilon=EPSILON, path_length=PATH_LENGTH):
    return abs_value / (epsilon * path_length)

def read_and_prepare_data(file_path, delimiter=','):
    """
    Reads a file, skipping lines until header "Time", returns a DataFrame
    with columns: Time, Product_Concentration, etc.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Identify header line
        header_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('"Time'):
                header_index = i
                break

        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            skiprows=header_index,
            quoting=1
        )

        if df.shape[1] < 2:
            logging.warning(f"File {file_path} has < 2 columns.")
            return None

        df.columns = ['Time', 'Abs'] + list(df.columns[2:])
        df['Abs'] = df['Abs'].astype(str).str.replace(r'["\s]', '', regex=True)

        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df['Abs']  = pd.to_numeric(df['Abs'], errors='coerce')
        df.dropna(subset=['Time','Abs'], inplace=True)

        df['Product_Concentration'] = df['Abs'].apply(convert_abs_to_conc)
        if df.empty:
            logging.warning(f"No valid data after cleaning {file_path}.")
            return None

        return df

    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

# ---------------------------------------------------------------------
# FITTING
# ---------------------------------------------------------------------
def fit_piecewise_burst(t_fit, P_fit, t0):
    if len(t_fit) < 3:
        return None, None, None

    t_prime = t_fit - t0

    # Let's try different initial guesses
    Pi_init = max(P_fit)*0.5   # 50% of max
    kb_init = 0.01            # slower reaction guess
    Vss_init = 0.0

    # Or if data goes all the way from 0 to some plateau:
    # Vss_init = (P_fit[-1] - Pi_init) / (t_prime[-1] + 1e-9)

    def burst_shifted(tprime, Pi, kb, Vss):
        return Pi*(1.0 - np.exp(-kb*tprime)) + Vss*tprime

    p0 = [Pi_init, kb_init, Vss_init]

    try:
        popt, pcov = curve_fit(
            burst_shifted,
            t_prime,
            P_fit,
            p0=p0,
            maxfev=100000,
            # Example bounds if you like, 
            # lower bounds = 0 to avoid negative rates:
            # bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf])
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, pcov
    except RuntimeError as e:
        logging.error(f"Curve fitting failed: {e}")
        return None, None, None

# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------
def plot_piecewise_model(
    df_all, df_fit,
    popt, perr,
    t0, r_squared,
    file_name, output_dir
):
    """
    Plots:
      - All raw data (time 0..end)
      - Fitted line from first *fitting* data point to the max time
      - Residuals only for the fitting subset
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Unpack parameters
        Pi, kb, Vss = popt
        Pi_err, kb_err, Vss_err = perr

        # All data
        t_all = df_all['Time'].values
        P_all = df_all['Product_Concentration'].values

        t_min_fit = df_fit['Time'].min()  # earliest point used for fitting
        t_max = df_all['Time'].max()

        # Prepare for plotting the line from t_min_fit to t_max
        t_plot = np.linspace(t_min_fit, t_max, 300)

        # Evaluate the piecewise function
        def piecewise_func(t):
            out = np.zeros_like(t)
            mask = (t >= t0)
            shifted = t[mask] - t0
            out[mask] = Pi*(1.0 - np.exp(-kb*shifted)) + Vss*shifted
            return out

        P_plot = piecewise_func(t_plot)

        # Residuals for the subset
        t_fit = df_fit['Time'].values
        P_fit_data = df_fit['Product_Concentration'].values
        P_model_fit = piecewise_func(t_fit)
        residuals = P_fit_data - P_model_fit

        plt.figure(figsize=(10, 8))

        # Top subplot: raw data + model line
        plt.subplot(2,1,1)
        plt.plot(t_all, P_all, 'o', label='All Data', markersize=4)
        plt.plot(t_plot, P_plot, '-', label='Fitted Model', linewidth=2)

        annot_text = (
            f"Pi = {Pi:.3g} ± {Pi_err:.3g}\n"
            f"kb = {kb:.3g} ± {kb_err:.3g}\n"
            f"Vss = {Vss:.3g} ± {Vss_err:.3g}\n"
            f"t0 = {t0:.1f}\n"
            f"R² = {r_squared:.4f}"
        )
        plt.text(
            0.05, 0.95, annot_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round", fc="w")
        )

        plt.title(f'Piecewise Burst Fit: {file_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('[P] (M)')
        plt.legend()
        plt.grid(True)

        # Bottom subplot: residuals
        plt.subplot(2,1,2)
        plt.plot(t_fit, residuals, 'o', markersize=4)
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.xlabel('Time (s) [Fitting Subset]')
        plt.ylabel('Residuals')
        plt.grid(True)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{file_name}_fit.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logging.info(f"Plot saved to {out_path}.")

    except Exception as e:
        logging.error(f"Error in plotting for {file_name}: {e}")

# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------
def save_regression_parameters(params, file_name, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        param_file = os.path.join(output_dir, f"{file_name}_parameters.json")
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=4)
        logging.info(f"Parameters saved to {param_file}.")
    except Exception as e:
        logging.error(f"Error saving parameters for {file_name}: {e}")

def save_summary(summary_list, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        df_summary = pd.DataFrame(summary_list)
        summary_file = os.path.join(output_dir, 'summary.csv')
        df_summary.to_csv(summary_file, index=False)
        logging.info(f"Summary saved to {summary_file}.")
    except Exception as e:
        logging.error(f"Error saving summary: {e}")

# ---------------------------------------------------------------------
# PROCESS EACH FILE
# ---------------------------------------------------------------------
def process_file(file_path, config):
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        logging.info(f"Processing {file_name}...")

        df_all = read_and_prepare_data(file_path, delimiter=config.get('delimiter', ','))
        if df_all is None or df_all.empty:
            logging.warning(f"No valid data in {file_name}.")
            return None

        # ------------------------------------------------------------
        # 1) Get the cutoff info (time and offset) for this file
        # ------------------------------------------------------------
        cutoff_info = FILE_SPECIFIC_CUTOFFS.get(file_name, {"time_cutoff": 0.0, "abs_offset": 0.0})
        t0 = cutoff_info["time_cutoff"]
        offset = cutoff_info["abs_offset"]

        # ------------------------------------------------------------
        # 2) Subtract offset from the Abs column BEFORE converting to conc
        #    If you've already converted to concentration, do it analogously.
        #    But typically we do it at the absorbance level.
        # ------------------------------------------------------------
        df_all["Abs"] = df_all["Abs"] - offset

        # Recompute concentration using the updated absorbance
        # or if you already have it, just do it once up front in read_and_prepare_data().
        # E.g., if read_and_prepare_data() calls convert_abs_to_conc right away,
        # then either:
        #   A) move the offset-subtraction inside read_and_prepare_data before convert_abs_to_conc, or
        #   B) do the offset subtraction here and *then* recalculate Product_Concentration
        df_all["Product_Concentration"] = df_all["Abs"].apply(
            lambda x: x / (EPSILON * PATH_LENGTH)
        )

        # ------------------------------------------------------------
        # 3) Filter for t >= t0
        # ------------------------------------------------------------
        df_fit = df_all[df_all["Time"] >= t0].copy()
        if df_fit.empty:
            logging.warning(f"No data >= {t0} for {file_name}. Skipping.")
            return None

        # Now do the usual piecewise fit...
        t_fit = df_fit["Time"].values
        P_fit = df_fit["Product_Concentration"].values

        popt, perr, pcov = fit_piecewise_burst(t_fit, P_fit, t0)
        if popt is None:
            logging.warning(f"Fitting failed for {file_name}.")
            return None

        # Evaluate R^2 on the fitting subset
        Pi, kb, Vss = popt
        P_model_subset = piecewise_burst_phase(t_fit, Pi, kb, Vss, t0)
        residuals = P_fit - P_model_subset
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((P_fit - np.mean(P_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Plot as normal
        plot_piecewise_model(
            df_all, df_fit, popt, perr,
            t0, r_squared, file_name,
            config['output_plot_dir']
        )

        # Save parameters
        Pi_err, kb_err, Vss_err = perr
        param_dict = {
            "Pi": float(Pi),    "Pi_err": float(Pi_err),
            "kb": float(kb),    "kb_err": float(kb_err),
            "Vss": float(Vss),  "Vss_err": float(Vss_err),
            "t0": float(t0),
            "Abs_Offset_Subtracted": float(offset),
            "R_squared": float(r_squared),
            "Covariance_Matrix": pcov.tolist()
        }
        save_regression_parameters(param_dict, file_name, config['output_params_dir'])

        summary = {
            "File": file_name,
            "Pi": Pi,       "Pi_err": Pi_err,
            "kb": kb,       "kb_err": kb_err,
            "Vss": Vss,     "Vss_err": Vss_err,
            "t0": t0,
            "Abs_Offset_Subtracted": offset,
            "R_squared": r_squared
        }
        logging.info(f"Done processing {file_name}.")
        return summary

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    # Load the config and set up logging
    config = load_config(CONFIG_FILE)
    setup_logging(config['log_dir'])

    # Gather files
    pattern = os.path.join(config['data_dir'], config['file_pattern'])
    data_files = glob.glob(pattern)
    if not data_files:
        logging.error(f"No files found matching {pattern}")
        return

    logging.info(f"Found {len(data_files)} files to process.")

    # Ensure output dirs exist
    os.makedirs(config['output_plot_dir'], exist_ok=True)
    os.makedirs(config['output_params_dir'], exist_ok=True)
    os.makedirs(config['output_summary_dir'], exist_ok=True)

    # Parallel processing
    num_workers = min(config.get('num_workers', 1), multiprocessing.cpu_count())
    logging.info(f"Using {num_workers} workers for parallel processing.")

    with Pool(num_workers) as pool:
        process_partial = partial(process_file, config=config)
        results = list(tqdm(pool.imap(process_partial, data_files), total=len(data_files)))

    summary = [res for res in results if res is not None]
    if summary:
        save_summary(summary, config['output_summary_dir'])
    else:
        logging.warning("No valid results to summarize.")

    logging.info("All files processed.")

if __name__ == "__main__":
    main()
