import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import argparse
import json
import glob
import time

# --- Configuration ---
FALLBACK_D_MAP = {
    # Self Dual
    4:2, 6:2, 8:2, 10:2, 12:4, 14:4, 16:4, 18:4, 20:4, 22:6, 24:6, 26:6, 28:6, 30:6, 32:8,
    # Dual Containing
    7:3, 17:5, 23:7, 47:11, 79:15, 103:19, 167:23
}

def calc_ci(k, n, alpha=0.05):
    """Clopper-Pearson confidence intervals (Vectorized for Series)."""
    k = np.array(k)
    n = np.array(n)
    
    # Calculate lower bound
    lower = scipy.stats.beta.ppf(alpha / 2, k, n - k + 1)
    lower = np.nan_to_num(lower, nan=0.0)
    
    # Calculate upper bound
    upper = scipy.stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    upper = np.nan_to_num(upper, nan=1.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        p_hat = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n!=0)
        
    return p_hat - lower, upper - p_hat

def load_and_merge_files(files_list):
    """Reads a list of CSV files and returns one combined DataFrame."""
    if not files_list:
        print("No files provided.")
        return pd.DataFrame()

    print(f"Found {len(files_list)} files. Loading...")
    t0 = time.time()
    
    dfs = []
    for f in files_list:
        if not os.path.exists(f):
            print(f"Skipping missing file: {f}")
            continue
        try:
            # Low_memory=False is faster for large chunks but uses more RAM
            df = pd.read_csv(f, skipinitialspace=True, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Cleaning: Remove repeated headers
    if 'shots' in full_df.columns:
        full_df = full_df[full_df['shots'].astype(str) != 'shots']

    print(f"Merged into {len(full_df)} rows in {time.time()-t0:.2f}s.")
    return full_df

def process_data(df):
    """Parses metadata and aggregates stats."""
    if df.empty: return df

    # 1. Parse JSON Metadata (OPTIMIZED)
    if 'json_metadata' in df.columns:
        print("Parsing JSON metadata (Optimized)...")
        t0 = time.time()
        
        # Convert column to list of strings first (faster iteration)
        json_strings = df['json_metadata'].fillna('{}').astype(str).tolist()
        
        # Fast parsing using list comprehension
        try:
            parsed_data = [json.loads(s) for s in json_strings]
            meta_df = pd.DataFrame(parsed_data)
            
            # Drop original and concat
            df = df.drop(columns=['json_metadata'], errors='ignore')
            df = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
            print(f"JSON parsed in {time.time()-t0:.2f}s.")
            
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return pd.DataFrame()

    # 2. Ensure Numeric Types
    print("Converting numeric types...")
    cols_to_numeric = ['shots', 'errors', 'seconds', 'n', 'p', 'd']
    for c in cols_to_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 3. Drop invalid rows
    df = df.dropna(subset=['n', 'p', 'shots'])

    # 4. Aggregate
    group_cols = [c for c in ['n', 'p', 'd', 'noise_model', 'code_type'] if c in df.columns]
    
    if not group_cols:
        print("Error: No grouping columns (n, p) found.")
        return pd.DataFrame()

    print(f"Aggregating {len(df)} rows by {group_cols}...")
    agg_df = df.groupby(group_cols, as_index=False)[['shots', 'errors', 'seconds']].sum()

    # 5. Calculate Rates
    agg_df['total_logical_error_rate'] = agg_df['errors'] / agg_df['shots']
    agg_df['average_cpu_time_seconds'] = agg_df['seconds'] / agg_df['shots']

    print(f"Final condensed dataset has {len(agg_df)} points.")
    return agg_df

def plot_results(df, output_base):
    if df.empty: return

    noise_models = df['noise_model'].unique() if 'noise_model' in df.columns else ['Unknown']

    for model in noise_models:
        subset_model = df if len(noise_models) == 1 else df[df['noise_model'] == model]
        
        ns = sorted(subset_model['n'].unique())
        ps = sorted(subset_model['p'].unique())

        fig, (ax_ler, ax_time) = plt.subplots(1, 2, figsize=(16, 6))

        # --- LER Plot ---
        for n in ns:
            data = subset_model[subset_model['n'] == n].sort_values('p')
            
            if 'd' in data.columns:
                d = int(data['d'].iloc[0])
            else:
                d = FALLBACK_D_MAP.get(int(n), '?')

            low, high = calc_ci(data['errors'], data['shots'])
            yerr = [low, high]

            ax_ler.errorbar(data['p'], data['total_logical_error_rate'], yerr=yerr,
                            marker='o', capsize=3, label=f'n={n}, d={d}')

        ax_ler.plot(ps, ps, 'k--', alpha=0.3, label="Break-even")
        ax_ler.set_xscale('log'); ax_ler.set_yscale('log')
        ax_ler.set_xlabel('Physical Error Rate (p)')
        ax_ler.set_ylabel('Logical Error Rate (LER)')
        ax_ler.set_title(f'Logical Error Rate ({model})')
        ax_ler.grid(True, which="both", ls="--", alpha=0.4)
        ax_ler.legend()

        # --- CPU Time Plot ---
        for n in ns:
            data = subset_model[subset_model['n'] == n].sort_values('p')
            ax_time.plot(data['p'], data['average_cpu_time_seconds'], marker='^', linestyle='-.', label=f'n={n}')
        
        ax_time.set_xscale('log'); ax_time.set_yscale('log')
        ax_time.set_xlabel('Physical Error Rate (p)')
        ax_time.set_ylabel('Seconds per Shot')
        ax_time.set_title('Decoding Time')
        ax_time.grid(True, which="both", ls="--", alpha=0.4)
        ax_time.legend()

        plt.tight_layout()
        fname = f"{output_base}_{model}.png"
        plt.savefig(fname)
        print(f"Plot saved to {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot merged Sinter shards.")
    parser.add_argument("files", nargs='+', help="List of CSV files")
    parser.add_argument("--out", default="plot", help="Output filename base")
    
    args = parser.parse_args()
    
    final_file_list = []
    for f in args.files:
        if any(char in f for char in ['*', '?', '[']):
            final_file_list.extend(glob.glob(f))
        else:
            final_file_list.append(f)

    raw_df = load_and_merge_files(final_file_list)
    clean_df = process_data(raw_df)
    plot_results(clean_df, args.out)