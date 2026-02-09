import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import argparse
import sys

# --- CONFIGURATION & METADATA ---
OP1_DICT = {4:2, 6:2, 8:2, 10:2, 12:4, 14:4, 16:4, 18:4, 20:4, 22:6, 24:6, 26:6, 28:6, 30:6, 32:8, 34:6, 36:8, 38:8, 40:8, 42:8, 44:8, 46:8, 48:8, 50:8, 52:10, 54:8, 56:10, 58:10, 60:12, 62:10, 64:10}
OP2_DICT = {7:3, 17:5, 23:7, 47:11, 79:15, 103:19, 167:23}

CODE_CONFIGS = {
    "self_dual": { "source": "local", "dist_dict": OP1_DICT },
    "dual_containing": { "source": "local", "dist_dict": OP2_DICT },
    "cubic": { "source": "qcodeplot3d" },
    "tetrahedral": { "source": "qcodeplot3d" },
    "triangular": { "source": "qcodeplot3d" },
    "square": { "source": "qcodeplot3d" }
}

ALPHA = 0.05

def get_code_metadata(code_type, n, d_from_csv):

    config = CODE_CONFIGS.get(code_type, {})
    source = config.get("source", "unknown")
    
    if source == "local":
        d_theo = config.get("dist_dict", {}).get(n, d_from_csv)
        return f"n={n}, d={d_theo}", n
    else:
        return f"d={d_from_csv} (n={n})", d_from_csv

def load_and_aggregate_data(directory):

    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return pd.DataFrame()

    all_files = glob.glob(os.path.join(directory, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {directory}")
        return pd.DataFrame()
    
    print(f"Found {len(all_files)} files. Loading and Aggregating...")
    df_list = []
    
    for f in all_files:
        try:
            temp_df = pd.read_csv(f)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not df_list: return pd.DataFrame()
    
    raw_df = pd.concat(df_list, ignore_index=True)
    
    # Fill defaults
    defaults = {'code_type': 'unknown', 'noise_model': 'unknown', 'n': 0, 'd': 0, 'p': 0.0, 'shots': 0, 'errors': 0}
    for col, val in defaults.items():
        if col not in raw_df.columns:
            raw_df[col] = val

    # Aggregate
    group_cols = ['code_type', 'noise_model', 'n', 'd', 'p']
    agg_rules = {'shots': 'sum', 'errors': 'sum'}
    
    if 'mean_objective_value' in raw_df.columns:
        agg_rules['mean_objective_value'] = 'mean'
    if 'average_cpu_time_seconds' in raw_df.columns:
        agg_rules['average_cpu_time_seconds'] = 'mean'
        
    agg_df = raw_df.groupby(group_cols, as_index=False).agg(agg_rules)
    agg_df['total_logical_error_rate'] = agg_df['errors'] / agg_df['shots']
    
    print(f"Aggregated {len(raw_df)} raw rows into {len(agg_df)} unique data points.")
    return agg_df

def plot_data(df, output_dir="."):
    if df.empty:
        print("No data to plot.")
        return

    groups = df.groupby(['code_type', 'noise_model'])

    for (code_type, noise_model), group_df in groups:
        print(f"Plotting: {code_type} - {noise_model}")
        
        curves = group_df.groupby(['n', 'd'])
        config = CODE_CONFIGS.get(code_type, {})
        is_geometric = config.get("source") == "qcodeplot3d"
        
        sorted_curves = []
        for (n, d), curve_data in curves:
            label, sort_key = get_code_metadata(code_type, n, d)
            sorted_curves.append({
                'n': n, 'd': d, 'data': curve_data.sort_values('p'),
                'label': label, 'sort_key': sort_key
            })
        
        # Sort the list of curves
        sorted_curves.sort(key=lambda x: x['sort_key'])
        
        ps = sorted(group_df['p'].unique())
        confidence_percent = int((1 - ALPHA) * 100)
        
        # PLOT SETUP
        plt.rcParams.update({
            'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,     
            'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12
        })

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 6, height_ratios=[3, 2])

        ax_total = fig.add_subplot(gs[0, :])
        ax_obj = fig.add_subplot(gs[1, 0:3])
        ax_time = fig.add_subplot(gs[1, 3:6])

        # --- 1. Total Logical Error Rate ---
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_curves)))
        
        for i, curve in enumerate(sorted_curves):
            subset = curve['data']
            p_L = subset['total_logical_error_rate'].values
            shots_val = subset['shots'].values
            errors = subset['errors'].values

            # Confidence Intervals (Clopper-Pearson)
            lower_b = scipy.stats.beta.ppf(ALPHA / 2, errors, shots_val - errors + 1)
            lower_b[errors == 0] = 0.0
            upper_b = scipy.stats.beta.ppf(1 - ALPHA / 2, errors + 1, shots_val - errors)
            upper_b[errors == shots_val] = 1.0
            
            y_err = [p_L - lower_b, upper_b - p_L]

            ax_total.errorbar(subset['p'], p_L, yerr=y_err, marker='o', linestyle='-', 
                              label=curve['label'], color=colors[i], capsize=3, alpha=0.8)
            
            # --- 2. Obj Value ---
            if 'mean_objective_value' in subset.columns:
                ax_obj.plot(subset['p'], subset['mean_objective_value'], 
                            marker='s', linestyle=':', color=colors[i], label=curve['label'])

            # --- 3. CPU Time ---
            if 'average_cpu_time_seconds' in subset.columns:
                ax_time.plot(subset['p'], subset['average_cpu_time_seconds'], 
                             marker='^', linestyle='-.', color=colors[i])

        # --- Formatting Main Plot ---
        ax_total.plot(ps, ps, linestyle='--', color='black', alpha=0.4, label="Break-even")
        
        # Resolution Limit
        max_shots = group_df['shots'].max()
        res_limit = 1.0 / max_shots if max_shots > 0 else 1e-9
        ax_total.axhline(y=res_limit, color='gray', linestyle=':', alpha=0.5)
        x_min, x_max = min(ps), max(ps)
        ax_total.fill_between([x_min * 0.5, x_max * 1.5], 1e-12, res_limit, 
                              color='lightgray', alpha=0.3, label=f'Unreliable (< {res_limit:.1e})')

        ax_total.set_xscale('log')
        ax_total.set_yscale('log')
        ax_total.set_xlabel('Physical Error Rate (p)')
        ax_total.set_ylabel(f'Logical Error Rate ({confidence_percent}% CI)')
        ax_total.set_title(f'Logical Error Rate: {code_type} ({noise_model})')
        ax_total.grid(True, which="both", ls="--", alpha=0.3)
        ax_total.set_xlim(min(ps)*0.8, max(ps)*1.2)
        ax_total.legend(loc='center right', bbox_to_anchor=(1.0, 0.5))

        # --- Formatting Subplots ---
        ax_obj.set_xscale('log')
        ax_obj.set_xlabel('p')
        ax_obj.set_ylabel('Weight')
        ax_obj.set_title('Mean MWPF Objective')
        ax_obj.grid(True, which="both", ls="--", alpha=0.3)

        ax_time.set_xscale('log')
        ax_time.set_yscale('log')
        ax_time.set_xlabel('p')
        ax_time.set_ylabel('Seconds')
        ax_time.set_title('Avg Decoding Time')
        ax_time.grid(True, which="both", ls="--", alpha=0.3)

        # Save
        fig.suptitle(f'{code_type.replace("_", " ").title()} Analysis', fontsize=22)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        out_file = os.path.join(output_dir, f"plot_{code_type}_{noise_model}.png")
        plt.savefig(out_file, dpi=300)
        print(f"Saved plot to: {out_file}")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot QEC Simulation Results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .csv results")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save plots (default: input_dir)")
    args = parser.parse_args()

    input_path = os.path.expandvars(args.input_dir)
    output_path = os.path.expandvars(args.output_dir) if args.output_dir else input_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = load_and_aggregate_data(input_path)
    plot_data(df, output_path)

if __name__ == "__main__":
    main()