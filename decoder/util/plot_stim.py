import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# --- CONFIGURATION ---
INPUT_DIR = os.path.expandvars("$HOME/Desktop/Qiskit-CSS-T/decoder/data/results")
# INPUT_DIR = os.path.expandvars("$HOME/work/Qiskit-CSS-T/decoder/data/1/results")
ALPHA = 0.05

# Distance Dictionaries
OP1_DICT = {4:2, 6:2, 8:2, 10:2, 12:4, 14:4, 16:4, 18:4, 20:4, 22:6, 24:6, 26:6, 28:6, 30:6, 32:8, 34:6, 36:8, 38:8, 40:8, 42:8, 44:8, 46:8, 48:8, 50:8, 52:10, 54:8, 56:10, 58:10, 60:12, 62:10, 64:10}
OP2_DICT = {7:3, 17:5, 23:7, 47:11, 79:15, 103:19, 167:23}

def get_dist_dict(code_type):
    """Returns the correct distance dictionary based on code_type."""
    if "dual_containing" in str(code_type):
        return OP2_DICT
    return OP1_DICT

def load_and_aggregate_data(directory):
    """
    1. Loads all CSVs.
    2. Aggregates shots/errors in memory (SUM).
    3. Returns a clean DataFrame for plotting.
    """
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {directory}")
        return pd.DataFrame()
    
    print(f"Found {len(all_files)} files. Loading and Aggregating...")
    df_list = []
    
    # 1. Load Raw Data
    for f in all_files:
        try:
            temp_df = pd.read_csv(f)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not df_list: return pd.DataFrame()
    
    raw_df = pd.concat(df_list, ignore_index=True)
    
    # Fill missing columns to ensure code doesn't break
    defaults = {'code_type': 'self_dual', 'noise_model': 'unknown', 'n': 0, 'd': 0, 'p': 0.0, 'shots': 0, 'errors': 0}
    for col, val in defaults.items():
        if col not in raw_df.columns:
            raw_df[col] = val

    # 2. Aggregate In-Memory
    group_cols = ['code_type', 'noise_model', 'n', 'd', 'p']
    
    agg_rules = {
        'shots': 'sum',
        'errors': 'sum',
    }
    # Optional columns (take mean)
    if 'mean_objective_value' in raw_df.columns:
        agg_rules['mean_objective_value'] = 'mean'
    if 'average_cpu_time_seconds' in raw_df.columns:
        agg_rules['average_cpu_time_seconds'] = 'mean'
        
    agg_df = raw_df.groupby(group_cols, as_index=False).agg(agg_rules)
    
    # 3. Recalculate Logical Error Rate (Total Errors / Total Shots)
    agg_df['total_logical_error_rate'] = agg_df['errors'] / agg_df['shots']
    
    print(f"Aggregated {len(raw_df)} raw rows into {len(agg_df)} unique data points.")
    return agg_df

# --- MAIN PLOTTING LOGIC ---

df_all = load_and_aggregate_data(INPUT_DIR)

if not df_all.empty:
    groups = df_all.groupby(['code_type', 'noise_model'])

    for (code_type, noise_model), df in groups:
        print(f"Plotting: {code_type} - {noise_model}")
        
        ns = sorted(df['n'].unique())
        ps = sorted(df['p'].unique())
        dist_dict = get_dist_dict(code_type)
        confidence_percent = int((1 - ALPHA) * 100)
        
        # PLOT STYLING
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
        for n in ns:
            subset = df[df['n'] == n].sort_values('p')
            if subset.empty: continue

            d = dist_dict.get(n, subset['d'].iloc[0])
            p_L = subset['total_logical_error_rate'].values
            shots_val = subset['shots'].values
            errors = subset['errors'].values

            # Confidence Intervals
            lower_b = scipy.stats.beta.ppf(ALPHA / 2, errors, shots_val - errors + 1)
            lower_b[errors == 0] = 0.0
            
            upper_b = scipy.stats.beta.ppf(1 - ALPHA / 2, errors + 1, shots_val - errors)
            upper_b[errors == shots_val] = 1.0
            
            y_err = [p_L - lower_b, upper_b - p_L]

            ax_total.errorbar(subset['p'], p_L, yerr=y_err, marker='o', linestyle='-', 
                              label=f'n={n}, d={d}', capsize=3, alpha=0.8)

        # Break-even line
        ax_total.plot(ps, ps, linestyle='--', color='black', alpha=0.4, label="Break-even")
        
        # --- RESOLUTION LIMIT (WITH SHADING) ---
        max_shots = df['shots'].max()
        res_limit = 1.0 / max_shots
        
        # Draw the line
        ax_total.axhline(y=res_limit, color='gray', linestyle=':', alpha=0.5)
        
        # Shade the area below (using 1e-10 as effective zero for log plot)
        # We extend the x-range slightly to cover the full width
        x_min, x_max = min(ps), max(ps)
        ax_total.fill_between([x_min * 0.5, x_max * 1.5], 1e-10, res_limit, 
                              color='lightgray', alpha=0.3, label=f'Unreliable (< {res_limit:.1e})')

        ax_total.set_xscale('log')
        ax_total.set_yscale('log')
        ax_total.set_xlabel('Physical Error Rate (p)')
        ax_total.set_ylabel(f'Logical Error Rate ({confidence_percent}% CI)')
        ax_total.set_title(f'Total Logical Error: {code_type} ({noise_model})')
        ax_total.grid(True, which="both", ls="--", alpha=0.3)
        
        # Reset X-limits to fit data tightly (removes extra whitespace from shading)
        ax_total.set_xlim(min(ps)*0.8, max(ps)*1.2)
        
        handles, labels = ax_total.get_legend_handles_labels()
        ax_total.legend(handles, labels, loc='center right', bbox_to_anchor=(1.0, 0.5))

        # --- 2. Mean Objective Value ---
        if 'mean_objective_value' in df.columns:
            for n in ns:
                subset = df[df['n'] == n].sort_values('p')
                ax_obj.plot(subset['p'], subset['mean_objective_value'], marker='s', linestyle=':', label=f'n={n}')
            ax_obj.set_xscale('log')
            ax_obj.set_xlabel('Physical Error Rate (p)')
            ax_obj.set_ylabel('Weight')
            ax_obj.set_title('Mean MWPF Objective Value')
            ax_obj.grid(True, which="both", ls="--", alpha=0.3)

        # --- 3. CPU Time ---
        if 'average_cpu_time_seconds' in df.columns:
            for n in ns:
                subset = df[df['n'] == n].sort_values('p')
                ax_time.plot(subset['p'], subset['average_cpu_time_seconds'], marker='^', linestyle='-.')
            ax_time.set_xscale('log')
            ax_time.set_yscale('log')
            ax_time.set_xlabel('Physical Error Rate (p)')
            ax_time.set_ylabel('Seconds / Shot')
            ax_time.set_title('Avg Decoding Time')
            ax_time.grid(True, which="both", ls="--", alpha=0.3)

        # Final Formatting
        fig.suptitle(f'{code_type.replace("_", " ").title()} - {noise_model.capitalize()}', fontsize=22)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_filename = f"plot_{code_type}_{noise_model}.png"
        plt.savefig(output_filename, dpi=300)
        print(f"Saved plot to {output_filename}")
        plt.show()

else:
    print("No data found to plot.")