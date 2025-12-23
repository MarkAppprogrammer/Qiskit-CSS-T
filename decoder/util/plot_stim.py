import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import argparse

length_dist_dict = {7:3, 17:5, 23:7, 47:11, 79:15, 103:19, 167:23}

def calc_ci_for_df(k, n, alpha):
    """Calculate Clopper-Pearson confidence intervals for error bars."""
    lower = scipy.stats.beta.ppf(alpha / 2, k, n - k + 1)
    lower = np.nan_to_num(lower, nan=0.0)
    upper = scipy.stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    upper = np.nan_to_num(upper, nan=1.0)
    p_hat = k / n
    return p_hat - lower, upper - p_hat

def plot_from_csv(csv_filepath, alpha_val=0.05):
    if not os.path.exists(csv_filepath):
        print(f"Error: File '{csv_filepath}' not found.")
        return

    df = pd.read_csv(csv_filepath)

    # Extract noise model name for the title
    if 'noise_model' in df.columns:
        # Assuming the file contains one type of noise model, take the first one
        noise_model_name = df['noise_model'].iloc[0]
    else:
        noise_model_name = "Unknown"

    ns = sorted(df['n'].unique())
    ps = sorted(df['p'].unique())
    confidence_percent = int((1 - alpha_val) * 100)

    # Styling
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16})
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])
    
    ax_total = fig.add_subplot(gs[0, :])
    ax_obj = fig.add_subplot(gs[1, 0])
    ax_time = fig.add_subplot(gs[1, 1])
    
    ler_axes = [ax_total]
    plot_configs = [
        {'key': 'total_logical_error_rate', 'title': 'Total Logical Error'},
    ]

    for i, config in enumerate(plot_configs):
        ax = ler_axes[i]
        key = config['key']
        
        for n in ns:
            subset = df[df['n'] == n].sort_values('p')
            if subset.empty or key not in subset.columns: continue
            
            d = length_dist_dict.get(n, '?')
            
            # Calculate error bars using the 'shots' column from the CSV
            if 'errors' in subset.columns and 'shots' in subset.columns:
                err_low, err_high = calc_ci_for_df(subset[key]*subset['shots'], subset['shots'], alpha_val)
                y_err = [err_low, err_high]
            else:
                y_err = None

            ax.errorbar(subset['p'], subset[key], yerr=y_err, marker='o', label=f'n={n}, d={d}', capsize=3)

        ax.plot(ps, ps, linestyle='--', color='black', alpha=0.3, label="Break-even")
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(config['title'])
        ax.grid(True, which="both", ls="--", alpha=0.4)
        if i == 0: ax.set_ylabel(f'LER ({confidence_percent}% CI)')

    # --- Mean Objective ---
    if 'mean_objective_value' in df.columns:
        for n in ns:
            subset = df[df['n'] == n].sort_values('p')
            ax_obj.plot(subset['p'], subset['mean_objective_value'], marker='s', linestyle=':', label=f'n={n}')
        ax_obj.set_xscale('log')
        ax_obj.set_title('Mean Objective Value')
        ax_obj.set_ylabel('Weight')
        ax_obj.grid(True, which="both", ls="--", alpha=0.4)

    # --- CPU Time ---
    if 'average_cpu_time_seconds' in df.columns:
        for n in ns:
            subset = df[df['n'] == n].sort_values('p')
            ax_time.plot(subset['p'], subset['average_cpu_time_seconds'], marker='^', linestyle='-.')
        ax_time.set_xscale('log'); ax_time.set_yscale('log')
        ax_time.set_title('Avg Computation Time')
        ax_time.set_ylabel('Seconds / Shot')
        ax_time.grid(True, which="both", ls="--", alpha=0.4)

    # Legend and Title
    fig.legend(*ax_total.get_legend_handles_labels(), loc='center right', bbox_to_anchor=(1.1, 0.5))
    
    # Dynamic title using extracted noise model name
    fig.suptitle(f'Hyperion Decoder Performance for {noise_model_name} Noise Model', fontsize=22)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save the plot to a file
    output_filename = os.path.splitext(csv_filepath)[0] + ".png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simulation results from CSV.")
    parser.add_argument("csv_file", type=str, help="Path to the results CSV file")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha value for confidence intervals (default: 0.05)")
    
    args = parser.parse_args()
    
    plot_from_csv(args.csv_file, args.alpha)