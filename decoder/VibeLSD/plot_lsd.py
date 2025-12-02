import pandas as pd
import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Dictionary to map code size n to distance d
length_dist_dict = {
    4: 2, 6: 2, 8: 2, 10: 2, 12: 4, 14: 4, 16: 4, 18: 4, 20: 4, 22: 6, 24: 6,
    26: 6, 28: 6, 30: 6, 32: 8, 34: 6, 36: 8, 38: 8, 40: 8, 42: 8, 44: 8, 46: 8,
    48: 8, 50: 8, 52: 10, 54: 8, 56: 10, 58: 10, 60: 12, 62: 10, 64: 10
}

def plot_from_csv(csv_filepath, bias_factor=None, alpha_val=None, shots_val=None):
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        return

    # Attempt to find 'shots' in CSV if not provided via args
    if shots_val is None:
        if 'shots' in df.columns:
            shots_val = df['shots'].iloc[0]
        else:
            match_shots = re.search(r"shots(\d+)", csv_filepath)
            if match_shots:
                shots_val = int(match_shots.group(1))

    ns = sorted(df['n'].unique())
    ps = sorted(df['p'].unique())

    # Determine confidence level string
    if alpha_val is not None:
        confidence_percent = int((1 - alpha_val) * 100)
    else:
        confidence_percent = 95 # Default

    # Global Font Settings
    plt.rcParams.update({
        'font.size': 14,          
        'axes.titlesize': 18,     
        'axes.labelsize': 16,     
        'xtick.labelsize': 14,    
        'ytick.labelsize': 14,    
        'legend.fontsize': 14,    
        'lines.linewidth': 2,     
        'lines.markersize': 8     
    })

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 2])

    ax_total = fig.add_subplot(gs[0, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_z = fig.add_subplot(gs[0, 2])
    ax_time = fig.add_subplot(gs[1, :])

    ler_axes = [ax_total, ax_x, ax_z]
    
    # Updated configs for asymmetric CI columns
    plot_configs = [
        {'key': 'total_logical_error_rate', 'low': 'total_err_low', 'high': 'total_err_high', 'title': 'Total Logical Error'},
        {'key': 'x_logical_error_rate', 'low': 'x_err_low', 'high': 'x_err_high', 'title': 'Logical X Error'},
        {'key': 'z_logical_error_rate', 'low': 'z_err_low', 'high': 'z_err_high', 'title': 'Logical Z Error'}
    ]

    # Plot Logical Error Rates
    for i, config in enumerate(plot_configs):
        ax = ler_axes[i]
        key = config['key']
        low_key = config['low']
        high_key = config['high']
        
        for n in ns:
            subset = df[df['n'] == n].sort_values('p')
            if not subset.empty:
                d = length_dist_dict.get(n, '?')
                p_L = subset[key]
                
                # Check for asymmetric error columns
                if low_key in subset.columns and high_key in subset.columns:
                    y_err = [subset[low_key].values, subset[high_key].values]
                else:
                    # Fallback for old CSVs
                    old_std_key = key.replace('logical_error_rate', 'std_error')
                    if old_std_key in subset.columns:
                        y_err = subset[old_std_key]
                    else:
                        y_err = np.zeros_like(p_L)
                
                ax.errorbar(subset['p'], p_L, yerr=y_err, marker='o', linestyle='-', label=f'n={n}, d={d}', capsize=3, alpha=0.8)

        # Break-even line
        ax.plot(ps, ps, linestyle='--', color='black', alpha=0.5)
        ax.text(ps[-1], ps[-1], "  Break-even", va='center', fontsize=14, color='black', alpha=0.7)

        # Draw Resolution Limit
        if shots_val:
            res_limit = 1.0 / shots_val
            ax.axhline(y=res_limit, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.text(ps[0], res_limit * 1.1, f" Resolution Limit (1/N)", color='gray', fontsize=12, va='bottom')
            ax.fill_between(ps, 0, res_limit, color='gray', alpha=0.1)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Physical Error Rate (p)')
        ax.set_title(config['title'])
        ax.grid(True, which="both", ls="--", alpha=0.4)
        if i == 0: ax.set_ylabel(f'Logical Error Rate ({confidence_percent}% CI)')

    # Share Y-axis
    for ax in ler_axes[1:]:
        ax.sharey(ler_axes[0])
        plt.setp(ax.get_yticklabels(), visible=False)

    # Plot Average Computation Time
    if 'average_cpu_time_seconds' in df.columns:
        for n in ns:
            subset = df[df['n'] == n].sort_values('p')
            if not subset.empty:
                d = length_dist_dict.get(n, '?')
                ax_time.plot(subset['p'], subset['average_cpu_time_seconds'], marker='^', linestyle='-.', label=f'n={n}, d={d}')
        
        ax_time.set_xscale('log')
        ax_time.set_yscale('log')
        ax_time.set_xlabel('Physical Error Rate (p)')
        ax_time.set_ylabel('Avg Time per Shot (s)')
        ax_time.set_title('Average Computation Time per Round')
        ax_time.grid(True, which="both", ls="--", alpha=0.4)
    else:
        ax_time.text(0.5, 0.5, "No 'average_cpu_time_seconds' column", ha='center', va='center')

    # Legend and Titles
    handles, labels = ler_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), title="Code Parameters")

    suptitle = 'BP+LSD Logical Error Rates'
    params = []
    if bias_factor is not None: params.append(f"Bias: {bias_factor}")
    if alpha_val is not None: params.append(f"Alpha: {alpha_val}")
    if params: suptitle += f" ({', '.join(params)})"

    fig.suptitle(suptitle, fontsize=24)
    
    base_name = os.path.splitext(os.path.basename(csv_filepath))[0]
    plot_filename = f"{base_name}_plot.png"
    
    fig.tight_layout(rect=[0, 0, 0.9, 0.96]) 
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Plot saved as {plot_filename}")
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot logical error rates from a BP+LSD simulation CSV file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--bias", type=float, nargs='?', default=None, help="Bias factor.")
    parser.add_argument("--alpha", type=float, nargs='?', default=None, help="Alpha value.")
    parser.add_argument("--shots", type=int, nargs='?', default=None, help="Total shots (N).")
    
    args = parser.parse_args()

    # Attempt to auto-detect parameters from filename if missing
    if args.bias is None:
        match = re.search(r"bias([\d\.]+)", args.input_file)
        if match: args.bias = float(match.group(1))

    if args.alpha is None:
        match = re.search(r"alpha([\d\.]+)", args.input_file)
        if match: args.alpha = float(match.group(1))

    if args.shots is None:
        match = re.search(r"shots(\d+)", args.input_file)
        if match: args.shots = int(match.group(1))

    plot_from_csv(args.input_file, args.bias, args.alpha, args.shots)

if __name__ == "__main__":
    main()