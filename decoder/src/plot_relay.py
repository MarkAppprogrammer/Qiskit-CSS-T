import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

def plot_results(csv_file, output_file=None):
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        sys.exit(1)
    
    if 'n' not in df.columns:
        print(f"Error: CSV {csv_file} must contain 'n' column.")
        sys.exit(1)
        
    ns = sorted(df['n'].unique())
    
    plt.figure(figsize=(12, 8))
    
    for n in ns:
        subset = df[df['n'] == n].sort_values('p')
        plt.plot(subset['p'], subset['total_logical_error_rate'], marker='o', linestyle='-', label=f'n={n}')
        
    min_p = df['p'].min()
    max_p = df['p'].max()
    plt.plot([min_p, max_p], [min_p, max_p], linestyle='--', color='black', label='Break-even (y=x)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Physical Error Rate (p)')
    plt.ylabel('Logical Error Rate')
    plt.title(f'Relay-BP Performance ({os.path.basename(csv_file)})')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which="both", ls="--")
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        base = os.path.splitext(csv_file)[0]
        out = f"{base}.png"
        plt.savefig(out)
        print(f"Plot saved to {out}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Relay-BP Simulation Results")
    parser.add_argument("csv_file", type=str, help="Path to the CSV results file")
    parser.add_argument("--out", type=str, default=None, help="Output PNG file path")
    
    args = parser.parse_args()
    
    plot_results(args.csv_file, args.out)
