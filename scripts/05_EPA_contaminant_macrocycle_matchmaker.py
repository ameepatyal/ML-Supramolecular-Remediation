#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import sys

def generate_matchmaker(input_file, output_file, top_n=3):
    print(f"Loading raw prediction matrix from: {input_file}")
    try:
        df_matrix = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find the file {input_file}. Please check the path.")
        sys.exit(1)
        
    if 'Guest' not in df_matrix.columns:
        print("Error: The input matrix must contain a 'Guest' column.")
        sys.exit(1)
        
    df_matrix.set_index('Guest', inplace=True)

    print("Calculating row-by-row relative metrics (Z-Scores)...")
    # 1. Calculate statistical baselines for EACH contaminant across all hosts
    row_means = df_matrix.mean(axis=1)
    row_stds = df_matrix.std(axis=1)

    # 2. Compute Z-scores (more negative = stronger relative affinity)
    df_zscore = df_matrix.sub(row_means, axis=0).div(row_stds, axis=0)

    # 3. Melt to long-form sheets
    df_z_long = df_zscore.reset_index().melt(id_vars=['Guest'], var_name='Host', value_name='Selectivity_Z')
    df_raw_long = df_matrix.reset_index().melt(id_vars=['Guest'], var_name='Host', value_name='Raw_Energy')

    # 4. Merge together
    df_combined = pd.merge(df_z_long, df_raw_long, on=['Guest', 'Host'])

    # 5. Extract Top N closest macrocycle options for each individual guest
    print(f"Compiling Top {top_n} macrocycle options per contaminant...")
    df_top = df_combined.sort_values(by=['Guest', 'Selectivity_Z'], ascending=[True, True])
    df_top = df_top.groupby('Guest').head(top_n).copy()

    # Add a ranking tracking index (1st, 2nd, 3rd choice...)
    df_top['Rank'] = df_top.groupby('Guest').cumcount() + 1

    # Pivot into a clean, publication-ready summary document
    summary_table = df_top.pivot(index='Guest', columns='Rank', values=['Host', 'Raw_Energy', 'Selectivity_Z'])

    # Flatten the multi-index columns for clean CSV headers
    summary_table.columns = [f'{col[0]}_Choice_{col[1]}' for col in summary_table.columns]

    # Reorder columns dynamically so they read logically: Host 1, Energy 1, Z 1, Host 2...
    ordered_cols = []
    for i in range(1, top_n + 1):
        ordered_cols.extend([f'Host_Choice_{i}', f'Raw_Energy_Choice_{i}', f'Selectivity_Z_Choice_{i}'])
    
    # Ensure we only try to order columns that actually exist (in case of very small datasets)
    ordered_cols = [col for col in ordered_cols if col in summary_table.columns]
    summary_table = summary_table[ordered_cols]

    # Round numeric columns to 2 decimals
    summary_table = summary_table.round(2)

    # Save results
    summary_table.to_csv(output_file)
    print(f"Done! Matchmaker summary saved to: {output_file}")

if __name__ == "__main__":
    # Setup the argument parser for terminal execution
    parser = argparse.ArgumentParser(description="Generate Top Macrocycle Matches based on Z-Scores")
    
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to the input Screening Matrix CSV")
    
    parser.add_argument("--output", type=str, default="EPA_Contaminant_Sorbent_Matchmaker.csv", 
                        help="Path to save the final Matchmaker CSV")
    
    parser.add_argument("--top_n", type=int, default=3, 
                        help="Number of top macrocycle choices to output per contaminant (default: 3)")
    
    args = parser.parse_args()
    
    # Execute the function with the provided arguments
    generate_matchmaker(args.input, args.output, args.top_n)