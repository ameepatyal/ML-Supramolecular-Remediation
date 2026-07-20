#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge host, guest, and binding energy data into a master ML dataset.
"""

import argparse
import pandas as pd

def merge_datasets(be_file, guest_file, host_file, output_file):
    print(f"Loading binding energies from: {be_file}")
    df_be = pd.read_csv(be_file)
    
    print(f"Loading guest features from: {guest_file}")
    df_guest_features = pd.read_csv(guest_file)
    
    print(f"Loading host features from: {host_file}")
    df_host_features = pd.read_csv(host_file)
    
    # Merge Guest features into the Binding Energy data (matching on 'Guest')
    print("Merging guest features...")
    df_merged = pd.merge(df_be, df_guest_features, on='Guest', how='left')
    
    # Merge Host features into the master dataframe (matching on 'Host')
    print("Merging host features...")
    df_master = pd.merge(df_merged, df_host_features, on='Host', how='left')
    
    # Drop the SMILES column
    if 'Guest_SMILES' in df_master.columns:
        df_master = df_master.drop(columns=['Guest_SMILES'])
    elif 'SMILES' in df_master.columns: # Catching 'SMILES' just in case
        df_master = df_master.drop(columns=['SMILES'])
        
    # Quality Check
    print("\n--- Quality Check: Missing values per column ---")
    missing_counts = df_master.isna().sum()
    print(missing_counts)
    
    if missing_counts.sum() > 0:
        print("\nWARNING: Missing values detected! ML models (like Random Forest) may fail if NaNs are present.")
    
    # Save the ML-ready dataset
    df_master.to_csv(output_file, index=False)
    print(f"\nMerge complete! Saved ML-ready dataset to: {output_file}")

if __name__ == "__main__":
    # accept terminal commands
    parser = argparse.ArgumentParser(description="Merge Host, Guest, and Binding Energy CSVs into one dataset.")
    
    # Define required arguments
    parser.add_argument("--be_file", type=str, required=True, help="Path to the Binding Energy CSV")
    parser.add_argument("--guest_file", type=str, required=True, help="Path to the Guest Features CSV")
    parser.add_argument("--host_file", type=str, required=True, help="Path to the Host Features CSV")
    
    # Define optional argument for the output file (defaults to FINAL_ML_DATASET.csv)
    parser.add_argument("--output", type=str, default="FINAL_ML_DATASET.csv", help="Path to save the merged output CSV")
    
    args = parser.parse_args()
    
    # Execute the merge function using the provided arguments
    merge_datasets(args.be_file, args.guest_file, args.host_file, args.output)