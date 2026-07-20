#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Mute warnings for clean terminal output
warnings.filterwarnings('ignore')
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# ---------------------------------------------------------
# 1. CORE FUNCTIONS
# ---------------------------------------------------------

def get_guest_features(smiles_string):
    """Generates 3D structure and calculates RDKit descriptors from a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: 
            return [np.nan] * 11
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        npr1 = rdMolDescriptors.CalcNPR1(mol)
        npr2 = rdMolDescriptors.CalcNPR2(mol)
        asa = rdMolDescriptors.CalcLabuteASA(mol)
        vol = AllChem.ComputeMolVolume(mol)
        rg = rdMolDescriptors.CalcRadiusOfGyration(mol)
        
        return [mw, logp, tpsa, rot_bonds, hbd, hba, npr1, npr2, asa, vol, rg]
    except Exception:
        return [np.nan] * 11 

def train_rf_model(train_file, feature_cols):
    """Loads training data, scales features, and trains the Random Forest model."""
    print(f"Training Final Model on data from: {train_file}")
    df_train = pd.read_csv(train_file).dropna()

    X_train = df_train[feature_cols].values
    y_train = df_train['Binding_Energy'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_split=4, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    return rf_model, scaler

def process_guest_library(guest_file):
    """Loads a list of SMILES and calculates their RDKit features."""
    print(f"Loading Contaminant Database from: {guest_file}")
    df_epa = pd.read_csv(guest_file)[['Preferred Name', 'SMILES']].dropna() 

    tqdm.pandas(desc="Calculating RDKit Features")
    features = df_epa['SMILES'].progress_apply(get_guest_features)
    
    feature_names = ["Guest_MW", "Guest_LogP", "Guest_TPSA", "Guest_RotBonds", 
                     "Guest_HBD", "Guest_HBA", "Guest_NPR1", "Guest_NPR2", 
                     "Guest_ASA", "Guest_Vol", "Guest_Rg"]
    
    features_df = pd.DataFrame(features.tolist(), columns=feature_names)
    df_guests = pd.concat([df_epa.reset_index(drop=True), features_df], axis=1).dropna()
    df_guests = df_guests.rename(columns={'Preferred Name': 'Guest'})
    
    return df_guests

def build_and_predict(df_hosts, df_guests, rf_model, scaler, feature_cols, output_file):
    """Merges host and guest data, runs ML predictions, and pivots into a final matrix."""
    print("Building the Screening Matrix...")
    df_hosts['key'] = 1
    df_guests['key'] = 1
    df_screen = pd.merge(df_hosts, df_guests, on='key').drop('key', axis=1)

    X_screen = df_screen[feature_cols].values

    print("Predicting Binding Energies...")
    X_screen_scaled = scaler.transform(X_screen)
    df_screen['Predicted_Binding_Energy'] = rf_model.predict(X_screen_scaled)

    results = df_screen[['Host', 'Guest', 'Predicted_Binding_Energy']]
    matrix = results.pivot(index='Guest', columns='Host', values='Predicted_Binding_Energy')

    matrix.to_csv(output_file)
    print(f"Complete! Results saved to {output_file}")

# ---------------------------------------------------------
# 2. MAIN EXECUTION BLOCK (Command Line Interface)
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="High-Throughput ML Virtual Screen for Macrocyclic Sorbents")
    
    # Define terminal arguments
    parser.add_argument("--train_data", type=str, default="FINAL_ML_DATASET.csv", help="Path to training data CSV")
    parser.add_argument("--host_data", type=str, default="host_features.csv", help="Path to host macrocycle features CSV")
    parser.add_argument("--guest_data", type=str, default="Contaminant_list.csv", help="Path to target SMILES CSV")
    parser.add_argument("--output", type=str, default="ML_screening_matrix.csv", help="Path to save the output matrix CSV")
    
    args = parser.parse_args()

    # Define exact feature columns expected by the model
    feature_cols = ['Core Cavity Depth', 'Functionalized Depth', 'Effective Free Diameter', 
                    'HBA', 'HBD', 'Aromatic Rings', 'Guest_MW', 'Guest_LogP', 'Guest_TPSA', 
                    'Guest_RotBonds', 'Guest_HBD', 'Guest_HBA', 'Guest_NPR1', 'Guest_NPR2', 
                    'Guest_ASA', 'Guest_Vol', 'Guest_Rg']

    # Execute workflow using the parsed arguments
    rf_model, scaler = train_rf_model(args.train_data, feature_cols)
    df_hosts = pd.read_csv(args.host_data)
    df_guests = process_guest_library(args.guest_data)
    
    build_and_predict(df_hosts, df_guests, rf_model, scaler, feature_cols, args.output)

if __name__ == "__main__":
    main()