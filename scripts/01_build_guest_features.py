#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import rdMolDescriptors

def get_guest_features(smiles_string):

    mol = Chem.MolFromSmiles(str(smiles_string))
    if mol is None:
        return [None] * 11
    
    mol = Chem.AddHs(mol)
    
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        return [None] * 11
    
    # Calculate Descriptors
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    npr1 = rdMolDescriptors.CalcNPR1(mol)
    npr2 = rdMolDescriptors.CalcNPR2(mol)
    asa = rdMolDescriptors.CalcLabuteASA(mol)
    vol = AllChem.ComputeMolVolume(mol)
    rg = rdMolDescriptors.CalcRadiusOfGyration(mol)
    
    return mw, logp, tpsa, rot_bonds, hbd, hba, npr1, npr2, asa, vol, rg

def process_molecule_dataframe(df, name_col, smiles_col, output_file):

    print(f"Processing {len(df)} molecules...")
    features_list = []
    
    for index, row in df.iterrows():
        name = row[name_col]
        smiles = row[smiles_col]
        
        features = get_guest_features(smiles)
        
        features_list.append({
            "Guest": name,
            "SMILES": smiles,
            "Guest_MW": features[0],
            "Guest_LogP": features[1],
            "Guest_TPSA": features[2],
            "Guest_RotBonds": features[3],
            "Guest_HBD": features[4],
            "Guest_HBA": features[5],
            "Guest_NPR1": features[6],
            "Guest_NPR2": features[7],
            "Guest_ASA": features[8],
            "Guest_Vol": features[9],
            "Guest_Rg": features[10]
        })

    rounding_dict = {
        'Guest_MW': 2, 'Guest_LogP': 3, 'Guest_TPSA': 3, 
        'Guest_NPR1': 3, 'Guest_NPR2': 3, 'Guest_ASA': 2,
        'Guest_Vol': 2, 'Guest_Rg': 3
    }

    df_features = pd.DataFrame(features_list)
    df_features = df_features.round(rounding_dict)
    
    # Clean up any rows where RDKit failed
    df_features = df_features.dropna(subset=['Guest_MW'])
    
    df_features.to_csv(output_file, index=False)
    print(f"Complete! Saved {len(df_features)} valid molecules to {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate RDKit 3D/2D features from SMILES.")
    
    parser.add_argument("--input", type=str, help="Path to input CSV containing Guest and SMILES.")
    parser.add_argument("--output", type=str, default="guest_ml_features.csv", help="Path to save output CSV.")
    parser.add_argument("--name_col", type=str, default="Guest", help="Name of the column containing identifiers.")
    parser.add_argument("--smiles_col", type=str, default="SMILES", help="Name of the column containing SMILES.")
    
    args = parser.parse_args()
    
    if args.input:
        # 1. REAL DATA WORKFLOW (Triggered when you provide --input)
        print(f"Loading dataset from {args.input}...")
        df_input = pd.read_csv(args.input).dropna(subset=[args.smiles_col])
        
    else:
        # 2. DEFAULT TOY DATASET WORKFLOW (Triggered if no --input is provided)
        print("No input file provided. Using the default 17-contaminant toy dataset for demonstration...")
        guest_smiles = {
            "11DCE": "C=C(Cl)Cl",
            "CDCE":"C(=C\Cl)\Cl",
            "TCE":"C(=C(Cl)Cl)Cl",
            "dioxane": "C1COCCO1",
            "PFBA":"C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O",
            "PFPrS":"C(C(F)(F)F)(C(F)(F)S(=O)(=O)O)(F)F",
            "PFBS":"C(C(C(F)(F)S(=O)(=O)O)(F)F)(C(F)(F)F)(F)F",
            "PFOA": "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CF",
            "PFOS":"C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F",
            "GenX": "C(=O)(C(C(F)(F)F)(OC(C(C(F)(F)F)(F)F)(F)F)F)O",
            "Acetaminophen":"CC(=O)NC1=CC=C(C=C1)O",
            "Atrazine": "CCNC1=NC(=NC(=N1)Cl)NC(C)C",
            "Carbamazepine":"C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N",
            "Diuron":"CN(C)C(=O)NC1=CC(=C(C=C1)Cl)Cl",
            "Imidacloprid":"C1CN(/C(=N/[N+](=O)[O-])/N1)CC2=CN=C(C=C2)Cl",
            "water":"O",
            "MeOH":"CO"
        }
        # Convert dictionary to DataFrame automatically
        df_input = pd.DataFrame(list(guest_smiles.items()), columns=[args.name_col, args.smiles_col])
        
        # Save this to the toy folder so downstream scripts can use it
        import os
        os.makedirs("data/toy", exist_ok=True)
        df_input.to_csv("data/toy/toy_contaminants.csv", index=False)
        print("Saved default contaminants to data/toy/toy_contaminants.csv")

    # Run the processing function
    process_molecule_dataframe(df_input, args.name_col, args.smiles_col, args.output)