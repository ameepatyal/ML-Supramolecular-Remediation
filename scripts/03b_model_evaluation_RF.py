#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using a Random Forest Regressor with 5-Fold Cross-Validation.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def load_and_preprocess(input_file):
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file).dropna()
    
    feature_names = df.drop(columns=['Host', 'Guest', 'Binding_Energy']).columns
    y = df['Binding_Energy'].values
    X = df.drop(columns=['Host', 'Guest', 'Binding_Energy']).values
    
    return X, y, feature_names, df

def plot_parity(y_true, y_pred, r2, rmse, out_dir):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, c='teal', alpha=0.7, edgecolor='k', s=50, 
                label=f'Blind CV Predictions\n($R^2$={r2:.2f}, RMSE={rmse:.1f})')
    
    min_val = min(y_true.min(), y_pred.min()) - 10
    max_val = max(y_true.max(), y_pred.max()) + 10
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Fit')
    
    plt.xlabel("DFT Calculated Binding Energy (kJ/mol)", fontsize=12)
    plt.ylabel("ML Predicted Binding Energy (kJ/mol)", fontsize=12)
    plt.title("Random Forest Performance (5-Fold CV)", fontsize=14)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, "RF_KFold_Parity_Plot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Parity Plot to {save_path}")

def plot_feature_importances(pipeline, X, y, feature_names, out_dir):
    pipeline.fit(X, y)
    
    # feature importances
    rf_model = pipeline.named_steps['rf']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names[indices],
        'Importance': importances[indices]
    })
    
    csv_path = os.path.join(out_dir, "RF_Feature_Importances.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved Feature Importances CSV to {csv_path}")
    
    top_n = min(15, len(feature_names))
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances", fontsize=14)
    plt.bar(range(top_n), importances[indices[:top_n]], color="teal", align="center", edgecolor='k', alpha=0.8)
    plt.xticks(range(top_n), feature_names[indices[:top_n]], rotation=45, ha="right", fontsize=10)
    plt.xlim([-1, top_n])
    plt.ylabel("Relative Importance (MDI)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, "RF_Feature_Importances.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_energy_distribution(y, out_dir):
    plt.figure(figsize=(7, 5))
    sns.histplot(y, bins=20, kde=True, color='teal')
    plt.title(f"Distribution of DFT Binding Energies (N = {len(y)})", fontsize=14)
    plt.xlabel("Binding Energy (kJ/mol)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, "Energy_Distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    X, y, feature_names, df = load_and_preprocess(args.input)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_split=4, random_state=42))
    ])
    
    print("Running 5-Fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    y_pred_cv = cross_val_predict(pipeline, X, y, cv=kf)
    

    cv_r2 = r2_score(y, y_pred_cv)
    cv_rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
    print(f"\nGlobal CV R²   : {cv_r2:.3f}")
    print(f"Global CV RMSE : {cv_rmse:.2f} kJ/mol\n")
    

    plot_parity(y, y_pred_cv, cv_r2, cv_rmse, args.out_dir)
    plot_feature_importances(pipeline, X, y, feature_names, args.out_dir)
    plot_energy_distribution(y, args.out_dir)
    
    print("All tasks completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Random Forest on Host-Guest data.")
    parser.add_argument("--input", type=str, default="FINAL_ML_DATASET.csv", help="Path to the input CSV dataset")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save output plots and CSVs")
    
    args = parser.parse_args()
    main(args)