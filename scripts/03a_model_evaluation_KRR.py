#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kernel Ridge Regression (KRR) Evaluation Model
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error

def load_and_preprocess_data(filepath):
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath).dropna()
    
    y = df['Binding_Energy'].values
    # X is everything EXCEPT the tracking names and the target
    X = df.drop(columns=['Host', 'Guest', 'Binding_Energy']).values
    
    return X, y

def train_and_evaluate_model(X, y):
    # Train/Test Split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("Running Grid Search to find optimal parameters...")
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        'gamma': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
    }

    krr = KernelRidge(kernel='rbf')
    grid_search = GridSearchCV(krr, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_krr = grid_search.best_estimator_
    print(f"Best Parameters Found: {grid_search.best_params_}")

    # Generate Predictions
    y_pred_train = best_krr.predict(X_train)
    y_pred_test = best_krr.predict(X_test)

    # Calculate Metrics
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("\n--- Optimized Model Performance ---")
    print(f"Training R² : {r2_train:.3f} (RMSE: {rmse_train:.2f} kJ/mol)")
    print(f"Testing R²  : {r2_test:.3f} (RMSE: {rmse_test:.2f} kJ/mol)")

    return grid_search, y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test

def plot_grid_search_heatmap(grid_search, out_dir):
    results_df = pd.DataFrame(grid_search.cv_results_)
    pivot_table = results_df.pivot_table(
        index='param_alpha', 
        columns='param_gamma', 
        values='mean_test_score'
    )

    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Grid Search Validation $R^2$", fontsize=14)
    plt.xlabel("Gamma ($\gamma$)", fontsize=12)
    plt.ylabel("Alpha ($\\alpha$)", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, "KRR_CV_GridSearch.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Grid Search heatmap to: {save_path}")

def plot_parity(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, out_dir):
    plt.figure(figsize=(7, 6))

    plt.scatter(y_train, y_pred_train, c='gray', alpha=0.5, edgecolor='k', label=f'Training ($R^2$={r2_train:.2f})')
    plt.scatter(y_test, y_pred_test, c='blue', s=60, edgecolor='k', label=f'Testing ($R^2$={r2_test:.2f})')

    min_val = min(np.min(y_train), np.min(y_pred_train), np.min(y_test), np.min(y_pred_test)) - 20
    max_val = max(np.max(y_train), np.max(y_pred_train), np.max(y_test), np.max(y_pred_test)) + 20
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Fit')

    plt.xlabel("DFT Calculated Binding Energy (kJ/mol)", fontsize=12)
    plt.ylabel("ML Predicted Binding Energy (kJ/mol)", fontsize=12)
    plt.title("KRR Model Performance: Host-Guest Binding", fontsize=14)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, "KRR_Parity_Plot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Parity Plot to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate a KRR Model.")
    parser.add_argument("--input", type=str, default="FINAL_ML_DATASET.csv", help="Path to input dataset CSV.")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save output plots.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X, y = load_and_preprocess_data(args.input)
    
    grid_search, y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test = train_and_evaluate_model(X, y)
    
    plot_grid_search_heatmap(grid_search, args.out_dir)
    plot_parity(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, args.out_dir)

if __name__ == "__main__":
    main()