# ML-Supramolecular-Remediation

### Machine-Learning-Guided Discovery of Broad-Spectrum Macrocyclic Sorbents for the Sequestration of Emerging Water Contaminants

This repository contains the computational workflow and Python scripts used to predict the host-guest binding affinities of over 17,000 emerging environmental contaminants across 15 distinct macrocyclic architectures (Cyclodextrins, Cucurbiturils, and Pillararenes). 

This repository accompanies our upcoming manuscript.

---

## ⚠️ Data Availability Note
**The raw density functional theory (DFT) training dataset (N=186) and the final predictions for all 17,933 EPA designated contaminants are currently withheld pending peer review and publication.** 

Once the manuscript is published, the full dataset, along with all 186 DFT-optimized VASP format (POSCAR) structure files, will be made publicly available via Zenodo and linked here. 

In the meantime, this repository contains a `toy_dataset.csv` (located in `data/toy/`) that allows users to test the machine learning pipeline and verify the reproducibility of our Random Forest regression models.

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/                 # Ignored (Real training data)
│   ├── processed/           # Ignored (Screening matrices)
│   └── toy/                 # Uploaded (Dummy data for testing scripts)
├── scripts/                 # Python workflow scripts
│   ├── 01_build_guest_features.py
│   ├── 02_host_features_merge_data.py
│   ├── 03a_model_evaluation_KRR.py
│   ├── 03b_model_evaluation_RF.py
│   ├── 04_virtual_screen.py
│   └── 05_EPA_contaminant_macrocycle_matchmaker.py
├── results/                 # Ignored (Parity plots, feature importance charts)
├── .gitignore               # Ensures private data is not uploaded
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```
## 🚀 Quick Start (Testing with Toy Data)
To verify the computational pipeline on your local machine using the provided toy dataset, follow these steps:

1. Install Dependencies
```
bash
pip install -r requirements.txt
```
2. Evaluate the Machine Learning Models
Test the Kernel Ridge Regression (KRR) and Random Forest (RF) algorithms on the dummy dataset:
```
bash
python scripts/03a_model_evaluation_KRR.py --input data/toy/toy_dataset.csv
python scripts/03b_model_evaluation_RF.py --input data/toy/toy_dataset.csv
```
3. Run the Virtual Screen & Matchmaker Pipeline
Execute the high-throughput screening algorithm and rank the top macrocycles using the Comparative Selectivity Index (Z-score):
```
bash
python scripts/04_virtual_screen.py \
  --train_data data/toy/toy_dataset.csv \
  --guest_data data/toy/toy_contaminants.csv \
  --host_data data/toy/toy_host_features.csv \
  --output toy_screening_matrix.csv

python scripts/05_EPA_contaminant_macrocycle_matchmaker.py \
  --input toy_screening_matrix.csv \
  --output toy_Matchmaker_Results.csv
```
## 📄 Citation
If you utilize this workflow or the associated DFT datasets in your research, please cite our upcoming publication

## Authors

This database was created at Professor Joshua D. Howe's lab at Texas Tech University.

Please contact Ameevardhan Singh Patyal (amee.patyal@gmail.com) with questions or corrections to this database.

## Acknowledgment

This research was supported by the National Institute of Environmental Health Sciences (NIEHS) Superfund Research Program (Grant No. R01ES032692) and the Department of Defense (DoD) SERDP (Project No. ER21-C1-1256). In silico calculations were performed using the High-Performance Computing Center resources at Texas Tech University.
