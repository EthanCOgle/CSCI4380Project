Data-Driven Player Efficiency Rating (PER) Prediction
Predicting NBA Rookie PER from College and Draft Combine Data

Sam Peters, Ethan Ogle, Matthew Mullis, Joseph Murphy
CSCI 4380 • December 5, 2025

Overview

This repository contains all code, notebooks, and outputs for a project investigating whether college statistics, NBA Draft Combine measurements, and biographical data can predict a player’s Rookie Player Efficiency Rating (PER).

We merged three public datasets into a unified table (217 players, 97 features) and trained multiple machine learning models. Ensemble tree methods significantly outperformed linear models and neural networks, with XGBoost achieving the strongest predictive performance.

The repo includes:

Data preparation and cleaning

Feature engineering

Model training and evaluation (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, Neural Network)

Visualizations and outputs

Final written report

Repository Structure
.
├── Comparisons/                     # Model comparison files and outputs
├── outputs/                         # Cleaned datasets, processed CSVs, model results
├── CSCI4380Project.ipynb            # Old notebook with initial data ingestion and cleanup
├── Data_driven_Player_Efficiency_Rating_Prediction.ipynb  # Updated notebook with final analysis
└── README.md                        # This file
