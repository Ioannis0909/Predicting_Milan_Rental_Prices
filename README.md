# Milan Rental Prices Forecasting

This repository contains my individual project for the 20596 Machine Learning class at Bocconi University. The goal of this project is to predict rental prices for Milan apartments using various machine learning models, primarily based on Random Forests and ensemble methods.

## Author

- Ioannis Thomopoulos  
- Master of Science in Business Analytics and Data Science  
- Bocconi University (AY 2024–2025)  
- Course: 20596 Machine Learning cl.23  
- Professor: Daniele Durante

## Problem Description

The dataset contains 7334 rental listings from Milan collected from Immobiliare.it:

- 4500 listings include the rental price (training set).
- 2834 listings only include features (test set), where rental price must be predicted.

Each listing contains 11 input features:

- `square_meters`: size in sqm
- `contract_type`: rental contract type
- `availability`: when apartment is available
- `description`: room description
- `other_features`: list of apartment features
- `conditions`: apartment condition
- `floor`: apartment floor level
- `elevator`: elevator present or not
- `energy_efficiency_class`: energy class A-G
- `condominium_fees`: condominium costs
- `zone`: zone in Milan

The weight variable `w` is discarded.

## Data Preparation

The following preprocessing was performed:

- Cleaning and standardizing categorical features.
- Extracting bedrooms, bathrooms, kitchen type, and disability features from text descriptions.
- Geocoding zones using Nominatim API to obtain latitude/longitude.
- Calculating distances to:
  - Duomo (city center).
  - Nearest Metro station (using Comune di Milano Metro dataset).
- Engineering boolean flags for 18 amenities.
- Generating multiple interaction terms between key variables.

## Modeling Pipeline

Multiple model variants were developed:

### 1. Baseline Random Forest
- Simple RF without tuning.
- Validation MAE: 357.36.

### 2. Random Forest with Hyperparameter Tuning
- RandomizedSearchCV used to tune: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- Best Validation MAE: ~347.08.

### 3. Random Forest with Interaction Terms
- Added engineered interaction terms.
- Validation MAE: ~349.29.

### 4. Random Forest with Metro Proximity
- Added distance to nearest Metro station.
- Validation MAE: ~347.08.

### 5. Final Random Forest Model
- Combined best features with optimized hyperparameters:
  - `n_estimators=1000`
  - `max_depth=30`
  - `min_samples_split=2`
  - `min_samples_leaf=1`
  - `max_features=0.5`
- Final Validation MAE: 327.96

### 6. Histogram Gradient Boosting (HGB)
- Gradient boosting model using sklearn’s `HistGradientBoostingRegressor`.
- Validation MAE: 304.41.

### 7. Stacked Ensemble Model
- Combined HGB, XGBoost, and CatBoost using LassoCV meta-learner.
- Final Validation MAE: 302.07.

## Final Model Scores

| Model            | Validation MAE |
|-------------------|----------------|
| Random Forest     | 327.96         |
| Gradient Boosting | 304.41         |
| Stacking Ensemble | 302.07         |

## External Data Sources

- **Zone Coordinates:** [OpenStreetMap Nominatim API](https://nominatim.openstreetmap.org/)
- **Metro Coordinates:** [Comune di Milano Open Data](https://dati.comune.milano.it)

## Code Execution Notes

- Code written fully in Python.
- Internet connection required for geocoding.
- Heavy grid-search operations may require significant compute time.
- All required dependencies are listed in the notebook.

## Data Files

- `train.csv`: Training dataset with target prices.
- `test.csv`: Test dataset for prediction.

## Requirements

- Python 3.x
- pandas, numpy, scikit-learn, xgboost, catboost, matplotlib, tqdm, requests, joblib

## Submission Notes

- This repository includes:
  - Full Jupyter notebook with all code.
  - Final project report.
  - All raw data files used.