# Social Media Ads - Purchase Prediction

Predicting whether a user will purchase a product based on their demographics and ad interaction data. Built as a classification project using multiple ML algorithms.

## Dataset

- **Source:** Social Network Ads dataset (400 samples)
- **Target:** Purchased (0 = No, 1 = Yes) — 257 negative, 143 positive
- **Features (9):** Gender, Age, Income, Clicks, Location, Ad Type, Ad Topic, CTR, Conversion Rate
- **Split:** 75/25 train-test with stratification

## Preprocessing Pipeline

1. Label encoding for categorical features (Gender, Location, Ad Type, Ad Topic)
2. Handled data leakage in Clicks column (added noise — raw Clicks had near-perfect correlation with target)
3. MinMaxScaler normalization (0-1 range)
4. Correlation analysis and PCA for feature reduction insights
5. Log transformation on skewed features

## Models Trained

| Model | Train Acc | Test Acc | Precision | Recall | F1 | AUC |
|-------|-----------|----------|-----------|--------|------|------|
| Logistic Regression | 90.67% | 91% | 86.49% | 88.89% | 0.88 | 0.97 |
| Random Forest | 99.33% | 95% | 94.29% | 91.67% | 0.93 | 0.99 |
| **Gradient Boosting** | **100%** | **96%** | **94.44%** | **94.44%** | **0.94** | **0.99** |
| SVM (RBF) | 98.67% | 91% | 93.55% | 80.56% | 0.87 | 0.97 |
| KNN (k=7) | 100% | 90% | 93.33% | 77.78% | 0.85 | 0.96 |

**Best model: Gradient Boosting (96% accuracy)**

All models exceed 90% test accuracy.

## Files

```
DS_Project/
├── data.ipynb                      # Full analysis notebook
├── train_models.py                 # Training script
├── social_media_ads_with_target.csv
├── best_model.pkl                  # Best model (Gradient Boosting)
├── scaler_weights.pkl              # MinMaxScaler weights
├── feature_names.pkl
├── model_info.json
├── model_comparison_results.csv
├── logistic_regression_model.pkl
├── random_forest_model.pkl
├── gradient_boosting_model.pkl
├── svm_model.pkl
├── knn_model.pkl
└── README.md
```

## How to Run

```bash
# train all models
python train_models.py

# or open the notebook
jupyter notebook data.ipynb
```

## Usage

```python
import pickle

scaler = pickle.load(open('scaler_weights.pkl', 'rb'))
model = pickle.load(open('best_model.pkl', 'rb'))

X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```

## Tech Stack

- Python 3.13
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
