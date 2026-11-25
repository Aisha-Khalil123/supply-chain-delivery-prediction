# Late Delivery & Profit Prediction: DataCo Supply Chain

Predicting which orders will arrive late, and how much profit each order makes, using the DataCo Smart Supply Chain dataset (180,519 orders). Built for the *AI & Intelligent Systems* course at Al Hussein Technical University (Sep 2025).

## Problem

A logistics company wants to use its historical order data to:

1. **Flag orders at risk of late delivery** (binary classification, target: `Late_delivery_risk`)
2. **Estimate profit per order** (regression, target: `Order Profit Per Order`)

## Data

- Source: [DataCo Smart Supply Chain dataset (Mendeley Data)](https://data.mendeley.com/datasets/8gx2fvg2k6/5), file `DataCoSupplyChainDataset.csv`
- 180,519 orders, 53 columns: shipping mode, scheduled shipping days, customer and order region, product, price, discount, sales, profit
- The dataset is not included in this repo because of its size. Download it from the link above and place it next to the notebooks.

## Notebooks

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | `1_data_cleaning.ipynb` | Removes empty columns and customer personal data (names, emails, passwords), drops duplicate columns found through correlation, label-encodes categorical features, and analyses profit outliers with IQR |
| 2 | `2_regression.ipynb` | Linear regression to predict order profit, evaluated with RMSE and R² |
| 3 | `3_classification_knn.ipynb` | K-Nearest Neighbours for late-delivery risk, tuning k, with accuracy, precision, recall and F1 |
| 4 | `4_neural_network.ipynb` | Feed-forward neural network (Keras: 16 → 8 → 1, sigmoid output) for late-delivery risk |

## Results

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Late delivery | KNN (k = [ ]) | Accuracy / F1 | [ ] / [ ] |
| Late delivery | Neural network | Accuracy / Recall | [ ] / [ ] |
| Profit per order | Linear regression | RMSE / R² | [ ] / [ ] |

## Avoiding data leakage

The first version of the models scored 99.6% to 100% accuracy on late delivery. That score was too good to be real. In this dataset, `Late_delivery_risk` is 1 exactly when `Days for shipping (real)` is greater than `Days for shipment (scheduled)`, and `Delivery Status` says "Late delivery" directly. Neither value is known until the order arrives, so the models were reading the answer.

The final models use only information available **when the order is placed**. `Days for shipping (real)` and `Delivery Status` are removed for classification. `Order Item Profit Ratio` is removed for regression, because it is profit divided by sales.

## Ethics and privacy

Customer names, emails and passwords were removed during cleaning, before any modelling.

## Tools

Python, Pandas, NumPy, scikit-learn, Keras/TensorFlow, Matplotlib, Seaborn, Google Colab

## How to run

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

Run the notebooks in order (1 → 4). Notebook 1 creates `PreprocessedData.csv`, which the others read.
