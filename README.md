# Insurance Cost Prediction Project

This project analyzes the insurance dataset to predict medical insurance charges based on demographic and lifestyle data.

## Dataset

The dataset `insurance.csv` contains the following columns:

- age: Age of primary beneficiary
- sex: Gender (male, female)
- bmi: Body Mass Index
- children: Number of children
- smoker: Smoking status (yes, no)
- region: Residential region
- charges: Medical insurance cost

## Analysis

The notebook `insurance_analysis.ipynb` performs:

1. Data loading and exploration
2. Preprocessing (encoding categorical variables)
3. Exploratory data analysis with visualizations
4. Model building with Linear Regression, Decision Tree, and Random Forest
5. Model evaluation
6. Feature importance analysis
7. Fairness assessment across groups

## Requirements

Install dependencies with:

```
pip install -r requirements.txt
```

## Running the Notebook

Open `insurance_analysis.ipynb` in Jupyter and run all cells.

## Results

- Models are evaluated on RMSE, MAE, R2
- Feature importance shows key factors like smoker, age, bmi
- Fairness check reveals differences in charges by gender and region, with potential concerns for equity.