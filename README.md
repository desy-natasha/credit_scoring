# Credit Scoring Prototype

## Overview
This is a credit scoring model that compares the performance of two machine learning algorithm: Logistic Regression and Random Forest. The process includes data preprocessing, exploratory data analysis (EDA), and hyperparameter tuning.

## Installation
- Python 3.8
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

Clone this repository and install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Dataset Preparation
Ensure the dataset can be found in `[ROOT_DIR]/credit_scoring/data`

### 2. Run the Scripts

- **Data Preprocessing**:
  ```bash
  python src/data_processing.py
  ```
- **Exploratory Data Analysis**:
  ```bash
  python src/eda.py
  ```
- **Model Training and Evaluation**:
  ```bash
  python src/model.py
  ```

### 3. Model Training
Credit scoring model training will be performed in the `[ROOT_DIR]/credit_scoring/notebook`:
```bash
jupyter notebook notebook/credit-scoring-prototype.ipynb
```
