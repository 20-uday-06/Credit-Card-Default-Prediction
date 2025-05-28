#cell 1

import pandas as pd

try:
    train_df = pd.read_csv('train_dataset_final1.csv', sep=',', engine='python', on_bad_lines='skip')
    validate_df = pd.read_csv('validate_dataset_final.csv', sep=',', engine='python', on_bad_lines='skip')
    display(train_df.head())
    display(validate_df.head())
except FileNotFoundError:
    print("Error: One or both of the CSV files were not found.")
except pd.errors.ParserError:
    print("Error: There was an issue parsing the CSV file(s).  Check the file format and separators.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

#cell 2
import matplotlib.pyplot as plt
import seaborn as sns

# Basic Exploration
print("Shape of the DataFrame:", train_df.shape)
print("\nData Types:\n", train_df.dtypes)
print("\nDescriptive Statistics:\n", train_df.describe())
print("\nMissing Values:\n", train_df.isnull().sum())

# Distribution Analysis for Numerical Features
numerical_features = ['LIMIT_BAL', 'age', 'Bill_amt1', 'Bill_amt2', 'Bill_amt3', 'Bill_amt4', 'Bill_amt5', 'Bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6', 'AVG_Bill_amt', 'PAY_TO_BILL_ratio']
plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_features):
  plt.subplot(5, 4, i + 1)
  sns.histplot(train_df[col], kde=True)
  plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Distribution Analysis for Categorical Features
categorical_features = ['sex', 'education', 'marriage', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
plt.figure(figsize=(20, 15))
for i, col in enumerate(categorical_features):
    plt.subplot(3, 3, i + 1)
    sns.countplot(x=col, data=train_df)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# Distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='next_month_default', data=train_df)
plt.title('Distribution of Target Variable')
plt.show()

# Class Imbalance Calculation
class_counts = train_df['next_month_default'].value_counts()
class_percentages = class_counts / len(train_df) * 100
print("\nClass Imbalance:")
print(class_counts)
print(class_percentages)


# Correlation Analysis
plt.figure(figsize=(12, 10))
sns.heatmap(train_df[numerical_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


# Pairplots
sns.pairplot(train_df[numerical_features[:5] + ['next_month_default']], hue='next_month_default')
plt.show()

#cell 3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Handle missing values
for df in [train_df, validate_df]:
    # Impute missing 'age' with the median
    df['age'] = df['age'].fillna(df['age'].median())

    # Investigate and remove rows with missing 'Customer_ID'
    df.dropna(subset=['Customer_ID'], inplace=True)

# Outlier Treatment (Winsorization)
numerical_features = ['LIMIT_BAL', 'age', 'Bill_amt1', 'Bill_amt2', 'Bill_amt3', 'Bill_amt4', 'Bill_amt5', 'Bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6', 'AVG_Bill_amt', 'PAY_TO_BILL_ratio']
for df in [train_df, validate_df]:
    for col in numerical_features:
        df[col] = np.clip(df[col], df[col].quantile(0.01), df[col].quantile(0.99))

# Data Encoding (Label Encoding for categorical features)
categorical_cols = ['sex', 'education', 'marriage', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
for df in [train_df, validate_df]:
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Data Consistency Checks
for df in [train_df, validate_df]:
    # Check 'PAY_0' to 'PAY_6' for consistency
    pay_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    for col in pay_cols:
        df[col] = np.clip(df[col], -2, 8)

    # Check 'LIMIT_BAL'
    df['LIMIT_BAL'] = np.clip(df['LIMIT_BAL'], 10000, 1000000)

    # Check 'age'
    df['age'] = np.clip(df['age'], 20, 80)

display(train_df.head())
display(validate_df.head())

#cell 4
import matplotlib.pyplot as plt
import seaborn as sns

# Target Variable Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='next_month_default', data=train_df)
plt.title('Distribution of Target Variable')
plt.show()

class_counts = train_df['next_month_default'].value_counts()
class_percentages = class_counts / len(train_df) * 100
print("\nClass Imbalance:")
print(class_counts)
print(class_percentages)

# Feature vs. Target Relationships
numerical_features = ['LIMIT_BAL', 'age', 'Bill_amt1', 'Bill_amt2', 'Bill_amt3', 'Bill_amt4', 'Bill_amt5', 'Bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6', 'AVG_Bill_amt', 'PAY_TO_BILL_ratio']
categorical_features = ['sex', 'education', 'marriage', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']

plt.figure(figsize=(20, 30))
for i, col in enumerate(numerical_features):
    plt.subplot(6, 3, i + 1)
    sns.boxplot(x='next_month_default', y=col, data=train_df)
    plt.title(f'{col} vs. Default')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 30))
for i, col in enumerate(categorical_features):
    plt.subplot(3, 3, i + 1)
    sns.countplot(x=col, hue='next_month_default', data=train_df)
    plt.title(f'{col} vs. Default')
plt.tight_layout()
plt.show()

# Financial Insights
# Credit Limit Analysis
plt.figure(figsize=(8, 6))
sns.boxplot(x='next_month_default', y='LIMIT_BAL', data=train_df)
plt.title('Credit Limit vs. Default')
plt.show()

# Repayment History Analysis
pay_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
plt.figure(figsize=(15, 10))
for i, col in enumerate(pay_cols):
    plt.subplot(2, 3, i + 1)
    sns.countplot(x=col, hue='next_month_default', data=train_df)
    plt.title(f'{col} vs. Default')
plt.tight_layout()
plt.show()

# Bill/Payment Amounts Analysis
bill_amt_cols = ['Bill_amt1', 'Bill_amt2', 'Bill_amt3', 'Bill_amt4', 'Bill_amt5', 'Bill_amt6']
pay_amt_cols = ['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']

plt.figure(figsize=(15, 10))
for i, col in enumerate(bill_amt_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='next_month_default', y=col, data=train_df)
    plt.title(f'{col} vs. Default')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(pay_amt_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='next_month_default', y=col, data=train_df)
    plt.title(f'{col} vs. Default')
plt.tight_layout()
plt.show()

# Calculate and visualize credit utilization
train_df['credit_utilization'] = train_df['Bill_amt1'] / train_df['LIMIT_BAL']
plt.figure(figsize=(8, 6))
sns.boxplot(x='next_month_default', y='credit_utilization', data=train_df)
plt.title('Credit Utilization vs. Default')
plt.show()

#cell 5
# Summarize Key Findings
print("## Summary of Exploratory Data Analysis and Financial Insights")
print("\n**Target Variable Distribution:**")
print("- The target variable 'next_month_default' exhibits class imbalance, with approximately 81% of customers not defaulting and 19% defaulting.")

print("\n**Feature vs. Target Relationships:**")
print("- Numerical features show varying distributions across default and non-default classes, suggesting their potential predictive power.")
print("- Categorical features also demonstrate differences in default rates across categories, highlighting their importance in the prediction task.")

print("\n**Financial Insights:**")
print("\n**Credit Limit Analysis:**")
print("- Customers with lower credit limits appear to have a higher risk of default. Further analysis is needed to quantify this observation.")

print("\n**Repayment History Analysis:**")
print("- Customers with a history of delayed payments (represented by 'PAY_0', 'PAY_2', etc.) show a significantly higher tendency to default.")
print("- A clear trend of increasing default risk is observed with increasing payment delays.")

print("\n**Bill/Payment Amounts Analysis:**")
print("- The relationship between bill amounts ('BILL_AMT1', etc.) and payment amounts ('PAY_AMT1', etc.) with default risk is complex and requires further investigation.")
print("- Consistently high credit utilization (bill amount / credit limit) seems associated with a higher default probability.")

print("\n**Credit Utilization:**")
print("- Initial analysis of credit utilization suggests a potential link with default risk.  Higher credit utilization may indicate higher risk.")

print("\n**Overall:**")
print("- The EDA and financial analysis provide valuable insights into the factors driving credit default.")
print("- The class imbalance needs to be addressed in future modeling steps.")
print("- The identified features are promising candidates for building a predictive model.")

#cell 6
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.impute import SimpleImputer

# Separate features (X) and target variable (y)
X = train_df.drop('next_month_default', axis=1)
y = train_df['next_month_default']

# Fill NaN values in the target variable with the most frequent value
most_frequent_class = y.mode()[0]
y.fillna(most_frequent_class, inplace=True)

# Impute missing values in X using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Apply SMOTE to oversample the minority class
try:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
except ValueError as e:
    print(f"SMOTE error: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit(1)

# Combine the oversampled features and target variable back into a DataFrame
train_df_oversampled = pd.DataFrame(X_resampled, columns=X.columns)
train_df_oversampled['next_month_default'] = y_resampled

# Verify class distribution
print(train_df_oversampled['next_month_default'].value_counts())

#cell 7
from sklearn.model_selection import train_test_split

# Separate features (X) and target variable (y)
X = train_df_oversampled.drop('next_month_default', axis=1)
y = train_df_oversampled['next_month_default']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print shapes of the resulting DataFrames/Series
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

#cell 8
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Initialize models
logreg_model = LogisticRegression(solver='liblinear', max_iter=1000)  # Changed solver
decision_tree_model = DecisionTreeClassifier()
xgboost_model = XGBClassifier()
lightgbm_model = LGBMClassifier()

# Train models
logreg_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
xgboost_model.fit(X_train, y_train)
lightgbm_model.fit(X_train, y_train)

#cell 9
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline

# Define parameter grids for each model
param_dist_logreg = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

param_dist_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_dist_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0]
}

param_dist_lgbm = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [31, 50, 100],
    'subsample': [0.8, 0.9, 1.0]
}

# Create pipelines (if needed) and instantiate RandomizedSearchCV
scoring = {'f1': make_scorer(f1_score), 'roc_auc': make_scorer(roc_auc_score)}
n_iter_search = 10  # Reduce iterations for faster execution

random_search_logreg = RandomizedSearchCV(logreg_model, param_distributions=param_dist_logreg, n_iter=n_iter_search, cv=5, scoring=scoring, refit='f1', random_state=42)
random_search_dt = RandomizedSearchCV(decision_tree_model, param_distributions=param_dist_dt, n_iter=n_iter_search, cv=5, scoring=scoring, refit='f1', random_state=42)
random_search_xgb = RandomizedSearchCV(xgboost_model, param_distributions=param_dist_xgb, n_iter=n_iter_search, cv=5, scoring=scoring, refit='f1', random_state=42)
random_search_lgbm = RandomizedSearchCV(lightgbm_model, param_distributions=param_dist_lgbm, n_iter=n_iter_search, cv=5, scoring=scoring, refit='f1', random_state=42)

# Fit RandomizedSearchCV to the training data
random_search_logreg.fit(X_train, y_train)
random_search_dt.fit(X_train, y_train)
random_search_xgb.fit(X_train, y_train)
random_search_lgbm.fit(X_train, y_train)

# Store the best estimators
best_logreg_model = random_search_logreg.best_estimator_
best_dt_model = random_search_dt.best_estimator_
best_xgb_model = random_search_xgb.best_estimator_
best_lgbm_model = random_search_lgbm.best_estimator_

#cell 10
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

# Predict using the best models
y_pred_logreg = best_logreg_model.predict(X_val)
y_pred_dt = best_dt_model.predict(X_val)
y_pred_xgb = best_xgb_model.predict(X_val)
y_pred_lgbm = best_lgbm_model.predict(X_val)

# Evaluate performance
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return [model_name, accuracy, precision, recall, f1, roc_auc]

results = []
results.append(evaluate_model(y_val, y_pred_logreg, "Logistic Regression"))
results.append(evaluate_model(y_val, y_pred_dt, "Decision Tree"))
results.append(evaluate_model(y_val, y_pred_xgb, "XGBoost"))
results.append(evaluate_model(y_val, y_pred_lgbm, "LightGBM"))


# Create a summary table
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"])
display(results_df)


#cell 11
import pandas as pd
import numpy as np

# Calculate credit utilization for the validation set
validate_df['credit_utilization'] = validate_df['Bill_amt1'] / validate_df['LIMIT_BAL']

# Predict probabilities for the positive class (default=1)
predicted_probabilities = best_lgbm_model.predict_proba(validate_df.drop('Customer_ID', axis=1))[:, 1]

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({'Customer_ID': validate_df['Customer_ID'], 'predicted_probability': predicted_probabilities})

# Save predictions to a CSV file
predictions_df.to_csv('lgbm_predictions.csv', index=False)


#cell 12
# Check features in training and validation sets
print("Training Features:", X_train.columns.tolist())
print("\nValidation Features:", validate_df.columns.tolist())

# Identify missing features
missing_features = set(X_train.columns) - set(validate_df.columns)
print("\nMissing Features:", missing_features)

# If there are missing features in the validation set, add them with appropriate values
if missing_features:
    for feature in missing_features:
        if feature == 'next_month_default': # Skip this column as it's the target
            continue
        validate_df[feature] = 0  # Or use a more suitable imputation method

# Predict probabilities
predicted_probabilities = best_lgbm_model.predict_proba(validate_df.drop('Customer_ID', axis=1))[:, 1]
predictions_df = pd.DataFrame({'Customer_ID': validate_df['Customer_ID'], 'predicted_probability': predicted_probabilities})
predictions_df.to_csv('lgbm_predictions.csv', index=False)

#cell 13
# Reorder the columns in validate_df to match X_train
validate_df = validate_df[X_train.columns]

# Predict probabilities
predicted_probabilities = best_lgbm_model.predict_proba(validate_df.drop('Customer_ID', axis=1))[:, 1]
predictions_df = pd.DataFrame({'Customer_ID': validate_df['Customer_ID'], 'predicted_probability': predicted_probabilities})
predictions_df.to_csv('lgbm_predictions.csv', index=False)



