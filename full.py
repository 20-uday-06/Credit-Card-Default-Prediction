### Block 0: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import time
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


### Block 1: Load Data
training_data = pd.read_csv('train_dataset_final1.csv')
testing_data = pd.read_csv('validate_dataset_final.csv')

training_data['marriage'] = training_data['marriage'].astype('category')
training_data['sex'] = training_data['sex'].astype('category')
training_data['education'] = training_data['education'].astype('category')

### Block 2: EDA
default_counts = training_data['next_month_default'].value_counts()
labels = ['No Default (0)', 'Default (1)']
colors = ['#05316b', '#5c0000']

training_data['age'] = pd.to_numeric(training_data['age'], errors='coerce')
bins = [20, 30, 40, 50, 60, 70, 80]
labels = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80']
training_data['AGE_GROUP'] = pd.cut(training_data['age'], bins=bins, labels=labels)
plt.figure(figsize=(10,3))
sns.countplot(x='AGE_GROUP', hue='next_month_default', data=training_data , palette={0: 'green', 1: 'Yellow'})
plt.title('Default Status by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

#Leaving other plots and EDA steps out as they are not directly related to the model training and evaluation.

### Block 3: Feature Engineering
#cell 1
# Count of late payments (status > 0 means late)

pay_cols = ['pay_0' , 'pay_2' , 'pay_3' , 'pay_4' , 'pay_5' , 'pay_6']
training_data['total_late_payments'] = (training_data[pay_cols] > 0).sum(axis=1)

# Max delinquency severity
training_data['max_delinquency'] = training_data[pay_cols].max(axis=1)

# Current delinquency streak
def get_current_streak(row):
    streak = 0
    for i in pay_cols:  # Check PAY_0 first (most recent)
        if row[f'{i}'] > 0:
            streak += 1
        else:
            break
    return streak

training_data['current_delinquency_streak'] = training_data.apply(get_current_streak, axis=1)
training_data

#cell 2
# Count of months with only minimum payment (status = 0)
pay_cols = ['pay_0' , 'pay_2' , 'pay_3' , 'pay_4' , 'pay_5' , 'pay_6']
training_data['min_payment_months'] = (training_data[pay_cols] == 0).sum(axis=1)

# Count of months with full payment (status < 0)
training_data['full_payment_months'] = (training_data[pay_cols] < 0).sum(axis=1)

# Count of months with delayed payment (status > 0)
training_data['delayed_months'] = (training_data[pay_cols] > 0).sum(axis=1)

# Payment consistency
pay_amt_cols = [f'pay_amt{i}' for i in range(1, 7)]
training_data['payment_consistency'] = training_data[pay_amt_cols].std(axis=1)
training_data

#cell 3
def generate_features(df):
    df = df.copy()

    #Repayment Consistency 
    repayment_cols = [f'pay_amt{i}' for i in range(1, 7)]
    df['Repayment_StdDev'] = df[repayment_cols].std(axis=1)

    #Monthly Repayment Ratios (Payment / Bill) 
    for i in range(1, 7):
        pay_col = f'pay_amt{i}'
        bill_col = f'Bill_amt{i}'
        df[f'Repay_Ratio_{i}'] = df[pay_col] / df[bill_col].replace(0, np.nan)

    #Overpayment Frequency (Negative bills)
    bill_columns = [f'Bill_amt{i}' for i in range(1, 7)]
    df['Overpay_Count'] = df[bill_columns].lt(0).sum(axis=1)

    # Utilization Ratio (Avg Bill / Limit) 
    df['Utilization_Rate'] = df['AVG_Bill_amt'] / df['LIMIT_BAL'].replace(0, np.nan)

    # Delinquency Features
    pay_status_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    df['Delinquency_Total'] = df[pay_status_cols].ge(1).sum(axis=1)
    df['Max_Delinquency'] = df[pay_status_cols].max(axis=1)

    return df

# apply to both datasets
training_data = generate_features(training_data)
testing_data = generate_features(testing_data) 


#cell 4
for i in range(1, 7):
    training_data[f'UTILIZATION_{i}'] = training_data[f'Bill_amt{i}'] / training_data['LIMIT_BAL']

util_cols = [f'UTILIZATION_{i}' for i in range(1, 7)]
util = training_data.groupby('next_month_default')[util_cols].mean()

plt.figure(figsize=(12, 6))
for status in [0, 1]:
    plt.plot(util_cols, util.loc[status],
             label=f"Default = {status}",
             marker='o')
plt.title('Average Credit Utilization Trends')
plt.xlabel('Month')
plt.ylabel('Utilization Ratio')
plt.legend()
plt.grid()
plt.show()

### Block 4: Data preprocessing
#cell 1
training_data_clean = training_data.drop(columns='Customer_ID', errors='ignore').copy()
testing_data_clean = testing_data.drop(columns='Customer_ID', errors='ignore').copy()

#use knn imputer for missing values
from sklearn.impute import SimpleImputer

numeric_cols = ['LIMIT_BAL', 'age', 'AVG_Bill_amt', 'PAY_TO_BILL_ratio', 'Utilization_Rate', 'Delinquency_Total', 'Max_Delinquency','Repayment_StdDev', 'Overpay_Count'
] + [f'pay_amt{i}' for i in range(1, 7)] + [f'Bill_amt{i}' for i in range(1, 7)] +  [f'Repay_Ratio_{i}' for i in range(1, 7)]

imputer = SimpleImputer(strategy='mean')

training_data_clean[numeric_cols] = imputer.fit_transform(training_data_clean[numeric_cols])
testing_data_clean[numeric_cols] = imputer.transform(testing_data_clean[numeric_cols]) # Use transform only for validation


#cell 2
# categorical columns to encode
categorical_cols = ['sex', 'education', 'marriage']

#Apply one-hot encoding to both train and validation datasets
training_data_encoded = pd.get_dummies(training_data_clean.copy(), columns=categorical_cols, drop_first=True)
testing_data_encoded = pd.get_dummies(testing_data_clean.copy(), columns=categorical_cols, drop_first=True)

training_data_encoded, testing_data_encoded = training_data_encoded.align(testing_data_encoded, join='left', axis=1, fill_value=0)

print("Train shape:", training_data_encoded.shape)
print("Testing shape:", testing_data_encoded.shape)


#cell 3
from sklearn.preprocessing import StandardScaler 
import joblib

# scale numerical features ---
def scale_features(df, features, scaler=None):
    df = df.copy()

    existing_features = [feat for feat in features if feat in df.columns]

    if scaler is None:
        scaler = StandardScaler()
        df[existing_features] = scaler.fit_transform(df[existing_features])
    else:
        df[existing_features] = scaler.transform(df[existing_features])

    return df, scaler

numerical_features = ['LIMIT_BAL', 'age', 'AVG_Bill_amt', 'PAY_TO_BILL_ratio','Utilization_Rate', 'Delinquency_Total', 'Max_Delinquency','Repayment_StdDev', 'Overpay_Count'
] + [f'pay_amt{i}' for i in range(1, 7)] +  [f'Bill_amt{i}' for i in range(1, 7)] +  [f'Repay_Ratio_{i}' for i in range(1, 7)]

# --- Scale train data ---
training_data_encoded, scaler = scale_features(training_data_encoded, numerical_features)

# --- Save scaler for future use ---
joblib.dump(scaler, 'scaler.pkl')


#cell 4
training_data_encoded.drop(['AGE_GROUP', 'education_label', 'Marriage_status','sex_label'], axis=1, inplace=True)
training_data_encoded.info()

#cell 5
# Handling Class Imbalance since default came out to be 20%
# I will do it with SMOTE
from imblearn.over_sampling import SMOTE

X_train = training_data_encoded.drop(columns='next_month_default')
y_train = training_data_encoded['next_month_default']

# Initialize and fit the imputer on the training features (X_train)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Convert the imputed array back to a DataFrame to maintain column names
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Fit and resample using the imputed data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)


#cell 6
print(pd.Series(y_train_resampled).value_counts())

### Block 5: Model Bulding And Training
def evaluate_model(model, X, y, model_name):
    print(f"\n🧪 Evaluating: {model_name}")
    start = time.time()

    # Cross-validated AUC scores
    auc_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

    # Cross-validated predictions
    y_pred = cross_val_predict(model, X, y, cv=5)

    # Metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    end = time.time()
  

    # Output
    print(f"AUC Scores (5-fold): {auc_scores}")
    print(f"Mean AUC: {auc_scores.mean():.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training Time: {end - start:.2f} seconds")

# 1. Logistic Regression with Yeo-Johnson transformation
log_reg_pipeline = Pipeline([
    ('yeojohnson', PowerTransformer(method='yeo-johnson')),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear'))
])
evaluate_model(log_reg_pipeline, X_train, y_train, "Logistic Regression (Yeo-Johnson)")

# 2. Decision Tree
tree = DecisionTreeClassifier(class_weight='balanced', random_state=42)
evaluate_model(tree, X_train, y_train, "Decision Tree")

# 3. LightGBM (using resampled data)
lgbm = LGBMClassifier(n_estimators=100, force_row_wise='true', verbose=-1)
evaluate_model(lgbm, X_train_resampled, y_train_resampled, "LightGBM")

# 4. XGBoost
xgb = XGBClassifier(n_estimators=100, eval_metric='logloss', use_label_encoder=False)
evaluate_model(xgb, X_train, y_train, "XGBoost")

# 5. Random Forest
forest = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
evaluate_model(forest, X_train, y_train, "Random Forest")

#cell 2
from lightgbm import LGBMClassifier
import joblib

# Initialize the LGBM model
final_model_lgb = LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Train the model on resampled data
final_model_lgb.fit(X_train_resampled, y_train_resampled)

# Save the trained model
model_path = 'best_lgbm_model.pkl'
joblib.dump(final_model_lgb, model_path)
print(f"✅ LightGBM model saved successfully at: {model_path}")



### Block 6 Hyperparameter Tuning
# --- Define Pipelines ---
logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(solver='liblinear', max_iter=2000))
])

# --- Define Parameter Grids ---
param_dist_logreg = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

param_dist_dt = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6]
}

param_dist_xgb = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

param_dist_lgbm = {
    'n_estimators': [200, 300, 400, 500],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70, 100, 150],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'min_child_samples': [10, 20, 30, 50],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0]
}


# --- Set up and Run RandomizedSearchCV ---
n_iter_search = 50 
scoring = {'f1': make_scorer(f1_score), 'roc_auc': make_scorer(roc_auc_score)}

# Initialize the search objects with n_jobs=-1 to use all CPU cores
random_search_logreg = RandomizedSearchCV(logreg_pipeline, param_distributions=param_dist_logreg, n_iter=10, cv=5, scoring=scoring, refit='f1', random_state=42, n_jobs=-1) # <-- ADD THIS
random_search_dt = RandomizedSearchCV(decision_tree_model, param_distributions=param_dist_dt, n_iter=n_iter_search, cv=5, scoring=scoring, refit='f1', random_state=42, n_jobs=-1) # <-- ADD THIS
random_search_xgb = RandomizedSearchCV(xgboost_model, param_distributions=param_dist_xgb, n_iter=n_iter_search, cv=5, scoring=scoring, refit='f1', random_state=42, n_jobs=-1) # <-- ADD THIS
random_search_lgbm = RandomizedSearchCV(lightgbm_model, param_distributions=param_dist_lgbm, n_iter=n_iter_search, cv=5, scoring=scoring, refit='f1', random_state=42, n_jobs=-1) # <-- ADD THIS

# Fit RandomizedSearchCV to the training data
print("Tuning Logistic Regression...")
random_search_logreg.fit(X_train, y_train)

print("Tuning Decision Tree...")
random_search_dt.fit(X_train, y_train)

print("Tuning XGBoost...")
random_search_xgb.fit(X_train, y_train)

print("Tuning LightGBM...")
random_search_lgbm.fit(X_train, y_train)

# Store the best estimators
best_logreg_model = random_search_logreg.best_estimator_
best_dt_model = random_search_dt.best_estimator_
best_xgb_model = random_search_xgb.best_estimator_
best_lgbm_model = random_search_lgbm.best_estimator_

print("Hyperparameter tuning complete!")

#cell 2
# Predict using the best models
# y_pred_logreg = best_logreg_model.predict(X_val)
# y_pred_dt = best_dt_model.predict(X_val)
# y_pred_xgb = best_xgb_model.predict(X_val)
# y_pred_lgbm = best_lgbm_model.predict(X_val)

#predict using models before hyperparameter tuning
y_pred_logreg = logreg_model.predict(X_val)
y_pred_dt = decision_tree_model.predict(X_val)
y_pred_xgb = xgboost_model.predict(X_val)
y_pred_lgbm = lightgbm_model.predict(X_val)

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

### Block 7: Tunnning the threshold
from sklearn.metrics import precision_recall_curve
import numpy as np

# Ensure lgbm is trained — if not, fit it first:
lgbm.fit(X_train_resampled, y_train_resampled)

# Get predicted probabilities (positive class)
y_probs_lgbm = lgbm.predict_proba(X_train_resampled)[:, 1]  # Probability of class 1

# Compute precision-recall pairs
precisions, recalls, thresholds = precision_recall_curve(y_train_resampled, y_probs_lgbm)

# Compute F2 score
beta = 2
f2_scores = (1 + beta*2) * (precisions * recalls) / (beta*2 * precisions + recalls + 1e-8)

# Find best threshold
best_idx = np.argmax(f2_scores)
best_threshold = thresholds[best_idx]

# Print best values
print(f"📊 Best Threshold for LightGBM (F2): {best_threshold:.4f}")
print(f"⭐ Best F2-Score: {f2_scores[best_idx]:.4f}")

### Block 8: Final Predictionsimport joblib
import numpy as np
import pandas as pd

# List of numerical features to scale (same as training)
numerical_features = [
    'LIMIT_BAL', 'age', 'AVG_Bill_amt', 'PAY_TO_BILL_ratio','Utilization_Rate', 'Delinquency_Total', 'Max_Delinquency','Repayment_StdDev', 'Overpay_Count'
] + [f'pay_amt{i}' for i in range(1, 7)] + \
    [f'Bill_amt{i}' for i in range(1, 7)] + \
    [f'Repay_Ratio_{i}' for i in range(1, 7)]

# Load saved scaler
scaler = joblib.load('scaler.pkl')

# Drop the target column for prediction since this is unlabeled data scenario
X_val_unlabeled = testing_data_encoded.drop(columns=['next_month_default'])

# Scale numerical features of validation data
X_val_unlabeled[numerical_features] = scaler.transform(X_val_unlabeled[numerical_features])

# Your trained best Random Forest model (load or keep in memory)
# Example: best_rf_model = joblib.load('random_forest_model.pkl')
best_lgb_model = joblib.load('best_lgbm_model.pkl')        # replace with your loaded/trained Random Forest model

# Predict probabilities for the positive class
y_val_probs = best_lgb_model.predict_proba(X_val_unlabeled , predict_disable_shape_check = 'true')[:, 1]

# Apply threshold aligned with bank’s risk appetite
best_threshold = 0.5
y_val_pred = (y_val_probs >= best_threshold).astype(int)

# Prepare dataframe with results
predictions_df = pd.DataFrame({
    'probability_default': y_val_probs,
    'predicted_default': y_val_pred
})

print(f"Predictions generated with threshold {best_threshold}")
print(predictions_df.head())
print(predictions_df['predicted_default'].value_counts())

predictions_df.insert(0, 'Customer_ID', testing_data['Customer_ID'].values)
predictions_df

predictions_df.to_csv('validation_predictions.csv', index=False)

# # Drop the target column for prediction since this is unlabeled data scenario
X_val_unlabeled = testing_data_encoded.drop(columns=['next_month_default'])

# # Scale numerical features of validation data
X_val_unlabeled[numerical_features] = scaler.transform(X_val_unlabeled[numerical_features])

# # Your trained best Random Forest model (load or keep in memory)
# # Example: best_rf_model = joblib.load('random_forest_model.pkl')
best_lgb_model = joblib.load('best_lgbm_model.pkl')        # replace with your loaded/trained Random Forest model

# # Predict probabilities for the positive class
y_val_probs = best_lgb_model.predict_proba(X_val_unlabeled , predict_disable_shape_check = 'true')[:, 1]

import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, precision_recall_curve
from lightgbm import LGBMClassifier

# ✅ Define and train LightGBM
final_model_lgbm = LGBMClassifier(n_estimators=100, force_row_wise='true', verbose=-1)
final_model_lgbm.fit(X_train_resampled, y_train_resampled)


# ✅ Final predicted labels using best threshold
y_pred_final = (y_probs >= best_threshold).astype(int)

# ✅ Export to CSV
output_df = pd.DataFrame({
    # "Actual": y_val,
    "Predicted": y_pred_final,
    "Probabilities": y_probs
})
output_df.to_csv("lightgbm_predictions.csv", index=False)
print("✅ Predictions saved to 'lightgbm_predictions.csv'")


### Block 9 : business impact
#cell 11.5 - Business Impact Analysis
from sklearn.metrics import confusion_matrix
def calculate_business_impact(y_true, y_pred, loan_amount_avg=50000):
    """Calculate business impact of the model"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Assume average loan amount and costs
    avg_loan = loan_amount_avg
    cost_of_default = avg_loan * 0.4  # 40% loss on default
    cost_of_investigation = avg_loan * 0.05  # 5% cost to investigate/reject
    
    # Calculate costs
    cost_false_negatives = fn * cost_of_default  # Missed defaults
    cost_false_positives = fp * cost_of_investigation  # Unnecessary investigations
    total_cost = cost_false_negatives + cost_false_positives
    
    # Potential savings compared to no model
    total_defaults = tp + fn
    cost_without_model = total_defaults * cost_of_default
    savings = cost_without_model - total_cost
    
    print(f"Business Impact Analysis:")
    print(f"Total Cost with Model: ${total_cost:,.2f}")
    print(f"Cost without Model: ${cost_without_model:,.2f}")
    print(f"Potential Savings: ${savings:,.2f}")
    print(f"Cost of False Negatives: ${cost_false_negatives:,.2f}")
    print(f"Cost of False Positives: ${cost_false_positives:,.2f}")
    
    return savings

# Calculate business impact
business_savings = calculate_business_impact(y_val, y_pred_lgbm)



