#Final Logistic Regression model with all features and threshold tuning

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#==================================================
#1. Load processed data
#==================================================
#Read CSV file
df = pd.read_csv("/Users/joannajorise/Desktop/FYP_Project/data/processed/final_dataset.csv")

# Preview
print("Dataset shape:", df.shape)

#==================================================
#2. Prepare features & target
#==================================================
# Separate features (X) and target (y)
X = df.drop(columns=['target'])  
y = df['target']

# Convert categorical variables into numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

print("Features shape after encoding:", X.shape)

# Save model columns for dashboard use
model_columns = X.columns.tolist()

#==================================================
#3. Train-test split
#==================================================
# Split dataset into training and testing sets (80:20)
# Stratify to maintain class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


#==================================================
#4. Feature scaling with StandardScaler
#==================================================
# Logistic Regression is sensitive to feature scales, so scaling is applied to ensure all features contribute equally to the model
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#==================================================
#5. Model training with Logistic Regression
#==================================================
# Initialize Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Handle class imbalance
    random_state=42
    )

# Train the model
model.fit(X_train, y_train)

#==================================================
#6. Apply best threshold
#==================================================
best_threshold = 0.7 #selected threshold from threshold tuning

# Predict class labels based on the best threshold
y_prob = model.predict_proba(X_test)[:, 1]

# Convert probabilities to class predictions using selectd threshold
y_pred = (y_prob >= best_threshold).astype(int)

#==================================================
#7. Final evaluation with best threshold
#==================================================
print("\nFinal Model Evaluation:")
print("Best Threshold:", best_threshold)

print("\nConfusion Matrix using Best Threshold:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report using Best Threshold:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))

#==================================================
#8. Save the model and scaler
#==================================================
# save the trained model
joblib.dump(model, "/Users/joannajorise/Desktop/FYP_Project/models/saved_models/final_logistic_regression_model.pkl")

# applies the same scaling to new data during prediction, ensuring consistency with the training data
joblib.dump(scaler, "/Users/joannajorise/Desktop/FYP_Project/models/saved_models/scaler.pkl")

# Save model columns for dashboard use, ensuring the same feature order during prediction
joblib.dump(model_columns, "/Users/joannajorise/Desktop/FYP_Project/models/saved_models/model_columns.pkl")

# Save the best threshold for use in the dashboard, ensuring consistent prediction logic
joblib.dump(best_threshold, "/Users/joannajorise/Desktop/FYP_Project/models/saved_models/best_threshold.pkl")

print("\nFinal Logistic Regression model saved successfully!")
print("Saved files:")
print("- final_logistic_regression_model.pkl")
print("- scaler.pkl")
print("- model_columns.pkl")
print("- best_threshold.pkl")
