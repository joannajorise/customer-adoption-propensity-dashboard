#Epxeriment 1A: Logistic Regression with only customer features (excluding campaign features) and threshold tuning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
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
campaign_features = ['Reward', 'Mailer Type']
X = df.drop(columns=['target'] + campaign_features)  
y = df['target']

# Convert categorical variables into numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

print("Features used:", X.columns.tolist())
print("Features shape after encoding:", X.shape)

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
#6. Get predicted probabilities for threshold tuning
#==================================================
y_prob = model.predict_proba(X_test)[:, 1]

#==================================================
#7. Evaluate different thresholds
#==================================================
# Evaluate different thresholds to find the best one based on F1-score
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

results = []

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Threshold': threshold,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

print("\nThreshold Tuning Results:")
print(results_df)

#==================================================
#8. Select best threshold based on F1-score
#==================================================
best_result = results_df.loc[results_df['F1-Score'].idxmax()]
best_threshold = best_result['Threshold']

print(f"\nBest Threshold based on F1-Score:")
print(best_result)

#==================================================
#9. Final evaluation with best threshold
#==================================================
y_pred_best = (y_prob >= best_threshold).astype(int)

print("\nConfusion Matrix using Best Threshold:")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification Report using Best Threshold:")
print(classification_report(y_test, y_pred_best))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))
