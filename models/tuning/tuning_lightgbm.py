import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier

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
#4. Calculate class imbalance ratio
#==================================================
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) # Calculate scale_pos_weight for imbalanced data
print("Scale pos weight:", scale_pos_weight)

#==================================================
#5. Define LightGBM model
#==================================================
# Initialize LightGBM classifier
model = LGBMClassifier(
    random_state=42,
    class_weight='balanced', 
    verbose=-1
)

#==================================================
#6. Define parameter grid for hyperparameter tuning
#=================================================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, -1],  # -1 means no limit
    'learning_rate': [0.03, 0.05, 0.1],
    'num_leaves': [15, 31]
}

#==================================================
#7. Perform Grid Search with cross-validation
#==================================================
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=1
)

#Train the model with grid search
grid_search.fit(X_train, y_train)

#==================================================
#8. Display best parameters
#==================================================
print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(grid_search.best_score_)

#==================================================
#9. Evaluate best model
#==================================================
best_lightgbm = grid_search.best_estimator_

#Make predictions on the test set
y_pred = best_lightgbm.predict(X_test)

#Get predicted probabilities for ROC-AUC
y_prob = best_lightgbm.predict_proba(X_test)[:, 1]

#==================================================
#10.  Model evaluation
#==================================================
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))
