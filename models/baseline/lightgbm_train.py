import pandas as pd
from sklearn.model_selection import train_test_split
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
#5. Model training with LightGBM
#==================================================
# Initialize LightGBM classifier
model = LGBMClassifier(
    random_state=42,
    class_weight='balanced',  
    n_estimators=200, 
    learning_rate=0.05,
)

# Train the model
model.fit(X_train, y_train)

#==================================================
#6. Model prediction
#==================================================
# Predict class labels
y_pred = model.predict(X_test)

# Predict probabilities for ROC-AUC
y_prob = model.predict_proba(X_test)[:, 1]

#==================================================
#7. Model evaluation
#==================================================
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))