import pandas as pd

#==================================================
#1. Load data
#==================================================
# Read dataset
df = pd.read_csv("/Users/joannajorise/Desktop/FYP_Project/data/raw/credit_card_offer_acceptance.csv")

# Preview
print("Original dataset shape:", df.shape)
print("\nOriginal dataset preview:")
print(df.head())

#==================================================
# 2. Drop irrelevant columns
#==================================================
# 'index' and 'Customer Number' are identifiers
df = df.drop(columns=['index', 'Customer Number'])

#==================================================
# 3. Create target variable
#==================================================
# Convert 'Offer Accepted' to binary target variable
# Yes = 1, customers who accepted the offer
# No = 0, customers who did not accept the offer
df['target'] = df['Offer Accepted'].map({
    'Yes': 1, 
    'No': 0
    })

#Drop the original 'Offer Accepted' column as it's now redundant
df = df.drop(columns=['Offer Accepted'])

#==================================================
#4. Handle missing values
#==================================================
# The balance-related columns have missing values 
# Median imputation is used as balance values are numerical and may have outliers
balance_columns = [
    'Average Balance',
    'Q1 Balance',
    'Q2 Balance',
    'Q3 Balance',
    'Q4 Balance'
]

for col in balance_columns:
    df[col] = df[col].fillna(df[col].median())

#==================================================
# 5. Final checks
#==================================================
print("\nProcessed dataset preview:")
print(df.head())

print("\nTarget variable distribution:")
print(df['target'].value_counts())

print("\nTarget variable distribution (percentage):")
print(df['target'].value_counts(normalize=True) * 100)

print("\nMissing values per columnn:")
print(df.isnull().sum())

print("\nFinal dataset shape:")
print("Number of rows:", len(df)) 
print("Number of columns:", len(df.columns))

print("\nFinal dataset columns:")
print(df.columns)
#==================================================
#6. Save processed dataset
#==================================================
df.to_csv("/Users/joannajorise/Desktop/FYP_Project/data/processed/final_dataset.csv", index=False)

print("\nProcessed dataset saved successfully to /Users/joannajorise/Desktop/FYP_Project/data/processed/final_dataset.csv")