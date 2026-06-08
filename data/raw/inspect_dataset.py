import pandas as pd

df = pd.read_csv("/Users/joannajorise/Desktop/FYP_Project/data/raw/credit_card_offer_acceptance.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
print(df.isnull().sum())
print(df['Offer Accepted'].value_counts())
print(df['Offer Accepted'].value_counts(normalize=True))