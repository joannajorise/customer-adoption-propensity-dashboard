import pandas as pd

df = pd.read_csv("/Users/joannajorise/Desktop/FYP_Project/data/processed/final_dataset.csv")

balance_cols = [
    "Average Balance",
    "Q1 Balance",
    "Q2 Balance",
    "Q3 Balance",
    "Q4 Balance"
]

print(df[balance_cols].median())