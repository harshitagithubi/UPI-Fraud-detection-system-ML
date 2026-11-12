
import pandas as pd

# Load dataset
data = pd.read_csv("upi_fraud_dataset.csv")

# 1. Dataset dimensions
print("Dataset Shape:", data.shape)

# 2. Column information
print("\nColumn Information:")
print(data.info())

# 3. First five records
print("\nFirst Five Records:")
print(data.head())

# 4. Missing values
print("\nMissing Values:")
print(data.isnull().sum())

# 5. Fraud distribution
print("\nFraud Distribution:")
print(data['fraud_risk'].value_counts())

# 6. Fraud percentages
print("\nFraud Percentages:")
print(data['fraud_risk'].value_counts(normalize=True) * 100)

# 7. Statistical summary
print("\nStatistical Summary:")
print(data.describe(include='all'))
