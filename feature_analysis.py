
import pandas as pd

data = pd.read_csv("upi_fraud_dataset.csv")

# 1. Check duplicate records
print("\nDUPLICATE RECORDS")
print(data.duplicated().sum())

# 2. Check unique values in each column
print("\nUNIQUE VALUES")
print(data.nunique())

# 3. Check whether Id is unique
print("\nID DUPLICATES")
print(data['Id'].duplicated().sum())

# 4. Check UPI identifier uniqueness
print("\nUPI IDENTIFIER")
print("Unique UPI numbers:", data['upi_number'].nunique())

# 5. Fraud distribution by category
print("\nFRAUD BY CATEGORY")
print(pd.crosstab(
    data['category'],
    data['fraud_risk'],
    normalize='index'
).round(3))

# 6. Fraud distribution by state
print("\nFRAUD BY STATE")
print(pd.crosstab(
    data['state'],
    data['fraud_risk'],
    normalize='index'
).round(3))

# 7. Average transaction amount by fraud class
print("\nAVERAGE TRANSACTION AMOUNT")
print(data.groupby('fraud_risk')['trans_amount'].mean())

# 8. Fraud distribution across transaction IDs
print("\nFRAUD DISTRIBUTION BY ID RANGE")
data['id_group'] = pd.cut(data['Id'], bins=5)

print(pd.crosstab(
    data['id_group'],
    data['fraud_risk'],
    normalize='index'
).round(3))
