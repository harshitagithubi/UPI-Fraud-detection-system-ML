
import pandas as pd

data = pd.read_csv("upi_fraud_dataset.csv")

# 1. Inspect repeated IDs
print("\nREPEATED IDs")
print(
    data[data['Id'].duplicated(keep=False)]
    .sort_values('Id')
)

# 2. Check number of records per category
print("\nCATEGORY COUNTS")
print(data['category'].value_counts().sort_index())

# 3. Check number of records per state
print("\nSTATE COUNTS")
print(data['state'].value_counts().sort_index())

# 4. Check ZIP code frequency
print("\nZIP CODE ANALYSIS")
print("Unique ZIP codes:", data['zip'].nunique())
print("ZIP codes occurring once:",
      (data['zip'].value_counts() == 1).sum())

# 5. Check fraud distribution by ZIP
print("\nZIP FRAUD COUNTS")
print(
    data.groupby('zip')['fraud_risk']
    .agg(['count', 'mean'])
    .sort_values('count', ascending=False)
    .head(15)
)

# 6. Check whether repeated IDs have conflicting labels
print("\nREPEATED ID LABELS")
print(
    data.groupby('Id')['fraud_risk']
    .nunique()
    .value_counts()
)

# 7. Check date and ID ordering
print("\nID AND DATE CORRELATION")
print(
    data[['Id', 'trans_month', 'trans_day']]
    .corr()
)
