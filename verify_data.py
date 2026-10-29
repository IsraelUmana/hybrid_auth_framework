import pandas as pd
df = pd.read_csv("datasets/DSL-StrongPasswordData.csv")
assert len(df) == 20400, f"Wrong row count: {len(df)}"
assert df["subject"].nunique() == 51, f"Wrong subject count: {df['subject'].nunique()}"
print("Dataset verified: 20,400 rows, 51 subjects.")