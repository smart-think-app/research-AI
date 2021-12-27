import pandas as pd

df = pd.read_excel("Products.xlsx", sheet_name="BuyersShop")
print(df)
df = df[["TotalOrder", "TotalAmount", "Location"]].groupby(["Location"]).sum()
df = df.sort_values(["TotalOrder", "TotalAmount"], ascending=False)
print(df)
