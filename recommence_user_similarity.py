import pandas as pd

user_id_corr = 1
lenDF = 7
df = pd.read_excel("Products.xlsx", sheet_name="UserSimilarity")
df = df.assign(is_view=5)
df_user = df[df["UserId"] == user_id_corr]
buying_df = df.pivot_table(index=["ProductId"], columns=["UserId"], values="is_view")
buying_df = buying_df.fillna(0)
corr_matrix = buying_df.corr(method="pearson", min_periods=1)
print(corr_matrix)
tmp = corr_matrix[user_id_corr]
products_series = pd.Series()
for i in range(0, len(tmp.index)):
    if tmp.index[i] == user_id_corr:
        continue
    else:
        value = tmp[tmp.index[i]]
        if value > 0.3:
            products_series = products_series.append(buying_df[tmp.index[i]])

products_series = products_series.groupby(products_series.index).sum()
product_recommence_arr = []
for i in range(0, len(products_series.index)):
    product_id = products_series.index[i]
    if products_series[product_id] >= 5 and (product_id not in df_user["ProductId"].to_numpy()):
        product_recommence_arr.append(product_id)

print(product_recommence_arr)
