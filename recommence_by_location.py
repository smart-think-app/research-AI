import pandas as pd

df_weather = pd.read_excel("Products.xlsx", sheet_name="Weather")
df_purchase_weather = pd.read_excel("Products.xlsx", sheet_name="ProductPurchaseOnWeather")
df_products_base = pd.read_excel("Products.xlsx", sheet_name="ProductsDetail")
# df_weather.sort_values(ascending=)

arg_weather = "Cold"
arg_location = "HCM"

df_weather = df_weather[(df_weather["Weather"] == arg_weather) & (df_weather["Location"] == arg_location)]
date_arr = df_weather["Date"]
df_filter_weather = df_purchase_weather[(df_purchase_weather["Date"].isin(date_arr)) &
                                        (df_purchase_weather["Location"] == arg_location)]

df_filter_weather = df_filter_weather[["ProductId", "Quantity", "Search", "ViewDuration"]]
df_filter_weather = df_filter_weather.groupby("ProductId").sum().reset_index()
df_filter_weather = df_filter_weather.sort_values(["Quantity", "Search", "ViewDuration"], ascending=False)
df_filter_weather = pd.merge(df_filter_weather, df_products_base[["ProductId", "CategoryId", "Ads"]], on="ProductId")
df_filter_weather = df_filter_weather.fillna(0)
df_filter_weather = df_filter_weather[["Quantity", "Search", "ViewDuration", "CategoryId"]]
df_filter_weather = df_filter_weather.groupby("CategoryId").sum().reset_index()
df_filter_weather = df_filter_weather.sort_values(["Quantity", "Search", "ViewDuration"], ascending=False)
df_total = df_filter_weather[["Quantity"]].sum()
df_filter_weather = df_filter_weather.assign(percent_quantity=lambda x: x.Quantity / df_total["Quantity"] * 100)
print(df_filter_weather)