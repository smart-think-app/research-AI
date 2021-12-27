import pandas as pd
from sklearn import linear_model

df = pd.read_excel("Products.xlsx", sheet_name="ProductQuantity")
df = df.fillna(0)

mapLocation = {
    "HCM": 1
}

mapWeather = {
    "Hot": 1,
    "Cool": 2
}

df["Location"] = df["Location"].map(mapLocation)
df["Weather"] = df["Weather"].map(mapWeather)
print(df)
features = list(df.columns[:8])

Y = df["Quantity"]
X = df[features]

regr = linear_model.LinearRegression()
regr.fit(X, Y)

predicted = regr.predict([[1, 1, 1, 0, 0, 54, 9700, 10000]])
print(predicted)
