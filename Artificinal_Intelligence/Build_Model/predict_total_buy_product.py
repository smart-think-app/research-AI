import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import numpy as np
import pickle
import sklearn
scale = StandardScaler()

df_product = pd.read_excel("../DataTrain.xlsx", sheet_name="products_total_buy")
X = df_product[["famous_point", "cost_diffenrence_product_replace", "total_buyer_active"]]
Y = df_product["total_quantity"]

scaleX = scale.fit_transform(X)

model = linear_model.LinearRegression()
model.fit(scaleX, Y)
pickle.dump(model, open('../model/predict_total_buy.sav', 'wb'))
pickle.dump(scale, open('../model/predict_total_buy_scale.sav', 'wb'))
# model = pickle.load(open("../model/predict_total_buy.sav", 'rb'))
# scale = pickle.load(open("../model/predict_total_buy_scale.sav", 'rb'))
# features = np.array([[94, -13, 7000000]])
# value = model.predict(scale.transform(features))
# print(np.round(value))
