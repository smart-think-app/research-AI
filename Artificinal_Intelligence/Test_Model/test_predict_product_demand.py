import numpy as np
import pickle

model = pickle.load(open("../model/predict_total_buy.sav", 'rb'))
scale = pickle.load(open("../model/predict_total_buy_scale.sav", 'rb'))
features = np.array([[94, -13, 7000000]])
value = model.predict(scale.transform(features))
print(np.round(value))
