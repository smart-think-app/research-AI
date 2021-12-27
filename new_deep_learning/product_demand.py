import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_excel("data.xlsx", sheet_name="product_aircd_demand")


class AirCDModel:
    def __init__(self,
                 features_training_pr,
                 label_training_pr,
                 feature_test_pr,
                 label_test_pr):
        self.features_training_pr = features_training_pr
        self.label_training_pr = label_training_pr
        self.feature_test_pr = feature_test_pr
        self.label_test_pr = label_test_pr
        self.model = None

    def build_model(self):
        model = keras.Sequential()
        model.add(tf.keras.layers.Dense(256, activation="relu", input_shape=(7,)))
        model.add(tf.keras.layers.Dense(4, activation="softmax"))
        model.summary()
        self.model = model

    def fit_model(self):
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=keras.optimizers.RMSprop(),
                           metrics=["accuracy"])
        self.model.fit(self.features_training_pr, self.label_training_pr, batch_size=100,
                       epochs=10, verbose=2,
                       validation_data=(self.feature_test_pr, self.label_test_pr))


features = df.columns[:7]
data_x = df[features]

scale = StandardScaler()
data_x = scale.fit_transform(data_x)

data_x_training = data_x[:50]
data_x_test = data_x[50:]

data_y = df["demand"]
data_y_training = data_y.iloc[:50]
data_y_test = data_y.iloc[50:]

data_y_training = data_y_training.to_numpy()
data_y_test = data_y_test.to_numpy()

data_y_training = keras.utils.to_categorical(data_y_training, 4)
data_y_test = keras.utils.to_categorical(data_y_test, 4)

airCDModel = AirCDModel(data_x_training, data_y_training,
                        data_x_test, data_y_test)
airCDModel.build_model()
airCDModel.fit_model()
x = data_x[1]

tmpX = np.array([[33, 410, 15000, 1, 2, 3, 3]])

tmpX = scale.transform(tmpX)
#
tmpX = tmpX.reshape(1, 7)
x = x.reshape(1, 7)
pre = airCDModel.model.predict(tmpX)
print(np.argmax(pre))
