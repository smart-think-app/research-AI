import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_excel("../DataTrain.xlsx", sheet_name="comment")
print(df)

vectorize = CountVectorizer()
counts = vectorize.fit_transform(df["comment"].values)
classifier = MultinomialNB()

targets = df["label"].values
classifier.fit(counts, targets)
pickle.dump(classifier, open('../model/predict_comment.sav', 'wb'))
pickle.dump(vectorize, open('../model/predict_comment_vector.sav', 'wb'))
# text = ["Hàng kém"]
# text_tf = vectorize.transform(text)
#
# prediction = classifier.predict(text_tf)
# print(prediction)