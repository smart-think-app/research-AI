import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1: thích # 2: giao hàng chậm trễ #4: sản phẩm không tốt #3  : Thái độ không tốt
df = pd.read_excel("Products.xlsx", sheet_name="BuyersComment")

vectorize = CountVectorizer()
counts = vectorize.fit_transform(df["Comment"].values)

classifier = MultinomialNB()

targets = df["Label"].values
classifier.fit(counts, targets)

text = ["Giao hàng chậm"]
text_tf = vectorize.transform(text)

prediction = classifier.predict(text_tf)
print(prediction)
