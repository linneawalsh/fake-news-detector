import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# convert DataFrame of features to bag of words model
def format_features(unformattedFeatures):
   ps = PorterStemmer()
   corpus = []
   for i in range(0, len(unformattedFeatures)):
      review = re.sub('[^a-zA-Z]', ' ', unformattedFeatures.iloc[i, 0])
      review = review.lower()
      review = review.split()
      
      review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
      review = ' '.join(review)
      corpus.append(review)

   cv = CountVectorizer(min_df=5)
   feat = cv.fit_transform(corpus).toarray()
   return feat, cv

data = pd.read_csv("cleanData.csv", index_col=0)
# data from https://www.kaggle.com/competitions/fake-news/overview
# then unneccesary columns removed
# 0 = reliable, 1 = unreliable

data = data.dropna()

features = data.iloc[:,:-1]
feat, cv = format_features(features)

lab = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(feat, lab, test_size = 0.1)

model = Sequential([
   Dense(250, activation='relu'),
   Dropout(0.3),
   Dense(50, activation='relu'),
   Dropout(0.3),
   Dense(10, activation='relu'),
   Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, verbose=2)

loss, acc = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy on test set: " + str(acc*100))
