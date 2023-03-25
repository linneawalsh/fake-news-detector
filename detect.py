import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def format_features(unformattedFeatures):
   ps = PorterStemmer()
   corpus = []
   for i in range(0, len(unformattedFeatures)):
      #review = re.sub('[^a-zA-Z]', ' ', unformattedFeatures.iloc[i, 0] + " " + unformattedFeatures.iloc[i, 2])
      review = re.sub('[^a-zA-Z]', ' ', unformattedFeatures.iloc[i, 0])
      review = review.lower()
      review = review.split()
      
      review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
      review = ' '.join(review)
      corpus.append(review)

   cv = CountVectorizer(min_df=2)
   feat = cv.fit_transform(corpus).toarray()
   return feat, cv

def format_labels(unformattedLabels):
   newLabels = np.zeros((len(unformattedLabels), 2))
   for i in range(0, len(unformattedLabels)):
      newLabels[i, unformattedLabels.iat[i]] = 1
   return pd.DataFrame(data=newLabels, index=unformattedLabels.index)

data = pd.read_csv("data.csv", index_col=0)
# data from https://www.kaggle.com/competitions/fake-news/overview
# 0 = reliable, 1 = unreliable

data = data.dropna()

features = data.iloc[:,:-1]
labels = data.iloc[:,-1]

feat, cv = format_features(features)
lab = format_labels(labels)

X_train, X_test, y_train, y_test = train_test_split(feat, lab, test_size = 0.1, random_state=21)

print(cv.get_feature_names_out()[0:20])