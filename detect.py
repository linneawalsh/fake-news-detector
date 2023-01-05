import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("data.csv", index_col=0)
# data from https://www.kaggle.com/competitions/fake-news/overview
# 0 = reliable, 1 = unreliable

data = data.dropna()

features = data.iloc[:,:-1]
labels = data.iloc[:,-1]

ps = PorterStemmer()
corpus = []
for i in range(0, len(features)):
   #review = re.sub('[^a-zA-Z]', ' ', features.iloc[i, 0] + " " + features.iloc[i, 2])
   review = re.sub('[^a-zA-Z]', ' ', features.iloc[i, 0])
   review = review.lower()
   review = review.split()
    
   review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
   review = ' '.join(review)
   corpus.append(review)

cv = CountVectorizer()
feat = cv.fit_transform(corpus).toarray()

print(cv.get_feature_names()[:20])