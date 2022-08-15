# Before NORMALIZATION/SCALING
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
x1 = np.random.randint(1,50,30)
x1 = np.sort(x1)
x2 = np.random.randint(10000,70000,30)
plt.plot(x1,x2)

# after NORMALIZATION/SCALING
x1min = min(x1)
x1max = max(x1)
x2min = min(x2)
x2max = max(x2)
x1norm = (x1-x1min)/(x1max-x1min)
x2norm = (x2-x2min)/(x2max-x2min)
plt.plot(x1norm,x2norm)

#LOGISTIC REGRESSION - IT IS A CLASSIFICATION TECHNIQUE
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/Social_Network_Ads.csv')
df
df.info()

#divide into i/p and o/p
x = df.iloc[:,2:4].values #2 dimensional
y = df.iloc[:,4].values #1 dimensional
#I want to count how many have purchased and how many have not purchased
df['Purchased'].value_counts()
#train and test variables
#when ever you take traintestsplit,you get 4 variables
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)
x.shape
x_train.shape
x_test.shape

#scaling or normalization(ONLY FOR INPUTS)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#apply REGRESSOR/CLASSIFIER
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#fitting the model
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
y_test

#to check the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)*100

#SUPPORT VECTOR MACHINE
#https://github.com/diazoniclabs - for dataset
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/diazoniclabs/Machine-Learning-using-sklearn/master/Datasets/spam.tsv',sep ='\t')
df
df.info()
#the count of spam and ham messages
df['label'].value_counts()

#VISUALIZATION
df['label'].value_counts().plot(kind = 'bar')
#divide into i/p and o/p
x = df['message'].values #- only here in SVM ,x is 1 dimensional
y = df.iloc[:,0].values
#train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)

#Apply TFIDF Vectorizer - Very imp for text based dataset
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
x_train_v = vect.fit_transform(x_train)
x_test_v = vect.transform(x_test)
#apply svm OR SVC
from sklearn.svm import SVC
model = SVC()
model.fit(x_train_v,y_train)
y_pred =model.predict(x_test_v)
y_pred
y_test
#accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)*100
text ='Win free tickets'
text = vect.transform([text])
model.predict(text)
text ='Hi there'
text = vect.transform([text])
model.predict(text)