import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
music_data=pd.read_csv('music.csv')
#print (df)
#print(df.shape)
#df.describe
#df.values
X=music_data.drop(columns=('genre'))
y=music_data['genre']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
#pickle.dump(model,'music_recommendation')
preds=model.predict(X_test)
#print (preds)
score=accuracy_score(y_test,preds)
#print (score)