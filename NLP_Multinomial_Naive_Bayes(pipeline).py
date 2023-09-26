import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# lendo o dataset
df = pd.read_csv("IMDB Dataset.csv", encoding='utf-8')

#print(df.shape)

#print(df.head())
#criando mais uma coluna no dataset para representar "positivo" como (1) e negativo como (0)
df['binsentiment'] = df['sentiment'].apply(lambda x: 1 if x =='positive' else 0)
#print(df.head())

#contado a quantidade de reviews positivas e negativas
#df.binsentiment.value_counts()

#formalizando o dataset para ser usado posteriormente
X_train, X_test, y_train, y_test = train_test_split(df.review, df.binsentiment, test_size=0.2)


#realizando processo de embeding e instanciando o modelo
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))