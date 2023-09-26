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


#criando mais uma coluna no dataset para representar "positivo" como (1) e negativo como (0)
df['binsentiment'] = df['sentiment'].apply(lambda x: 1 if x =='positive' else 0)


#contado a quantidade de reviews positivas e negativas
#df.binsentiment.value_counts()

#formalizando o dataset para ser usado posteriormente
X_train, X_test, y_train, y_test = train_test_split(df.review, df.binsentiment, test_size=0.2)

#criando instancia de objeto
v = CountVectorizer()

#transformando as reviews em vetores de numeros (encoding)
X_train_cv = v.fit_transform(X_train)


#criando instancia de RandomForest
model = RandomForestClassifier(n_estimators=50, criterion='entropy')


#treinando o modelo com os dados
model.fit(X_train_cv, y_train)

#encoding dos dados de teste
X_test_cv = v.transform(X_test)

y_pred = model.predict(X_test_cv)

print(classification_report(y_test, y_pred))

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]

emails_count = v.transform(emails)
print(model.predict(emails_count))


clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])