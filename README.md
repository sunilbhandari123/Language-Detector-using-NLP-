
# Language Identification using-NLP

This project focuses on identifying different languages such as English, Spanish, Hindi, and others. The dataset, sourced from Kaggle, contains 10,267 text samples. I implemented a Naive Bayes classifier (MultinomialNB) for this task and achieved an accuracy of 98%.

**Link to Website :**https://languagedetectionnlp.streamlit.app/
## Table of Content

1. Importing all the essiantial libraries.
2. Loading the data from kaggle.
3. Data Exploration and cleaning.
4. Dependent Variable vs Independent Variable.
5. Bag of Words.
6. Dividing the dataset.
7. Train test Split
8. Model and prediction
9. Translating
## Table of Content

1. Importing all the essiantial libraries.
2. Loading the data from kaggle.
3. Data Exploration and cleaning.
4. Dependent Variable vs Independent Variable.
5. Bag of Words.
6. Dividing the dataset.
7. Train test Split
8. Model and prediction
9. Translating
## Importing all essential libraries

```http
- import pandas as pd
- import numpy as np
- import re
- import seaborn as sns
- import matplotlib.pyplot as plt
- import pickle
- import warnings
- warnings.filterwarnings("ignore")
- from sklearn.model_selection import train_test_split
```

## Loading datasets form kaggle


```http
- !pip install opendatasets
- import opendatasets as od
- od.download('https://www.kaggle.com/datasets/basilb2s/language-detection?resource=download')
- data= pd.read_csv("/content/language-detection/Language Detection.csv")
```
## Data exploration and cleaning

```http
data.Language.value_counts()
data.isnull().sum()
data.dtypes
```
## Dependent variable vs Independent Variable


```http
x=data["Text"]
y=data['Language']
```
## Bag of Words.

```http
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(x).toarray()
```

## Train Test Split

```http
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
```
## Model and Prediction

```http
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train,y_train)

model.score(X_test,y_test)
```
## Identification (Screenshot)

![screenshot](https://github.com/sunilbhandari123/Language-Detector-using-NLP-/blob/main/Screenshot%202025-08-06%20185106.png)

##  Deployment Using Streamlit
``` http import streamlit as st
import joblib


model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🌐 Language Detection App")
st.write("Detects whether a sentence is in English,Malayalam,Hindi,Tamil, Kannada, French, Spanish, Portuguese, Italian, Sweedish, Dutch, Arabic, Turkish, German, Danish, Greek.")





user_input = st.text_input("Enter a sentence:")

if user_input:
    vectorized_text = vectorizer.transform([user_input])
    prediction = model.predict(vectorized_text)
    st.success(f"Predicted Language: **{prediction[0]}**")```