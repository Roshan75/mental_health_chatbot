import string

import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from wordcloud import WordCloud as wc
import matplotlib.pyplot as plt
import spacy

import nltk
from nltk.stem import WordNetLemmatizer, RSLPStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras import Model
from imblearn.over_sampling import SMOTE

nltk.download('punkt')
nltk.download('stopwards')
nltk.download('preprocess_texts')

if __name__ == '__main__':


    df = pd.read_csv('../datasets/raw/Combined_Data.csv')
    # print(df.head())
    # print(df['statement'].isnull().sum())
    df = df.dropna()
    # print(df.isnull().sum())
    # print(df['status'].unique(), df['status'].nunique())
    sentiment_counts = df['status'].value_counts()
    # print(sentiment_counts)

    df['statement_length'] = df['statement'].apply(len)

    # print(df['statement_length'])
    q1 = df['statement_length'].quantile(0.25)
    q3 = df['statement_length'].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr

    filtered_df = df[(df['statement_length'] >= lower_bound) & (df['statement_length'] <= upper_bound)]

    def generate_word_cloud(text, title):
        wordcloud = wc(width=800, height=400).generate(text)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title)
        plt.axis('off')
        plt.show()

    statuses = df['status'].unique()

    for status in statuses:
        status_text = ' '.join(df[df['status'] == status]['statement'])
        # generate_word_cloud(status_text, title=status)

    sample_size = 20 #20000
    df_sample = df.sample(n=sample_size, random_state=1)

    nlp = spacy.load("en_core_web_sm")

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = text.lower()

        text = text.translate(str.maketrans('', '', string.punctuation))

        doc = nlp(text)

        tokens = [token.lemma_ for token in doc if not token.is_stop]

        return ' '.join(tokens)

    def preprocess_texts(texts):
        return [preprocess_text(text) for text in texts]

    # num_cores = cpu_count()
    # df_split = np.array_split(df_sample, 1)
    #
    # with Pool(1) as pool:
    #     results = pool.map(preprocess_texts, [batch['statement'].tolist() for batch in df_split])
    #
    #
    # df_sample['cleaned_statement'] = [item for sublist in results for item in sublist]

    # print(df_sample[['statement', 'cleaned_statement']])

    df_split = np.array_split(df_sample, 1)
    results = [preprocess_texts(batch['statement'].tolist()) for batch in df_split]
    df_sample['cleaned_statement'] = [item for sublist in results for item in sublist]

    processedtext = df_sample['cleaned_statement']
    sentiment = df_sample['status']

    X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment, train_size=0.05, random_state=0)

    print(f'X_train size: {len(X_train)}')
    print(f'X_test size: {len(X_test)}')
    print(f'y_train size: {len(y_train)}')
    print(f'y_test size: {len(y_test)}')

    vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
    vectoriser.fit(X_train)

    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)

    smote = SMOTE(random_state=0)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
    }

    bnb = BernoulliNB()

    grid_search = GridSearchCV(estimator=bnb, param_grid=param_grid, cv=5, scoring='accuracy')

    grid_search.fit (X_train_resampled, y_train_resampled)

    best_bnb = grid_search.best_estimator_
    best_bnb.fit(X_train_resampled, y_train_resampled)

    y_pre_best_bnb = best_bnb.predict(X_test)
    print(classification_report(y_test, y_pre_best_bnb))



