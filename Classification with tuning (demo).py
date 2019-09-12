#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import logging
from numpy import random
#import gensim
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("preProcessedData.csv")
# fraction of rows
# here you get 75% of the rows
train = data.sample(frac=0.75, random_state=99)
test = data.loc[~data.index.isin(train.index), :]


# In[3]:


# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train['Text'].values.astype('U'))
X_train_counts.shape


# In[4]:


# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[6]:


# Machine Learning
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, train.Label)
# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(train['Text'].values.astype('U'), train.Label)
# Performance of NB Classifier
predicted = text_clf.predict(test['Text'].values.astype('U'))
np.mean(predicted == test.Label)


# In[8]:


# Grid Search
# Here, we are creating a list of parameters for which we would like to do performance tuning. 
# All the parameters name start with the classifier name (remember the arbitrary name we gave). 
# E.g. vect__ngram_range; here we are telling to use unigram and bigrams and choose the one which is optimal.

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
# Next, we create an instance of the grid search by passing the classifier, parameters 
# and n_jobs=-1 which tells to use multiple cores from user machine.

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(train['Text'].values.astype('U'), train['Label'].values.astype('U'))

# To see the best mean score and the params, run the following code

gs_clf.best_score_
gs_clf.best_params_
predicted = gs_clf.predict(test['Text'].values.astype('U'))
np.mean(predicted == test.Label)


# In[9]:


get_ipython().run_cell_magic('time', '', "\nprint('accuracy %s' % accuracy_score(predicted, test.Label))\nprint(classification_report(test.Label, predicted,target_names=tags))")


# In[11]:


# Training Support Vector Machines - SVM and calculating its performance

from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=15, random_state=42))])

text_clf_svm = text_clf_svm.fit(train['Text'].values.astype('U'), train['Label'].values.astype('U'))
predicted_svm = text_clf_svm.predict(test['Text'].values.astype('U'))
np.mean(predicted_svm == test['Label'].values.astype('U'))


# In[13]:


# Similarly doing grid search for SVM
from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(train['Text'].values.astype('U'), train['Label'].values.astype('U'))


gs_clf_svm.best_score_
gs_clf_svm.best_params_
predicted_svm = gs_clf_svm.predict(test['Text'].values.astype('U'))
np.mean(predicted_svm == test['Label'].values.astype('U'))


# In[14]:


get_ipython().run_cell_magic('time', '', "from sklearn.metrics import classification_report\n\nprint('accuracy %s' % accuracy_score(predicted_svm, test.Label))\nprint(classification_report(test.Label, predicted_svm,target_names=tags))")


# In[9]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
text_clf_lsvc = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf-lsvc', LinearSVC())])
text_clf_lsvc = text_clf_lsvc.fit(train['Text'].values.astype('U'), train['Label'].values.astype('U'))

parameters_lsvc = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False)}

gs_clf_lsvc= GridSearchCV(text_clf_lsvc, parameters_lsvc, n_jobs=-1)
gs_clf_lsvc = gs_clf_lsvc.fit(train['Text'].values.astype('U'), train['Label'].values.astype('U'))


gs_clf_lsvc.best_score_
gs_clf_lsvc.best_params_
predicted_lsvc = gs_clf_lsvc.predict(test['Text'].values.astype('U'))
np.mean(predicted_lsvc == test['Label'].values.astype('U'))


# In[11]:


get_ipython().run_cell_magic('time', '', "tags = data.Label.unique()\nfrom sklearn.metrics import classification_report\n\nprint('accuracy %s' % accuracy_score(predicted_lsvc, test.Label))\nprint(classification_report(test.Label, predicted_lsvc,target_names=tags))")


# In[ ]:




