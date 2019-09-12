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

# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_tfidf.shape


# In[4]:


from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
text_clf_lsvc = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf-lsvc', LinearSVC())])
text_clf_lsvc = text_clf_lsvc.fit(train['Text'].values.astype('U'), train['Label'].values.astype('U'))

#Predict the response for test dataset
predicted_lsvc = text_clf_lsvc.predict(test['Text'].values.astype('U'))
np.mean(predicted_lsvc == test['Label'].values.astype('U'))


# In[5]:


get_ipython().run_cell_magic('time', '', "tags = data.Label.unique()\nfrom sklearn.metrics import classification_report\n\nprint('accuracy %s' % accuracy_score(predicted_lsvc, test.Label))\nprint(classification_report(test.Label, predicted_lsvc,target_names=tags))")


# In[7]:


from sklearn.model_selection import GridSearchCV

parameters_lsvc = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False)}

gs_clf_lsvc= GridSearchCV(text_clf_lsvc, parameters_lsvc, n_jobs=-1)
gs_clf_lsvc = gs_clf_lsvc.fit(train['Text'].values.astype('U'), train['Label'].values.astype('U'))


gs_clf_lsvc.best_score_
gs_clf_lsvc.best_params_
predicted_lsvc = gs_clf_lsvc.predict(test['Text'].values.astype('U'))
np.mean(predicted_lsvc == test['Label'].values.astype('U'))


# In[8]:


get_ipython().run_cell_magic('time', '', "tags = data.Label.unique()\nfrom sklearn.metrics import classification_report\n\nprint('accuracy %s' % accuracy_score(predicted_lsvc, test.Label))\nprint(classification_report(test.Label, predicted_lsvc,target_names=tags))")


# In[21]:


from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(test['Label'], predicted_lsvc)
conf_mat


# In[ ]:




