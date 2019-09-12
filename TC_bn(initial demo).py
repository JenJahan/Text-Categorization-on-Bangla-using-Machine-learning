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


# In[3]:


data = pd.read_csv("data.csv") 


# In[4]:


data.dropna(inplace=True)
data.isnull().sum()
data.shape


# In[6]:


print (data['Label'].value_counts())


# In[104]:


plt.figure(figsize=(10,4))
data.Label.value_counts().plot(kind='bar');


# In[105]:


tags = data.Label.unique()
tags


# In[106]:


p = ["!", "@",'–', "#", "|", "%", "(", ")", "।", "—", ".", "-", "", ",", "’", "•", "‘", ":", "*", "?",
          "০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯"]
for i in range(len(p)):
    data['Text'] = data['Text'].str.replace(p[i],'')


# In[107]:


data['Text'].apply(lambda x: len(x.split(' '))).sum()


# In[108]:


data['split'] = np.random.randn(data.shape[0], 1)
msk = np.random.rand(len(data)) <= 0.25
d1 = data[msk]
d2 = data[~msk]


# In[109]:


# fraction of rows
# here you get 75% of the rows
train = d1.sample(frac=0.75, random_state=99)
test = d1.loc[~d1.index.isin(train.index), :]


# In[112]:


train_size = train.shape[0]
test_size  = test.shape[0]


# In[113]:


train_size, test_size


# In[114]:


# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train['Text'].values.astype('U'))
X_train_counts.shape


# In[115]:


# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[121]:


# Machine Learning
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, train.Label)


# In[122]:


# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(train['Text'].values.astype('U'), train.Label)


# In[123]:


# Performance of NB Classifier
predicted = text_clf.predict(test['Text'].values.astype('U'))
np.mean(predicted == test.Label)


# In[124]:


get_ipython().run_cell_magic('time', '', "from sklearn.metrics import classification_report\n\nprint('accuracy %s' % accuracy_score(predicted, test.Label))\nprint(classification_report(test.Label, predicted,target_names=tags))")


# In[125]:


# Training Support Vector Machines - SVM and calculating its performance

from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=15, random_state=42))])

text_clf_svm = text_clf_svm.fit(train['Text'].values.astype('U'), train['Label'].values.astype('U'))
predicted_svm = text_clf_svm.predict(test['Text'].values.astype('U'))
np.mean(predicted_svm == test['Label'].values.astype('U'))


# In[126]:


get_ipython().run_cell_magic('time', '', "print('accuracy %s' % accuracy_score(predicted_svm, test.Label))\nprint(classification_report(test.Label, predicted_svm,target_names=tags))")


# In[129]:


# Grid Search
# Here, we are creating a list of parameters for which we would like to do performance tuning. 
# All the parameters name start with the classifier name (remember the arbitrary name we gave). 
# E.g. vect__ngram_range; here we are telling to use unigram and bigrams and choose the one which is optimal.

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}


# In[130]:


# Next, we create an instance of the grid search by passing the classifier, parameters 
# and n_jobs=-1 which tells to use multiple cores from user machine.

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(train['Text'].values.astype('U'), train['Label'].values.astype('U'))


# In[131]:



# To see the best mean score and the params, run the following code

gs_clf.best_score_
gs_clf.best_params_


# In[134]:


predicted = gs_clf.predict(test['Text'].values.astype('U'))
np.mean(predicted == test.Label)


# In[135]:


get_ipython().run_cell_magic('time', '', "\nprint('accuracy %s' % accuracy_score(predicted, test.Label))\nprint(classification_report(test.Label, predicted,target_names=tags))")


# In[136]:



# Similarly doing grid search for SVM
from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(train['Text'].values.astype('U'), train['Label'].values.astype('U'))


gs_clf_svm.best_score_
gs_clf_svm.best_params_


# In[137]:


predicted_svm = gs_clf_svm.predict(test['Text'].values.astype('U'))
np.mean(predicted_svm == test['Label'].values.astype('U'))


# In[138]:


get_ipython().run_cell_magic('time', '', "from sklearn.metrics import classification_report\n\nprint('accuracy %s' % accuracy_score(predicted_svm, test.Label))\nprint(classification_report(test.Label, predicted_svm,target_names=tags))")


# In[139]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# Create Decision Tree classifer object
text_clf_dt = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-dt', DecisionTreeClassifier(criterion='gini',splitter='best',
                                                           max_depth=20))])

# Train Decision Tree Classifer
text_clf_dt = text_clf_dt.fit(train['Text'].values.astype('U'), train['Label'].values.astype('U'))

#Predict the response for test dataset
predicted_dt = text_clf_dt.predict(test['Text'].values.astype('U'))
np.mean(predicted_dt == test['Label'].values.astype('U'))


# In[140]:


get_ipython().run_cell_magic('time', '', "from sklearn.metrics import classification_report\n\nprint('accuracy %s' % accuracy_score(predicted_dt, test.Label))\nprint(classification_report(test.Label, predicted_dt,target_names=tags))")


# In[143]:


from sklearn.linear_model import LogisticRegression

text_clf_lr = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf-lr', LogisticRegression(n_jobs=1, C=1e5)),
               ])
text_clf_lr = text_clf_lr.fit(train['Text'].values.astype('U'), train['Label'].values.astype('U'))

#Predict the response for test dataset
predicted_lr = text_clf_lr.predict(test['Text'].values.astype('U'))
np.mean(predicted_lr == test['Label'].values.astype('U'))


# In[144]:


get_ipython().run_cell_magic('time', '', "from sklearn.metrics import classification_report\n\nprint('accuracy %s' % accuracy_score(predicted_lr, test.Label))\nprint(classification_report(test.Label, predicted_lr,target_names=tags))")


# In[ ]:




