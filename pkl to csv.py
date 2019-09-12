#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import pickle
from collections import Counter

import os


# In[35]:


with open('D:/Capstone/Dataset/40k_bangla_newspaper_article.p', 'rb') as f:
    data = pickle.load(f)
pik=pd.read_pickle('D:/Capstone/Dataset/40k_bangla_newspaper_article.p')
pik=pd.DataFrame(pik)


# In[44]:


pik.columns = ['Label','Text','Title']
pik.to_csv("data.csv", index=False, header=True)


# In[45]:


pik.to_csv(r'C:\\Users\\Jenny\\Downloads\\data.csv',index=False, header=['Label','Text','Title'])

