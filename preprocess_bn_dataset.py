#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[83]:


data = pd.read_csv("data.csv") 


# In[84]:


data.shape


# In[85]:


print(data.groupby('Label').size())


# In[86]:


plt.figure(figsize=(10,4)) # whole dataset visualisation
data.Label.value_counts().plot(kind='bar');


# In[87]:


p = ["!", "@",'–', "#", "|", "%", "(", ")", "।", "—", ".", "-", "", ",", "’", "•", "‘", ":", "*", "?",
          "০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯"]
for i in range(len(p)):
    data['Text'] = data['Text'].str.replace(p[i],'')


# In[88]:


data = data.drop_duplicates() #removes duplicate instances
data.dropna(inplace=True)
data.isnull().sum() #removes missing info instances
data.shape


# In[89]:


tags = data['Label'].unique()
tag_size = data['Label'].value_counts()
#print(data['Label'].unique())
tags


# In[90]:


# removing less instances categories
data = data[data.Label != 'art-and-literature']
data = data[data.Label != 'durporobash']
data = data[data.Label != 'northamerica']
data.shape


# In[91]:


print (data['Label'].value_counts()) #minimum instances number = 740


# In[92]:


# sampling each category with minimum number of instances
d_bn = data[data.Label == 'bangladesh']
d_op = data[data.Label == 'opinion']
d_ec = data[data.Label == 'economy']
d_sp = data[data.Label == 'sports']
d_en = data[data.Label == 'entertainment']
d_tc = data[data.Label == 'technology']
d_in = data[data.Label == 'international']
d_ls = data[data.Label == 'life-style']
d_ed = data[data.Label == 'education']

random_bn = d_bn.sample(n=740, replace=False, random_state=99)
random_op = d_op.sample(n=740, replace=False, random_state=99)
random_ec = d_ec.sample(n=740, replace=False, random_state=99)
random_sp = d_sp.sample(n=740, replace=False, random_state=99)
random_en = d_en.sample(n=740, replace=False, random_state=99)
random_tc = d_tc.sample(n=740, replace=False, random_state=99)
random_in = d_in.sample(n=740, replace=False, random_state=99)
random_ls = d_ls.sample(n=740, replace=False, random_state=99)


# In[93]:


newData = pd.concat([random_bn,random_op,random_ec,random_sp,random_en,random_tc,random_in,random_ls,d_ed])
newData.shape


# In[94]:


plt.figure(figsize=(10,4))
newData.Label.value_counts().plot(kind='bar');


# In[95]:


newData['Text'].apply(lambda x: len(x.split(' '))).sum()


# In[96]:


preProcessedData=pd.DataFrame(newData)
preProcessedData.columns = ['Label','Text','Title']
preProcessedData.to_csv("preProcessedData.csv", index=False, header=True)

