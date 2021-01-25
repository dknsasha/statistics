#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats


# In[3]:


data = pd.read_csv("winemag-data-130k-v2.csv")
data.head()


# In[4]:


df = data[[ 'country', 'description', 'designation', 'points', 'price', 'province', 'title', 'variety', 'winery']]
df.to_csv('wine.csv',index=False)


# In[5]:


dataNew = pd.read_csv("wine.csv")
dataNew.head()


# In[6]:


def isPrice(x):
    if x>=10 and x<=20:
        return  True             
    return False

Price = dataNew.groupby(dataNew.apply(lambda x: isPrice(x['price']) ,axis=1))
Price.groups


# In[7]:


NewPrice = Price.get_group(True)
NewPrice = NewPrice.dropna(axis=0)
NewPrice


# In[8]:


NewPrice.groupby('country').groups


# In[27]:


def isCountry(x):
    if x == 'Italy' or x=='Spain' or x == 'France' or x=='Chile':
        return  True             
    return False

NewPrice1 = NewPrice.groupby(dataNew.apply(lambda x: isCountry(x['country']) ,axis=1))
NewPrice1.groups


# In[28]:


NewPrice2 = NewPrice1.get_group(True)
NewPrice2 = NewPrice2.dropna(axis=0)
NewPrice2


# In[29]:


len(NewPrice2)


# In[30]:


NewPrice2['pointsbyprice']= NewPrice2.apply(lambda x: x['points']/(10+np.log( x['price'])), axis=1)


# In[31]:


NewPrice2


# In[33]:


stats.levene(NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Italy'],
             NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Spain'],
             NewPrice2['pointsbyprice'][NewPrice2['country'] == 'France'],
             NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Chile'])


# In[34]:


stats.shapiro(NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Italy'])


# In[35]:


stats.shapiro(NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Spain'])


# In[36]:


stats.shapiro(NewPrice2['pointsbyprice'][NewPrice2['country'] == 'France'])


# In[37]:


stats.shapiro(NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Chile'])


# В нашем случае получилось, что нормальности нет, как и равенства дисперсий, поэтому применяем критерий Краскела-Уоллиса

# In[47]:


stats.kruskal(NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Italy'],
             NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Spain'],
             NewPrice2['pointsbyprice'][NewPrice2['country'] == 'France'],
             NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Chile'])


# Отвергаем гипотезу, что средние равны

# In[46]:


stats.kruskal(NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Italy'],
             NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Spain'],
             NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Chile'])


# А вот между Италией, Испанией и Чили разница не очень большая, $p-value>0.05$

# In[39]:


pip install scikit_posthocs


# In[40]:


import scikit_posthocs as spp


# In[41]:


spp.posthoc_dunn([NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Italy'],
             NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Spain'],
             NewPrice2['pointsbyprice'][NewPrice2['country'] == 'France'],
             NewPrice2['pointsbyprice'][NewPrice2['country'] == 'Chile']], p_adjust='fdr_bh')


# In[42]:


sns.boxplot( y=NewPrice2['pointsbyprice'], x=NewPrice2['country'] );
plt.show()


# Вывод - Франция выделяется, качество вина оказалось выше, чем в других странах. 

# In[54]:


NP_crosstab = pd.crosstab(NewPrice2['points'], 
                            NewPrice2['country'],  
                               margins = False) 
print(NP_crosstab) 


# In[55]:


sp.stats.chi2_contingency(NP_crosstab)


# По оценкам различия есть

# In[ ]:




