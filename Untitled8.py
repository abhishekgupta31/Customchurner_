#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy 
from scipy.stats import stats
import numpy as np
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_curve
from sklearn.preprocessing import LabelEncoder,StandardScaler


# In[2]:


df=pd.read_csv('customer_churn.csv')
df.head()


# In[3]:


df.info(verbose=True)


# In[4]:


df.describe()


# In[5]:


df['Churn'].value_counts()


# In[6]:


df['Churn'].value_counts()/len(df['Churn'])*100
# so this is not balanced Data


# In[7]:


df.isnull().sum()


# In[8]:


df_n=df.copy()


# In[9]:


df_n.TotalCharges=pd.to_numeric(df_n.TotalCharges,errors='coerce')
df_n.isnull().sum()


# In[10]:


df_n.loc[df_n['TotalCharges'].isnull()==True]


# In[11]:


df_n.dropna(how='any',inplace=True)


# In[12]:


df_n.isnull().sum()


# In[13]:


print(df_n['tenure'].max())


# In[14]:


labels=["{0}-{1}".format(i,i+11) for i in range(1,72,12)]
df_n['tenure_group']=pd.cut(df.tenure,range(1,80,12),right=False,labels=labels)


# In[15]:


df_n['tenure_group'].value_counts()


# In[16]:


df_n_n=df_n.drop(['customerID','tenure'],axis=1)
df_n_n.head()


# In[17]:


df_n_n['Churn']=np.where(df_n_n.Churn == 'Yes',1,0)


# In[18]:


for i,var in enumerate(df_n_n.drop(columns=['Churn','TotalCharges','MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=df_n_n,x=var,hue='Churn')


# In[19]:


df_dummy=pd.get_dummies(df_n_n)
df_dummy.head()


# In[20]:


sns.lmplot(data=df_dummy,x='MonthlyCharges',y='TotalCharges')


# In[21]:


kde= sns.kdeplot(data=df_n_n.MonthlyCharges[(df_n_n['Churn']==0)],shade=True,color='blue')
kde= sns.kdeplot(data=df_dummy.MonthlyCharges[(df_dummy['Churn']==0)],shade=True,color='red')


# In[22]:


plt.figure(figsize=(20,15))
df_dummy.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')


# In[23]:


plt.figure(figsize=(30,25))
sns.heatmap(df_dummy.corr(),annot=True,cmap=None)


# In[24]:


pd.pandas.set_option('display.max_columns',None)
pd.pandas.set_option('display.max_rows',None)
df_dummy.head()


# In[25]:


#creating X and Y Variables


# In[26]:


x = df_dummy.drop(['Churn'],axis=1)
y = df_dummy['Churn']


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=101)


# In[28]:


dc=DecisionTreeClassifier(criterion='gini',random_state=101)
dc.fit(x_train,y_train)


# In[29]:


pred = dc.predict(x_test)
pred


# In[30]:


classification_report(y_test,pred,labels=[0,1])


# In[31]:


confusion_matrix(y_test,pred)


# In[32]:



from imblearn.combine import SMOTEENN


# In[33]:


sm=SMOTEENN()
x_resampled,y_resampled= sm.fit_resample(x,y)


# In[34]:


xr_train,xr_test,yr_train,yr_test=train_test_split(x_resampled,y_resampled,test_size=0.25,random_state=101)


# In[35]:


dc_sm=DecisionTreeClassifier(criterion='gini',random_state=101)
dc_sm.fit(xr_train,yr_train)


# In[36]:


pred_y_sm=dc_sm.predict(xr_test)
pred_y_sm


# In[37]:


confusion_matrix(yr_test,pred_y_sm)


# In[38]:


classification_report(yr_test,pred_y_sm)

Now we weill check on RANDOMFOREST
# In[39]:


from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators=100,criterion='gini',max_leaf_nodes=None)
rfc.fit(x_train,y_train)
pred_rfc=rfc.predict(x_test)
print(classification_report(y_test,pred_rfc))
print(confusion_matrix(y_test,pred_rfc))


# In[42]:


sm=SMOTEENN()
x_resampled,y_resampled= sm.fit_resample(x,y)


# In[44]:


xrr_train,xrr_test,yrr_train,yrr_test=train_test_split(x_resampled,y_resampled,test_size=0.25,random_state=101)


# In[48]:


from sklearn.ensemble import RandomForestClassifier
rfc_sm= RandomForestClassifier(n_estimators=100,criterion='gini',max_leaf_nodes=None)
rfc_sm.fit(xrr_train,yrr_train)
pred_rfc_sm=rfc_sm.predict(x_test)
print(classification_report(y_test,pred_rfc_sm))
print(confusion_matrix(y_test,pred_rfc_sm))


# In[49]:


import pickle
filename='model.sav'
pickle.dump(rfc_sm,open(filename,'wb'))


# In[51]:


load_model=pickle.load(open(filename,'rb'))


# In[60]:


load_model.score(xrr_test,yrr_test)


# In[ ]:




