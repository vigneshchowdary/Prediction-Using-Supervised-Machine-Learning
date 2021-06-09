#!/usr/bin/env python
# coding: utf-8

# In[20]:


GRIP : The Sparks Foundation

Data Science and Business Analyst

Author : Vignesh chowdary

Task 1 : Prediction Using Supervised Machine Learning

In(this, task, we, have, to, predict, the, percentage, of, an, student, based, on, the, no., of, study, hours.In, the, given, task, we, have, to, use, 2, variables, where, the, 1st, feature, is, no, of, hours, studied, and, the, target, value, is, the, percentage, score., This, given, task, can, be, solved, using, Simple, Linear, regression, /)


# In[21]:


#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


#Reading data from remote Url
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)


# In[8]:


#DATA EXPLORATION
print("Data imported successfully")

s_data.head(19)


# In[ ]:


#VISUALISATION OF DATA


# In[9]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='*')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[ ]:


There is a positive linear relation between the scores and the number of hours studied.

DATA PREPARATION

The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).


# In[10]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values


# In[11]:


#We first split the data into training data set and testing data set and then train the algorithm.
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)


# In[12]:


from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train, y_train) 

print("Training complete.")


# In[13]:


line = model.coef_*X+model.intercept_
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[15]:


print(X_test)
y_pred = model.predict(X_test)


# In[16]:



df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[18]:


from sklearn import metrics
print('Mean Squared :', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute :', metrics.mean_absolute_error(y_test, y_pred))


# In[19]:


hours = np.array([[8.12]])
prediction = model.predict(hours)
print('No of hours ={}'. format(hours))
print('Predicted score={}'.format(prediction[0]))


# In[ ]:


From the above result it is clear that if a student does not studies for 9.37 hours then the predicted score will be 81.85397687874413 and there is a significant drop off in the predicted value.

