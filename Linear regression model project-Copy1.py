#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES
# 

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


plt.style.use('dark_background')
import warnings
warnings.filterwarnings(action='ignore')


# In[9]:


data = pd.read_csv("Transformed_Housing_Data2.csv")


# In[10]:


data.head()


# # steps we followed to create Transformed_Housing_Data2.csv

# 1- Exploring the target variables and independent variables
# 2- Treating the outliers and missing values in independent and target variables
# 3- Transforming the categorical variables and numerical variables using dummy encoding

# # Scaling the Dataset

# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Y = data['Sale_Price'] #separating the target variable as Y
X = scaler.fit_transform(data.drop(columns=['Sale_Price'])) #scaler is instance of Standard Scaler to scale all the independent variables and store it on X
X = pd.DataFrame(data=X,columns=data.drop(columns=['Sale_Price']).columns) #using pandas dataframe for easy manipulation
X.head()


# # Checking the correlation among the independent variables

# X.corr() function ----> It gives us the correlation between every possible pair of variables in our dataset.

# # Checking and Removing Multicollinearity

# In[12]:


X.corr()


# In[13]:


# pair of independent variables with correlation greater than 0.5
k = X.corr()
z = [[str(i),str(j)] for i in k.columns for j in k.columns if (k.loc[i,j]>abs(0.5))&(i!=j)]
z,len(z)


# # Calculating VIF 

# In[14]:


# Importing_Variance _inflation_factor function from the Statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = X
# calculating VIF for every column 
VIF = pd.Series([variance_inflation_factor(vif_data.values,i) for i in range (vif_data.shape[1])], index = vif_data.columns)
VIF


# In[15]:


VIF[VIF==VIF.max()].index[0]


# In[16]:


def MC_remover(data):
    vif = pd.Series([variance_inflation_factor(data.values,i) for i in range(data.shape[1])],index = data.columns)
    if(vif.max()>5):
        print(vif[vif==vif.max()].index[0], 'has been removed')
        data = data.drop(columns=[vif[vif==vif.max()].index[0]])
        return data
    else:
        print("No Multicollinearity present anymore")
        return data


# In[17]:


for i in range(7):
    vif_data = MC_remover(vif_data)
vif_data.head()


# #  Remaining Columns

# In[18]:


# calculating VIF for remaing columns 
VIF = pd.Series([variance_inflation_factor(vif_data.values,i) for i in range(vif_data.shape[1])],index = vif_data.columns)
VIF,len(vif_data.columns)


# #  Train/Test set

# In[19]:


x=vif_data
y=data["Sale_Price"]


# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# # Linear Regression

# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
linear_regression = LinearRegression()
linear_regression.fit(x_train,y_train)


# In[27]:


linear_regression.coef_


# In[30]:


predictions = linear_regression.predict(x_test)
linear_regression.score(x_test, y_test)


# # 1. Residuals 

# In[31]:


residuals = predictions - y_test
residual_table = pd.DataFrame({'residuals':residuals,'predictions':predictions})
residual_table = residual_table.sort_values( by = 'predictions')


# In[35]:


z=[i for i in range(int(residual_table['predictions'].max()))]
k=[0 for i in range(int(residual_table['predictions'].max()))]
plt.figure(dpi=130,figsize=(17,7))
plt.scatter(residual_table['predictions'],residual_table['residuals'],color='red',s=2)
plt.plot(z, k, color='green',linewidth=3,label='regression')
plt.ylim(-800000,800000)
plt.xlabel('fitted points(order by predictions)')
plt.ylabel('residuals')
plt.le


# #  Model Coefficients

# In[36]:


coefficients_table = pd.DataFrame({'column':x_train.columns,'coefficients':linear_regression.coef_})
coefficient_table = coefficients_table.sort_values(by = 'coefficients')


# In[37]:


plt.figure(figsize=(8,6),dpi=120)
x=coefficient_table['column']
y=coefficient_table['coefficients']
plt.barh(x,y)
plt.xlabel("coefficients")
plt.ylabel("variables")
plt.title("Normalized coefficient plot")
plt.show()


# In[ ]:




