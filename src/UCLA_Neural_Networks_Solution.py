#!/usr/bin/env python
# coding: utf-8

# # Neural Networks

# ## Predicting Chances of Admission at UCLA

# ### Project Scope:
# 
# The world is developing rapidly, and continuously looking for the best knowledge and experience among people. This motivates people all around the world to stand out in their jobs and look for higher degrees that can help them in improving their skills and knowledge. As a result, the number of students applying for Master's programs has increased substantially.
# 
# The current admission dataset was created for the prediction of admissions into the University of California, Los Angeles (UCLA). It was built to help students in shortlisting universities based on their profiles. The predicted output gives them a fair idea about their chances of getting accepted.
# 
# 
# **Your Role:**
# 
# Build a classification model using **Neural Networks** to predict a student's chance of admission into UCLA.
# 
# 
# **Specifics:** 
# 
# * Machine Learning task: Classification model 
# * Target variable: Admit_Chance 
# * Input variables: Refer to data dictionary below
# * Success Criteria: Accuracy of 90% and above

# ### **Data Dictionary:**
# 
# The dataset contains several parameters which are considered important during the application for Masters Programs.
# The parameters included are : 
# 
# **GRE_Score:** (out of 340) \
# **TOEFL_Score:** (out of 120) \
# **University_Rating:**  It indicates the Bachelor University ranking (out of 5) \
# **SOP:** Statement of Purpose Strength (out of 5) \
# **LOR:** Letter of Recommendation Strength (out of 5) \
# **CGPA:** Student's Undergraduate GPA(out of 10) \
# **Research:** Whether the student has Research Experience (either 0 or 1) \
# **Admit_Chance:** (ranging from 0 to 1) 

# ### **Loading the libraries and the dataset**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")


# In[2]:


# load the data using the pandas `read_csv()` function. 
data = pd.read_csv('Admission.csv')
data.head()


# - In the above dataset, the target variable is **Admit_Chance**
# - To make this a classification task, let's convert the target variable into a categorical variable by using a threshold of 80%
# - We are assuming that if **Admit_Chance** is more than 80% then **Admit** would be 1 (i.e. yes) otherwise it would be 0 (i.e. no)

# In[3]:


# Converting the target variable into a categorical variable
data['Admit_Chance']=(data['Admit_Chance'] >=0.8).astype(int)


# In[4]:


data.head()


# #### Drop any unnecessary columns

# In[5]:


# Dropping columns
data = data.drop(['Serial_No'], axis=1)
data.head()


# Let's check the info of the data

# In[6]:


data.shape


# In[7]:


data.info()


# **Observations:**
# 
# - There are **500 observations and 8 columns** in the data
# - All the columns are of **numeric data** type.
# - There are **no missing values** in the data

# Let's check the summary statistics of the data

# In[10]:


data.describe().T


# **Observations:**
# 
# - The average GRE score of students applying for UCLA is ~316 out of 340. Some students scored full marks on GRE. 
# -  The average TOEFL score of students applying for UCLA is ~107 out of 120. Some students scored full marks on TOEFL.
# - There are students with all kinds of ratings for bachelor's University, SOP, and LOR - ratings ranging from 1 to 5.
# -  The average CGPA of students applying for UCLA is 8.57.
# - Majority of students (~56%) have research experience.
# - As per our assumption, on average 28.4% of students would get admission to UCLA.

# ### **Let's visualize the dataset to see some patterns**

# In[8]:


plt.figure(figsize=(15,8))
sns.scatterplot(data=data, 
           x='GRE_Score', 
           y='TOEFL_Score', 
           hue='Admit_Chance');


# **Observations:** 
# 
# - There is a linear relationship between GRE and TOEFL scores. This implies that students scoring high in one of them would score high in the other as well.
# - We can see a distinction between students who were admitted (denoted by orange) vs those who were not admitted (denoted by blue). We can see that majority of students who were admitted have GRE score greater than 320, TOEFL score greater than 105.

# ### **Data Preparation**

# This dataset contains both numerical and categorical variables. We need to treat them first before we pass them onto the neural network. We will perform below pre-processing steps - 
# *   One hot encoding of categorical variables
# *   Scaling numerical variables
# 
# An important point to remember: Before we scale numerical variables, we would first split the dataset into train and test datasets and perform scaling separately. Otherwise, we would be leaking information from the test data to the train data and the resulting model might give a false sense of good performance. This is known as **data leakage** which we would want to avoid.

# In this dataset, although the variable **University Rating** is encoded as a numerical variable. But it is denoting or signifying the quality of the university, so that is why this is a categorical variable and we would be creating one-hot encoding or dummy variables for this variable.

# In[11]:


data.head()


# In[9]:


# Create dummy variables for all 'object' type variables except 'Loan_Status'
data = pd.get_dummies(data, columns=['University_Rating','Research'])
data.head(2)


# ### Split the Data into train and test

# In[12]:


x = data.drop(['Admit_Chance'], axis=1)
y = data['Admit_Chance']


# In[13]:


# split the data
from sklearn.model_selection import train_test_split


# In[14]:


# Splitting the dataset into train and test data
xtrain, xtest, ytrain, ytest =  train_test_split(x, y, test_size=0.2, random_state=123)


# In[15]:


# import standard scaler
from sklearn.preprocessing import MinMaxScaler


# Now, we will perform scaling on the numerical variables separately for train and test sets. We will use `.fit` to calculate the mean and standard deviation and `.transform` to transform the data.

# In[16]:


# fit calculates the mean and standard deviation
scaler = MinMaxScaler()
scaler.fit(xtrain)


# In[17]:


# Now transform xtrain and xtest
Xtrain = scaler.transform(xtrain)
Xtest = scaler.transform(xtest)


# plt.subplot(2,2,1)
# sns.distplot(data['GRE_Score'])
# 
# plt.subplot(2,2,2)
# sns.distplot(Xtrain[:,0])
# 
# plt.subplot(2,2,3)
# sns.distplot(data['TOEFL_Score'])
# 
# plt.subplot(2,2,4)
# sns.distplot(Xtrain[:,1])
# 
# plt.show()

# ## **Neural Network Architecture**

# In neural networks, there are so many hyper-parameters that you can play around with and tune the network to get the best results. Some of them are - 
# 
# 
# 
# 1.   Number of hidden layers
# 2.   Number of neurons in each hidden layer
# 3.   Activation functions in hidden layers
# 4.   Batch size
# 5.   Learning rate
# 6.   Dropout

# In[18]:


# import the model
from sklearn.neural_network import MLPClassifier


# #### For this exercise, let's build a feed forward neural network with 2 hidden layers. Remeber, always start small.

# ### **Training the model**

# In[17]:


data.shape


# In[30]:


# fit/train the model. Check batch size.
MLP = MLPClassifier(hidden_layer_sizes=(3), batch_size=50, max_iter=100, random_state=123)
MLP.fit(Xtrain,ytrain)


# In[31]:


# make Predictions
ypred = MLP.predict(Xtest)


# In[32]:


# import evaluation metrices
from sklearn.metrics import confusion_matrix, accuracy_score


# In[33]:


confusion_matrix(ytest, ypred)


# In[34]:


# check accuracy of the model
accuracy_score(ytest, ypred)


# In[24]:


# Plotting loss curve
loss_values = MLP.loss_curve_

# Plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Loss', color='blue')
plt.title('Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# ### **Conclusion**

# In this case study,
# 
# - We have learned how to build a neural network for a classification task. 
# - **Can you think of a reason why, we could get such low accuracy?**
# - You can further analyze the misclassified points and see if there is a pattern or if they were outliers that our model could not identify.

# In[ ]:





# # Using Grid Search CV

# In[38]:


# cross validation using cross_val_score
from sklearn.model_selection import cross_val_score

# Import GridSearch CV
from sklearn.model_selection import GridSearchCV


# In[35]:


MLP.get_params


# In[45]:


# we will try different values for hyperparemeters
params = {'batch_size':[20, 30, 40, 50],
          'hidden_layer_sizes':[(2,),(3,),(3,2)],
         'max_iter':[50, 70, 100]}


# In[ ]:


# create a grid search
grid = GridSearchCV(MLP, params, cv=10, scoring='accuracy', rand)
grid.fit(x, y)


# In[47]:


grid.best_params_


# In[48]:


grid.best_score_


# In[49]:


grid.estimator


# In[ ]:




