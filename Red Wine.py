#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd

df= pd.read_csv("winequality-red.csv")
df


# In[54]:


#all the library which is use

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier

import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV 
from sklearn import metrics
import joblib
from sklearn import metrics


# # Exploratory Data Analysis (EDA)

# In[13]:


df.shape


# there are total 1599 rows and 12 columns present in our dataset.

# In[14]:


df.isnull().sum()


# we do not see any missing values in any of the columns of our dataset so we don't have to worry about handling missing data.

# In[15]:


df.info()


# none of the columns have any object data type values and our label is the only integer value making all the feature columns as float datatype i.e. similar datatype.

# In[16]:


df.describe()


# Using the describe method I can see the count, mean, standard deviation, minimum, maximum and inter quantile values of our dataset.

# In[17]:


df.skew()


# Here we see the skewness information present in our dataset. We will ignore quality since it is our target label in the dataset. Now taking a look at all the feature columns we see that fixed acidity, volatile acidity, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, sulphates and alcohol are all outside the acceptable range of +/-0.5. This skewness indicates outliers being present in our dataset that will need to be treated if required.

# # Visualization

# In[18]:


plt.figure(figsize=(10,7))
sns.countplot(x ='quality', data = df)
plt.xlabel('Quality of Red Wine')
plt.ylabel('Count of Rows in the dataset')
plt.show()


# in this we see the various categories of red wine quality and it shows that the number of data present for quality score 5 and 6 is way higher than it's counterparts. This indicates an imbalance which will need to be rectified so that our machine learning model do not get biased to a certain value during prediction.

# In[19]:


index=0
labels = df['quality']
features = df.drop('quality', axis=1)

for col in features.items():
    plt.figure(figsize=(10,5))
    sns.barplot(x=labels, y=col[index], data=df, color="deeppink")
plt.tight_layout()
plt.show()


# With the feature vs label barplot we are able to see the trend corresponding to the impact each has with respect to predicting the quality column (our target variable).
# 
# Observations regarding feature compared to the label are: 01. fixed acidity vs quality - no fixed pattern 02. volatile acidity vs quality - there is a decreasing trend 03. citric acid vs quality - there is an increasing trend 04. residual sugar vs quality - no fixed pattern 05. chlorides vs quality - there is a decreasing trend 06. free sulfur dioxide vs quality - no fixed pattern as it is increasing then decreasing 07. total sulfur dioxide vs quality - no fixed pattern as it is increasing then decreasing 08. density vs quality - no pattern at all 09. pH vs quality - no pattern at all 10. sulphates vs quality - there is an increasing trend 11. alcohol vs quality - there is an increasing trend
# 
# So here we can conclude that to get better quality wine citric acid, sulphates and alcohol columns play a major role.

# In[20]:


fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15,10))
index = 0
ax = ax.flatten()
for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()


# With the help of the above boxplot we are able to see the whisker details and outliers clearly. I am ignoring the continous outlier sections but the outliers that are single values and far away from the whiskers of the boxplot may need to be treated depending upon further analysis. Right now I am just trying to retain as much of data which is possible in the given dataset.

# In[21]:


fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15,10))
index = 0
ax = ax.flatten()
for col, value in df.items():
    sns.distplot(value, ax=ax[index], hist=False, color="g", kde_kws={"shade": True})
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()


# The distribution plots show that few of the columns are in normal distribution category showing a proper bell shape curve. However, we do see skewness in most of the feature columns like citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, sulphates and alcohol columns. We are going to ignore the label column since it is a categorical column and will need to fix the imbalance data inside it.
# 
# With respect to the treatment of skewness and outliers I will perform the removal or treatment after I can see the accuracy dependency of the machine learning models.

# # Correlation using a Heatmap
# 

# In[22]:


lower_triangle = np.tril(df.corr())
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, square=True, fmt='0.3f', 
            annot_kws={'size':10}, cmap="Spectral", mask=lower_triangle)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# 1. Positive correlation - A correlation of +1 indicates a perfect positive correlation, meaning that both variables move in the same direction together.
# 
# 2. Negative correlation - A correlation of –1 indicates a perfect negative correlation, meaning that as one variable goes up, the other goes down.

# the above heatmap shows the correlation matrix data wherein there are positive as well as negative correlations between the target label and other feture columns. A zero correlation indicates that there is no relationship between the variables. Looking at the above representation I see that quality column is positively correlated with alcohol and it is negatively correlated with the volatile acidity. The quality column is least correlated with residual sugar showing a coefficient value of 0.014 that close to 0. Similarly we can bifurcate all the other positively and negatively correlated feature columns with respect to the target label.

# # Dropping a column

# In[23]:


df = df.drop('free sulfur dioxide', axis=1)
df


# free sulfur dioxide and total sulfur dioxide are both indicating towards the same feature of sulfur dioxide therefore I am dropping the free option and keeping just the total option in our dataset.

# # Outlier removal

# In[24]:


df.shape


# Confirming the number of columns and rows before removing the outliers from the dataset.

# In[25]:


# Z-score method

z=np.abs(zscore(df))
threshold=3
np.where(z>3)

df=df[(z<3).all(axis=1)]
df


# I have used the Z score method to get rid of outliers present in our dataset that are not in the acceptable range of +/-0.5 value of skewness.

# In[26]:


df.shape


# Checking the number of rows present in the dataset after applying the outlier removal technique.

# In[27]:


# Percentage of Data Loss

data_loss=(1599-1464)/1599*100 
data_loss


# After removing the outliers we are checking the data loss percentage by comparing the rows in our original data set and the new data set post removal of the outliers.

# # Splitting the dataset into 2 variables namely 'X' and 'Y' for feature and label

# In[28]:


X = df.drop('quality', axis=1)
Y = df['quality']


# I have bifurcated the dataset into features and labels where X represents all the feature columns and Y represents the target label column.

# # Taking care of class imbalance

# In[29]:


Y.value_counts()


# the values of our label column to count the number of rows occupied by each category. This indicates class imbalance that we will need to fix by using the oversampling method.

# In[30]:


# adding samples to make all the categorical quality values same

oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)


# SMOTE is the over sampling mechanism that we are using to ensure that all the categories present in our target label have the same value.

# In[31]:


Y.value_counts()


# After applying over sampling we are one again listing the values of our label column to cross verify the updated information. Here we see that we have successfully resolved the class imbalance problem and now all the categories have same data ensuring that the machine learning model does not get biased towards one category.

# In[32]:


Y


# # Label Binarization

# In[33]:


Y = Y.apply(lambda y_value:1 if y_value>=7 else 0)
Y


# Using the label binarization technique we have tagged the categories present in our target label to 2 major class that are 0 for bad quality wine and 1 for good quality wine.

# In[34]:


X


# # Feature Scaling

# In[35]:


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X


# Even though all our feature columns were of float data type I was unhappy with the decimal place differences and was worried that it might make my model biased. Therefore I am using the Standard Scaler method to ensure all my feature columns have been standardized.

# # Creating the training and testing data sets

# In[36]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=21)


# # Machine Learning Model for Classification and Evaluation Metrics

# In[37]:


# Classification Model Function

def classify(model, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
    
    # Training the model
    model.fit(X_train, Y_train)
    
    # Predicting Y_test
    pred = model.predict(X_test)
    
    # Accuracy Score
    acc_score = (accuracy_score(Y_test, pred))*100
    print("Accuracy Score:", acc_score)
    
    # Classification Report
    class_report = classification_report(Y_test, pred)
    print("\nClassification Report:\n", class_report)
    
    # Cross Validation Score
    cv_score = (cross_val_score(model, X, Y, cv=5).mean())*100
    print("Cross Validation Score:", cv_score)
    
    # Result of accuracy minus cv scores
    result = acc_score - cv_score
    print("\nAccuracy Score - Cross Validation Score is", result)


# I have defined a class that will perform the train-test split, training of machine learning model, predicting the label value, getting the accuracy score, generating the classification report, getting the cross validation score and the result of difference between the accuracy score and cross validation score for any machine learning model that calls for this function.

# In[38]:


# Logistic Regression

model=LogisticRegression()
classify(model, X, Y)


# Created the Logistic Regression Model and checked for it's evaluation metrics.

# In[39]:


# Support Vector Classifier

model=SVC(C=1.0, kernel='rbf', gamma='auto', random_state=42)
classify(model, X, Y)


# Created the Support Vector Classifier Model and checked for it's evaluation metrics.

# In[40]:


# Decision Tree Classifier

model=DecisionTreeClassifier(random_state=21, max_depth=15)
classify(model, X, Y)


# Created the Decision Tree Classifier Model and checked for it's evaluation metrics.

# In[41]:


# Random Forest Classifier

model=RandomForestClassifier(max_depth=15, random_state=111)
classify(model, X, Y)


# Created the Random Forest Classifier Model and checked for it's evaluation metrics.

# In[42]:


# K Neighbors Classifier

model=KNeighborsClassifier(n_neighbors=15)
classify(model, X, Y)


# Created the K Neighbors Classifier Model and checked for it's evaluation metrics.

# In[43]:


# Extra Trees Classifier

model=ExtraTreesClassifier()
classify(model, X, Y)


# Created the Extra Trees Classifier Model and checked for it's evaluation metrics.

# In[51]:


# XGB Classifier

model=xgb.XGBClassifier(verbosity=0)
classify(model, X, Y)


# Created the XGB Classifier Model and checked for it's evaluation metrics.

# In[45]:


# LGBM Classifier

model=lgb.LGBMClassifier()
classify(model, X, Y)


# Created the LGBM Classifier Model and checked for it's evaluation metrics.

# # Hyper parameter tuning on the best ML Model

# In[46]:


# Choosing Support Vector Classifier

svc_param = {'kernel' : ['poly', 'sigmoid', 'rbf'],
             'gamma' : ['scale', 'auto'],
             'shrinking' : [True, False],
             'random_state' : [21,42,104],
             'probability' : [True, False],
             'decision_function_shape' : ['ovo', 'ovr'],
             'verbose' : [True, False]}


# After comparing all the classification models I have selected Support Vector Classifier as my best model and have listed down it's parameters above referring the sklearn webpage.

# In[47]:


GSCV = GridSearchCV(SVC(), svc_param, cv=5)


# I am using the Grid Search CV method for hyper parameter tuning my best model.

# In[48]:


GSCV.fit(X_train,Y_train)


# I have trained the Grid Search CV with the list of parameters I feel it should check for best possible outcomes.

# In[49]:


GSCV.best_params_


# Here the Grid Search CV has provided me with the best parameters list out of all the combinations it used to train the model.

# In[52]:


Final_Model = SVC(decision_function_shape='ovo', gamma='scale', kernel='rbf', probability=True, random_state=21,
                 shrinking=True, verbose=True)
Classifier = Final_Model.fit(X_train, Y_train)
fmod_pred = Final_Model.predict(X_test)
fmod_acc = (accuracy_score(Y_test, fmod_pred))*100
print("Accuracy score for the Best Model is:", fmod_acc)


# I have successfully incorporated the Hyper Parameter Tuning on my Final Model and received the accuracy score for it.

# # AUC ROC Curve

# In[83]:


disp = metrics.plot_roc_curve(Final_Model, X_test, Y_test)
disp.figure_.suptitle("ROC Curve")
plt.show()


# In[87]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve


from sklearn.metrics import roc_curve
from sklearn import metrics


# I have generated the ROC Curve for my final model and it shows the AUC score for my final model to be of 98%.

# 
# 
# # Confusion Matrix

# In[73]:


class_names = df.columns
metrics.plot_confusion_matrix(Classifier, X_test, Y_test, cmap='mako')
plt.title('\t Confusion Matrix for Decision Tree Classifier \n')
plt.show()


# With the help of above confusion matrix I am able to understand the number of times I got the correct outputs and the number of times my model missed to provide the correct prediction (depicting in the black boxes)

# # Saving the model

# In[74]:


filename = "FinalModel_3.pkl"
joblib.dump(Final_Model, filename)


# Finally I am saving my best classification model using the joblib library.
