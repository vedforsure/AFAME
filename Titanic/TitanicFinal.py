#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as numpy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

titanic_data = pd.read_csv("TrainingSetFull.csv")


# In[22]:


titanic_data = titanic_data.drop(['Unnamed: 0'], axis= 1)


# In[23]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, test_indices in split.split(titanic_data, titanic_data[["Survived", "Pclass", "Sex"]]):
    strat_train_set = titanic_data.loc[train_indices]
    strat_test_set = titanic_data.loc[test_indices]


# In[24]:


class AgeImputer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        imputer =  SimpleImputer(strategy="mean")
        X['Age'] = imputer.fit_transform(X[['Age']])
        return X
    
class FeatureEncoder (BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        encoder = OneHotEncoder()
        
        matrix = encoder.fit_transform(X[['Embarked']]).toarray()
        
        column_names = ["C", "S", "Q", "N"]
        
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
            
        matrix = encoder.fit_transform(X[['Sex']]).toarray()
        
        column_names = ["Female", "Male"]
        
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
              
        return X
    
class FeatureDropper(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(["Embarked", "Name", "Ticket", "Cabin", "Sex", "N"], axis=1, errors="ignore")


# In[25]:


pipeline = Pipeline([("ageimputer", AgeImputer()),
                     ("featureencoder", FeatureEncoder()),
                     ("featuredropper", FeatureDropper())])


# In[26]:


strat_train_set = pipeline.fit_transform(strat_train_set)
strat_test_set = pipeline.fit_transform(strat_test_set)


# In[28]:


X = strat_train_set.drop(['Survived'], axis=1)
y = strat_train_set['Survived']

scalar = StandardScaler()

X_data = scalar.fit_transform(X)
y_data = y.to_numpy()


# In[29]:


clf = RandomForestClassifier()

param_grid = [
    {"n_estimators" :[10, 100, 200, 500, 1000], "max_depth":[None, 5, 10], "min_samples_split": [2, 3, 4]}
]

grid_search = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_data, y_data)


# In[30]:


final_clf = grid_search.best_estimator_


# In[31]:


X_test = strat_test_set.drop(['Survived'], axis=1)
y_test = strat_test_set['Survived']

scalar = StandardScaler()

X_data_test = scalar.fit_transform(X_test)
y_data_test = y_test.to_numpy()


# In[33]:


final_titanic_data = pipeline.fit_transform(titanic_data)


# In[35]:


X_final = final_titanic_data.drop(['Survived'], axis=1)
y_final = final_titanic_data['Survived']

scalar = StandardScaler()

X_data_final = scalar.fit_transform(X_final)
y_data_final = y_final.to_numpy()


# In[36]:


prod_clf = RandomForestClassifier()

param_grid = [
    {"n_estimators" :[10, 100, 200, 500, 1000], "max_depth":[None, 5, 10], "min_samples_split": [2, 3, 4]}
]

grid_search = GridSearchCV(prod_clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_data_final, y_data_final)


# In[37]:


prod_final_clf = grid_search.best_estimator_


# In[51]:


#final test

titanic_test_data = pd.read_csv ("ModelTestSetFull.csv")
titanic_test_data = titanic_test_data.drop(['Unnamed: 0.1'], axis=1)


# In[52]:


final_test_data = pipeline.fit_transform(titanic_test_data)


# In[58]:


final_test_data = final_test_data.drop(['Unnamed: 0'], axis=1)


# In[59]:


X_final_test = final_test_data
X_final_test = X_final_test.fillna(method= "ffill")

scalar = StandardScaler()
X_data_final_test = scalar.fit_transform(X_final_test)


# In[60]:


predictions = prod_final_clf.predict(X_data_final_test)


# In[62]:


final_df = pd.DataFrame(titanic_test_data [['PassengerId', 'Name', 'Age']])
final_df['Survived'] = predictions


# In[64]:


final_df.to_csv("TitanicPredictions.csv", index=False)
final_df


# In[ ]:




