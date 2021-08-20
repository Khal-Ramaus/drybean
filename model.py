import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import pickle

df = pd.read_excel('Dry_Bean_Dataset.xlsx')
X = df.drop(columns=['MajorAxisLength', 'MinorAxisLength', 'Eccentricity','ConvexArea','EquivDiameter','Compactness','ShapeFactor2','ShapeFactor3','Class'])
y = df.Class

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

param = [
         {'classifier':[SVC(C=0.1, class_weight='balanced')], 
          'preprocessing':[StandardScaler()],
          'classifier__kernel':['linear','poly','rbf']},
         {'classifier':[RandomForestClassifier(class_weight='balanced')], 
          'preprocessing':[StandardScaler()],
          'classifier__n_estimators':[50,100,150,200],
          'classifier__max_features': [1, 2, 3]},
         {'classifier':[DecisionTreeClassifier(class_weight='balanced')], 
          'preprocessing':[StandardScaler()],
          'classifier__max_depth':[5,10,15,20,25]}
]

kFold=KFold(n_splits=5)

Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=0)

grid = GridSearchCV(pipe, param, cv=kFold)
grid.fit(Xtr, ytr)

best=grid.best_estimator_

# Saving model
pickle.dump(best, open('model.pkl','wb'))
