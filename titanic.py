'''
Kaggle Titanic: Machine Learning from Disater

'''
import pandas as pd
import numpy as np
import csv
from sklearn import tree

'''
1. import data as csv files
'''

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

'''
print (train.head())
print (test.head())
'''


'''
2. understand the data
'''
'''
print (train.describe())
print (train.shape)
'''

'''
3.print some specific imfo in the data
'''

'''
print(train["Survived"].value_counts())
print(train["Survived"].value_counts(normalize = True))
print(train["Survived"][train["Sex"]=='male'].value_counts())
print(train["Survived"][train["Sex"]=='female'].value_counts())
print(train["Survived"][train["Sex"]=='male'].value_counts(normalize = True))
'''

'''
4. manipulation. add a new column
'''

'''
train["Child"] = float('NaN')
train["Child"][train["Age"]<18] = 1
train["Child"][train["Age"]>=18] = 0
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))
print train["Survived"][train["Child"] == 0].value_counts(normalize = True)

'''
'''
5. a naive prediction based on majority Sex
'''
'''
test_one = test
test_one["Survived"] = 0
test_one["Survived"][test_one["Sex"] == 'female'] = 1
print test_one["Survived"]
'''
'''
6 cleaning and formatting your data
'''
train["Sex"][train["Sex"]=='male']=0
train["Sex"][train["Sex"]=='female']=1
train["Embarked"] = train["Embarked"].fillna('S')
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"][train["Embarked"]=='S'] = 0
train["Embarked"][train["Embarked"]=='Q'] = 2
train["Embarked"][train["Embarked"]=='C'] = 1
'''
print train["Embarked"]
'''

'''
7 create first decision tree

'''


target = train["Survived"].values 
'''
features_one = train[ [ "Pclass", "Age", "Sex", "Fare"]].values
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)


print my_tree_one.feature_importances_
print my_tree_one.score(features_one, target)
'''

'''
8 Predict and Submit to Kaggle
'''

test.Fare[152]= test["Fare"].median()
test["Sex"][test["Sex"]=='male'] = 0
test["Sex"][test["Sex"]=='female'] = 1
test["Age"] = test["Age"].fillna(test["Age"].median())

'''
print(test.describe())
print(test["Sex"].describe())
'''

'''
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

my_prediction = my_tree_one.predict(test_features)



PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print (my_solution)

'''
'''

print(my_solution.shape)
'''

'''
9 Over fitting and how to control it
'''
'''
features_two = train[["Pclass", "Age", "Sex","Fare", "SibSp", "Parch", "Embarked"]]
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)
print my_tree_two.score(features_two, target)
'''

'''
10 Feature Engineering
'''

'''
train_two = train.copy()
train_two["family_size"] = train_two["Parch"] + train_two["SibSp"]+1
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)
print my_tree_three.score(features_three, target)
'''
'''
11 Random Forest analysis in Python
'''

from sklearn.ensemble import RandomForestClassifier
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(max_depth = 10, min_samples_split = 2, n_estimators = 100)
my_forest = forest.fit(features_forest, target)
print my_forest.score(features_forest,target)

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print len(pred_forest)










