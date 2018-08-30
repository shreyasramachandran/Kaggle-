# import sklearn
import pandas as pd
# import seaborn as sns 
# import matplotlib.pyplot as plt 
import numpy as np 

# gender_submission = pd.read_csv('gender_submission.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = train['Survived'].astype(int)
train.drop(['PassengerId','Survived'],axis = 1, inplace = True)
test.drop(['PassengerId'],axis = 1,inplace = True)
train_test = pd.concat([train,test])

# extracting titles
Title_Dictionary = {
    "Capt"          : "Officer" ,
    "Col"           : "Officer" ,
    "Major"         : "Officer" ,
    "Jonkheer"      : "Royalty" ,
    "Don"           : "Royalty" ,
    "Dona"          : "Royalty" ,
    "Sir"           : "Royalty" ,
    "Dr"            : "Officer" ,
    "Rev"           : "Officer" ,
    "the Countess"  : "Royalty" ,
    "Mme"           : "Mrs"     ,
    "Mlle"          : "Miss"    ,
    "Ms"            : "Mrs"     ,
    "Mr"            : "Mr"      ,
    "Mrs"           : "Mrs"     ,
    "Miss"          : "Miss"    ,
    "Master"        : "Master"  ,
    "Lady"          : "Royalty"
}

train_test['Titles'] = train_test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
train_test['Titles'] = train_test['Titles'].map(Title_Dictionary)
train_test.drop(['Name'],axis = 1,inplace = True)

train_test["Age"]  = train_test.groupby(['Sex','Pclass','Titles'])['Age'].transform(lambda x: x.fillna(x.median()))

train_test["Fare"].fillna(train_test.groupby(["Pclass","Age",'Titles'])["Fare"].transform("median"), inplace=True)
train_test["Fare"] = train_test["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

train_test["Embarked"].fillna('S',inplace=True)

train_test["Cabin"].fillna('U',inplace=True)
# train_test.isnull().sum()
train_test["Cabin"] = train_test["Cabin"].map(lambda name:name[0])

train_test["Fsize"] = train_test["SibSp"] + train_test["Parch"] + 1
train_test['Single'] = train_test['Fsize'].map(lambda s: 1 if s == 1 else 0)
train_test['SmallF'] = train_test['Fsize'].map(lambda s: 1 if  s == 2  else 0)
train_test['MedF'] = train_test['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
train_test['LargeF'] = train_test['Fsize'].map(lambda s: 1 if s >= 5 else 0)

preTicket = []
for i in list(train_test.Ticket):
    if not i.isdigit() :
        preTicket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) 
    else:
        preTicket.append("X")
        
train_test["Ticket"] = preTicket

train_test = pd.get_dummies(train_test, columns=["Pclass","Sex","Ticket","Cabin","Embarked","Titles"],prefix=["Pclass","Sex","Ticket","Cabin","Embarked","Titles"], drop_first=True)

train = train_test[:len(train)]
test = train_test[len(train):]


# modeling
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, y_train)

features = pd.DataFrame()
features['Feature'] = train.columns
features['Importance'] = clf.feature_importances_
features.sort_values(by=['Importance'], ascending=True, inplace=True)
features.set_index('Feature', inplace=True)
# features['Importance']

model = SelectFromModel(clf,0.001093,prefit=True)
train_reduced = model.transform(train)
print(train_reduced.shape)

test_reduced = model.transform(test)
print(test_reduced.shape)

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()

models = [logreg, logreg_cv, rf, gboost]

for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=train_reduced, y=y_train, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')

run_gs = False
if run_gs:
    parameter_grid = {'max_depth' : [4, 6, 8],'n_estimators': [100,50, 10],'max_features': ['sqrt', 'auto', 'log2'],'min_samples_split': [2, 3, 10],'min_samples_leaf': [1, 3, 10],'bootstrap': [True, False],}
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(forest,scoring='accuracy',param_grid=parameter_grid,cv=cross_validation,verbose=1)
    grid_search.fit(train, y_train)
    model = grid_search
    parameters = grid_search.best_params
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50,'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    model = RandomForestClassifier(**parameters)
    model.fit(train, y_train)
    
output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output.shape

df_output[['PassengerId','Survived']].to_csv('Result4.csv', index=False,)