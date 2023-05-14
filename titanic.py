import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

def nextline():
    print("\n")

#Loading data
titanic = sns.load_dataset("titanic")

titanic = titanic.drop(["adult_male", "alive", "embark_town", "class", "alone", "who"], axis=1)
titanic = titanic.dropna(subset = ["embarked"])
print(titanic.head())
nextline()
print(titanic.info())
nextline()

#Handling missing data: Imputation
print(titanic.isna().sum())
cols_with_miss = [col for col in titanic.columns if titanic[col].isna().sum() > 0]

X = titanic.drop("survived", axis = 1)
y = titanic.survived

print(cols_with_miss)
category = ["pclass", "sex", "embarked", "deck"]
num = ["age", "sibsp", "parch", "fare"]
X_cat = X[category].values
X_num = X[num].values

#categorical
X_cat_train, X_cat_test, y_train, y_test = train_test_split(X_cat,y, random_state=13, train_size=0.8)

imputer = SimpleImputer(strategy="most_frequent")
X_cat_train = pd.DataFrame(imputer.fit_transform(X_cat_train))
X_cat_test = pd.DataFrame(imputer.transform(X_cat_test))

#numerical
X_num_train, X_num_test, y_train, y_test = train_test_split(X_num,y, random_state=13, train_size=0.8)

imputer1 = SimpleImputer(strategy="median")
X_num_train = pd.DataFrame(imputer1.fit_transform(X_num_train))
X_num_test = pd.DataFrame(imputer1.transform(X_num_test))

X_train = pd.concat([X_num_train, X_cat_train], axis=1)
X_test = pd.concat([X_num_test, X_cat_test], axis=1)

X_train.columns = num + category
X_test.columns = num + category

#converting column types
num_change = ["sibsp","parch"]
X_train[num_change] = X_train[num_change].astype("int64")
X_test[num_change] = X_test[num_change].astype("int64")

#Converting categorical features into numeric ones
object_cols = [col for col in X_train.columns if X_train[col].dtypes == "object"]
object_col = list(set(object_cols) - set(["pclass"]))

oh = OneHotEncoder(handle_unknown="ignore", sparse=False)

oh_X_train = pd.DataFrame(oh.fit_transform(X_train[object_col]))
oh_X_test = pd.DataFrame(oh.transform(X_test[object_col]))

#Set the name for features
oh_X_train.columns = oh.get_feature_names_out()
oh_X_test.columns = oh.get_feature_names_out()

#Return the index
oh_X_train.index = X_train.index
oh_X_test.index = X_test.index

#Concatenate
X_train = pd.concat([X_train.drop(object_col,axis=1), oh_X_train], axis=1)
X_test = pd.concat([X_test.drop(object_col,axis=1), oh_X_test], axis=1)

X = pd.concat([X_train,X_test]).values
y = y.values

#Scaling the data
scalerx = StandardScaler()

X_scaled_train = scalerx.fit_transform(X_train[num].values)
X_scaled_test = scalerx.transform(X_test[num].values)

X_train = pd.concat([pd.DataFrame(X_scaled_train, columns=num), X_train.drop(num, axis = 1)], axis = 1)
X_test = pd.concat([pd.DataFrame(X_scaled_test, columns=num), X_test.drop(num, axis = 1)], axis = 1)

#Redefine the data
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

#Choosing the best models
models = {"Logistic Regression":LogisticRegression(C = 1, penalty="l2", solver="liblinear"),
          "Support Vector Machine":SVC(C=1), "KNN":KNeighborsClassifier(n_neighbors=11),
          "DecisionTree":DecisionTreeClassifier(max_depth=5, min_samples_leaf=0.11),
          "RandomForest":RandomForestClassifier(n_estimators=300, max_depth=5,random_state=42)}
scores = []
means = []
std = []
con = []

for name, model in models.items():
    K = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=K)
    overall_score = sum(score) / len(score)
    scores.append(score)
    means.append(overall_score)
    std.append(np.std(score))
    con.append(np.quantile(score, [0.025, 0.975]))
    print(name + ": ",score)

#Visualization
plt.style.use("classic")
sns.set()
plt.grid()

plt.boxplot(scores, labels=models.keys())
plt.show()
plt.clf()

nextline()
print("Mean: ",means)
print("Standard deviation: ", std)
print("The confidence interval: ", con)
nextline()

#Tuning hyperparameters for 3 best models
name = ["KNN","SVC","RandomForest"]
model = [KNeighborsClassifier(),SVC(probability=True), RandomForestClassifier()]
params = [{"n_neighbors":np.arange(4,15,1), "weights":["uniform","distance"], "p":[1,2]},
          {"C":[1,8,12] , "gamma":["scale", "auto"], "kernel":["linear", "poly", "rbf"],
           "degree":[1,3,8,4]},
          {"n_estimators":[50,100,250,300], "criterion":["gini","entropy"], "max_depth":np.arange(4,8,1),
           "min_samples_leaf":[0.05,0.01,0.03,0.1]}]

i = 0
while (i<3):
    fold = KFold(n_splits=6, shuffle=True, random_state=42)
    grid = GridSearchCV(model[i], param_grid=params[i], cv = fold, n_jobs=-1, scoring="accuracy")
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:,1]
    print(name[i] + "\nScore: "+str(grid.best_score_)+"\nBest parameters: "+str(grid.best_params_))
    print("Accuracy score on training set: ", grid.score(X_train,y_train))
    print("Accuracy score on testing set: ", grid.score(X_test, y_test))
    print(classification_report(y_test, y_pred))
    print("Roc score: ",roc_auc_score(y_test, y_proba))
    nextline()
    i+=1


#Instantiating model
model1 = RandomForestClassifier(criterion="gini", max_depth=7, min_samples_leaf=0.01, n_estimators=100)

model2 = KNeighborsClassifier(n_neighbors=11, p=1, weights="uniform")

model3 = SVC(C=8, degree=1, gamma="scale", kernel="rbf")

model = VotingClassifier([("forest",model1), ("KNN",model2),("SVC",model3)], voting="hard", n_jobs=-1)

#Training
model.fit(X_train, y_train)

#Predicting
y_pred = model.predict(X_test)

#Evaluating performance
print("Training accuracy: ", accuracy_score(y_train, model.predict(X_train)))
print("Validation accuracy: ", accuracy_score(y_test, y_pred))

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n",classification_report(y_test, y_pred))









