import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import export_graphviz, DecisionTreeClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

dataSet = open("mushroom.txt", "r") #classification, handle missing data

pd.set_option("display.max_columns", 23)
colNames = ['label', 'cap-shape','cap-surface','cap-color','bruises','odor',
          'gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape',
          'stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',
          'stalk-color-above-ring','stalk-color-below-ring','veil-type',
          'veil-color','ring-number','ring-type','spore-print-color','population',
          'habitat']
readMushroom = pd.read_csv(dataSet, names=colNames)
readMushroom.replace('?',np.nan, inplace=True) #replace ? with NaN
labelEncoder = LabelEncoder()
# print(readMushroom)
#Categorical data converted into numberical data
#Using LabelEncoder
for columns in readMushroom.columns:
    readMushroom[columns] = labelEncoder.fit_transform(readMushroom[columns].astype(str))# Cannot accept float so converted to string.
# print(readMushroom.describe())

#Any columns with more than 20% missing values would be deleted, instead of just deleting the column
removeMissValues = readMushroom.isnull().apply(sum,axis=0)# count the number of nan in each column
for columns in readMushroom:
    if removeMissValues[columns] >= len(readMushroom) *0.2:
        del removeMissValues[columns]
# print(readMushroom)

#Deletes the column that has no value in it, because all of the value kept were == 0
df = readMushroom.drop(["veil-type"], axis=1)
# print(readMushroom.describe())

plt.figure()
pd.Series(df['label']).value_counts().sort_index().plot(kind='bar')
plt.ylabel("Count")
plt.xlabel("label")
plt.title('Number of poisonous/edible mushrooms (0=edible, 1=poisonous)')


X = df.drop(['label'], axis=1)
Y = df['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
# graph

features_list = X.columns.values
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(3,3))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()


y_prediction = clf.predict(X_test)
print("Accuracy of the Decision Tree", accuracy_score(Y_test, y_prediction))
print("Classification Report of Decision tree")
print(classification_report(Y_test, y_prediction))

logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, Y_train)
predicted = logisticRegression.predict(X_test)
accuracy = accuracy_score(Y_test,predicted)
print("Accuracy of Logistic Regression is %f "%accuracy)
print("Classification Report of Logistic Regression")
print(classification_report(Y_test, predicted))

cfm = confusion_matrix(Y_test,y_prediction)
sns.heatmap(cfm,annot=True,linewidths=.5, cbar=None)
plt.title('Decision Tree Classifier confusion matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted label')



#Gaussian Naive Bayes(Classifier)

gNB = GaussianNB()
gNB = gNB.fit(X_train,Y_train)
# print(gNB)
yPredgNB = gNB.predict(X_test)
# print(yPredgNB)
cfm = confusion_matrix(Y_test,yPredgNB)
# print(cfm)
sns.heatmap(cfm, annot = True,  linewidths=.5, cbar =None)
plt.title('Gaussian Naive Bayes confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print("Test data- Gaussian Naive Bayes report \n", confusion_matrix(Y_test, yPredgNB))
print("Accuracy of the Naive Bayes", accuracy_score(Y_test, yPredgNB))
print("Classification Report of Naive Bayes")
print(classification_report(Y_test, yPredgNB))

def ROC(Y_test,Y_pred,method):

    falsePosRate,truePosRate,tresholds = roc_curve(Y_test,Y_pred)
    rocAUC = auc(falsePosRate,truePosRate)
    plt.title('Reciever Operating Characteristic ')
    plt.plot(falsePosRate,truePosRate,color='darkorange',label='%s AUC = %0.6f'%(method, rocAUC))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.ylim([-0.1, 1.1])
    plt.xlim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

roc = ROC(Y_test, yPredgNB, "Gaussian Naive Bayes")
plt.show()


colorClass = df[['label','gill-color']].groupby(['gill-color'], as_index=False).mean().sort_values(by='label',ascending=False)
# print(colorClass)
colorVar = df[['label','gill-color']]
colorVar = colorVar[colorVar['gill-color'] < 3.5]
sns.catplot('label', col='gill-color', data=colorVar, strip='count', height=2.5, aspect=.8, col_wrap=4)
plt.show()
# def compAccuray(X,y,folds):
#     accuracy = []
#     foldAcc = []
#     depth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#     for i in depth:
#         kf = KFold(len(X),n_folds = folds)
#         for trainIndex, testIndex in kf:
#             X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size= 0.1)
#             clf = DecisionTreeClassifier(max_depth= i).fit(X_train,Y_train)
#             score = clf.score(X_test,Y_test)
#             accuracy.append(score)
#         foldAcc.append(np.mean(accuracy))
#         cvAccuracy = compAccuray(X, Y, folds=10)



















