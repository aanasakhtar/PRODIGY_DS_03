import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

dataset = pd.read_csv("C:/Users/Admin/Desktop/internship24/bank.csv", delimiter=";")

dataset.fillna(method="ffill", inplace=True)

labels_encoders = {}

binary_columns = ["default", "housing", "loan", "y"]

for col in binary_columns:
    labels_encoders[col] = LabelEncoder()
    dataset[col] = labels_encoders[col].fit_transform(dataset[col])

dataset = pd.get_dummies(dataset, columns=['job', 'education', "poutcome", "contact", "marital", "month"], drop_first=True)

X = dataset.drop(columns=['y'])
y = dataset['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42, max_depth = 4)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)

metrics_df = pd.DataFrame({
    "Metric" : ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    "value" : [accuracy, precision, recall, f1]
})

metrics_df.to_csv("C:/Users/Admin/Desktop/internship24/metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(20, 10))
fig.patch.set_facecolor('#666666')
ax.set_facecolor('lightyellow')
plot_tree(clf, filled=True, feature_names=list(X.columns), class_names=['no', 'yes'], rounded=True, fontsize = 12)
plt.show()
