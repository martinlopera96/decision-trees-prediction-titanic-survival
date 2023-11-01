import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

path = r'C:\Users\marti\Desktop\MARTÍN\DATA SCIENCE\Platzi\ML_projects\decision_trees\first_model\titanic.csv'
df = pd.read_csv(path, sep=',')

df.head(5)

columns_to_drop = ['Name', 'Fare']
df.drop(columns=columns_to_drop, axis='columns', inplace=True)

df.columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
print(df.shape)
print(df.dtypes)

df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
df.rename(columns={'Sex_male': 'Sex'}, inplace=True)

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=00000)

tree = DecisionTreeClassifier(max_depth=2, random_state=00000)
tree.fit(X_train, y_train)

train_prediction = tree.predict(X_train)
test_prediction = tree.predict(X_test)

train_accuracy = accuracy_score(y_train, train_prediction)
test_accuracy = accuracy_score(y_test, test_prediction)

print('Training Accuracy:', train_accuracy)
print('Test Accuracy:', test_accuracy)

importances = tree.feature_importances_
columns = X.columns

sns.barplot(y=importances, x=columns, palette='bright', saturation=2.0, edgecolor='black', linewidth=2)
plt.title('Feature Importance')
plt.show()


