import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Load dataset
dataset = pd.read_excel('titanic.xls')
dataset = dataset.rename(columns=lambda x: x.strip().lower())

# Clean missing values
dataset = dataset[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']]
dataset['sex'] = dataset['sex'].map({'male': 0, 'female': 1})
dataset['age'] = pd.to_numeric(dataset['age'], errors='coerce')
dataset['age'] = dataset['age'].fillna(np.mean(dataset['age']))
dataset['fare'] = dataset['fare'].fillna(np.mean(dataset['fare']))

# Handle missing embarked values and create dummy variables
dataset['embarked'] = dataset['embarked'].fillna('S')
embarked_dummies = pd.get_dummies(dataset['embarked'], prefix='embarked')
dataset = pd.concat([dataset, embarked_dummies], axis=1)
dataset = dataset.drop(['embarked'], axis=1)

# Features and target variable
X = dataset.drop(['survived'], axis=1)
y = dataset['survived']

# Check for any remaining missing values
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    raise ValueError("There are still missing values in the dataset.")

# Scaling features
sc = MinMaxScaler(feature_range=(0, 1))
X_scaled = sc.fit_transform(X)

# Model fitting
log_model = LogisticRegression(C=1)
log_model.fit(X_scaled, y)

# Saving model and scaler as pickle files
pickle.dump(log_model, open("titanic_survival_ml_model.sav", "wb"))
pickle.dump(sc, open("scaler.sav", "wb"))

# Optional: Evaluate the model (useful for debugging and validation)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
