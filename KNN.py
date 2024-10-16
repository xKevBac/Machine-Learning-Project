#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kevinbach
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler
#%%
# Load the dataset
df = pd.read_csv('stroke_data.csv')
#size of dataset
print("Data size:", df.shape)
print()
#%%
# Show the percentage distribution of the target class
target_distribution = df['stroke'].value_counts(normalize=True) * 100
print("Target Class Percentage Distribution:")
print(target_distribution)
print()
# Check for missing values in dataset columns
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)
print()
#%%
# Remove rows with missing values
df_cleaned = df.dropna()
# Show the percentage distribution of the target class
target_distribution = df_cleaned['stroke'].value_counts(normalize=True) * 100
print("Target Class Percentage Distribution After remove rows:")
print(target_distribution)
print()

X = df_cleaned.drop('stroke', axis=1)
y = df_cleaned['stroke']

#size of dataset
print("Data size after removing:", df_cleaned.shape)
print()

df_distribute = df_cleaned['stroke'].value_counts()
print(df_distribute)
#%%
# Standardization using StandardScaler
scaler = MinMaxScaler()  # Use MinMaxScaler instead of StandardScaler
X = scaler.fit_transform(X)

# Split the data into training and testing sets (80/20 split for demonstration)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
trainsize = 40907 * .8
print("Train set size:",trainsize)
print()
testssize = 40907 *.2
print("Test set size:",testssize)
print()
#%%
# Initialize the KNeighborsClassifier
# k=1; Accuraxy Score= 0.92
clf = KNeighborsClassifier(n_neighbors=3, weights='uniform')  # Set the number of neighbors as desired

clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

#%%
# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy Score:', accuracy)
print()
# Print the classification report
print(classification_report(y_test, y_pred))
print()
# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Test Error:', mse)
print()

# Perform cross-validation
cv_scores = cross_val_score(clf, X, y, cv=10)
# Print the cross-validation scores
print('Cross-Validation Scores:', cv_scores)
print()
print('Mean CV Score:', cv_scores.mean())
print()
#%%
#code for mean CV vs KNN:: comment out for the run time to print CM
"""
# Initialize lists to store results
k_values = list(range(1, 21))  # k values from 1 to 20
cv_scores_mean = []
cv_scores_std = []

# Perform cross-validation for different k values
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=10)  # 5-fold cross-validation
    cv_scores_mean.append(cv_scores.mean())
    cv_scores_std.append(cv_scores.std())

# Plotting the cross-validation scores
plt.errorbar(k_values, cv_scores_mean, yerr=cv_scores_std, fmt='-o', color='b')
plt.title('Cross-Validation Score vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Cross-Validation Score')
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()
"""
#%%
# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
