#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kevinbach
"""
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#%%
# Load the dataset
df = pd.read_csv('stroke_data.csv', sep=",")

#size of dataset
print("Data size:", df.shape)
print()

# Show the percentage distribution of the target class
target_distribution = df['stroke'].value_counts(normalize=True) * 100
print("Target Class Percentage Distribution:")
print(target_distribution)
print()
#%%
# Check for missing values in dataset columns
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)
print()

# Remove rows with missing values
df_cleaned = df.dropna()

# Show the percentage distribution of the target class
target_distribution = df_cleaned['stroke'].value_counts(normalize=True) * 100
print("Target Class Percentage Distribution After remove rows:")
print(target_distribution)
print()

X = df_cleaned.drop('stroke', axis=1)
y = df_cleaned['stroke']

print("Data size after removing:", df_cleaned.shape)
print()

df_distribute = df_cleaned['stroke'].value_counts()
print(df_distribute)
#%%
# Split the data into training and testing sets (80/20 split for demonstration)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
trainsize = 40907 * .8
print("Train set size:",trainsize)
print()
testssize = 40907 *.2
print("Test set size:",testssize)
print()
#%%
# Initialize the DecisionTreeClassifier
clf = DecisionTreeClassifier(max_leaf_nodes=300, criterion='gini')

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)
#%%
# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy Score:', accuracy)
print()

testerror = mean_squared_error(y_test, y_pred)
print('Test Error:', testerror)
print()

# Print the classification report
print('This is the Classification Report')
print(classification_report(y_test, y_pred))
print()

# Perform cross-validation
cv_scores = cross_val_score(clf, X, y, cv=10)
# Print the cross-validation scores
print('Cross-Validation Scores:', cv_scores)
print()
print('Mean CV Score:', cv_scores.mean())
print()
#%%
#Code to show the graph of mean CV vs Decision Tree:: comment out for the run time to print CM and tree cell below
#ran this seperately just for the graph
"""
# Perform cross-validation for different max leaf nodes
max_leaf_nodes_list = []
mean_cv_scores = []

for max_leaf_nodes in range(10, 501, 10):
    clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, criterion='gini')
    cv_scores = cross_val_score(clf, X, y, cv=10)
    max_leaf_nodes_list.append(max_leaf_nodes)
    mean_cv_scores.append(cv_scores.mean())

# Plot the line graph
plt.figure(figsize=(10, 6))
plt.plot(max_leaf_nodes_list, mean_cv_scores, marker='o')
plt.title('Mean Cross-Validation Score vs. Max Leaf Nodes')
plt.xlabel('Max Leaf Nodes')
plt.ylabel('Mean CV Score')
plt.grid(True)
plt.xticks(range(10, 501, 50))
plt.tight_layout()
plt.show()
"""
#%%
# Calculate and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)
print()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
#%%
# Plot the decision tree
plt.figure(figsize=(40, 40))
plot_tree(clf, filled=True, feature_names=['sex', 'age', 'hypertension', 'heart_disease', 'ever_married',
                                           'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
                                           'smoking_status'], class_names=['No Stroke', 'Stroke'])
plt.show()
#%%
# Get feature importance
feature_importance = clf.feature_importances_

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Print or visualize the significant attributes
print("Significant Attributes:")
print(feature_importance_df)

# Optionally, you can plot the feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()