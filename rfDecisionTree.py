#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
#%%
# Load the dataset
# Load the dataset
df = pd.read_csv('stroke_data.csv', sep=",")

# Handle missing data using SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

X = df_imputed.drop('stroke', axis=1)
y = df_imputed['stroke']

#%%
# Split the data into training and testing sets (80/20 split for demonstration)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(max_leaf_nodes=200)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

#%%
# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy Score:', accuracy)
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
# Calculate and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)
print()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Plot the decision tree
plt.figure(figsize=(20, 20))
plot_tree(clf.estimators_[0], filled=True, feature_names=X.columns, class_names=['No Stroke', 'Stroke'])
plt.title('Decision Tree from RandomForestClassifier')
plt.savefig('RF_decision_tree_plot.png')
plt.show()
