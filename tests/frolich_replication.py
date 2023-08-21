"""
This test script is for replicating Frolich's results.

First load the data from the dataloaders file and get the ICs and expert labels.
Then, run a nested CV setup and examine classifier performance.
"""
import os
from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV
from sklearn.linear_model import LogisticRegression  # Multinomial Logistic Regression
import numpy as np
import datetime

import BOWaves.utilities.dataloaders as dataloaders

#load all the ICs and labels

#create a data structure containing the ICs and labels

frolich_ics = {'ICs': [], 'labels': []}

#for file in directory frolich data
frolich_data = os.listdir('../data/frolich')

#filter out subdirectories such as /img
frolich_data = [file for file in frolich_data if not os.path.isdir(file)]

for file in frolich_data:
    ICs, labels = dataloaders.load_and_visualize_mat_file_frolich('../data/frolich/' + file, visualize=False)
    frolich_ics['ICs'].append(ICs)
    frolich_ics['labels'].append(labels)

#now we have a data structure containing all the ICs and labels

# Load your dataset and features
X, y = frolich_ics['ICs'], frolich_ics['labels']  # Your data and labels

# want to classify ICs, not subjects. So arrange it so that X is all ICs, and y is all the labels, concatenated
X = np.concatenate(X, axis=1)
y = np.concatenate(y, axis=1)

print(X, y)

# Define the directory for output
output_directory = '../data/frolich/results'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Define the output file path
output_file_path = os.path.join(output_directory, f'results_{date}.txt')

# Outer Leave-One-Out (LOO) cross-validation
outer_cv = LeaveOneOut()
outer_scores = []

with open(output_file_path, "w") as output_file:
    for train_index, test_index in outer_cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Inner 5-fold cross-validation
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Define the hyperparameters grid
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }

        # Inner GridSearchCV
        inner_search = GridSearchCV(estimator=LogisticRegression(multi_class="multinomial"),
                                    param_grid=param_grid, cv=inner_cv)
        inner_search.fit(X_train, y_train)

        # Get the best hyperparameters from inner CV
        best_params = inner_search.best_params_

        # Train the model with the best hyperparameters on the entire training set
        model = LogisticRegression(multi_class="multinomial", **best_params)
        model.fit(X_train, y_train)

        # Evaluate the model on the outer test set
        score = model.score(X_test, y_test)
        outer_scores.append(score)

        # Write results to the output file
        output_file.write(f'Test Index: {test_index}\n')
        output_file.write(f'Best Hyperparameters: {best_params}\n')
        output_file.write(f'Model Score: {score}\n')
        output_file.write('-' * 40 + '\n')

# Calculate the average performance over all outer folds
average_score = np.mean(outer_scores)
with open(output_file_path, "a") as output_file:
    output_file.write(f'Average Accuracy: {average_score}\n')

#run this on Caviness, it won't run on my laptop
