import pyrootutils
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from numpy.random import default_rng
import joblib, datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

pyrootutils.set_root(path='/work/cniel/ajmeek/BOWaves/BOWaves', pythonpath=True)

import BOWaves.utilities.dataloaders as dataloaders
from BOWaves.sikmeans.sikmeans_core import shift_invariant_k_means, _assignment_step


def bag_of_waves(raw_ics, codebooks):
    """
    Creates a bag-of-words representation of the input data using the codebooks.

    The codebooks should be a concatenation of all the centroids learned above, one for each class.

    Parameters
    ----------
    codebooks - the array of centroids. I.e., neural['centroids']

    Returns
    -------
    X: matrix of shape (n_ics, n_features)
        The bag-of-words representation of the input data.
        (note to self, this should definitely be a sparse matrix right?)
    """

    #n_ics is the number of ICs from all classes in the codebooks list
    n_ics = sum(len(codebook['ICs']) for codebook in codebooks)

    #n_centroids is the number of centroids from all classes in the codebooks list
    n_centroids = sum(len(codebook['centroids']) for codebook in codebooks)

    x_squared_norms = None
    X = np.zeros((n_ics, n_centroids), dtype=codebooks[0]['centroids'].dtype)

    for ic in range(n_ics):
        for centroid in range(n_centroids):
            nu, _, _ = _assignment_step(codebooks[centroid]['ICs'][ic], codebooks[centroid]['centroids'], x_squared_norms)

            nu, counts = np.unique(nu, return_counts=True)

            i_feature = nu + centroid * n_centroids
            X[ic, i_feature] = counts
    return X

#frolich_ics = {'ICs': np.array([]), 'labels': np.array([])}
frolich_ics = {'ICs': [], 'labels': []}

#for file in directory frolich data
frolich_data = os.listdir('../data/frolich')

#filter out subdirectories such as /img
frolich_data = [file for file in frolich_data if not os.path.isdir(file)]

for file in frolich_data:
    ICs, labels = dataloaders.load_and_visualize_mat_file_frolich('../data/frolich/' + file, visualize=False)
    frolich_ics['ICs'].extend(ICs)
    frolich_ics['labels'].extend(labels)

X_train, X_test, y_train, y_test = train_test_split(frolich_ics['ICs'], frolich_ics['labels'], test_size=0.2, random_state=42)

if len(X_train) != len(y_train):
    raise ValueError('X_train and y_train are not the same length.')


# Forgot what the classes were. check on Caviness
neural = {'name': 'neural', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
blink = {'name': 'blink', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
muscle = {'name': 'muscle', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
mixed = {'name': 'mixed', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
lateyes = {'name': 'lateyes', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
heart = {'name': 'heart', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}

all_classes = [neural, blink, muscle, mixed, lateyes, heart]

for i in range(len(X_train)):
    if y_train[i] == 'neural':
        neural['ICs'].append(X_train[i])
    elif y_train[i] == 'blink':
        blink['ICs'].append(X_train[i])
    elif y_train[i] == 'muscle':
        muscle['ICs'].append(X_train[i])
    elif y_train[i] == 'mixed':
        mixed['ICs'].append(X_train[i])
    elif y_train[i] == 'lateyes':
        lateyes['ICs'].append(X_train[i])
    elif y_train[i] == 'heart':
        heart['ICs'].append(X_train[i])
    else:
        raise ValueError('Unknown class label: ' + y_train[i])

#grab windows from the ICs
#not all ICs have the same length, so must take windows. This is what we pass in to sikmeans anyways. Different windows for each class.

#what is the window len? Set hyperparams above. Later on will modify to be chosen from grid search in nested cross validation.
window_len = 256 #what's the sampling rate again?

windows_per_class = {'neural': [], 'blink': [], 'muscle': [], 'mixed': [], 'lateyes': [], 'heart': []}

tot_num_windows = 0 #to house total number of windows for all ICs
for label in all_classes:
    #iterate through class and collect info about sequence lengths
    ic_lengths = []
    n_ics = len(label['ICs']) #need to change, this thinks it's talking about a str not the dict. way to do what I want in python?
    for ic in label['ICs']:
        ic_lengths.append(len(ic))

    #Currently, assuming that we are not taking a subset of the ICs at all. Carlos had the option for that in his earlier window code.
    #So the number of windows per ic will just be the length of each ic / win len.
    n_windows_per_ic = [ic_len // window_len for ic_len in ic_lengths]
    tot_num_windows += sum(n_windows_per_ic)

    #Now that we have the number of windows per ic, we can create the windows.

    rng = np.random.RandomState(42)

    X = np.zeros((tot_num_windows, window_len)) #X is for each class. Stack later
    win_start = 0
    for label in all_classes:
        for ic in label['ICs']:
            windows_per_ic = len(ic) // window_len
            time_idx = np.arange(0, len(ic)-window_len+1, window_len)
            time_idx = rng.choice(time_idx, size=windows_per_ic, replace=False)
            time_idx = time_idx[:, None] + np.arange(window_len)[None, :]

            # #print all relevant info up to this point for debugging
            # print("----------------------------------")
            # #print('ic: ' + str(ic))
            # print('windows_per_ic: ' + str(windows_per_ic))
            # #print('time_idx: ' + str(time_idx))
            # print('win_start: ' + str(win_start))
            # print('win_start + windows_per_ic: ' + str(win_start + windows_per_ic))
            # #print('ic[time_idx]: ' + str(ic[time_idx]))
            # #print('X[win_start:win_start+windows_per_ic]: ' + str(X[win_start:win_start+windows_per_ic]))
            # print('ic[time_idx].shape: ' + str(ic[time_idx].shape))
            # print('X[win_start:win_start+windows_per_ic].shape: ' + str(X[win_start:win_start+windows_per_ic].shape))
            # print('X.shape: ' + str(X.shape))
            # print('ic.shape: ' + str(ic.shape))

            #There seems to be an off by one error here.
            #The very last IC goes past the total num of windows in X.
            #Not sure if it's off by one, since the first portion of X is filled. Perhaps it's about how I calc
            # the total num of ICs? - don't think so. So cut off at the last time.

            if(win_start == tot_num_windows):
                break

            X[win_start:win_start+windows_per_ic] = ic[time_idx]
            win_start += windows_per_ic

    windows_per_class[label['name']] = X
    print("type of first element at every iteration: \t", type(windows_per_class['neural']))
    print(label['name'] + ': ' + str(type(windows_per_class[label['name']])))
    print("type of X: ", type(X))

from BOWaves.sikmeans.sikmeans_core import shift_invariant_k_means
metric, init = 'cosine', 'random'
num_clusters = 16
centroid_len = 256
n_runs = 3
n_jobs = 1
rng = 42#np.random.RandomState(42)

#error checking of type in windows per class
print("after training codebooks")
for label in windows_per_class:
    print(label + ': ' + str(type(windows_per_class[label])))

#need to do this per class.
neural['centroids'], neural['labels'], neural['shifts'], neural['distances'], neural['inertia'], _ = \
    shift_invariant_k_means(windows_per_class['neural'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
blink['centroids'], blink['labels'], blink['shifts'], blink['distances'], blink['inertia'], _ = \
    shift_invariant_k_means(windows_per_class['blink'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
muscle['centroids'], muscle['labels'], muscle['shifts'], muscle['distances'], muscle['inertia'], _ = \
    shift_invariant_k_means(windows_per_class['muscle'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
mixed['centroids'], mixed['labels'], mixed['shifts'], mixed['distances'], mixed['inertia'], _ = \
    shift_invariant_k_means(windows_per_class['mixed'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
lateyes['centroids'], lateyes['labels'], lateyes['shifts'], lateyes['distances'], lateyes['inertia'], _ = \
    shift_invariant_k_means(windows_per_class['lateyes'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
heart['centroids'], heart['labels'], heart['shifts'], heart['distances'], heart['inertia'], _ = \
    shift_invariant_k_means(windows_per_class['heart'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)

#sanity check
for label in windows_per_class:
    print(label + ': ' + str(windows_per_class[label].shape))

codebooks = neural['centroids'] + blink['centroids'] + muscle['centroids'] + mixed['centroids'] + lateyes['centroids'] + heart['centroids']

#split by subject for LOO

subjects = [i+1 for i in range(12)]
bowav_rep_by_sub = [] #bag of waves representation by subject. above list will constitute the labels

#for file in directory frolich data
frolich_data = os.listdir('../data/frolich')

#filter out subdirectories such as /img
frolich_data = [file for file in frolich_data if not os.path.isdir(file)]

for file in frolich_data:
    ICs, labels = dataloaders.load_and_visualize_mat_file_frolich('../data/frolich/' + file, visualize=False)
    bowav_rep = bag_of_waves(ICs, codebooks)
    bowav_rep_by_sub.append(bowav_rep)


#list of classifier hyperparameters


#get data here - want raw_ICs, codebooks. Split between subjects.
#note - should I make a new function for it as Carlos did? - just run the above cell


#pipeline set up, params just what Carlos used
pipe = Pipeline([('scaler', TfidfTransformer()), ('clf', LogisticRegression())])

clf_params = dict(
    clf__class_weight='balanced',
    clf__solver='saga',
    clf__penalty='elasticnet', #Carlos' default. could also be none, l1, or l2
    clf__random_state=np.random.RandomState(13),
    clf__multi_class='multinomial',
    clf__warm_start=True,
    clf__max_iter=1000, #Carlos' default.
)
pipe.set_params(**clf_params)

#will likely want to add alpha and beta params for elasticnet, as Dr. B suggested in a slack message
candidate_params = dict(
    clf__C=[0.1,1,10], #regularization factor
    clf__l1_ratio=[0, 0.2, 0.4, 0.6, 0.8, 1],
)

#I'm not going to parallelize this for now, and I don't need to worry about expert label masks, so just use sklearn's built-in Grid search CV method.
# results = (
#     pipe,
#     candidate_params,
#     X,
#     y,
#     cv,
#     n_jobs=1 #parallelization
# )

grid_search = GridSearchCV(pipe, candidate_params, cv=LeaveOneOut(), scoring='accuracy')
grid_search.fit(bowav_rep_by_sub, subjects)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

cv_results = grid_search.cv_results_

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Assuming 'best_model' is the best model obtained from GridSearchCV
joblib.dump(best_model, f'results/frolich/models/{date}.pkl')

# Assuming 'best_model' is the best model obtained from GridSearchCV
y_pred = best_model.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
#plt.show()

plt.savefig(f'results/frolich/conf_matrices/{date}.png')