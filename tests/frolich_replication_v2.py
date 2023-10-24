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
import re
import argparse

pyrootutils.set_root(path='/work/cniel/ajmeek/BOWaves/BOWaves', pythonpath=True)

import BOWaves.utilities.dataloaders as dataloaders
from BOWaves.sikmeans.sikmeans_core import shift_invariant_k_means, _assignment_step

print("Error check - Caviness jobs don't seem to be running") #it's not. will try working on this again later.

parser = argparse.ArgumentParser(description="Your script description here")

parser.add_argument("--loo_subj", type=int, help="The subject to leave out for LOO cross validation.")
args = parser.parse_args()

#error check, print root directory
#print("root directory: ", pyrootutils.find_root(search_from=__file__, indicator=".git"))

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

def train_and_store_codebooks(frolich_ICs_by_subj, loo_subj = None):
    """
    This calculates a codebook for each class, per subject. These are then stored in separate files.

    We want to store by subject and codebook so that we can train on specific subjects and test using codebooks from
    specific held out subjects. This way we avoid contamination for LOO.

    Files saved in data/frolich/codebooks. Other params etc are not necessary currently.

    Sept. 14
    Changing this so that you just pass in all ICs and labels at once, split by subject. Then loo_subj
    calculates the test for that and train for everything else.
    The way I'll pass in the data is through the frolich_ics_by_subj dictionary. Make sure the nested list works.
    Returns
    -------

    """

    # TODO - standardize with an object of command line args, as Carlos had it. clean this script.

    # Here are the hyperparams. Remember, this is for cue dataset with sampling rate 500 Hz (see Frolich et al 2015)

    # Carlos has recommended a centroid length of one second, and a window length of 1.5 times centroid length.

    window_len = 750
    metric, init = 'cosine', 'random'
    num_clusters = 16
    centroid_len = 500
    n_runs = 3
    n_jobs = 1
    rng = np.random.RandomState(42)
    #rng = 42

    # train_subjects = [f'{i+1:02}' for i in range(12)]
    # train_subjects.remove(loo_subj)

    neural_train = {'name': 'neural', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
    blink_train = {'name': 'blink', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
    muscle_train = {'name': 'muscle', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
    mixed_train = {'name': 'mixed', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
    lateyes_train = {'name': 'lateyes', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
    heart_train = {'name': 'heart', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}

    all_classes_train = [neural_train, blink_train, muscle_train, mixed_train, lateyes_train, heart_train]

    neural_test = {'name': 'neural', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
    blink_test = {'name': 'blink', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
    muscle_test = {'name': 'muscle', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
    mixed_test = {'name': 'mixed', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
    lateyes_test = {'name': 'lateyes', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}
    heart_test = {'name': 'heart', 'ICs': [], 'centroids': [], 'labels': [], 'shifts': [], 'distances': [], 'inertia': []}

    all_classes_test = [neural_test, blink_test, muscle_test, mixed_test, lateyes_test, heart_test]

    # train_data = [frolich_ICs_by_subj[subject] for subject in train_subjects]

    for subject in frolich_ICs_by_subj:
        if subject['subject'] != loo_subj:
            print(subject['labels'])

            if len(subject['ICs']) != len(subject['labels']):
                raise ValueError('ICs and labels are not the same length.')

            for i, label in enumerate(subject['labels']):
                if label == 'neural':
                    neural_train['ICs'].append(subject['ICs'][i])
                elif label == 'blink':
                    blink_train['ICs'].append(subject['ICs'][i])
                elif label == 'muscle':
                    muscle_train['ICs'].append(subject['ICs'][i])
                elif label == 'mixed':
                    mixed_train['ICs'].append(subject['ICs'][i])
                elif label == 'lateyes':
                    lateyes_train['ICs'].append(subject['ICs'][i])
                elif label == 'heart':
                    heart_train['ICs'].append(subject['ICs'][i])
                else:
                    raise ValueError('Unknown class label: ' + label)
        else:
            print(subject['labels'])

            if len(subject['ICs']) != len(subject['labels']):
                raise ValueError('ICs and labels are not the same length.')

            for i, label in enumerate(subject['labels']):
                if label == 'neural':
                    neural_test['ICs'].append(subject['ICs'][i])
                elif label == 'blink':
                    blink_test['ICs'].append(subject['ICs'][i])
                elif label == 'muscle':
                    muscle_test['ICs'].append(subject['ICs'][i])
                elif label == 'mixed':
                    mixed_test['ICs'].append(subject['ICs'][i])
                elif label == 'lateyes':
                    lateyes_test['ICs'].append(subject['ICs'][i])
                elif label == 'heart':
                    heart_test['ICs'].append(subject['ICs'][i])
                else:
                    raise ValueError('Unknown class label: ' + label)



    # window_len = 256  # what's the sampling rate again?

    # where am I saving this or calling sikmeans? Think I did this wrong
    # oh. I was doing that before wrapping it in this function.
    # I need to grab the windows first, and then pass all of it to sikmeans.
    windows_per_class = {'neural': [], 'blink': [], 'muscle': [], 'mixed': [], 'lateyes': [], 'heart': []}

    tot_num_windows = 0  # to house total number of windows for all ICs
    for label in all_classes_train:
        # iterate through class and collect info about sequence lengths
        ic_lengths = []
        n_ics = len(label['ICs'])  # need to change, this thinks it's talking about a str not the dict. way to do what I want in python?
        for ic in label['ICs']:
            ic_lengths.append(len(ic))

        # Currently, assuming that we are not taking a subset of the ICs at all. Carlos had the option for that in his earlier window code.
        # So the number of windows per ic will just be the length of each ic / win len.
        n_windows_per_ic = [ic_len // window_len for ic_len in ic_lengths]
        tot_num_windows += sum(n_windows_per_ic)

        # Now that we have the number of windows per ic, we can create the windows.

        rng = np.random.RandomState(42)

        X = np.zeros((tot_num_windows, window_len))  # X is for each class. Stack later
        win_start = 0
        # for label in all_classes:
        for ic in label['ICs']:
            windows_per_ic = len(ic) // window_len
            time_idx = np.arange(0, len(ic) - window_len + 1, window_len)
            time_idx = rng.choice(time_idx, size=windows_per_ic, replace=False)
            time_idx = time_idx[:, None] + np.arange(window_len)[None, :]


            # There seems to be an off by one error here.
            # The very last IC goes past the total num of windows in X.
            # Not sure if it's off by one, since the first portion of X is filled. Perhaps it's about how I calc
            # the total num of ICs? - don't think so. So cut off at the last time.

            if (win_start == tot_num_windows):
                break

            X[win_start:win_start + windows_per_ic] = ic[time_idx]
            win_start += windows_per_ic

        if label['name'] == 'neural':
            windows_per_class['neural'] = X
        windows_per_class[label['name']] = X
        print("type of first element at every iteration: \t", type(windows_per_class['neural']))
        print(label['name'] + ': ' + str(type(windows_per_class[label['name']])))
        print("type of X: ", type(X))

    path = pyrootutils.find_root(search_from=__file__, indicator=".git")

    #now calculate sikmeans on the above windows per class.

    # metric, init = 'cosine', 'random'
    # num_clusters = 16
    # centroid_len = 256
    # n_runs = 3
    # n_jobs = 1
    # rng = 42

    #error checking of type in windows per class
    print("after training codebooks")
    for label in windows_per_class:
        print(label + ': ' + str(type(windows_per_class[label])))

    #need to do this per class.
    neural_train['centroids'], neural_train['labels'], neural_train['shifts'], neural_train['distances'], neural_train['inertia'], _ = \
        shift_invariant_k_means(windows_per_class['neural'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    blink_train['centroids'], blink_train['labels'], blink_train['shifts'], blink_train['distances'], blink_train['inertia'], _ = \
        shift_invariant_k_means(windows_per_class['blink'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    muscle_train['centroids'], muscle_train['labels'], muscle_train['shifts'], muscle_train['distances'], muscle_train['inertia'], _ = \
        shift_invariant_k_means(windows_per_class['muscle'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    mixed_train['centroids'], mixed_train['labels'], mixed_train['shifts'], mixed_train['distances'], mixed_train['inertia'], _ = \
        shift_invariant_k_means(windows_per_class['mixed'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    lateyes_train['centroids'], lateyes_train['labels'], lateyes_train['shifts'], lateyes_train['distances'], lateyes_train['inertia'], _ = \
        shift_invariant_k_means(windows_per_class['lateyes'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    heart_train['centroids'], heart_train['labels'], heart_train['shifts'], heart_train['distances'], heart_train['inertia'], _ = \
        shift_invariant_k_means(windows_per_class['heart'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)


    #now save the codebooks trained on all ICs from each class in the train data.
    #make separate folder for each.
    for label in all_classes_train:
        out_file = path / f'data/codebooks/frolich/loo_{loo_subj}/train_{label["name"]}_minus_{loo_subj}.npz'
        with open(out_file, 'wb') as f:
            np.savez(out_file, centroids=label['centroids'], labels=label['labels'],
                     shifts=label['shifts'], distances=label['distances'], inertia=label['inertia'])

    #now redo whole process for the lone test subject
    # window_len = 256

    windows_per_class = {'neural': [], 'blink': [], 'muscle': [], 'mixed': [], 'lateyes': [], 'heart': []}

    tot_num_windows = 0  # to house total number of windows for all ICs
    for label in all_classes_test:
        # iterate through class and collect info about sequence lengths
        ic_lengths = []
        n_ics = len(label['ICs'])  # need to change, this thinks it's talking about a str not the dict. way to do what I want in python?
        for ic in label['ICs']:
            ic_lengths.append(len(ic))

        # Currently, assuming that we are not taking a subset of the ICs at all. Carlos had the option for that in his earlier window code.
        # So the number of windows per ic will just be the length of each ic / win len.
        n_windows_per_ic = [ic_len // window_len for ic_len in ic_lengths]
        tot_num_windows += sum(n_windows_per_ic)

        # Now that we have the number of windows per ic, we can create the windows.

        rng = np.random.RandomState(42)

        X = np.zeros((tot_num_windows, window_len))  # X is for each class. Stack later
        win_start = 0
        # for label in all_classes:
        for ic in label['ICs']:
            windows_per_ic = len(ic) // window_len
            time_idx = np.arange(0, len(ic) - window_len + 1, window_len)
            time_idx = rng.choice(time_idx, size=windows_per_ic, replace=False)
            time_idx = time_idx[:, None] + np.arange(window_len)[None, :]


            # There seems to be an off by one error here.
            # The very last IC goes past the total num of windows in X.
            # Not sure if it's off by one, since the first portion of X is filled. Perhaps it's about how I calc
            # the total num of ICs? - don't think so. So cut off at the last time.

            if (win_start == tot_num_windows):
                break

            X[win_start:win_start + windows_per_ic] = ic[time_idx]
            win_start += windows_per_ic

        if label['name'] == 'neural':
            windows_per_class['neural'] = X
        windows_per_class[label['name']] = X
        print("type of first element at every iteration: \t", type(windows_per_class['neural']))
        print(label['name'] + ': ' + str(type(windows_per_class[label['name']])))
        print("type of X: ", type(X))

    #now calculate sikmeans on the above windows per class.

    path = pyrootutils.find_root(search_from=__file__, indicator=".git")

    # metric, init = 'cosine', 'random'
    # num_clusters = 16
    # centroid_len = 256
    # n_runs = 3
    # n_jobs = 1
    # rng = 42#np.random.RandomState(42)

    #error checking of type in windows per class
    print("after training codebooks")
    for label in windows_per_class:
        print(label + ': ' + str(type(windows_per_class[label])))

    #need to do this per class.
    neural_test['centroids'], neural_test['labels'], neural_test['shifts'], neural_test['distances'], neural_test['inertia'], _ = \
        shift_invariant_k_means(windows_per_class['neural'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    blink_test['centroids'], blink_test['labels'], blink_test['shifts'], blink_test['distances'], blink_test['inertia'], _ = \
        shift_invariant_k_means(windows_per_class['blink'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    muscle_test['centroids'], muscle_test['labels'], muscle_test['shifts'], muscle_test['distances'], muscle_test['inertia'], _ = \
        shift_invariant_k_means(windows_per_class['muscle'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    mixed_test['centroids'], mixed_test['labels'], mixed_test['shifts'], mixed_test['distances'], mixed_test['inertia'], _ = \
        shift_invariant_k_means(windows_per_class['mixed'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    lateyes_test['centroids'], lateyes_test['labels'], lateyes_test['shifts'], lateyes_test['distances'], lateyes_test['inertia'], _ = \
        shift_invariant_k_means(windows_per_class['lateyes'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    heart_test['centroids'], heart_test['labels'], heart_test['shifts'], heart_test['distances'], heart_test['inertia'], _ = \
        shift_invariant_k_means(windows_per_class['heart'], num_clusters, centroid_len, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)


    #now save the codebooks trained on all ICs from each class in the train data.
    #make separate folder for each.
    for label in all_classes_test:
        out_file = path / f'data/codebooks/frolich/loo_{loo_subj}/test_{label["name"]}_{loo_subj}.npz'
        with open(out_file, 'wb') as f:
            np.savez(out_file, centroids=label['centroids'], labels=label['labels'],
                     shifts=label['shifts'], distances=label['distances'], inertia=label['inertia'])


#test codebooks with 8 train ICs and 2 test ICs


#frolich_ics = {'ICs': np.array([]), 'labels': np.array([])}
frolich_ics = {'ICs': [], 'labels': []}
frolich_ics_by_subject = [{'subject': i, 'ICs': [], 'labels': []} for i in range(12)]

#for file in directory frolich data
frolich_data = os.listdir('../data/frolich')

#filter out subdirectories such as /img
frolich_data = [file for file in frolich_data if not os.path.isdir(file)]

#for regex
pattern = r'\d+'

for file in frolich_data:
    ICs, labels = dataloaders.load_and_visualize_mat_file_frolich('../data/frolich/' + file, visualize=False)
    #frolich_ics['ICs'].extend(ICs)
    #frolich_ics['labels'].extend(labels)

    if [int(match.group()) for match in re.finditer(pattern, file)]:
        subject = [int(match.group()) for match in re.finditer(pattern, file)][0]
        frolich_ics_by_subject[subject-1]['ICs'].extend(ICs)
        frolich_ics_by_subject[subject-1]['labels'].extend(labels)
        print("subject: ", subject)



# TODO - switch to held out by subject. includes modifying the above code to load the data by subject.
#X_train, X_test, y_train, y_test = train_test_split(frolich_ics['ICs'], frolich_ics['labels'], test_size=0.2, random_state=42)
#X_train, y_train = frolich_ics_by_subject[0:10]['ICs'], frolich_ics_by_subject[0:10]['labels']
#X_test, y_test = frolich_ics_by_subject[10:12]['ICs'], frolich_ics_by_subject[10:12]['labels']


# TODO - modify below to do LOO by subject, calling train and store codebooks each time.

# X_train, X_test, y_train, y_test = [], [], [], []
#
# #print("list: ", frolich_ics_by_subject[:10])
# selected_dataframes = [frolich_ics_by_subject[index] for index in range(10)]
# X_train.extend(df['ICs'][0] for df in selected_dataframes)
# y_train.extend(df['labels'][0] for df in selected_dataframes)
#
# selected_dataframes = [frolich_ics_by_subject[index] for index in [10, 11]]
# X_test.extend(df['ICs'][0] for df in selected_dataframes)
# y_test.extend(df['labels'][0] for df in selected_dataframes)
#
# print(selected_dataframes[0]['labels'])
# print("y_train labels: ", y_train)
# if len(X_train) != len(y_train) or len(X_test) != len(y_test):
#     raise ValueError('X and y are not the same length.')

loo_subj = args.loo_subj
train_and_store_codebooks(frolich_ics_by_subject, loo_subj=loo_subj)


# below here is some other notes / code on training clf. just use this file to do codebooks for now

#
# subjects = [i+1 for i in range(12)]
# bowav_rep_by_sub = [] #bag of waves representation by subject. above list will constitute the labels
#
# #for file in directory frolich data
# frolich_data = os.listdir('../data/frolich')
#
# #filter out subdirectories such as /img
# frolich_data = [file for file in frolich_data if not os.path.isdir(file)]
#
# for file in frolich_data:
#     ICs, labels = dataloaders.load_and_visualize_mat_file_frolich('../data/frolich/' + file, visualize=False)
#     bowav_rep = bag_of_waves(ICs, codebooks)
#     bowav_rep_by_sub.append(bowav_rep)
#
#
# #list of classifier hyperparameters
#
#
# #get data here - want raw_ICs, codebooks. Split between subjects.
# #note - should I make a new function for it as Carlos did? - just run the above cell
#
#
# #pipeline set up, params just what Carlos used
# pipe = Pipeline([('scaler', TfidfTransformer()), ('clf', LogisticRegression())])
#
# clf_params = dict(
#     clf__class_weight='balanced',
#     clf__solver='saga',
#     clf__penalty='elasticnet', #Carlos' default. could also be none, l1, or l2
#     clf__random_state=np.random.RandomState(13),
#     clf__multi_class='multinomial',
#     clf__warm_start=True,
#     clf__max_iter=1000, #Carlos' default.
# )
# pipe.set_params(**clf_params)
#
# #will likely want to add alpha and beta params for elasticnet, as Dr. B suggested in a slack message
# candidate_params = dict(
#     clf__C=[0.1,1,10], #regularization factor
#     clf__l1_ratio=[0, 0.2, 0.4, 0.6, 0.8, 1],
# )
#
# #I'm not going to parallelize this for now, and I don't need to worry about expert label masks, so just use sklearn's built-in Grid search CV method.
# # results = (
# #     pipe,
# #     candidate_params,
# #     X,
# #     y,
# #     cv,
# #     n_jobs=1 #parallelization
# # )
#
# grid_search = GridSearchCV(pipe, candidate_params, cv=LeaveOneOut(), scoring='accuracy')
# grid_search.fit(bowav_rep_by_sub, subjects)
#
# best_model = grid_search.best_estimator_
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
#
# cv_results = grid_search.cv_results_
#
# date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#
# # Assuming 'best_model' is the best model obtained from GridSearchCV
# joblib.dump(best_model, f'results/frolich/models/{date}.pkl')
#
# # Assuming 'best_model' is the best model obtained from GridSearchCV
# y_pred = best_model.predict(X_test)
#
# # Create a confusion matrix
# cm = confusion_matrix(y_test, y_pred)
#
# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# #plt.show()
#
# plt.savefig(f'results/frolich/conf_matrices/{date}.png')