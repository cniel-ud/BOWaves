import pickle
from pathlib import Path
import os

path = Path('candidate_964_split_7.pkl')
#test_file = pickle.load('candidate_964_split_7.pkl')

with open(path, 'rb') as f:
    test_file = pickle.load(f)
    print(test_file)

# load all pickle files. sort by test_score.
# among the top 100, see if there are any parameters identical to all.

# then rerun all jobs with those parameters fixed.
# this may take a lot of memory to sort everything - see how large it is. my laptop has 4 gigs RAM

# store the name of each file in memory with test score, just compare based on that

files = []
candidates = {}

dir = 'clf-lr_penalty-elasticnet_solver-saga_C-0.001_0.01_0.1_1.0_10.0_100.0_1000.0_l1_ratio-0_0.2_0.4_0.6_0.8_1_expert_weight-1.0_2.0_4.0_8.0_16.0_32.0_cbookMinPerIC-50.0_cbookICsPerSubj-2_bowavNorm-l_1'

for file in os.listdir(dir):
    print(file)
    if file.endswith('.pkl'):

        candidate_num = file.split('_')[1]
        if candidate_num not in candidates:
            candidates[candidate_num] = []
        split = file.split('_')[3]
        split = split.split('.')[0]

        # if file == 'candidate_547_split_14.pkl':
        #     # seems to have been corrupted
        #     continue
        try:
            path = Path(dir + '/' + file)
            with open(path, 'rb') as f:
                test_file = pickle.load(f)
                file_info = [test_file['test_scores'], test_file['parameters'], file]
                files.append(file_info)
                candidates[candidate_num].append([split, test_file['test_scores'], test_file['parameters'], file])
        except:
            continue

# sort by first index of list
files.sort(key = lambda x: x[0], reverse=True)

list_of_candidates_with_nonmatching_params_or_insufficient_splits = []
candidates_mean_scores = {}
# get mean test score by candidate and assert that all params are the same
for key in candidates.keys():
    scores = []
    params = []
    for split in candidates[key]:
        scores.append(split[1])
        params.append(split[2])

    if key not in candidates_mean_scores:
        candidates_mean_scores[key] = []

    print(key)
    if len(scores) != 27:
        list_of_candidates_with_nonmatching_params_or_insufficient_splits.append(key)
        continue

    #list_of_splits_with_nonmatching_params_or_insufficient_splits = []
    # check that all params are the same for each candidate
    for i in range(1, len(params)):
        print(key)
        if params[i] != params[0]:
            #list_of_splits_with_nonmatching_scores.append(key)
            list_of_candidates_with_nonmatching_params_or_insufficient_splits.append(key)
            break

    if key not in list_of_candidates_with_nonmatching_params_or_insufficient_splits:
        candidates_mean_scores[key].append([sum(scores)/len(scores), params])

    print('Candidate ' + key + ' mean test score: ' + str(sum(scores)/len(scores))
          + ' number of splits: ' + str(len(scores)))


# now we have a sorted list of files by test score. we can now compare the parameters of the top 100 files
# and see if any are identical. if they are, we can store the parameters in a list and rerun all jobs with those parameters fixed.
identical_params = {}
for i in range(10):
    if i == 0:
        identical_params = files[i][1]
    else:
        for key in identical_params.keys():
            if identical_params[key] != files[i][1][key]:
                identical_params[key] = None

    print(files[i][1])


# count how many in the files have a test score of 1.0
count = 0
for file in files:
    if file[0] == 1.0:
        count += 1

print('Number of files with test score of 1.0: ' + str(count)) # need to count over all splits, get the mean.

print(identical_params)


# turn candidates into list
candidates_list = []
for key in candidates_mean_scores.keys():
    if key not in list_of_candidates_with_nonmatching_params_or_insufficient_splits:
        candidates_list.append([key, candidates_mean_scores[key][0][0], candidates_mean_scores[key][0][1]])

# sort by candidates mean score
candidates_list.sort(key = lambda x: x[1], reverse=True)

print(candidates_list)

# now check if there are any identical parameters among the top 100 candidates
identical_params = {}
for i in range(15):
    if i == 0:
        identical_params = candidates_list[i][2][0] # first param set of the 27 splits fine
    else:
        for key in identical_params.keys():
            if identical_params[key] != candidates_list[i][2][0][key]:
                identical_params[key] = None

    print(candidates_list[i][1], ' : ', candidates_list[i][2][0])

print()