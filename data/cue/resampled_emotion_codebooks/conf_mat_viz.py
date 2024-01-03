"""
Load the data in this directory from every file and output a conf matrix for each in jpg format

"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
import os

from BOWaves.utilities.visualization import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

files = os.listdir()
files = [f for f in files if f.endswith('.txt')]

import re


def read_lists_from_file(file_path):
    with open(file_path, 'r') as file:
        #lines = file.readlines()
        mylist = file.read().splitlines()

    y_pred_list = mylist[1] + mylist[2]
    y_pred_list = y_pred_list[2:-1]
    #make into a list with only the numbers
    y_pred_list = y_pred_list.split()
    # now turn those into ints
    y_pred_list = [int(i) for i in y_pred_list]

    #add a comma after every numeral in y_pred_list
    #y_pred_list = y_pred_list.replace(" ", ", ")

    labels_list = mylist[4:]
    labels_list[0] = labels_list[0][1:]
    first_label = list(labels_list[0])
    first_label[0] = ' '
    first_label = ''.join(first_label)
    labels_list[0] = first_label
    labels_list[-1] = labels_list[-1][:-1]

    labels_list_reformatted = []
    for i in labels_list:
        i = i.split()
        i = i[0][1]
        #print(i)
        #i = [int(j) for j in i]
        labels_list_reformatted.append(int(i))

    return y_pred_list, labels_list_reformatted

for file in files:
    y_pred, labels = read_lists_from_file(file)
    print("file: ", file)
    print("predicted: ", y_pred)
    print("true labels: ", labels)

    # BoWav confusion matrix on expert data
    disp_labels_x = ['brain', 'muscle', 'eye', 'Heart',
                     'Line Noise', 'Channel Noise', 'Other']
    #disp_labels_y = ['brain', 'muscle', 'eye']
    disp_labels_y = ['blink', 'neural', 'heart', 'lat eye', 'muscle', 'mixed']

    # cm = confusion_matrix(
    #     y_pred,
    #     labels,
    #     labels=np.arange(7),
    #     normalize='true'
    # )
    # cm = cm[:3]
    cm = confusion_matrix(
        labels,
        y_pred,
        labels=np.arange(7),
        normalize='true'
    )
    cm = cm[:6]
    fig, ax = plot_confusion_matrix(
        cm,
        cmap='viridis',
        title=file[:2],
        display_labels=[disp_labels_x, disp_labels_y],
        xticks_rotation='vertical'
    )

    #show plot
    plt.show()

    # save plot with appropriate filename per subj
    # plt.savefig(file[:-4] + '.jpg')
