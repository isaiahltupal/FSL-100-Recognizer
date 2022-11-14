"""
Author: isaiah Tupal
This is the script to preprocess the video clips by category. The outputs are this is a folder with the name of the category having the following filenames

X_nodes_train.npy
X_frames_train.npy
Y_label_train.npy
X_nodes_test.npy
X_frames_test.npy
Y_label_test.npy
"""

import pandas as pd
import numpy as np
import pickle
import os

import tools as tl
import STSGraph as STS
import GCN as gcn
import tensorflow as tf
from tensorflow.python.client import device_lib

#path to information
#path to project
path_to_project = "C:\\Users\\isaia\\code\\FSL-100-RECOGNIZER\\FSL-100-Recognizer\\"
#path to preprocessed folder
path_to_processed = path_to_project + "training\\processed\\"

path_to_labels = path_to_project + "labels.csv"
path_to_test = path_to_project + "test.csv"
path_to_train = path_to_project + "train.csv"

#load the key dataframe
labels = pd.read_csv(path_to_labels)
test = pd.read_csv(path_to_test)
train = pd.read_csv(path_to_train)



def process_category(category):


    path_to_category = path_to_processed + category + "\\" #get to the path ofthe processed category
    if not os.path.exists(path_to_category):
      
        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(path_to_category)
    

    subset_train = train.loc[train['category'] == category]
    subset_test = test.loc[test['category']==category]


    """
    ====================== load testing data
    """

    
    X_nodes_test,X_frames_test ,Y_label_test = tl.load_data_from_df(subset_test)
    Y_label_test = Y_label_test-np.min(Y_label_test) #to remove offset since model only takes zero index stuff

   
    #save pre-processed data
    X_nodes_test_path = path_to_category + "X_nodes_test.npy" #to rename
    X_frames_test_path = path_to_category + "X_frames_test.npy" #to rename
    Y_label_test_path = path_to_category + "Y_label_test.npy" #to rename 

    with open(X_nodes_test_path, 'wb') as f:
        np.save(f, X_nodes_test)

    with open(X_frames_test_path, 'wb') as f:
        np.save(f, X_frames_test)

    with open(Y_label_test_path, 'wb') as f:
        np.save(f, Y_label_test)




    """
    ======================= load training data
    """

    X_nodes_train,X_frames_train ,Y_label_train = tl.load_data_from_df(subset_train)
    Y_label_train = Y_label_train-np.min(Y_label_train) #to remove offset since model only takes zero index stuff

   
    #save pre-processed data
    X_nodes_train_path = path_to_category + "X_nodes_train.npy" #to rename
    X_frames_train_path = path_to_category + "X_frames_train.npy" #to rename
    Y_label_train_path = path_to_category + "Y_label_train.npy" #to rename 

    with open(X_nodes_train_path, 'wb') as f:
        np.save(f, X_nodes_train)

    with open(X_frames_train_path, 'wb') as f:
        np.save(f, X_frames_train)

    with open(Y_label_train_path, 'wb') as f:
        np.save(f, Y_label_train)



if __name__ == "__main__":
    

    cat_list = labels['category'].unique()
    print(cat_list)
    for i in range(len(cat_list)):
        process_category(cat_list[i])
    
