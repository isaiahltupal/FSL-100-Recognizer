import pandas as pd
import numpy as np

#path to project
path_to_project = "C:\\Users\\isaia\\code\\FSL-100-RECOGNIZER\\FSL-100-Recognizer\\"

import tools as tl

#path to information
path_to_labels = path_to_project + "labels.csv"
path_to_test = path_to_project + "test.csv"
path_to_train = path_to_project + "train.csv"


if __name__ == "__main__":
    """
    shit to do
    1. load data
    2. create graph
    3. run model and save history and prediction in a json file 
    """
    labels = pd.read_csv(path_to_labels)
    test = pd.read_csv(path_to_test)
    train = pd.read_csv(path_to_train)
    subset = train.loc[train['category'] == "GREETING"]

    print(labels.tail)

    """
    for full 100 dataset
    """
    subset_train = train.loc[train['category'] == "GREETING"]
    subset_test = test.loc[test['category']=="GREETING"]
    X_nodes_train,X_frames_train,Y_label_train = tl.load_data_from_df(subset_test)
    X_nodes_test,X_frames_test,Y_label_test = tl.load_data_from_df(subset_test)
    print(X_nodes_test.shape)
    print("cum")h