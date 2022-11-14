"""
Author: Isaiah Tupal

This is my main scratch file where I test the funcitons of this project

"""


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

#path to project
path_to_project = "C:\\Users\\isaia\\code\\FSL-100-RECOGNIZER\\FSL-100-Recognizer\\"
#path to results of main
path_to_results = "C:\\Users\\isaia\\code\\FSL-100-RECOGNIZER\\FSL-100-Recognizer\\results\\"

import tools as tl
import STSGraph as STS
import GCN as gcn
import tensorflow as tf
from tensorflow.python.client import device_lib

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
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    
    labels = pd.read_csv(path_to_labels)
    test = pd.read_csv(path_to_test)
    train = pd.read_csv(path_to_train)

    
    def train_category(category):
        """     
        trains a category set (signs of 10-13 ) and then has the following output:
        ----------------------------------------------------
        Saves: 
            1. History (accuracy and loss) of the training
            2. the model itself
            3. Predicts the test and saves it in numpy

        """

        subset_train = train.loc[train['category'] == category]
        subset_test = test.loc[test['category']==category]

        X_nodes_train, _ ,Y_label_train = tl.load_data_from_df(subset_train)

        Y_label_train = Y_label_train-np.min(Y_label_train) #to remove offset since model only takes zero index stuff

        """         
        #save pre-processed data
        X_nodes_path = path_to_project + "X_nodes.npy" #to rename
        X_frames_path = path_to_project + "X_frames.npy" #to rename
        Y_label_path = path_to_project + "Y_label_path.npy" #to rename 
        
        X_nodes_train = np.load(X_nodes_path,allow_pickle=True)
        #X_frames_train = np.load(X_frames_path,allow_pickle=True)
        Y_label_train = np.load(Y_label_path,allow_pickle=True)

        """

        spektral_data = STS.STS_DATASET(X_nodes_train)

        #train model
        history,model = gcn.train_model(spektral_data,Y_label_train)
        del X_nodes_train

        #save history 
        history_path = path_to_results + category + "history" #to rename
        with open(history_path, 'wb') as file:
            pickle.dump(history.history, file)

        #save model
        model_path = path_to_results + category + "model"
        model.save(model_path)

        #test results


        #test data results is a dict of {"evaluate":evaluate,"predict":predict}
        """         
        test_results = {}
        test_results["evaluate"] = model.evaluate(spektral_data.read_as_tensor(),Y_label_test)
        test_results["predict"] = model.predict(X_nodes_test.read_as_tensor())
        test_results_path = path_to_results + category  + "test_results"
        with open(test_results_path, 'wb') as file:
            pickle.dump(test_results, file) 
        """
    
    
    def train_GREETING():
        """     
        trains a category set (signs of 10-13 ) and then has the following output:
        ----------------------------------------------------
        Saves: 
            1. History (accuracy and loss) of the training
            2. the model itself
            3. Predicts the test and saves it in numpy

        """

        #subset_train = train.loc[train['category'] == category]
        #subset_test = test.loc[test['category']==category]

        #X_nodes_train, _ ,Y_label_train = tl.load_data_from_df(subset_train)

        #save pre-processed data
        X_nodes_path = path_to_project + "X_nodes.npy" #to rename
        X_frames_path = path_to_project + "X_frames.npy" #to rename
        Y_label_path = path_to_project + "Y_label_path.npy" #to rename 
        
        X_nodes_train = np.load(X_nodes_path,allow_pickle=True)
        #X_frames_train = np.load(X_frames_path,allow_pickle=True)
        Y_label_train = np.load(Y_label_path,allow_pickle=True)

    

        spektral_data = STS.STS_DATASET(X_nodes_train)

        #train model
        history,model = gcn.train_model_generator(spektral_data,Y_label_train)
        del X_nodes_train

        #save history 
        history_path = path_to_results + "GREETINGS_gen" + "history" #to rename
        with open(history_path, 'wb') as file:
            pickle.dump(history.history, file)

        #save model
        model_path = path_to_results + "GREETINGS_gen" + "model1"
        model.save(model_path)

        #test results


        #test data results is a dict of {"evaluate":evaluate,"predict":predict}
        """         
        test_results = {}
        test_results["evaluate"] = model.evaluate(spektral_data.read_as_tensor(),Y_label_test)
        test_results["predict"] = model.predict(X_nodes_test.read_as_tensor())
        test_results_path = path_to_results + category  + "test_results"
        with open(test_results_path, 'wb') as file:
            pickle.dump(test_results, file) 
        """
    
    train_GREETING()
