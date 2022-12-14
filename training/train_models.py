"""
Author: Isaiah Tupal

This is to train the models. run this when you already have the pre_processed folder

TODO:
Change the Filepath from absolute to pathlib coz OMG youre gonna run this in a linux machine 


"""



import pandas as pd
import numpy as np
import pickle




import STSGraph as STS
import GCN as gcn
import os


from pathlib import Path



#path to information
path_to_training= Path(os.getcwd() )
path_to_project = Path(os.path.abspath(os.path.join(path_to_training, os.pardir)))
path_to_models = path_to_project  / "training" / "trained_models" 
path_to_labels = path_to_project / "labels.csv"
path_to_test = path_to_project / "test.csv"
path_to_train = path_to_project / "train.csv"


#load the key dataframe
labels = pd.read_csv(path_to_labels)
test = pd.read_csv(path_to_test)
train = pd.read_csv(path_to_train)
    

#train per category    
def train_category(category):
    """     
    trains a category set (signs of 10-13 ) and then has the following output:
    ----------------------------------------------------
    Saves: 
        1. History (accuracy and loss) of the training
        2. the model itself
        3. Predicts the test and saves it in numpy

    """
    path_to_preprocessed = path_to_project / "training" / "processed" / category
     
    #save pre-processed data
    X_nodes_path = path_to_preprocessed / "X_nodes_train.npy" #to rename
    #X_frames_path = path_to_preprocessed / "X_frames_train.npy" #to rename
    Y_label_path = path_to_preprocessed / "Y_label_train.npy" #to rename 
    
    X_nodes_train = np.load(X_nodes_path,allow_pickle=True)
    #X_frames_train = np.load(X_frames_path,allow_pickle=True)
    Y_label_train = np.load(Y_label_path,allow_pickle=True)

    #convert nodes to graph
    spektral_data = STS.STS_DATASET(X_nodes_train)

    #get output size of model
    unique = np.unique(Y_label_train)
    output_size = unique.shape[0]

    #train_model
    history,model = gcn.train_model_generator(spektral_data,Y_label_train,output_size=output_size)
    del X_nodes_train

    #save history 
    history_path = path_to_models / Path(category + "history") #to rename
    with open(history_path, 'wb') as file:
        pickle.dump(history.history, file)

    #save model
    model_path = path_to_models / Path(category + "model")
    model.save(model_path)

    

if __name__ == "__main__":

    cat_list = labels['category'].unique()
    print(cat_list)
    for i in range(len(cat_list)):
        train_category(cat_list[i])
        break


    