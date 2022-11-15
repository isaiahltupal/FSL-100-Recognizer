"""
Author: Isaiah Tupal

This is the Graph Convolutional Neural Network model file. Edit the learning rate and the epoch in the file to your liking
"""


from spektral.data import SingleLoader, BatchLoader
from spektral.datasets import TUDataset
from spektral.layers import GCSConv, GlobalSumPool, GraphMasking, MinCutPool, GCNConv, \
GlobalAttentionPool, GeneralConv, GINConv

import tensorflow as tf
import tools as tl


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

LEARNING_RATE = 0.001
total_nodes =tl.SEQ_LENGTH*tl.KEYPOINT_NUMBERS
EPOCHS = 500
DECAY_STEPS = int(EPOCHS)*5 
DECAY_RATE = 0.1
#good decay rate is 0.01 and lr 0.001 at epoch = 25
#best so far is lr 0.005, decay 100*5, lr at .001
"""
The model being used tfor the project, edit this to make changes on your model
"""
def gcn(output_size):


    #GCN branch: Spektral library
    #node_features = 
    #preprocess adjacency matrix -- self loops


    
    node_feat_input = tf.keras.layers.Input(shape=(total_nodes,tl.FEATURES_PER_NODE), name='node_feature_inp_layer')
    graph_input_adj = tf.keras.layers.Input(shape=(total_nodes,total_nodes), sparse=True, name='graph_adj_layer')
    x = GCNConv(256, activation="relu",kernel_regularizer=tf.keras.regularizers.L2(0.01))([node_feat_input, graph_input_adj])  
    x = GCNConv(256, activation="relu",kernel_regularizer=tf.keras.regularizers.L2(0.01))([x, graph_input_adj])  

    x = tf.keras.layers.Dropout(0.5, seed=69)(x)
    

    #gnn_branch = GlobalAttentionPool(4)(x)
    


    x = GlobalSumPool()(x)


    
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)


    #output layer: action prediciton
    output_layer = tf.keras.layers.Dense(output_size, activation = 'Softmax')(x)
    #put model together
    merged_model = tf.keras.models.Model(inputs=[ node_feat_input, graph_input_adj],
                                        outputs=[output_layer])
    #compile mode
    merged_model.compile(optimizer='adam', 
                         weighted_metrics=['acc'],
                         loss='mse')

    return merged_model


def get_model(path,output_size):
    
    model = gcn(output_size)
    learning_rate = LEARNING_RATE
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE)




    opt = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["acc"])
    return model



#trains the model
def train_model(spektral_data,Y_label,output_size=10):

    model = gcn(output_size)
    learning_rate = LEARNING_RATE
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE)




    opt = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["acc"])
    print(model.summary())
    ylabels = Y_label.astype(int).reshape(Y_label.shape)
    print(ylabels.shape)


    history = model.fit_generator(x=spektral_data.read_as_tensor(), y=ylabels, batch_size=16 ,epochs=EPOCHS)
    return history,model




#function to make it train in batches instead of putting the entirety of it in the model
def train_model_generator(spektral,Y_label,batch_size=32,output_size=10):
    
    model = gcn(output_size)
    learning_rate = LEARNING_RATE
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE)

    #save best in terms of accuracy

    
    opt = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["acc"])
    print(model.summary())

    steps_per_epoch = int( spektral.num_vid / batch_size) 
    #initialize generator
    train_generator = spektral.batch_generator(Y_label,batch_size)
    history = model.fit(x=train_generator,epochs=EPOCHS,batch_size=batch_size,steps_per_epoch=steps_per_epoch)
    return history,model