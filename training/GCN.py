from spektral.data import SingleLoader, BatchLoader
from spektral.datasets import TUDataset
from spektral.layers import GCSConv, GlobalSumPool, GraphMasking, MinCutPool, GCNConv, \
GlobalAttentionPool, GeneralConv, GINConv

import tensorflow as tf
import tools as tl
import pickle

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


total_nodes =tl.SEQ_LENGTH*tl.KEYPOINT_NUMBERS
EPOCHS = 500

def gcn():


    #GCN branch: Spektral library
    #node_features = 
    #preprocess adjacency matrix -- self loops

    node_feat_input = tf.keras.layers.Input(shape=(total_nodes,tl.FEATURES_PER_NODE), name='node_feature_inp_layer')
    graph_input_adj = tf.keras.layers.Input(shape=(total_nodes,total_nodes), sparse=True, name='graph_adj_layer')
    x = GCNConv(24, activation="relu",kernel_regularizer='l2')([node_feat_input, graph_input_adj])  
    x = GCNConv(16, activation="relu")([x,graph_input_adj])
    #gnn_branch = GlobalAttentionPool(4)(x)
    gnn_branch = GlobalSumPool()(x)


    dense = tf.keras.layers.Dense(512, activation = 'relu')(gnn_branch)
    #output layer: action prediciton
    output_layer = tf.keras.layers.Dense(10, activation = 'Softmax')(dense)
    #put model together
    merged_model = tf.keras.models.Model(inputs=[ node_feat_input, graph_input_adj],
                                        outputs=[output_layer])
    #compile mode
    merged_model.compile(optimizer='adam', 
                         weighted_metrics=['acc'],
                         loss='mse')

    return merged_model


def get_model(path):
    
    model = gcn()
    learning_rate = .001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=100,
        decay_rate=0.9)




    opt = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["acc"])
    return model



#trains the model
def train_model(spektral_data,Y_label):

    model = gcn()
    learning_rate = .001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10000,
        decay_rate=0.9)




    opt = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["acc"])
    print(model.summary())
    ylabels = Y_label.astype(int).reshape(Y_label.shape)
    print(ylabels.shape)


    history = model.fit_generator(x=spektral_data.read_as_tensor(), y=ylabels, batch_size=16 ,epochs=EPOCHS)
    return history,model




#function to make it train in batches instead of putting the entirety of it in the model
def train_model_generator(spektral,Y_label,batch_size=32):
    
    model = gcn()
    learning_rate = .001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10000,
        decay_rate=0.9)

    
    opt = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["acc"])
    print(model.summary())

    steps_per_epoch = int( spektral.num_vid / batch_size) 
    #initialize generator
    train_generator = spektral.batch_generator(Y_label,batch_size)
    history = model.fit(x=train_generator,epochs=EPOCHS,batch_size=batch_size,steps_per_epoch=steps_per_epoch)
    return history,model