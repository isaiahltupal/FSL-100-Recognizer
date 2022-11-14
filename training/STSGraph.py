"""
Author: Isaiah Jassen L. Tupal
Description: this file is for the Spatio-temporal Skeleton graph traphs
"""


from spektral.data import Dataset, DisjointLoader, Graph
import tensorflow as tf
import numpy as np

BODY_KEYPOINTS = 25
HAND_KEYPOINTS  = 21

KEYPOINT_NUMBERS = BODY_KEYPOINTS  + HAND_KEYPOINTS*2



class STS_DATASET(Dataset):
    """
    A dataset of spatio-temporal skeleton graphs describing someone 

    """

    def __init__(self, X_nodes,  **kwargs): #X_nodes is (num_vid, frames per vid, nodes, features per node)
        self.num_samples = X_nodes.shape[0]
        self.STS_Graph_list = [] #empty

        self.X_list = []
        self.adj_list = []

        self.num_vid = X_nodes.shape[0]
        self.frames_per_vid = X_nodes.shape[1]
        self.nodes_per_frame = X_nodes.shape[2]
        self.features_per_node = X_nodes.shape[3]

        self.adj_template = self.build_adj_mat() # template for the STS graph
        for i in range(self.num_samples):
          ithgraph,ithgraph_bare = self.make_STS_graph(X_nodes[i])
          self.STS_Graph_list.append(ithgraph)
          self.X_list.append(ithgraph_bare[0])
          self.adj_list.append(ithgraph_bare[1])

        super().__init__(**kwargs)


    
    def make_STS_graph(self,X_graph):
      #build an sts graph
      #total_nodes is nodes*frames per vid

      nodes_per_frame = X_graph.shape[1]
      frame_num = X_graph.shape[0]
      feature_per_node = X_graph.shape[2]

      x_arr = X_graph.reshape((nodes_per_frame*frame_num, feature_per_node)) #reshape to make X matrix

      a_arr = self.adj_template #single graph template for all of the classes

      return Graph(x=x_arr,a = a_arr),(x_arr,a_arr) # make the graph



    #builds the adjaceny matrix
    def build_adj_mat(self):
      num_frames = self.frames_per_vid
      num_nodes_per_frame = self.nodes_per_frame
      


      if num_nodes_per_frame == 67: # if 24 mediapipe nodes
        #build zero array
        total_nodes = num_frames*num_nodes_per_frame
        adj = np.zeros((total_nodes,total_nodes))

        #build self connections
        for i in range(num_nodes_per_frame):
          adj = self.make_undirected_edge((i,i),adj)

        #build temporal connections first 
        for ith_frame in range(num_frames-1):
          for joints in range(num_nodes_per_frame):
            curr_joint_node = num_nodes_per_frame * ith_frame + joints
            next_frame_joint_node = num_nodes_per_frame * (ith_frame+1) + joints
            adj = self.make_undirected_edge((curr_joint_node,next_frame_joint_node),adj) #the previous keypoint in that node and the future node is connected. this is not bidirectional btw
        
        #build skeleton connection
        for ith_frame in range(num_frames):
          adj = self.build_skeleton_edges(adj,ith_frame)

        return adj




    def build_skeleton_edges(self,adj,ith_frame):
      offset = ith_frame*self.nodes_per_frame
      
      skeleton_pairs = [ (0,1), (0,4), (2,1), (2,3), (3,7), (4,5), (5,6), (6,8), (10,9),# face
              (12,11),(12,24),(24,23),(23,11), #body
              (11,13),(13,15),(15,21),(15,17),(15,19),(19,17), #right arm
              (12,14),(14,16),(16,22),(16,18),(16,20),(20,18) #left arm
      ]

      hand_pair = [(0,1), (1,2), (2,3), (3,4), #thumb
              (0,5),(0,6),(0,7),(0,8), #index finger
              (9,10), (10,11), (11,12), #middle finger
              (13,14) , (14,15), (15,16), #ring 
              (0,16), (17,18), (18,19), (19,20), #pinky
              (5,9), (9,13), (13,17), #knuckles
              ]
 
      for edge_pair in skeleton_pairs:
        adj = self.make_undirected_edge(edge_pair,adj,offset=offset)
      
      for edge_pair in hand_pair:
        adj = self.make_undirected_edge(edge_pair,adj,offset=offset + BODY_KEYPOINTS)
        adj = self.make_undirected_edge(edge_pair,adj,offset=offset + BODY_KEYPOINTS+HAND_KEYPOINTS)

      return adj

    
    #makes an undirected edge
    def make_undirected_edge(self,pair,adj,offset=0):
      source = pair[0] + offset
      dest = pair[1] + offset
      adj[source][dest] = 1
      adj[dest][source] = 1
      return adj

    
    #makes a directed edge between two nodes. the format is adjacency[source][destination]
    def make_directed_edge(self,source,dest,adj):
      adj[source][dest] = 1
      return adj

    #returns the value as a tensor. it makes it faster. Returns the list [ X[numvid][totalnodes][features] adj[numvid][totalnodes][totalnodes] ]
    def read_as_tensor(self):
      X_tf = tf.convert_to_tensor(self.X_list,dtype=tf.float32)
      Adj_tf = tf.convert_to_tensor(self.adj_list,dtype=tf.float32)
      return  [X_tf,Adj_tf]

    #returns the graph (d)
    def read(self):
        return self.STS_Graph_list

    #returns the np array of the graph dataset
    def read_bare(self):
        return (np.array(self.X_list),np.array(self.adj_list))


    #data generator function using this class
    def batch_generator(self,Y_labels, batch_size = 32):
      indices = np.arange(self.num_vid)
      batch_adj=[]
      batch_X =[]
      batch_Y=[]
      while True:
              # it might be a good idea to shuffle your data before each epoch
              np.random.shuffle(indices) 
              for i in indices:
                  batch_X.append(self.X_list[i])  #pretty self explanatory
                  batch_adj.append(self.adj_list[i])
                  batch_Y.append(Y_labels[i])
                  if len(batch_X) == batch_size:
                      batch_X =  np.array(batch_X)
                      batch_adj = np.array(batch_adj)
                      yield [batch_X, batch_adj], np.array(batch_Y)
                      batch_adj=[] # list is emptyu to reduce memory usage
                      batch_X =[]
                      batch_Y=[]