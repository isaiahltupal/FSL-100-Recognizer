"""
Author: Isaiah Jassen L. Tupal
Description: this file is for functions for loading the video and then getting the keypoints as well as the frames 

TODO

fix singular data loader

"""

from tqdm import tqdm
from pathlib import Path



import numpy as np
import mediapipe as mp
import cv2
import os


#constants
IMG_SIZE = 64
STANDARD_FPS = 8

BODY_KEYPOINTS = 25
HAND_KEYPOINTS  = 21

KEYPOINT_NUMBERS = BODY_KEYPOINTS  + HAND_KEYPOINTS*2
FEATURES_PER_NODE = 4
RGB_CHANNELS = 3
SEQ_LENGTH = 30
#mediapipe tools

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

#path to dataset
path_to_training= Path(os.getcwd() )
path_to_project = Path(os.path.abspath(os.path.join(path_to_training, os.pardir)))
path_to_dataset = path_to_project / "dataset" 


def load_video(path, max_frames=SEQ_LENGTH,standard_fps = STANDARD_FPS):
    print(path)
    cap = cv2.VideoCapture(str(path))
  
    fps = cap.get(cv2.CAP_PROP_FPS) # get fps
    frame_skip = int(fps/standard_fps)-1
    frames = []
    try:
        with tqdm(total = SEQ_LENGTH ,position=0, leave=True, desc="video loader progress") as load_pbar:
          while True:
              load_pbar.update()
              for i in range(frame_skip):
                ret, frame = cap.read() 
              ret, frame = cap.read() 
              if not ret:
                cap.release()
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
              frame = frame[:, :, [2, 1, 0]]
              frames.append(frame)
              if len(frames) == max_frames:
                  break
    finally:
        cap.release()
    return np.array(frames)



def get_body_keypoints(image):
  """
  gets the total keypoints of a single image using the mediapipe framework

  parameters
  ------------
  image: np.array

  """

  #"get the normalized 0 to 1 pose of the results of the keypoints"

  #get key points till 25
  #for now, only get up 
  #features: x, y, visibility (z can be included next time but i just dont see its point rn)

  with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0,
    min_detection_confidence=0.5) as pose:
    image_height, image_width, _ = image.shape
    results = pose.process(image)


    if not results.pose_landmarks:
      return np.zeros(BODY_KEYPOINTS,FEATURES_PER_NODE) #increase dimensionality if you are gonna include 3d 
    
    #get  keypoints
    body_keypoint_features = [ (results.pose_landmarks.landmark[i].x,
                                results.pose_landmarks.landmark[i].y,
                                results.pose_landmarks.landmark[i].z,
                                results.pose_landmarks.landmark[i].visibility) 
                                for i in range(BODY_KEYPOINTS) ]

    keypoint_features = np.array(body_keypoint_features)

    #return shape is node x feature

    return keypoint_features



def get_hand_keypoints(image):  

  #hand shape = (21,4)  
  #generate list of keypoints ill use
  #hand_keypoints = [i for i in range(HAND_KEYPOINTS)]
  empty_template = np.zeros((HAND_KEYPOINTS,FEATURES_PER_NODE))
  empty_hand_keypoints = np.append(empty_template,empty_template,axis=0)

  with mp_hands.Hands(
    static_image_mode=True,
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
    # Convert the BGR image to RGB before processing.
    results = hands.process(image)
    
    # Print handedness and draw hand landmarks on the image.

    if not results.multi_hand_landmarks:
      return np.copy(empty_hand_keypoints)
    image_height, image_width, _ = image.shape
    key_point_one = empty_template
    key_point_two = empty_template

    #get keypoints of first hand
    result_one = results.multi_hand_landmarks[0]
    key_point_one = np.array([(result_one.landmark[i].x,
      result_one.landmark[i].y,
      result_one.landmark[i].z,
      1) for i in range(HAND_KEYPOINTS)])
    
    #check if hands exists
    if len(results.multi_hand_landmarks) == 2:
      result_two = results.multi_hand_landmarks[1]
      key_point_two = np.array([(result_two.landmark[i].x,
      result_two.landmark[i].y,
      result_two.landmark[i].z,
      1) for i in range(HAND_KEYPOINTS)])

    #stack in order of left right
    if results.multi_handedness[0] == "Left":
      hand_keypoint = np.append(key_point_one,key_point_two,axis=0)
      return hand_keypoint
    #else rearrange the hand_keypoints
    else:
      hand_keypoint = np.append(key_point_two,key_point_one,axis=0)
      return hand_keypoint



def get_keypoints(image):
  hand_keypoints = get_hand_keypoints(image)
  body_keypoints = get_body_keypoints(image)
  return np.append(body_keypoints,hand_keypoints,axis=0)



#data loader
'''
input: df of the video you want to load 
outputs:
  keypoints [videos, frames,nodes, features per node] 
  rgb data [videos, size, size]

'''

#helper for data loader, makes sure that the aspect ratio is saved 
def pad_to_square(frame):
  y, x = frame.shape[0:2]
  if y>x:
    #padd horizontally
    padding = int((y-x)/2)
    frame = cv2.copyMakeBorder(frame, 0, 0, padding, padding, cv2.BORDER_CONSTANT, None, value = 0)
  elif x>y:
    #padd vertically
    padding = int((x-y)/2)
    frame = cv2.copyMakeBorder(frame, padding, padding, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)
  return frame



def get_features_from_video(video,resize=(IMG_SIZE, IMG_SIZE)):
    
    '''
      keypoints [videos, frames,nodes, features per node] 
  rgb data [videos sequence length, size, size]
    '''
    
    nodes = np.empty((SEQ_LENGTH,KEYPOINT_NUMBERS,FEATURES_PER_NODE))
    frames = np.empty((SEQ_LENGTH,IMG_SIZE,IMG_SIZE,RGB_CHANNELS))
    with tqdm(total = SEQ_LENGTH ,position=0, leave=True, desc="keypoint extraction progress") as feature_pbar:
      for ith_frame in range(SEQ_LENGTH):

        #get keypoints/nodes first 
        frame_nodes = get_keypoints(video[ith_frame])

        #comment for removing append
        #frame_nodes = np.expand_dims(frame_nodes,axis=0) #for the shape to be 1, nodes, features per node
        #nodes =  np.append(nodes,frame_nodes,axis=0)
        nodes[ith_frame] = frame_nodes
        
        frame = np.copy(video[ith_frame])
        frame = pad_to_square(frame) #test, so nothign will be removed
        frame = cv2.resize(frame, resize) 
        #frame = np.expand_dims(frame,axis=0) #for the shape to be 1, size size, 3
        #frames = np.append(frames,frame,axis=0) 
        frames[ith_frame] = frame
        feature_pbar.update()
      return nodes,frames



def load_data_from_df(df):
  df.reset_index()

  video_num = df.shape[0]

  X_nodes = np.empty((video_num,SEQ_LENGTH,KEYPOINT_NUMBERS,FEATURES_PER_NODE))
  X_frames = np.empty((video_num,SEQ_LENGTH,IMG_SIZE,IMG_SIZE,RGB_CHANNELS))
  Y_label  = np.empty((video_num))

  with tqdm(total = video_num ,position=0, leave=True, desc="total_video_processed") as total_pbar:
    for i in range(df.shape[0]):
      
      path = path_to_dataset / Path(df.iloc[i].vid_path) # coz i used ms filepath format
      video = load_video(path)
      nodes,frames = get_features_from_video(video) # get the features
      
      #nodes = np.expand_dims(nodes,axis=0) # add dims to be able to add to the data
      #frames = np.expand_dims(frames,axis=0) # add dims to be able to add to the data

      Y_label[i] = int(df.iloc[i].id_label)
      X_nodes[i] = nodes
      X_frames[i] = frames

      total_pbar.update()
  return X_nodes,X_frames,Y_label



def load_singular_data(complete_path,label=0):


    X_nodes = np.empty((0,SEQ_LENGTH,KEYPOINT_NUMBERS,FEATURES_PER_NODE))
    X_frames = np.empty((0,SEQ_LENGTH,IMG_SIZE,IMG_SIZE,RGB_CHANNELS))
    Y_label  = np.empty([])


      
    video = load_video(complete_path)
    nodes,frames = get_features_from_video(video) # get the features
    
    nodes = np.expand_dims(nodes,axis=0) # add dims to be able to add to the data
    frames = np.expand_dims(frames,axis=0) # add dims to be able to add to the data

    Y_label = np.append(Y_label,label)
    X_nodes = np.append(X_nodes,nodes,axis=0)
    X_frames = np.append(X_frames,frames,axis=0)

    return X_nodes,X_frames,Y_label
