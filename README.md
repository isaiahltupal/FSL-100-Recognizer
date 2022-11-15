# FSL-100-Recognizer
This is the repository for the project titled <b>"RECOGNIZING FILIPINO SIGN LANGUAGE VIDEO SEQUENCES USING DEEP LEARNING 
TECHNIQUES"</b>
WARNING: this repo is only currently compatible with windows machine. unix compatible code will be published soon
# HOW TO RUN
### STEP 0 Required software and cloning
Make sure you have the latest conda in your environment (Just search how to install it)
### STEP 1 INITIAL SETUP
1. download the clips from the source  NOTE: link of the dataset is yet to be published. if you wish to access it email isaiah_tupal@dlsu.edu.ph with the subject "FSL100 DATASET REQUEST"
2. extract the files and paste it in the ```dataset\clips\``` directory. it should look like this:

```ascii
└───FSL-100-Recognizer
    ├───.vscode
    ├───dataset
    │   └───clips
    │       ├───0
    │       │   └───0.mov
    │       │   └───2.mov
    │       │   └───.
    │       │   └───19.mov
    │       ├───1
    |       |   .
    |       |   .
    |       |   .
    │       ├───101
    │       ├───104

```
3. open a terminal in the repo and go the training directory
```cmd
conda env create -f environment.yml
```
### STEP 2 Pre-processing
1. Preprocess the existing data by running preprocess.py be patient for this will take a while
```
python preprocess.py
```
2. check processed if .pynb files are inside it
```
└───FSL-100-Recognizer
    └───training
        ├───processed
        │   ├───CALENDAR
        │   ├───COLOR
        │   │   ├───X_frames_test.npy
        │   │   ├───X_nodes_test.npy
        │   │   ├───Y_label_test.npy
        │   │   ├───X_frames_train.npy
        │   │   ├───X_nodes_train.npy
        │   |   └───Y_label_train.npy
```
### Step 3: training
1. Preprocess the existing data by running train_models.py. be patient for this will take a while

python train_models.py
2. check if the models are in the ```training\trained_models``` diretory

```
└───FSL-100-Recognizer
    └───training
        ├───trained_models
        │   ├───CALENDARmodel
        │   │   ├───assets
        │   │   └───variables
        │   ├───CALENDARhistory
        .
        .
        .
        .
        │   ├───GREETINGRmodel
        │   │   ├───assets
        │   │   └───variables
            └───GREETINGhistory
```

### optional step: evaluate
run the ```model_analysis.ipynb``` file to get results