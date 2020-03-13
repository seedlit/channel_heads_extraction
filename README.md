# channel_heads_extraction
Extracting channel heads using Digital Elevation Models (DEM).

The input to this script should be in the form of .csv file. Suitable changes in the script should be made according to the dataset.
The dataset should be split into training set, validation set, and the test set prior to running this script.

Update: March 13, 2020
Use train_channel_heads_v2.py to train the model, and predict_channel_heads_v2.py to extract the channel heads.
The input to train_channel_heads_v2.py should be DEM (Digital Elevation Model) in grid format, and channel heads 
shapefile.
