from osgeo import gdal
import shapefile
import numpy as np
import os
import time
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
import pickle


def generate_slope_curvature_rasters(dem_path):
    slp_path = dem_path.split('.')[0] + '_slope.sdat'
    plan_curv_path = dem_path.split('.')[0] + '_plan_curv.sdat'
    profile_curv_path = dem_path.split('.')[0] + '_profile_curv.sdat'
    cmd = 'saga_cmd ta_morphometry 0 -UNIT_SLOPE 1 -ELEVATION {} -SLOPE {} -C_PROF {} -C_PLAN {}'.format(dem_path, slp_path, profile_curv_path, plan_curv_path)
    os.system(cmd)
    return slp_path, plan_curv_path, profile_curv_path

def load_train_data(dem_path, slope_path, facc_path, plan_curv_path, profile_curv_path):
    """
    Args:
        dem_path {string}: path to DEM raster
        slope_path {string}: path to Slope raster
        facc_path {string}: path to Flow Accumulation raster
        plan_curv_path {string}: path to Plan Curvature raster
        profile_curv_path {string}: path to Profile Curvature raster
    Returns:        
        data_array {numpy array}: a numpy array of 'n' depth (here n = 5) and size = size of the DEM raster.
        dem {gdal_instance}: DEM raster loaded in the variable dem.
    """        
    # the idea is to get a numpy array having n-depth, with following values in each layers: elevation values,
    # upstream area (A), slope (S), plain curvature, profile curvature, and lastly labels 1 or 0 corresponding 
    # to each cell. These labels will act as ground truth, with 1 indicating that the cell is a channel head, and 
    # 0 indicating otherwise.              
    # the assumption is that all these rasters (DEM, slope, curvatures) will be of same shape
    # TODO; maybe applying fillsinks/fill nodata will be a good idea    
    time_start = time.time()
    dem_array = gdal.Open(dem_path).ReadAsArray()    
    slope_array = gdal.Open(slope_path).ReadAsArray()
    facc_array = gdal.Open(facc_path).ReadAsArray()
    plan_curv_array = gdal.Open(plan_curv_path).ReadAsArray()
    profile_curv_array = gdal.Open(profile_curv_path).ReadAsArray()
    print("shape of dem is ", dem_array.shape)
    print("shape of slope is {}, {}", slope_array.shape)
    print("shape of facc is {}, {}", facc_array.shape)
    print("shape of plan_curv is {}, {}", plan_curv_array.shape)
    print("shape of profile_curv is {}, {}", profile_curv_array.shape)
    data_array = np.dstack((dem_array, slope_array, facc_array, plan_curv_array, profile_curv_array))
    print("it took {} seconds to load all data and convert it to numpy array".format(time.time() - time_start))
    return data_array, dem

def get_channel_heads_pixels(dem, channel_heads_shapefile_path):
    """
    Args:
        dem : gdal instance of the DEM (Digital Elevation Model)
        channel_heads_shapefile_path {string}: path of the channels heads shapefile. (Channel heads should be present as points).
    Returns:
        returns a list of lists containing coordinates of those raster cells on whcih a channel head lies.
    """    
    gt = dem.GetGeoTransform()
    # gt is a list containing the information about the top left coordinate, rotation, and resolution of the dem
    # gt[0] = xmin
    # gt[1] = xres
    # gt[2] = rotation #TODO: update
    # gt[3] = ymax
    # gt[4] = rotation #TODO: update
    # gt[5] = yres
    shp = shapefile.Reader(channel_heads_shapefile_path)
    features = shape.shapeRecords()    
    xy_list = []
    for i in range(len(features)):
        coordinates = features[i].shape.__geo_interface__['coordinates']
        x = (coordinates[0] - gt[0]) // gt[1]
        y = (coordinates[1] - gt[3]) // gt[5]
        xy_list.append([x,y])
    return xy_list

def assign_groundtruth_labels(data_array, xy_list):
    """
    Args:
        data_array {numpy array}: a numpy array of 'n' depth (here n = 5) and size = size of the DEM raster
        xy_list {list}: a list of lists containing coordinates of those raster cells on whcih a channel head lies.
    Returns:
        returns data_array {numpy array} with 'n+1' depth (here n = 5) having ground truth values.
    """
    time_start = time.time()
    data_array = np.dstack(np.zeros((data_array.shape[0], data_array.shape[1])))
    depth = data_array.shape(2) - 1
    for i in range(len(xy_list)):
        x,y = xy_list[i]
        data_array[x,y,depth] = 1
    print("it took {} seconds to assign ground truths".format(time.time() - time_start))
    return data_array

def get_all_data(data_array):
    """
    Args:
        data_array {numpy array}: a numpy array of 'n+1' depth (here n = 5) having ground truth values.
    Returns:
        returns a list with num_columns = n+1 and num_rows = total pixels in the DEM raster.
    """
    time_start = time.time()    
    all_data = np.empty((0, data_array.shape[2]), int)
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            all_data = np.vstack(all_data, data_array[i, j, :])
    np.random.shuffle(all_data)
    print("it took {} seconds to get all data and shuffle it".format(time.time() - time_start))
    return all_data
    
def train_val_split(all_data, validation_percent):
    """
    Args:
        all_data {list}: a list with num_columns = n+1 and num_rows = total pixels in the DEM raster.
        validation_percent {integer}: percentage of data we want to use for validation of our trained model
    Returns:
        train_x, train_y, val_x, val_y
        Splits the data into training and validation sets. 
    """
    train_data = all_data[0:int(all_data.shape[0] * validation_percent / 100), :]
    val_data = all_data[int(all_data.shape[0] * validation_percent / 100) :, :]
    train_x = train_data[:, :train_data.shape[1] - 1]
    train_y = train_data[:, train_data.shape[1] - 1 : train_data.shape[1]]
    val_x = val_data[:, :val_data.shape[1] - 1]    
    val_y = val_data[:, val_data.shape[1] - 1 : val_data.shape[1]]
    return train_x, train_y, val_x, val_y

def oversample_trainData(train_x, train_y):
    """
    Args:
        train_x {list}: list containing training data
        train_y {list}: list containing grountruth labels corresponding to training data
    Returns:
        train_x_resampled {list} and train_y_resampled {list}. Returns dataset mitigating the class imbalance problem.
    """
    resample = RandomOverSampler()
    train_x_resampled, train_y_resampled = resample.fit_sample(train_x, train_y)
    return train_x_resampled, train_y_resampled

def train_and_export(train_x_resampled, train_y_resampled):
    """
    Args:
        train_x_resampled {list}: oversampled training data
        train_y_resampled {list}: groundtruths corresponding to oversampled training data
    Returns:
        Trains and exports the trained model at the specified directory.
    """
    random_forest = RandomForestRegressor(n_estimators=8, max_depth=6, max_features=5, random_state=42) # play with these numbers to get optimum hyperparameters
    random_forest.fit(train_x_resampled, train_y_resampled)
    print("training score = {}".format(random_forest.score(train_x_resampled, train_y_resampled)))
    # print("valdation score = {}".format(random_forest.score(val_x, val_y)))
    val_y_predicted = np.round(random_forest.predict(val_x))
    print("validation set confusion matrix")
    print(confusion_matrix(val_y, val_y_predicted))
    print("validation classification report")
    print(classification_report(val_y, val_y_predicted))
    pickle.dump(random_forest, open(os.path.join(model_out_path, model_name), 'wb'))

#---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # this is part of preprocessing/data creation step. we want to extract those pixel/cell value from DEM/tiff file
    # on which a channel head lies

    dem_path = ""
    channel_heads_shapefile_path = ""
    facc_path = ""
    validation_percent = 5          # percentage of data we want to use for validation 
    model_out_path = ""             # where we want to save our model
    model_name = ""                 # the name we want to give to our trained model

    # TODO: the DEM and the channel heads shapefile must be in the same CRS (Coordinate Reference System), insert a check for this

    slope_path, plan_curv_path, profile_curv_path = generate_slope_curvature_rasters(dem_path)
    loaded_data, dem = load_train_data(dem_path, slope_path, facc_path, plan_curv_path, profile_curv_path)
    xy_list = get_channel_heads_pixels(dem, channel_heads_shapefile_path)
    loaded_data = assign_groundtruth_labels(loaded_data, xy_list)
    all_data = get_all_data(loaded_data)
    train_x, train_y, val_x, val_y = train_val_split(all_data)
    # Our dataset is heavily imbalanced. No. of pixels that are channel heads are much less than the ones which are not.
    # Hence, we will over sample our training data by generating copies of pixels that are channel heads.
    train_x_resampled, train_y_resampled = oversample_trainData(train_x, train_y)
    train_and_export(train_x_resampled, train_y_resampled)