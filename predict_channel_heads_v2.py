import pickle
import numpy as np
from osgeo import gdal, ogr, osr
import time
import os

def generate_slope_curvature_rasters(dem_path):
    slp_path = dem_path.split('.')[0] + '_slope.sdat'
    plan_curv_path = dem_path.split('.')[0] + '_plan_curv.sdat'
    profile_curv_path = dem_path.split('.')[0] + '_profile_curv.sdat'
    cmd = 'saga_cmd ta_morphometry 0 -UNIT_SLOPE 1 -ELEVATION {} -SLOPE {} -C_PROF {} -C_PLAN {}'.format(dem_path, slp_path, profile_curv_path, plan_curv_path)
    os.system(cmd)
    return slp_path, plan_curv_path, profile_curv_path

def load_data(dem_path, slope_path, facc_path, plan_curv_path, profile_curv_path):
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
    time_start = time.time()
    dem_array = gdal.Open(dem_path).ReadAsArray()    
    slope_array = gdal.Open(slope_path).ReadAsArray()
    facc_array = gdal.Open(facc_path).ReadAsArray()
    plan_curv_array = gdal.Open(plan_curv_path).ReadAsArray()
    profile_curv_array = gdal.Open(profile_curv_path).ReadAsArray()
    print("shape of dem is ", dem_array.shape)
    print("shape of slope is ", slope_array.shape)
    print("shape of facc is ", facc_array.shape)
    print("shape of plan_curv is ", plan_curv_array.shape)
    print("shape of profile_curv is ", profile_curv_array.shape)
    data_array = np.dstack((dem_array, slope_array, facc_array, plan_curv_array, profile_curv_array))
    print("it took {} seconds to load all data and convert it to numpy array".format(time.time() - time_start))
    return data_array, dem

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
            all_data = np.vstack((all_data, data_array[i, j, :]))
    print("it took {} seconds to get all data".format(time.time() - time_start))
    return all_data

def generate_predictions(model_path, test_data):
    """
    Args:
        model_path {string}: path to our saved model
        test_data {list}: a list with num_columns = n+1 and num_rows = total pixels in the DEM raster
    Returns:
        test_predictions {list}: a list of generated predictions corresponding to the test data
    """
    model = pickle.load(open(model_path, 'rb'))
    test_predictions = np.round(model.predict(test_data))
    print("generated channel head predictions")
    return test_predictions

def generate_prediction_raster(test_predictions, dem):  
    """
    Args:
        test_predictions {list}: a list of generated predictions corresponding to the test data
        dem {gdal_instance}: DEM raster loaded in the variable dem.
    Returns:
        Exports the predictions raster to the specified path (raster_out_path).
    """
    test_predictions = test_predictions.reshape(dem.RasterYSize, dem.RasterXSize)
    gt = dem.GetGeoTransform()
    driver = gdal.GetDriverByName('GTiff')
    proj = dem.GetProjection()
    prediction_raster = driver.Create(raster_out_path, dem.RasterXSize, dem.RasterYSize, gdal.GDT_Byte)
    prediction_raster.SetProjection(proj)
    prediction_raster.GetRasterBand(1).WriteArray(test_predictions)
    prediction_raster.FlushCache()  # saving the channel heads predictions raster     
    print("generated prediction raster!")
    return test_predictions

def generate_prediction_shapefile(test_predictions, dem):
    """
    Args:
        test_predictions {list}: a list of generated predictions corresponding to the test data
        dem {gdal_instance}: DEM raster loaded in the variable dem.
    Returns:
        Exports the predictions shapefile to the specified path (shapefile_out_path).
    """
    srs = osr.SpatialReference()    
    srs.ImportFromWkt(dem.GetProjection())
    driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    shapeData = driver.CreateDataSource(shapefile_out_path)
    layer = shapeData.CreateLayer('ogr_pts', srs, ogr.wkbPoint)
    layerDefinition = layer.GetLayerDefn()
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = dem.GetGeoTransform()
    (y_index, x_index) = np.nonzero(test_predictions == 1)    
    i = 0
    for x_coord in x_index:
        x = x_index[i] * x_size + upper_left_x + (x_size / 2) # adding half the cell size to centre the point    
        y = y_index[i] * y_size + upper_left_y + (y_size / 2) 
        point = osgeo.ogr.Geometry(osgeo.ogr.wkbPoint)
        point.SetPoint(0, x, y)
        feature = osgeo.ogr.Feature(layerDefinition)
        feature.SetGeometry(point)
        feature.SetFID(i)
        layer.CreateFeature(feature)
        i += 1
    shapeData.Destroy()
    print("Generated Shapefile of predcited channel heads. Total no. of channel heads found = {}".format(i))

#-----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    model_path = ""
    dem_path = ""
    facc_path = ""
    raster_out_path = ""
    shapefile_out_path = ""

    slope_path, plan_curv_path, profile_curv_path = generate_slope_curvature_rasters(dem_path)
    loaded_data, dem = load_data(dem_path, slope_path, facc_path, plan_curv_path, profile_curv_path)
    test_data = get_all_data(loaded_data)
    test_predictions = generate_predictions(model_path, test_data)
    test_predictions = generate_prediction_raster(test_predictions, dem)
    generate_prediction_shapefile(test_predictions, dem)
