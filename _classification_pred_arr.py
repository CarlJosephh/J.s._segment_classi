# # # Classify prediction arrays
# # imports
import geopandas as gpd
from rasterio.mask import mask
import rasterio
import numpy as np
import glob
import os
import time

# # import the X y training arrays

# training data directory
train_dir = r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU\Data' \
            r'\Zwergstrauch_Praktikum\Data\classification_arrays\training_X_y'

# testing data directory
test_dir = r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU\Data' \
           r'\Zwergstrauch_Praktikum\Data\classification_arrays\testing_X_y'

# create lists of all the paths contained per directory
train_paths = glob.glob(os.path.join(train_dir, '*train.npy'))
test_paths = glob.glob(os.path.join(test_dir, '*test.npy'))
for path in test_paths:
    print(path)

# dictionaries for the types of training data
X_y_cls_all = {'name': 'cls_all'}  # data split into classes
X_y_bin_all = {'name': 'bin_all'}  # all absence classes merged
X_y_bin_abs = {'name': 'bin_abs'}  # only manually sampled absences used for training

# iterating througha list of the training dictionaries to append the respective training datasets to them
dict_list = [X_y_cls_all, X_y_bin_all, X_y_bin_abs]
for dict in dict_list:
    dict['X_train'] = np.load([path for path in train_paths if dict['name'] + '_X' in path][0])
    dict['y_train'] = np.load([path for path in train_paths if dict['name'] + '_y' in path][0])
    ##############
    dict['X_test'] = np.load([path for path in test_paths if dict['name'] + '_X' in path][0])
    dict['y_test'] = np.load([path for path in test_paths if dict['name'] + '_y' in path][0])
# the dataset containing all unique training classes is further used
print(X_y_cls_all['name'])
print(X_y_cls_all['X_train'].shape)
print(X_y_cls_all['y_train'].shape)
print(X_y_cls_all['X_test'].shape)
print(X_y_cls_all['y_test'].shape)

# # set paths for data import (segmentation & prediction images) and create lists with filepaths

# segmentation files clipped to the habitats of interest, where Juniperon sabinae occurs
hoi_dir = r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU' \
          r'\Data\Zwergstrauch_Praktikum\Data\ecognition\ROI_full_segmentation\segments\Final\clipped\hoi'
# using glob and os to create a list containing the path to each mapsheet's segmentation
hoi_paths = glob.glob(os.path.join(hoi_dir, '*hoi_clip.shp'))

# prediction images
pred_dir = r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU\Data' \
           r'\Zwergstrauch_Praktikum\Data\raster\classi_orthos\images_with_predictors'
# using glob and os to create a list containing the path to each mapsheet's predictor image
pred_paths = glob.glob(os.path.join(pred_dir, '*mrg.tif'))

# make a list containing all mapsheet numbers of the area that is moddeled
ms_numb = [path[-21:-17] for path in hoi_paths]

# print('dataset lengths: \n',
#       len(hoi_paths),'segment paths; \n',
#       len(pred_paths), 'seg clipped; \n',
#       len(ms_numb), 'mapsheets \n''
# )

# print(hoi_paths[0])
# print(pred_paths[0])
# print(ms_numb[0])
print(ms_numb)


# # list paths to prediction arrays and indices with dropped rows
pred_arr_dir = r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU\Data' \
               r'\Zwergstrauch_Praktikum\Data\classification_arrays\prediction\hoi/'
pred_arr_paths = glob.glob(os.path.join(pred_arr_dir, '*arr.npy'))

crop_ind_dir = r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU\Data' \
               r'\Zwergstrauch_Praktikum\Data\classification_arrays\prediction\crop_indices\hoi/'
crop_ind_paths = glob.glob(os.path.join(crop_ind_dir, '*ind.npy'))

# # check training data
print(np.unique(X_y_cls_all['y_train']))
print(X_y_cls_all['X_train'].shape)

# # Train random forest classifier
X_y = X_y_cls_all # define training data used
# import  sklearn classifier
from sklearn.ensemble import RandomForestClassifier
# create and train the classifier
rand_f = RandomForestClassifier(max_features='sqrt', criterion='entropy', random_state=0)
rand_f.fit(X_y['X_train'], X_y['y_train'])


# # create functions to add rules to the classified shapefile
# calculating a local ndvi mean for each Juniperus segment and adding it to a row
# mean consists of mean ndvi of segments within a specified radius (excluding other juniperus segments)
def calculate_local_mean_noj(row):
    if row['LC18_27_6'] == '90':
        segment_geometry = row['geometry']
        # Create a buffer around the segment's geometry with the specified radius
        buffer_geometry = segment_geometry.buffer(proximity_radius)
        # Get the indices of segments that intersect with the buffer
        possible_neighbors = list(classi_poly_sindex.intersection(buffer_geometry.bounds))
        # Filter the GeoDataFrame to include only potential neighbors
        neighbor_candidates = classified_clean.iloc[possible_neighbors]
        # take Juniperus out of the neighbour candidates
        filtered_neighbor_candidates = neighbor_candidates[neighbor_candidates["LC18_27_6"] != '90']
        # Calculate the mean NDVI of the potential neighbors
        local_mean = filtered_neighbor_candidates['mean_NDVI'].mean()
        return local_mean
    else:
        return None

# mean consists of mean ndvi of segments within a specified radius (including other juniperus segments)
def calculate_local_mean(row):
    if row['LC18_27_6'] == '90':
        segment_geometry = row['geometry']
        # Create a buffer around the segment's geometry with the specified radius
        buffer_geometry = segment_geometry.buffer(proximity_radius)
        # Get the indices of segments that intersect with the buffer
        possible_neighbors = list(classi_poly_sindex.intersection(buffer_geometry.bounds))
        # Filter the GeoDataFrame to include only potential neighbors
        neighbor_candidates = classified_clean.iloc[possible_neighbors]
        # Calculate the mean NDVI of the potential neighbors
        local_mean = neighbor_candidates['mean_NDVI'].mean()
        return local_mean
    else:
        return None


# defining threshholds and based on that decide wheather to accept juniperus segments
def assign_attribute(row):
    if row['LC18_27_6'] == '90':
        if (row['mean_NDVI'] > row['loc_m_nojs'] + 0.065 and
                row['2'] <= 0.1 and
                row['80'] <= 0.355 and
                row['VHM'] <= 3):
            return 1
        else:
            return 2
    else:
        return 0


# predict data for the whole image
class_list = ['1', '2', '3', '31', '32', '4', '45', '5', '6', '80', '90']
for number in ms_numb:
    if os.path.isfile(r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU\Data' \
                      r'\Zwergstrauch_Praktikum\Data\shapes\classified_JS\hoi/' + number + 'prediction.shp'):
        # if wanted to overwrite old files, comment the following two lines
        pass
    else:
        part_start_time = time.time()
        pred_array_path = [path for path in pred_arr_paths if number in path][0]
        poly_path = [path for path in hoi_paths if number in path][0]
        crop_index = [path for path in crop_ind_paths if number in path][0]

        predictors = np.load(pred_array_path)
        classi_ar = rand_f.predict(predictors)  # y-file with labels that can be joined with polygons
        proba_prediction = rand_f.predict_proba(predictors)  # [:,-1]  for js (90)
        print(np.unique(classi_ar, return_counts=True), 'classified array')

        # find indices of Js predicted segments
        # pres_tuple = np.where(classi_ar == '90')
        # pres_index = np.unique(pres_tuple[0])
        # print(len(pres_index))
        # open the segmentation shapefile
        segmentation = gpd.read_file(poly_path, engine="pyogrio")
        print(len(segmentation), 'segments')

        # delete segments with invalid data
        classified_clean = segmentation.drop(index=np.load(crop_index))
        print(len(classified_clean), 'cleaned segments')
        # add class column to segment gdf
        classified_clean['LC18_27_6'] = classi_ar
        classified_clean['mean_NDVI'] = predictors[:, 4]
        classified_clean['VHM'] = predictors[:, 11]
        classified_clean['DHM'] = predictors[:, 6]
        # add prediction probabilities to the shapefile
        for i in range(11):
            classified_clean[class_list[i]] = proba_prediction[:, i]
        # select segments predicted as Js
        # predicted_js = segmentation_clean.iloc[pres_index]
        # print(len(predicted_js))
        # print(predicted_js.type)
        proximity_radius = 10
        classi_poly_sindex = classified_clean.sindex
        classified_clean['local_mean_ndvi'] = classified_clean.apply(calculate_local_mean, axis=1)
        classified_clean['loc_m_nojs'] = classified_clean.apply(calculate_local_mean_noj, axis=1)
        classified_clean['js_filter'] = classified_clean.apply(assign_attribute, axis=1)

        Juniperus_sabina = classified_clean[
            (classified_clean['LC18_27_6'] == '90') & (classified_clean['js_filter'] == 1)]

        # write Js prediction to a shapefile and n array
        classified_clean.to_file(r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU\Data' \
                                 r'\Zwergstrauch_Praktikum\Data\shapes\classified_JS\hoi/' + number + '_prediction.shp')

        Juniperus_sabina.to_file(r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU\Data' \
                                 r'\Zwergstrauch_Praktikum\Data\shapes\classified_JS\hoi\J_sabinae/' + number + '_js_prediction.shp')

        np.save(r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU\Data' \
                r'\Zwergstrauch_Praktikum\Data\classification_arrays\classified\hoi/' + number + '_classif_ar.npy',
                np.column_stack([predictors, classi_ar]))
        part_end_time = time.time()
        print(f'time for classifying mapsheet {number}: ', part_end_time - part_start_time)

# In[ ]:


# create one cell (to write detailed file and another) to directly apply the rules


# In[ ]:


# abs_tuple = np.where(classi_ar != '90')
# abs_index = np.unique(abs_tuple[0])
# js_probas = np.delete(proba_prediction, abs_index, 0)
# print(len(js_probas))
# plt.boxplot(js_probas)