# # # Extract prediction features as mean values per segment to a numpy array
# # imports
import geopandas as gpd
from rasterio.mask import mask
import rasterio
import numpy as np
from shapely.geometry import mapping
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

# # # create prediction arrays from prediction images and segmentation gdf's
# function to create data arrays from segments (training data and all segments)

def seg_to_feat_array(seg_path, image_path, d_type):  # distinguish between TD and PD, the latter only needs X
    # read segmentation file and exract geometry values
    seg = gpd.read_file(seg_path, engine="pyogrio")
    seg = seg.set_crs('EPSG:2056')
    geoms = seg.geometry.values

    # extract the raster values within the segmentation shapefile
    with rasterio.open(image_path) as src:
        band_count = src.count  # count image bands
        no_data = src.nodata  # store the image nodata value

        X = np.array([], dtype=np.float_).reshape(0, band_count*2)  # create an array to store segment mean and std
        y = np.array([], dtype=np.string_)  # labels for training

        # for loop iterating through segments by index
        # for every segment calculate mean and std per band
        for index, geom in enumerate(geoms):
            feature = [mapping(geom)]

            # the mask function returns an array of the raster pixels within this feature
            out_image, out_transform = mask(src, feature, crop=True)
            out_image = out_image.astype(np.float32)  # maybe change to float 64
            # print(out_image[11,1,:])
            # print('out_image', out_image)

            # eliminate all the pixels with 0 values for all bands - AKA not actually part of the shapefile
            out_image_trimmed = out_image[:, ~np.all(out_image == 0, axis=0)]
            # eliminate all the pixels with 255 values for all bands - AKA not actually part of the shapefile
            out_image_trimmed = out_image_trimmed[:, ~np.all(out_image_trimmed == 255, axis=0)]
            # eliminate all the pixels with no_data values for all bands - AKA not actually part of the shapefile
            out_image_trimmed = out_image_trimmed[:, ~np.all(out_image_trimmed == no_data, axis=0)]
            # eliminate all the pixels with values >= 65530 for all bands (frequent no data 65535, sometimes sketchy
            # values close to it.
            out_image_trimmed = out_image_trimmed[:, ~np.all(out_image_trimmed >= 65530, axis=0)]
            # print('out_image_trimmed', out_image_trimmed)

            # append to the one dimensional label array (y) for TD
            if d_type == 'TD':
                # append the labels to the y array
                y = np.append(y, [seg["LC18_27_6"][index]])  # * out_image_reshaped.shape[0])

            # stack a list of stats (mean, std) onto the stat array (X)
            stat_values = []
            for i in range(0, band_count):
                s_mean = np.mean(out_image_trimmed[i, :])
                stat_values.append(s_mean)

            for i in range(0, band_count):
                s_std = np.std(out_image_trimmed[i, :])
                stat_values.append(s_std)

            X = np.vstack((X, stat_values))  # out_image_reshaped))


    # make warning, if any segments contain nan values
    if X.size == 0:
        print('no shapefile within:', image_path)
    else:
        if np.isnan(np.min(X)):
            print('segments with no data found', image_path)
            nan_tuple = np.where(np.isnan(X))
            nan_index=np.unique(nan_tuple[0])
            print('where are nan: ', nan_index)
            print('no. of shapefiles with no data: :', len(nan_index))
        else:
            # print('all segments have values')
            pass
    # return X, y if d_type == 'TD' else X
    if d_type == 'TD':
        y = np.array([item.decode() for item in y])
        return X, y
    else:
        return X


start_time = time.time()
for number in ms_numb:
    # check if the file already exists, if so skip to next number
    if os.path.isfile(r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU\Data'\
                       r'\Zwergstrauch_Praktikum\Data\classification_arrays\prediction\hoi/' + number + '_hoi_pred_arr.npy'):
        # if wanted to overwrite old files, comment the following two lines
        pass
    else:
        print(f'no {number} extraction started')
        part_start_time = time.time()  # timestamp
        # find paths of image and segmentation based on the mapsheet number
        image_path = [path for path in pred_paths if number in path][0]
        poly_path = [path for path in hoi_paths if number in path][0]

        # run data extraction function
        X_tmp = seg_to_feat_array(seg_path=poly_path, image_path=image_path, d_type='test')
        print(f'X_y sampled{number}')
        ## clean the dataset from nan values etc.
        # find training rows with invalid data (+-inf, >65530)
        bad_tuple = np.where(np.any(X_tmp>= 65530, axis=1))  
        inf_tuple = np.where(np.isinf(X_tmp))# find training rows with invalid data (+-inf, >65530
        nan_tuple = np.where(np.isnan(X_tmp))
        print(f'{number}values above 65530:', np.count_nonzero(X_tmp>= 65530))
        print('inf values:', np.count_nonzero(np.isinf(X_tmp)))
        print('rows with values above 65530:', np.count_nonzero(np.sum(X_tmp>= 65530, axis=1)))
        print('rows with inf values', np.sum(np.any(np.isinf(X_tmp), axis=1)), '\n')

        # Get the indices of rows with 'inf' values or values above the threshold
        bad_index = np.unique(bad_tuple[0])
        inf_index = np.unique(inf_tuple[0])
        nan_index = np.unique(nan_tuple[0])
        inv_data_index = np.unique(np.hstack((bad_index, inf_index, nan_index)))

        # drop invalid rows from the training dataset
        X_cleaned = np.delete(X_tmp, inv_data_index, 0)
        print('length of cleaned array: ', len(X_cleaned), 'length of old array: ', len(X_tmp))


        np.save(r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU\Data'\
                r'\Zwergstrauch_Praktikum\Data\classification_arrays\prediction\hoi/' + number + '_hoi_pred_arr.npy', X_cleaned)

        np.save(r'\\speedy11-12-fs\data_15\_PROJEKTE\2018_Lebensraumkarte_BAFU\Data'\
                r'\Zwergstrauch_Praktikum\Data\classification_arrays\prediction\crop_indices\hoi/' + number + '_hoi_inv_data_ind.npy', inv_data_index)
        part_end_time = time.time()
        print(f'time for mapsheet {number}: ', part_end_time - part_start_time)

end_time=time.time()
print('elapsed time: ', end_time - start_time)
