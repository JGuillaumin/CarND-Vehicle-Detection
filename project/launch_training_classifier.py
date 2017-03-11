import numpy as np
import time
import glob
import argparse
import os
from project.settings_classifier import settings_classifier
from project.utils import extract_features
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pickle

if __name__ == "__main__":
    t_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name for the model/classifier ",
                        default='classifier')
    parser.add_argument("--output_dir", help="output folder, where the trained model will be saved",
                        default='data/')

    args = parser.parse_args()

    # ############################# ARGUMENTS #############################
    output_dir = args.output_dir
    name = args.name

    if not os.path.isdir(output_dir):
        print("Ouput_dir not found")
        exit()

    # ######################################################################

    # data
    cars_files = glob.glob('data/vehicles/*/*')
    notcars_files = glob.glob('data/non-vehicles/*/*')

    color_space = settings_classifier['color_space']
    spatial_size = settings_classifier['spatial_size']
    hist_bins = settings_classifier['hist_bins']
    orient = settings_classifier['orient']
    pix_per_cell = settings_classifier['pix_per_cell']
    cell_per_block = settings_classifier['cell_per_block']
    hog_channel = settings_classifier['hog_channel']
    spatial_feat = settings_classifier['spatial_feat']
    hist_feat = settings_classifier['hist_feat']
    hog_feat = settings_classifier['hog_feat']

    t = time.time()
    car_features = extract_features(cars_files, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars_files, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract the features ...')

    # Normalize features and create labels
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    print(" {} samples".format(len(y)))
    print("\t {} cars".format(len(car_features)))
    print("\t {} notcars".format(len(notcar_features)))
    print("\t {} features".format(X.shape))

    # Shuffle and split data for training and validation
    rand_state = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    clf = Pipeline([('scaling', StandardScaler()),
                    ('classification', LinearSVC(loss='hinge')),
                    ])

    # Check the training time for the SVC
    t = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    # Check the prediction time for a single sample

    # To save :
    # - X_scaler
    # - settings_classifier
    # - svc

    # pickle.dump((clf, settings_classifier), open(os.path.join(output_dir, name+'.pkl'), "wb")
    joblib.dump({'model': clf, 'settings': settings_classifier}, os.path.join(output_dir, name+'.pkl'))
    print("Classifier, Scaler and setting are saved.")

    print("Script took {} seconds. ".format(time.time() - t_start))
