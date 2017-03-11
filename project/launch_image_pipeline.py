import re
import os
import time
import argparse
import pickle
import cv2
from sklearn.externals import joblib

from project.utils import pipeline

if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="path to the folder that contains all images for calibration",
                        default='test_images/')
    parser.add_argument("--model", help="path to a .pkl file that contains model+scaler+settings_classifier",
                        default='data/classif_RGB.pkl')
    parser.add_argument("--output_dir", help="path to the folder that contains all images for calibration",
                        default='output_images')
    parser.add_argument("--save_inter", help="save intermediate images (optional)", action='store_true')

    args = parser.parse_args()

    # ###################################### ARGUMENTS ######################################
    input_dir = args.input_dir
    output_dir = args.output_dir
    save_inter = args.save_inter
    model_file = args.model

    print("=================================")
    print("input_dir : {}".format(input_dir))
    print("save_inter : {}".format(save_inter))
    print("output_dir : {}".format(output_dir))
    print("model file : {}".format(model_file))
    print("=================================")
    ###########################################################################################

    # regex pattern to image file
    image_regex = re.compile(r'.*\.(jpg|png|gif)$')

    # list files into `input_dir` that match to the correct regex form
    # using regex (instead of glog.glob) can handle multiple file formats.
    input_files = [filename for filename in os.listdir(os.path.abspath(input_dir))
                  if image_regex.match(filename) is not None]

    # Add 'input_dir'
    input_files = [os.path.join(input_dir, file) for file in input_files]

    print("Files are : ")
    [print("\t{}".format(file)) for file in input_files]

    # load calibration coeff
    # with open(model_file, mode='rb') as f:
        #svc, scaler, settings_classifier = pickle.load(f)

    data = joblib.load(model_file)
    # data = joblib.load('models/clf_9869.pkl')
    # svc = data['model']
    clf = data['model']
    settings_classifier = data['settings']

    print("\nsvc : {}".format(type(clf)))
    print("settings_classifier : \n\t{}\n".format(settings_classifier))

    for file in input_files:
        bounding_boxes_image = pipeline(file, output_dir, save_inter, clf, settings_classifier, filepath=True)
        image_name = os.path.split(file)[-1]
        cv2.imwrite(os.path.join(output_dir, 'final_'+image_name), bounding_boxes_image)

    print("\nScript took {} seconds\n\n".format(time.time() - start))
