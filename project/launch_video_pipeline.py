import pickle
import os
import time
import argparse
import re

import cv2
from moviepy.editor import VideoFileClip
from sklearn.externals import joblib

from project.settings_classifier import settings_classifier
from project.settings_pipeline import settings_pipeline
from project.utils import pipeline

if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", help="path to a video where apply the pipeline",
                        default='project_video.mp4')
    parser.add_argument("--model", help="path to a .pkl file that contains model+scaler+settings_classifier",
                        default='data/classif_YUV.pkl')
    parser.add_argument("--output_video", help="complete path (path + name) for the output",
                        default='output_project_video.mp4')
    args = parser.parse_args()

    # ###################################### ARGUMENTS ######################################
    input_video = args.input_video
    output_video = args.output_video
    model_file = args.model

    print("=================================")
    print("input_video : {}".format(input_video))
    print("output_dir : {}".format(output_video))
    print("model file : {}".format(model_file))
    print("=================================")
    ###########################################################################################

    # load calibration coeff
    #with open(model_file, mode='rb') as f:
    #    svc, scaler, settings_classifier = pickle.load(f)

    data = joblib.load(model_file)
    # data = joblib.load('models/clf_9869.pkl')
    # svc = data['model']
    clf = data['model']
    settings_classifier = data['settings']

    print("\nsvc : {}".format(type(clf)))
    print("settings_classifier : \n\t{}\n".format(settings_classifier))

    print("\nsvc : {}".format(type(clf)))
    print("settings_classifier : \n\t{}\n".format(settings_classifier))

    # the function create a custom pipeline with camera matrix 'mtx' and distortion coefficients 'dist'.
    # file is set to false, the first argument 'file' represents an image as array.
    def create_image_pipeline(svc, settings_classifier):
        def image_pipeline(file):
            return pipeline(file, None, False, svc, settings_classifier, filepath=False)
        # returns a function the pre-configurated arguments (python closure)
        return image_pipeline

    image_pipeline = create_image_pipeline(clf, settings_classifier)

    # load input_video
    clip1 = VideoFileClip(input_video)

    # apply the pipeline to the video
    output_clip = clip1.fl_image(image_pipeline)

    # save the new video
    output_clip.write_videofile(output_video, audio=False)

    print("\nScript took {} seconds\n\n".format(time.time() - start))