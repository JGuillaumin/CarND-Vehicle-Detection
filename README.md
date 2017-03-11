# Vehicle Detection

[//]: # (Image References)
[image2]: ./output_images/windowstest3.jpg
[image3]: ./output_images/hot_windowstest3.jpg
[image4]: ./output_images/final_test3.jpg


## HOG features and Vehicle classifier

I wrote a script that trains a classifier (SVM from sklearn) given parameters saved in `project/settings_classifier.py`.

To launch this script :

```bash
python launch_training_classifier.py --name "classifier_YUV" --output_dir "data/"
```

It trains a new classifier, with data stored into `data/`.

It saves the trained classifier as a Sklearn Pipeline in 'output_dir'.
(Pipeline : scaler + classifier).


My best classifier was obtained with those parameters :

```python
settings_classifier = dict()
settings_classifier['color_space'] = 'YUV'
settings_classifier['spatial_size'] = (16, 16)
settings_classifier['hist_bins'] = 16
settings_classifier['orient'] = 8
settings_classifier['pix_per_cell'] = 8
settings_classifier['cell_per_block'] = 2
settings_classifier['hog_channel'] = 0
settings_classifier['spatial_feat'] = True
settings_classifier['hist_feat'] = True
settings_classifier['hog_feat'] = True
```

Feature extraction is the same as in the course.
With those parameters :
 - 8792 cars
 - 8968 notcars
 - **2384 features per sample**

HOG features are extracted with
    `get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True)`

In `project/utils.py`, line 16 through 34.

Other features are extracted in ` extract_features()` line 60 through 112.

With the selected HOG features, I trained a linear SVM, within a Sklearn Pipeline :

(lines 70 to 93 in `project/launch_training_classifier.py`)

```python
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    clf = Pipeline([('scaling', StandardScaler()),
                    ('classification', LinearSVC(loss='hinge')),
                    ])

    clf.fit(X_train, y_train)

    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
```


It gave me a classifier with **accuracy=0.9859** on test set.

Then, the classifier is saved with `joblib` from sklearn, for later use in Window Search part.


## Sliding Window Search

I used a multi-scale sliding window search to find cars in images.

I kept 3 scale : 64x64, 96x96 and 128x128

```python
    scales = [(64, 64), (96, 96), (128, 128)]
    overlaps = [0.75, 0.75, 0.75]
    y_start_stops = [[400, 500], [400, 500], [400, 600]]
    x_start_stops = [[None, None], [None, None], [None, None]]
    colors = [(255, 0, 0), (128, 128, 0), (0, 128, 128)]
```

It gives us this search policy / grid search :

[Image2]


Hot (activated by the classifier) windows are (confidence threshold = 0.7) :

[Image3]


## HeatMap estimation and False positive

I compute the HeatMap with the same method as in the course.

Then I use thresholds to eliminate false positive detections.

In addition, I apply ROI extraction, to set to zero the heatmap in some parts of the image (like the left bottom of the image).

With the thresholded HeatMap, I apply `label` from `scipy.ndimage.measurements` to draw one bounding bax per car.

[Image4]

## Image and Video Pipeline :

I created two scripts to launch the pipeline :
- `project/launch_image_pipeline.py` : apply the pipeline on images and save intermediate outputs
- `project/launch_video_pipeline.py` : apply the pipeline to a video


```bash
python project/launch_image_pipeline.py --input_dir "test_images/" --output_dir "output_images"  \
    --model "data/classif_YUV.pkl" --save_inter
```


```bash
python project/launch_video_pipeline.py --input_ideo "project_video.mp4" --output_video "output_project_video.mp4"  \
    --model "data/classif_YUV.pkl"
```


See `output_project_video.mp4`.

## Discussion

- The classifier seems to have difficulties to detect the back car, it gives me the same result when I choose other channels for HOG feature extraction.

- There is no time dependencies between frames. So the bounding boxes are not really stable. So one possible improvement might be an history of the center and size of the bounding box of past frames.