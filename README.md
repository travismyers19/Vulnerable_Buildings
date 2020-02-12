# Vulnerable Building Detection
A soft-story building is a multiple-story building in which the first floor is "soft", meaning it consists mostly of garage door openings or windows or other openings.  Soft-story buildings are particularly vulnerable to earthquakes, and thus the ability to quickly inventory soft-story buildings in an area may be of use to city or state officials.  This repo contains code to create and train a custom Inception V3 image classifier to detect soft-story buildings and deploy the model in a user-friendly Streamlit app.

## Overview
The Streamlit app takes as input from the user latitude and longitude coordinates specifying a bounding box region of the world and also a number of addresses.  It takes random latitude and longitude locations within that bounding box and uses reverse geocoding to obtain the addresses.  Then it uses those addresses to obtain Google Street View images.  The images are then sent through the image classifier, the bad images are discarded, and the results are presented to the user: "Soft-Story" means that the image is of a soft-story building, and "Non-Soft-Story" means that the image is of a non-soft-story buildng.

There are two types of models that can be used:

- Ternary classification:  A single model classifies the images into the three categories:  "Soft", "Non-Soft", and "Bad Image" (a bad image is one in which there is no building in the image or the building is obscured or it's unclear which building the image is an image of).

- Binary classification:  One model classifies the image as good or bad, and then a second model classifies the image as soft or non-soft.

## Installation
Clone the Github repo:

```
git clone https://github.com/travismyers19/Vulnerable_Buildings
```

## Initial Setup

### If using the Ubuntu Deep Learning AMI:

Activate the tensorflow 2.0 with Python 3.6 environment:

```
source activate tensorflow2_p36
```

Install Streamlit:

```
pip install streamlit
```

### If not using the Ubuntu Deep Learning AMI:
Create a conda environment from "configs/environment.yml:

```
conda env create -f configs/environment.yml
```

Activate the conda environment:

```
source activate tensorflow2_p36
```

## Setting the PYTHONPATH and changing directories:
Move into the `Vulnerable_Buildings` folder:

```
cd Vulnerable_Buildings
```

Use "build/environment.sh" to add the "Modules" folder to the PYTHONPATH:

```
source build/environment.sh
```

All command line commands below assume that the user is in the top level directory of this repo.

## Modules
The `Modules` folder contains all of the custom modules in this repo.

### `addresses.py`
This module contains the class Addresses which provides functionality for grabbing random addresses using reverse geocoding and for getting images from Google Street View corresponding to given addresses.

### `buildingclassifier.py`
This module contains the class BuildingClassifier which provides functionality for creating and training custom Inception V3 models for either ternary classification or binary classification, as well as functions for evaluating a model to determine the following statistics given a directory containing test images:

- Accuracy:  Percentage of the test images labeled correctly.
- Soft Precision:  Precision in determining soft vs. non-soft.
- Soft Recall:  Recall in determining soft vs. non-soft.
- Good Precision:  Precision in determining good image vs. bad image.
- Good Recall:  Recall in determining good image vs. bad image.

### `customlosses.py`
This module contains custom loss functions for binary crossentropy and categorical cross entropy which incorporate the focal loss modification described in this paper:  https://arxiv.org/abs/1708.02002

### `imagefunctions.py`
This module contains functions for saving a single image and for loading a single image into a numpy array that can be fed to the model for prediction.

## Creating a Custom Inception Model
Run `Training/create_inception_model.py` to create and save a custom Inception V3 model:

```
python Training/create_inception_model.py
```

Set the following variables within the python script:
- `model_filename`: the location where the model is to be saved.  The default setting will create `test_model.h5` in the `Training` folder.
- `number_categories`:  1 if binary classifier, 3 if ternary classifier.
- `dense_layer_sizes`:  a list containing the sizes of the dense layers to add to the end of the pre-trained portion of the Inception V3 model.
- `dropout_fraction`:  the dropout fraction to be used after each dense layer.
- `unfrozen_layers`:  the number of layers to be used for training; all other layers will keep the pre-trained weights.

## Training the Model
The folder `Small_Data` contains a small amount of data to train the model on for testing purposes.
Run "./Training/train.sh" to train the model:

```
./Training/train.sh
```

Set the following variables within the bash script:

- `HOSTS`:  if you want to use distributed computing, list all hosts that will be used.  If only one host is listed, it doesn't matter what that host is because only the localhost will be used.
- `GPUS`:  the number of GPUs on each host.
- `TRAINING_DIRECTORY`:  the directory where the training images are located.  If using binary classification, there should be two subfolders in this directory; if using ternary classification, there should be three subfolders.  By default, the directory is `Small_Data`.
- `TEST_DIRECTORY`:  the directory where the test images are located (for the purpose of calculating loss and accuracy).  By default, the directory is `Small_Data`.
- `MODEL_FILENAME`:  the location of the model to be trained.
- `TRAINED_MODEL_FILENAME`:  the location to save the trained model.
- `METRICS_FILENAME`:  the location to save the loss and accuracy.  It will be saved as a numpy array where the first row is the accuracy in each epoch and the second row is the loss in each epoch.
- `WEIGHTS`:  the weights to apply to each class to combat class imbalance.
- `BINARY`:  set to 0 if ternary classification, set to 1 if binary classification.
- `EPOCHS`:  the number of epochs to train.

## Plotting Training Metrics Using Streamlit
Modify line 7 of `Training/plot_metrics.py` to point to the metrics file created during training (by default it points to the metrics file created by the default `train.sh` file) and then run the following in the command line:

```
streamlit run Training/plot_metrics.py
```

## Launching the Streamlit App
Modify line 17 of "Product/product.py" to point to the model(s) you wish to serve with the app (by default, it points to the model created by the default `train.sh` file).  Then run the following in the command line:

```
streamlit run Product/product.py
```

This will output an external http address that any browser can view.

## Collecting More Street View Images
The "Pre-Processing" folder contains scripts for getting random addresses and Google Street View Images:

- `get_random_addressees.py`:  Gets random addresses from a region specified by latitude and longitude coordinates and writes them to a csv file.

- `get_soft_story_images.py`:  Gets Google Street View images given a csv file of addresses.

- `get_non_soft_story_images.py`:  Given a csv file of addresses, displays the Google Street View image of each one for the user to manually label as non-soft-story or bad image.