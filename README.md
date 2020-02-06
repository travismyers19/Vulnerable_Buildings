# Vulnerable Building Detection
A soft-story building is a multiple-story building in which the first floor is "soft", meaning it consists mostly of garage door openings or windows or other openings.  Soft-story buildings are particularly vulnerable to earthquakes, and thus the ability to quickly inventory soft-story buildings in an area may be of use to city or state officials.  This repo contains code to create and train a custom Inception V3 image classifier to detect soft-story buildings and deploy the model in a user-friendly Streamlit app.

##Overview
The Streamlit app takes as input from the user latitude and longitude coordinates specifying a bounding box region of the world and also a number of addresses.  It takes random latitude and longitude locations within that bounding box and uses reverse geocoding to obtain the addresses.  Then it uses those addresses to obtain Google Street View images.  The images are then sent through the image classifier and the results are presented to the user: "Soft-Story" means that the image is of a soft-story building, "Non-Soft-Story" means that the image is of a non-soft-story buildng, and "Bad Image" means that the image does not contain any buildings or it is unclear which building the image is supposed to refer to.

There are two types of models that can be used:

Ternary classification:  A single model classifies the images into the three categories.

Binary classification:  One model classifies the image as good or bad, and then a second model classifies the image as soft or non-soft.

##Installation and Dependencies
Clone the repo

##Preparing the Environment
Before running any scripts in this repo, source the "build/environment.sh" bash script to add "Modules" to the PYTHONPATH environment variable:

```
source build/environment.sh
```

##Modules
The "Modules" folder contains all of the custom modules in this repo.

###addresses.py
This module contains the class Addresses which provides functionality for grabbing random addresses using reverse geocoding and for getting images from Google Street View corresponding to given addresses.

###buildingclassifier.py
This module contains the class BuildingClassifier which provides functionality for creating and training custom Inception V3 models for either ternary classification or binary classification, as well as functions for evaluating a model to determine the following statistics given a directory containing test images:

Accuracy:  Percentage of the test images labeled correctly.
Soft Precision:  Precision in determining soft vs. non-soft.
Soft Recall:  Recall in determining soft vs. non-soft.
Good Precision:  Precision in determining good image vs. bad image.
Good Recall:  Recall in determining good image vs. bad image.

###customlosses.py
This module contains custom loss functions for binary crossentropy and categorical cross entropy which incorporate the focal loss modification described in this paper:  https://arxiv.org/abs/1708.02002

###imagefunctions.py
This module contains functions for saving a single image and for loading a single image into a numpy array that can be fed to the model for prediction.

##Creating a Custom Inception Model
Run "Training/create_inception_model.py" to create and save a custom Inception V3 model:

```
python Training/create_inception_model.py
```

Set the following variables within the python script:
`model_filename`: the location where the model is to be saved.
`number_categories`:  1 if binary classifier, 3 if ternary classifier.
`dense_layer_sizes`:  a list containing the sizes of the dense layers to add to the end of the pre-trained portion of the Inception V3 model.
`dropout_fraction`:  the dropout fraction to be used after each dense layer.
`unfrozen_layers`:  the number of layers to be used for training; all other layers will keep the pre-trained weights.

##Training the Model
Run "Training/train.sh" to train the model:

```
Training/train.sh
```

Set the following variables within the bash script:


##Creating the



## Motivation for this project format:
- **Insight_Project_Framework** : Put all source code for production within structured directory
- **tests** : Put all source code for testing in an easy to find location
- **configs** : Enable modification of all preset variables within single directory (consisting of one or many config files for separate tasks)
- **data** : Include example a small amount of data in the Github repository so tests can be run to validate installation
- **build** : Include scripts that automate building of a standalone environment
- **static** : Any images or content to include in the README or web framework if part of the pipeline

## Setup
Clone repository and update python path
```
repo_name=Insight_Project_Framework # URL of your new repository
username=mrubash1 # Username for your personal github account
git clone https://github.com/$username/$repo_name
cd $repo_name
echo "export $repo_name=${PWD}" >> ~/.bash_profile
echo "export PYTHONPATH=$repo_name/src:${PYTHONPATH}" >> ~/.bash_profile
source ~/.bash_profile
```
Create new development branch and switch onto it
```
branch_name=dev-readme_requisites-20180905 # Name of development branch, of the form 'dev-feature_name-date_of_creation'}}
git checkout -b $branch_name
```

## Initial Commit
Lets start with a blank slate: remove `.git` and re initialize the repo
```
cd $repo_name
rm -rf .git   
git init   
git status
```  
You'll see a list of file, these are files that git doesn't recognize. At this point, feel free to change the directory names to match your project. i.e. change the parent directory Insight_Project_Framework and the project directory Insight_Project_Framework:
Now commit these:
```
git add .
git commit -m "Initial commit"
git push origin $branch_name
```

## Requisites

- List all packages and software needed to build the environment
- This could include cloud command line tools (i.e. gsutil), package managers (i.e. conda), etc.

#### Dependencies

- [Streamlit](streamlit.io)

#### Installation
To install the package above, pleae run:
```shell
pip install -r requiremnts
```

## Build Environment
- Include instructions of how to launch scripts in the build subfolder
- Build scripts can include shell scripts or python setup.py files
- The purpose of these scripts is to build a standalone environment, for running the code in this repository
- The environment can be for local use, or for use in a cloud environment
- If using for a cloud environment, commands could include CLI tools from a cloud provider (i.e. gsutil from Google Cloud Platform)
```
# Example

# Step 1
# Step 2
```

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- **DO NOT STORE CREDENTIALS IN THE CONFIG DIRECTORY!!**
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)


## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
