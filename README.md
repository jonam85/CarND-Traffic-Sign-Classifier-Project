## Project: Build a Traffic Sign Recognition Program

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

This project is as part of my Submission to the Project 2: Traffic Sign Classifier Project for the Udacity Self Driving Car Nano Degree Program.

In this project, deep neural networks and convolutional neural networks are used to classify traffic signs. The model is trained and validated using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) so it can classify traffic sign images. After the model is trained, the network is tried out on a set images of German traffic signs that is found on the web.

## The Project

The goals of this project are:

- To explore the dataset, provide summary and visualization of the data set
- Design, train and test a model architecture based on the Deep Learning and Convolutional Neural Network Architecture
- Test the model to make predictions on new images, Analyze the performance of the network on the new images
- Summarize the results with a written report

### Dependencies

The code for this project is written in Python and arranged in a Jupyter Notebook file [Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb).

This code shall run on a GPU, but it is recommended to run this code on an AWS GPU instance for faster execution (Note: this costs Money). 

This code uses TensorFlow, OpenCV, Numpy, Scipy and scikit libraries. It is recommended to use the Anaconda package which has all these libraries pre-installed or can be installed to specific environment.

### Dataset

1. Download the data set of pickled files from [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). The data set contains the pickled files separated as Train, Validation and Test files. Each contains the labelled data of German Traffic Signs images fixed to a size of 32x32x3 (Color images).
2. Once the dataset downloaded to the same path of the Jupyter Notebook file and with the all necessary library files available, the note book file shall be run from the browser.

### Review Set

1. The code of this project submission is at [Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb).

2. The write up for this project is at [WriteUp.md](WriteUp.md)

   ​



