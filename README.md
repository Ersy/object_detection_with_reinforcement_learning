# Object detection using reinforcement learning

## Installation Dependencies:

* Python 2.7
* TensorFlow 1.2.0
* keras 2.0.5
* OpenCV-Python

## Required Data

PASCAL VOC2007 and 2012 datasets
Available at: http://host.robots.ox.ac.uk/pascal/VOC/index.html

## Description

The RL agent learns to localise an object in an image of a given class by taking a series of actions from a discrete action space.

This RL implementation uses Q-learning with the Q-function being approximated with a deep fully connected network.

## To be Implemented

* Detect multiple instances of an object in a given image (implement IoR mechanism)
* Classification score for detections

## Further Ideas

* Lightweight classifier (Inception/Xception)
* Multiclass extension
