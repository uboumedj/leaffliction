# 42 Deep Learning Computer vision project: Leaffliction

This is a solo rework on a group project. The old version can be found here: https://github.com/thervieu/leaffliction

Why the rework ? Here are a few reasons:
- A lot of improvements could be made to the code
- I wanted to integrate a **fastAPI** interface to the project, which was not part of the assignment
- Updates to **plantCV** made the library better, but those same updates completely broke our own implementation

## Overview

The goal of the project is to use various tools to **analyse**, **augment**, and **classify** a dataset comprised of **plant leaf images**,
using deep learning and computer vision techniques.

The neural network model we chose to use was a **Convolutional Neural Network (CNN)**

## Libraries used

* The whole project was coded using python 3.11
* **matplotlib** (*data visualisation, training comparison*)
* **numpy** (*data structure*)
* **opencv** (*computer vision library*)
* **plantcv** (*plant-specialised computer vision library, based on OpenCV)
* **tensorflow** and **keras** (*deep learning models*)
* **click** (*program argument parser*)
* **pillow** (*image manipulation*)
