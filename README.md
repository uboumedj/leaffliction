# 42 Deep Learning Computer vision project: Leaffliction

This is a solo rework on a group project. The old version can be found here: https://github.com/thervieu/leaffliction

Why the rework ? Here are a few reasons:
- A lot of improvements could be made to the code
- I wanted to integrate a **fastAPI** interface to the project, which was not part of the assignment
- Updates to **plantCV** made the library better, but those same updates completely broke our own implementation
- We successfully finished the project by training **6 different models per plant type** (one per transformation)
and using a **voting system** to decide on the classification results. I wanted to know if
training a **single model** on the whole dataset (the original images from both plant types AND
the transformed ones together) could yield better results

## Overview

The goal of the project is to use various tools to **analyse**, **augment**, and **classify** a
dataset comprised of **plant leaf images**, using deep learning and computer vision techniques.

The neural network model we chose to use was a **Convolutional Neural Network (CNN)**

### Rework structure

For the rework, I separated the "legacy" code which is still present in the files beginning with `cli*`
from the "new" code in the `web*` files.

The two main differences are:
- The *web* files have less console output, since there will be a fastAPI interface
- The *web* files will interact with the whole dataset without separating the plant types or the
transformations, and yield a single model that will be used for the web predictions

## Libraries used

* The whole project was coded using python 3.11
* **matplotlib** and **seaborn** (*data visualisation*)
* **numpy** (*data structure*)
* **opencv** (*computer vision library*)
* **plantcv** (*plant-specialised computer vision library, based on OpenCV)
* **tensorflow** and **keras** (*deep learning models*)
* **click** (*program argument parser*)
* **pillow** (*image manipulation*)
