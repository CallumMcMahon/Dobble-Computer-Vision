# Dobble-automation
A computer vision project for playing the game of Dobble automatically

The project is explained in the [dissertation write-up](cm389-dissertation-paper.pdf) provided in the repo.

The basic idea of the project is to automatically detect cards within the video frame, crop them, then identify the 8 different symbols on each card. 

<img align="center" src="assets/README%20images/process.PNG" width="800" />

The orignial card images were aquired from [Leo Reynolds on Flickr](https://www.flickr.com/photos/lwr/sets/72157660922894042/)

The cards were then manually annotated using [labelImg](https://github.com/tzutalin/labelImg)

The datasets are then [synthesised](card_dataset_creator.ipynb) using openCV with a collection of [background images](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

The [bounding box](card_bbox_predictor.ipynb) and [multiclass classifier](symbol_multilabel_classifier.ipynb) models are created using [fastai](https://github.com/fastai/fastai), a library built on top of pytorch.

A [tkinter app](tkinter_app.py) loads the models and visualises predictions with a GUI.

The system can run at over 20 frames per second, correctly identifying the matching symbols between cards around 90% of the time on real world examples.
