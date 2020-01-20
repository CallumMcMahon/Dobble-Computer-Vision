# Dobble-automation
A computer vision project for playing the game of Dobble automatically

The project is explained in the [dissertation write-up](cm389-dissertation-paper.pdf) provided in the repo.

Primarily uses openCV, pytorch and the fastai libraries. 

The dataset_creator files creates the datasets needed

The card_bbox_predictor and symbol_multilabel_classifier files create the pytorch models.

tkinter_app loads the models with a GUI.
