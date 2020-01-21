# Dobble-automation
A computer vision project for playing the game of Dobble automatically

The project is explained in the [dissertation write-up](cm389-dissertation-paper.pdf) provided in the repo.

The basic idea of the project is to automatically perform these steps:

![cards](assets/README%20images/cards_on_table.jpg | width=100)

![bounding boxes](assets/README%20images/cards_on_table_bbox.PNG | width=100) 

![cropped card](assets/README%20images/single_card_crop.jpg | width=100) 

Primarily uses openCV, pytorch and the fastai libraries. 

The dataset_creator files creates the datasets needed

The card_bbox_predictor and symbol_multilabel_classifier files create the pytorch models.

tkinter_app loads the models with a GUI.
