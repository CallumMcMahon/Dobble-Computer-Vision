# Dobble-automation
A computer vision project for playing the game of Dobble automatically

The project is explained in the [dissertation write-up](cm389-dissertation-paper.pdf) provided in the repo.

The basic idea of the project is to automatically perform these steps:

![image](assets/README images/cards_on_table.jpg = 250x250)

![](assets/README images/cards_on_table_bbox.PNG | width=100) 

![](assets/README images/single_card_crop.jpg | width=100) 

Primarily uses openCV, pytorch and the fastai libraries. 

The dataset_creator files creates the datasets needed

The card_bbox_predictor and symbol_multilabel_classifier files create the pytorch models.

tkinter_app loads the models with a GUI.
