# AML

t

This is the Advanced Machine Learning project in 3rd year at ENSAE Paris. This project was designed by Antoine Rougier, Grégoire Brugère and Marin Petibon. 

The theoretical subject of this notebook is Batchnormalisation. And we're trying to predict a person's gender using voice data. To look at the effects of batchnormalization, we try to see if it influences the Internal Covariate Shift. To do this, we add Gaussian noise after each layer that has been batchnormalised and then compare the results with a reference layer.

We used Kaagle data for this project, you can find them directly in the SRC data file, or you can find here : https://www.kaggle.com/datasets/primaryobjects/voicegender 

We have different notebook :

import_data : is used to read the zip file and transform it into a csv file

data_analysis : these are the descriptive statistics for our dataset

NN : this is the notebook that compares the different neural networks with batchnormalisation 

NN_hand : we wanted to reproduce this by hand, but the results were not as good as we had hoped
