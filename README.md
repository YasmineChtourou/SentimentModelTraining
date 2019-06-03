# SentimentModelService


This is a multi-class text classification problem.

The purpose of this project is to classify sentiments from short sentences into 3 different categories ( positive / negative / neutral).

The model was built with Long short-term memory(LSTM).

In this project there are 2 folders:
   * Dataset : it contains the file data.csv
   * LSTM_Glove : it contains files 
       - TrainLSTM_preprocessing.py
       - preprocessing.py for the preprocessing of the data 
       

The dataset containts 2464 short texts ( reviews of people about services and products ). It was collected from different sources : amazon.com , yelp.com and imbd.com


## Install dependencies

To get a development environment running you should :

   * Install virtualenv :

``` pip install virtualenv ```

   * Create a new virtual environment and easily install all libraries by running the following command :

``` conda create  --name venv_name  --file requirements.txt ```

In the file requirements.txt you find all necessary dependencies for this project.

   * To activate the new environment :

``` source activate  venv_name ```


## In order to run this download Glove Vectors:

From this url https://nlp.stanford.edu/projects/glove/ downlowd glove.840B.300d.zip
Create a folder in your project named Glove and add the file glove.840B.300d.txt


## Reference:
 
-https://github.com/Harsh24893/EmotionRecognition 

