# Sentiment_Analysis_Net_Based_on_twitterDataset
Trained On Twitter dataset with sentiment label and mainly use LSTM

## Overview

Sentiment analysis involves determining the sentiment or emotion expressed in a piece of text. In this project, we use TensorFlow and LSTM to predict the sentiment (positive or negative) of tweets from the Twitter dataset.

## Dataset

The dataset consists of tweets with sentiment labels. Half of the dataset contains positive sentiments, while the other half contains negative sentiments. The dataset is sourced from [https://tianchi.aliyun.com/dataset/35761].


## Start the Training 
1. Download the dataset to the root directory, rename it as the train.csv,the format must be the CSV, download the globe.6B.200d.txt (http://nlp.stanford.edu/data/glove.6B.zip) to ./dataset/
2. CD the code directory, run the preprocess.py like:  python preprocess.py <the direction of the dataset> , the csv will be process and saved as train-processed.csv
3. Run the stats.py: python stats.py <the direction of the processed csv>, it will generates several files.
4. Run the lstm.py: python lstm.py
5. The model will be saved in ./models/


## Requirements
Tensorflow==2.4.0 <br>
python==3.8 <br>
numpy <br>
scikit-learn <br>
scipy <br>
nltk <br>
keras
