# Topic Categorization using the Bert 
To ensure the diversity of GPT model generation, we will build a topic model to classify all samples into five categories: sports, entertainment, technology, politics, and business. We used a web crawler to crawl 1500 samples from Twitter to construct our model.

## Dataset
The dataset contains the text and the labels (sports, entertainment, technology, politics, and business) associated with the text.
Due to some pravicy issues, we can not publish the dataset, but if you are proficient in using crawlers, it is very easy to obtain such datasets yourself. If you want to directly train on your data, make sure your data.csv follows the formot below.

      
| *category*     | *text*     | 
| -------- | -------- | 
| tech  |...| 
| politics |...| 
| sports  |...| 
|entertainment|...|
|business|...|
|...|...|

## Training
The training set, validation set and test set were randomly divided according to 80:10:10.
Before run the train.py, change the line 111 to change the dataset address to yours, you can DIY the number of epoch, learning rate to suit your dataset.

## Performance
The accuracy rate of the model on the validation set is 98.5%.
