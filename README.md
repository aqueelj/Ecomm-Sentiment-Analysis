# Ecomm-Sentiment-Analysis
Sentiment analysis of user reviews on an E-commerce website for women's clothing using RNN-LSTM and pre-trained Word Embeddings


In this project, we take user reviews from an E-commerce website and apply sentiment analysis to train model that is capable of predicitng
positive or negative emotions from a review or feedback.

It involves the following steps:

1. We have used the functions in Keras for preprocessing the review text. The text is tokenized and then mapped to a list of integers. Special
characters and punctuation marks have also been removed in this step.

2. Next we have used a pre-trained word embedding to help the model understand the review text better and provide better results. The GLoVe
embedding with 300 dimensions has been used here.

3. Lastly we have created an RNN model with LSTM cells which retain the context of long sentences better. The model is created using Keras on
tensorflow-gpu. A dropout layer has been added for regularization and the output layer has a softmax activation to predict negative or positive 
sentiment.
