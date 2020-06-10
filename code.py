

#Import packages

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from matplotlib import pyplot as plt
plt.style.use('dark_background')
import seaborn as sns


import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D


# Load csv file and set first row as header

path = 'C:/Users/user/Desktop/Files 2/Code/Womens_Ecomm/Womens_E-Comm.csv'
df = pd.read_csv(path, header=None)
df=df.drop(df.columns[0],axis=1)
df.columns = df.iloc[0]
df = df[1:]
#df.head()
#df.columns
#df.dtypes

# Drop title column and null values

df=df.drop(columns=['Title'])
df=df.dropna()
df.isna().sum()

#Visualizations

#Type conversion
df['Age']= df['Age'].astype('int')
df['Rating']= df['Rating'].astype('int')

#Breakdown of reviews

df.hist(column='Age')
# Majority reviews from women between ages 30-50

sns.boxplot(x=df['Rating'])
# Majority reviews are positive (4-5)

df['Department Name'].value_counts().plot(kind='bar')
#Tops and dresses have majority of reviews

df['Division Name'].value_counts().plot(kind='bar')




#Data Preprocessing

reviews = df[['Review Text','Rating']]
#reviews.shape

# Tokenization for wordcloud 

tokenized_review = reviews.apply(lambda x: x.split())
all_words = ' '.join(str(v) for v in tokenized_review)

#!pip install wordcloud
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()            
 

# Tokenization using Keras

X=df['Review Text']                              
y=df['Recommended IND']

tokenizer = Tokenizer(num_words=15000)

# Maximum of 15000 unique words in the vocabulary

tokenizer.fit_on_texts(X)

# Convert text tokens to a sequence of integers
sequences = tokenizer.texts_to_sequences(X)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences)

labels = to_categorical(np.asarray(y))

print('Shape of data tensor:', data.shape)
# Maximum length of a review is 116 characters
print('Shape of label tensor:', labels.shape)
# 5 ratings are one hot encoded + 0 rating

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)

# Glove Embedding

embeddings_index = {}
f = open('C:/Users/user/Desktop/Files 2/Code/glove/glove.6B.300d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim=300

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
        
        
embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=data.shape[1],
                            trainable=False)

# define the model
model = Sequential()
model.add(embedding_layer)
model.add(SpatialDropout1D(0.4))
model.add(LSTM(75, dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

batch_size=1000
epochs= 40
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# 91% accuracy on training set and 89% accuracy on test set


model.save('C:/Users/user/Desktop/Files 2/Code/Womens_Ecomm/saved_model.pb')
