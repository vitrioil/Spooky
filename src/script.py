# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

author_to_idx = {a:indx for indx,a in enumerate(set(train["author"]))}


train["author"] = [author_to_idx[a] for a in list(train["author"])]

word_to_idx = {}
word_to_idx["<UNK>"] = 0
word_to_idx["<PAD>"] = 1

word_set = set()
remove_punc = string.punctuation
for line in train["text"]:
    for word in line.split():
        for punc in remove_punc:
            word = word.strip(punc)
        word_set.add(word)

for indx,word in enumerate(word_set,len(word_to_idx.keys())):
    word_to_idx[word] = indx

idx_to_word = {idx:word for word,idx in word_to_idx.items()}

def text_to_word(df):
    global word_to_idx
    text = df["text"]
    indexed_text = []
    for line in list(text):
        indexed_line = []
        for word in line.split():
            for punc in remove_punc:
                word = word.strip(punc)
            indexed_line.append(word_to_idx.get(word,word_to_idx["<UNK>"]))
        indexed_text.append(indexed_line)
    df["text"] = indexed_text
    return df

train = text_to_word(train)
test = text_to_word(test)

train.drop(["id"],inplace=True,axis=1)
test.drop(["id"],inplace=True,axis=1)
train = train.values
test = test.values
import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import LSTM,TimeDistributed,Dense,Dropout,Input,Embedding,Flatten

def modelS(input_shape,total_words):
    X_input = Input(shape=input_shape,name="input_X")
    
    X = Embedding(total_words,100,input_length=None,name="embedding_1")(X_input)
    
    X = LSTM(64,return_sequences=True,name="lstm_1")(X)
    X = Dropout(0.1)(X)
    
    X = LSTM(64,return_sequences=True,name="lstm_2")(X)
    X = Dropout(0.1)(X)
    
    X = TimeDistributed(Dense(3,activation="softmax"),name="final")(X)
    return Model(inputs=X_input,outputs=X)

X_train,Y_train = train[:,0],train[:,1]
Y_train = keras.utils.to_categorical(Y_train,3)

epochs = 100
lr = 1e-4
mini_batch_size = 1

def data_generator(X_train,Y_train):
    while True:
        shuffle_indices = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_indices]
        Y_train = Y_train[shuffle_indices]
        for x,y in zip(X_train,Y_train):
            x = np.array(x)
            y = np.array(y)
            yield x[np.newaxis,...],y[np.newaxis,np.newaxis,...]


'''
if train_model:
	model = modelS((None,),len(word_to_idx))
	model.compile(optimizer=Adam(lr=lr),loss="categorical_crossentropy",metrics=["categorical_accuracy"])

	try:
    		model.fit_generator(data_generator(X_train,Y_train),epochs=epochs,steps_per_epoch=len(X_train))
	except KeyboardInterrupt as e:
	    pass
	model.save("model2.h5")
	model = keras.models.load_model("model.h5")
else:
	answer = []
	for i in test:
		i = np.array(i[0])[np.newaxis,...]
		prediction = model.predict(i)
		len(prediction[0])
		answer.append(prediction[0][-1])
	answer = np.array(answer)
	answer_df = {"id":["id"],"EAP":[i[0] for i in answer],"HPL":[i[1] for i in answer],"MWS":[i[2] for i in answer]}
	answer_df = pd.DataFrame(answer_df)
'''
