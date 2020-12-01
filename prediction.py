import numpy as np
import os
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import load_model
from collections import deque
## in shape
memory_length = 140

##load unique characters
unique_characters = np.load("unique_characters.npy")
print("Size of unique character matrix : ", unique_characters.shape)
##load one_hot_encoded data
character_data_onehotencoded = np.load("character_Data_onehotencoded.npy")
print("Size of onehot encoded data matrix : ",
      character_data_onehotencoded.shape)
## load int to char
with open('int_to_char.pickle', 'rb') as handle:
    int_to_char = pickle.load(handle)

in_shape = (memory_length,unique_characters.shape[0])

##load a dummy model
print("Building Model")
model = Sequential()
model.add(LSTM(256, input_shape=in_shape, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(unique_characters.shape[0], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
c = os.listdir('fromColab')
for _ in c:
#load weights:
    model.load_weights("fromColab/"+_)
    #model.load_weights("fromColab/weights-13-1.3717.hdf5")
    
    memory_length = 140
    indices_of_word = np.random.randint(0, high=200000, size=memory_length)
    #print(indices_of_word)
    x_in = deque(maxlen=memory_length)
    for i in range(memory_length):
        x_in.append(character_data_onehotencoded[indices_of_word[i]])
    prediction = []
    for i in range(100):
        out = model.predict(np.array(x_in,ndmin=3), verbose=0)
        x_new = np.zeros(unique_characters.shape[0])
        x_new[np.argmax(out[0])] = 1
        x_in.append(x_new)
        prediction.append(int_to_char[np.argmax(out)])
    
    output = ''.join(prediction)
    print("Predicted sentence: \""+output+"\"")
