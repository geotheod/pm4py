from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,TimeDistributed, Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import LSTM, Input, Bidirectional
from keras.models import load_model
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy



LSTM_CELLS = 50

#open file with train data
with open('train_log.txt', 'r') as myfile:
    data=myfile.read()

#be careful of special characters
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
#reads the words in data and gives an index for every words based on frequency
tokenizer.fit_on_texts([data])
print('Word index: ')
print(tokenizer.word_index)

#replace every word in the text to correspoding word index - returns list of list with one element so use [0] to get the one and only first list
encoded = tokenizer.texts_to_sequences([data])[0]
print('encoded: \n')
print(encoded)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# Bidirectional LSTM 3 timesteps - prepare data - encode 2 words -> 1 word
sequences = list()
for i in range(2, len(encoded)):
	sequence = encoded[i-2:i+1]
	sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
#print('Sequences before padding: \n')
#print(sequences)

max_length = max([len(seq) for seq in sequences])    #max_length is 3
# Pad sequence to be of the same length
# length of sequence must be 3 (maximum)
# 'pre' or 'post': pad either before or after each sequence
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)   													
#print('Sequences after padding: \n')
#print(sequences)
#convert list to array to get X,y train
sequences = array(sequences)

X, y = sequences[:,:-1],sequences[:,-1]


#convert y to binary vectors
y = to_categorical(y, num_classes=vocab_size)
#Bidirectional needs reshape of data
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(Bidirectional(LSTM(vocab_size), input_shape=(2, 1)))
model.add(Dropout(0.1))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=[categorical_accuracy])
model.fit(X, y, epochs=500, verbose=2,batch_size = 10)

# Evaluate network
print('Evaluate training')
score = model.evaluate(X, y, verbose=0)
print('Bidirectional Network Evaluation:\n')
print(score)

model.summary()
model.save('bidirectional_model.h5')  # creates a HDF5 file 
#Save model to drive
#torch.save(model.state_dict(), path)

del model  # deletes the existing model
