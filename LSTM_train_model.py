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
steps = 2


#read the file with train data
with open('train_log.txt', 'r') as myfile:
    data=myfile.read()

#be careful with special characters
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

# LSTM 3 timesteps - prepare data - encode 2 words -> 1 word
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
print('Sequences array: \n')
print(sequences)
X, y = sequences[:,:-1],sequences[:,-1]

#convert y to binary vectors
y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
#the first layer
# - the largest integer (i.e. word index) in the input should be no larger than vocabulary size 
# - The Embedding layer is initialized with random weights and will learn an embedding for all of the words in the training dataset.
# - output_dim (50): This is the size of the vector space in which words will be embedded (size of the embedding vectors). It defines the size of the output vectors from this layer for each word. For example, it could be 32 or 100 or even larger. Test different values for your problem.
# - input_length: This is the length of input sequences (here is 2)
# The Embedding layer has weights that are learned. If you save your model to file, this will include weights for the Embedding layer.
# The output of the Embedding layer is a 2D vector with one embedding for each word in the input sequence of words (input document).
# If you wish to connect a Dense layer directly to an Embedding layer, you must first flatten the 2D output matrix to a 1D vector using the Flatten layer.

model.add(Embedding(vocab_size+1, LSTM_CELLS, input_length=max_length-1))

model.add(LSTM(vocab_size))
model.add(Dropout(0.1))
model.add(Dense(vocab_size, activation='softmax'))
                                                                                                             
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=2,batch_size = 50)

# Evaluate network
print('LSTM Network Evaluation:\n')
print(X)
print(y)

#loss and accuracy
score = model.evaluate(X, y, verbose=0)
print(score)

print(model.summary())

model.save('lstm_model.h5')  # creates a HDF5 file 
del model  # deletes the existing model
