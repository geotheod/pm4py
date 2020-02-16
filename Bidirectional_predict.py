import math
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout,TimeDistributed, Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model




with open('test_log.txt', 'r') as myfile:
    data=myfile.read()

with open('train_log.txt', 'r') as myfile:
    traindata=myfile.read()

#Be carefull!!! Tokenize with training data!!!
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
#reads the words in data and gives an index for every words based on frequency
tokenizer.fit_on_texts([traindata])
print('Word index: ')
print(tokenizer.word_index)


encoded = tokenizer.texts_to_sequences([data])[0]

vocab_size = len(tokenizer.word_index) + 1
sequences = list()

for i in range(2, len(encoded)):
	sequence = encoded[i-2:i+1]
	sequences.append(sequence)
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]

#convert y to binary vectors
yl = to_categorical(y, num_classes=vocab_size)

#load trained model
model = load_model('lstm_model.h5')



#load trained model
model = load_model('bidirectional_model.h5')


print('Bidirectional Results: ')
print ('\n') 

#Evaluation without function
cnt = 0
for i in range(len(X)):
  yhat = model.predict_classes(X[i].reshape(1,2,1))
  #print('Expected:', y[i], 'Predicted', yhat)
  if (y[i] == yhat):
    cnt += 1

print('Total successful: ',cnt,' out of ', len(X), 'Percentage: ', cnt/len(X))

# Evaluate network with model.evaluate() function
X = X.reshape((X.shape[0], X.shape[1], 1))
score = model.evaluate(X, yl, verbose=0)
print('Bidirectional Network Evaluation:\n')
print(score)

