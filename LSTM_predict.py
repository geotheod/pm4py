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


#predict sequence of n_words activities
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
  #get input activity
  in_text = seed_text
  #print('in_text',in_text,'\n')
  #for the number of activities on sequence you want to predict
  for _ in range(n_words):
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    #pad if less than max text length
    encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
    #print('in text ',in_text)
    #predict one activity
    yhat = model.predict_classes(encoded, verbose=0)
    out_word = ''
    for word, index in tokenizer.word_index.items():
      #convert predicted activity to word
      if index == yhat:
        #print('Word',word,'\n')
        out_word = word
        break
    #feed the next input with the sequence of activities
    in_text += ' ' + out_word
    
  return in_text
 

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



# Evaluate network
print('LSTM Network Evaluation:\n')
score = model.evaluate(X, yl, verbose=0)
print(score)

print('LSTM Results: ')
print ('\n')

#sequence prediction
for i in tokenizer.word_index:
  #print(tokenizer.index_word)
  print(generate_seq(model, tokenizer, max_length-1, i , 3))

#load trained model
model = load_model('bidirectional_model.h5')
