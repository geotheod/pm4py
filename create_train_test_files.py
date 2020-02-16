from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.log.util import sorting
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.util import get_log_representation
import csv



#open empty files to write train and test data
train_file = open("/content/drive/My Drive/didactoriko/python/LSTM_BIDIRECTIONAL_RNN/train_log.txt","w")
test_file = open("/content/drive/My Drive/didactoriko/python/LSTM_BIDIRECTIONAL_RNN/test_log.txt","w")
#import event log
log = xes_import_factory.apply('/content/drive/My Drive/didactoriko/datasets/BPI Challenge 2017.xes')
#print(log)

#get the activities
activities = attributes_filter.get_attribute_values(log, "concept:name")
print(activities)

#sort event log based on timestamp
log = sorting.sort_timestamp(log)

#split data from event log - 80% for train and 20% for test
i=0
print(len(log))
for trace in log:
  for event in trace:
    if event['lifecycle:transition'] == 'start':
      #print(event['concept:name'], event['time:timestamp'])
      data = event['concept:name'].replace(' ','_')
      if i <= 0.8 * len(log):
        train_file.write(data)
        train_file.write(' ')
      else:
        test_file.write(data)
        test_file.write(' ') 
  if i <= 0.8 * len(log):
    train_file.write('\n')  
  else:
    test_file.write('\n') 
  i+=1

train_file.close()
test_file.close()
