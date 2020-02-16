from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.log.util import sorting
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.util import get_log_representation
import csv



#open empty file to write
train_file = open("/content/drive/My Drive/didactoriko/python/LSTM_BIDIRECTIONAL_RNN/train_log.txt","w")
test_file = open("/content/drive/My Drive/didactoriko/python/LSTM_BIDIRECTIONAL_RNN/test_log.txt","w")
#import event log
log = xes_import_factory.apply('/content/drive/My Drive/didactoriko/datasets/BPI Challenge 2017.xes')
#print(log)

activities = attributes_filter.get_attribute_values(log, "concept:name")
print(activities)
#act = 0
#for key in activities:
#  print(act)
#  act = act + activities[key]
#print(act)
#sort event log based on timestamp
log = sorting.sort_timestamp(log)


#str_trace_attributes = []
#str_event_attributes = ["concept:name"]
#num_trace_attributes = []
#num_event_attributes = ["amount"]
#data, feature_names = get_log_representation.get_representation(
#                           log, str_trace_attributes, str_event_attributes,
#                           num_trace_attributes, num_event_attributes)

#print(data)
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
