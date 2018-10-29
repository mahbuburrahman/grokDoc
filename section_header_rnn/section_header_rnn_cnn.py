
# coding: utf-8

# Developing a deep neural network for section header dataset

# In[ ]:

from sklearn import metrics, cross_validation
import tensorflow as tf
import skflow
import csv
from collections import defaultdict
import numpy as np
import sys
from sklearn.externals import joblib
import json
import re
import nltk
from collections import Counter
import codecs
from tensorflow.models.rnn import rnn, rnn_cell
csv.field_size_limit(sys.maxsize)
import string


# In[ ]:

#datafile= "data_for_weka_all.csv"
datafile1= "../s3/training_data/sample_train.csv"
datafile2= "../s3/training_data/sample_test.csv"


duplicate_samples=1
duplicate_samples_pos=1


# In[ ]:

# convert section header dataset for rnn neural network

sh_dataset = defaultdict(lambda : None)
sh_dataset['target_names'] =['no','yes']
sh_dataset['target'] =[]
sh_dataset['data'] =[]

with open(datafile1, 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        for i in range(duplicate_samples):
            if float(row['class']) > 0:
            #if row['class'] =="yes":
                for j in range(duplicate_samples_pos):
                    sh_dataset['target'].append(1)
                    sh_dataset['data'].append(row['text'])
            else:
                sh_dataset['target'].append(0)  
                sh_dataset['data'].append(row['text'])
with open(datafile2, 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        for i in range(duplicate_samples):
            if float(row['class']) > 0:
            #if row['class'] =="yes":
                for j in range(duplicate_samples_pos):
                    sh_dataset['target'].append(1)
                    sh_dataset['data'].append(row['text'])
            else:
                sh_dataset['target'].append(0)  
                sh_dataset['data'].append(row['text'])

            
        


# In[ ]:


X_train, X_test, y_train, y_test = cross_validation.train_test_split(sh_dataset['data'], sh_dataset['target'],
    test_size=0.5, random_state=42)

print "Total samples: ",len(sh_dataset['data'])
print "Training samples: ",len(X_train)
print "Test samples: ",len(X_test)


print "Total negative samples: ",y_train.count(0)+y_test.count(0)
print "Total positive samples: ",y_train.count(1)+y_test.count(1)



"""
X_train= np.array(X_train,dtype='float64')
X_test= np.array(X_test,dtype='float64')
y_train= np.array(y_train,dtype='float64')
y_test= np.array(y_test,dtype='float64')


# acceptance test data converting to numpy array 
X_acceptance= np.array(sh_acceptance['data'],dtype='float64')
y_acceptance= np.array(sh_acceptance['target'],dtype='float64')
"""


# In[ ]:

#Process vocabulary


MAX_DOCUMENT_LENGTH = 100

char_processor = skflow.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
X_train = np.array(list(char_processor.fit_transform(X_train)))
X_test = np.array(list(char_processor.transform(X_test)))

print X_train.shape
#print y_train.shape
print X_test.shape
#print y_test.shape


# Based on Character models

# In[ ]:

#HIDDEN_SIZE = 50
HIDDEN_SIZE = 10

N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2


def char_rnn_model(X, y):
    byte_list = skflow.ops.one_hot_matrix(X, 256)
    byte_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, byte_list)
    cell = rnn_cell.GRUCell(HIDDEN_SIZE)
    #cell = rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    _, encoding = rnn.rnn(cell, byte_list, dtype=tf.float32)
    return skflow.models.logistic_regression(encoding, y)
    


def char_cnn_model(X, y):
    """Character level convolutional neural network model to predict classes."""
    byte_list = tf.reshape(skflow.ops.one_hot_matrix(X, 256), 
        [-1, MAX_DOCUMENT_LENGTH, 256, 1])
    with tf.variable_scope('CNN_Layer1'):
        # Apply Convolution filtering on input sequence.
        conv1 = skflow.ops.conv2d(byte_list, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        # Add a RELU for non linearity.
        conv1 = tf.nn.relu(conv1)
        # Max pooling across output of Convlution+Relu.
        pool1 = tf.nn.max_pool(conv1, ksize=[1, POOLING_WINDOW, 1, 1], 
            strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
        # Transpose matrix so that n_filters from convolution becomes width.
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        # Second level of convolution filtering.
        conv2 = skflow.ops.conv2d(pool1, N_FILTERS, FILTER_SHAPE2,
            padding='VALID')
        # Max across each filter to get useful features for classification.
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
    # Apply regular WX + B and classification.
    return skflow.models.logistic_regression(pool2, y)


# In[ ]:

# configure GPU 
config_addon = skflow.addons.ConfigAddon(num_cores=5, gpu_memory_fraction=0.8)


# In[ ]:

#early stop set
val_monitor = skflow.monitors.ValidationMonitor(X_train, y_train,
                                                early_stopping_rounds=200,
                                                n_classes=2,
                                                print_steps=50)

classifier = skflow.TensorFlowEstimator(model_fn=char_rnn_model, n_classes=2,
    steps=1000, optimizer='Adam', learning_rate=0.01, continue_training=True, 
                                        config_addon=config_addon)


count=0
while count<10:
    # with early stop
    #classifier.fit(X_train, y_train, val_monitor, logdir='char_rnn')
    
    # without early stop
    classifier.fit(X_train, y_train, logdir='char_rnn')

    score = metrics.accuracy_score(y_test, classifier.predict(X_test))
    print('Accuracy: {0:f}'.format(score))
    count+=1
    

print "\n\nMore details:"
predicted = classifier.predict(X_test)
print(metrics.classification_report(y_test, predicted))

# Printing the confusion matrix
print "Confusion Matrix"
cm = metrics.confusion_matrix(y_test, predicted)
print(cm)

print "Done"

exit()

'''
# Predicting based on acceptance dataset
print "\n\nFor Acceptance test:"
predicted = classifier.predict(X_acceptance)
print(metrics.classification_report(y_acceptance, predicted))

# Printing the confusion matrix
print "Confusion Matrix"
cm = metrics.confusion_matrix(y_acceptance, predicted)
print(cm)
'''


# Process input data for classifier based on 15 features. 

# In[ ]:

class Bunch(dict):
    """Container object for datasets: dictionary-like object that
        exposes its keys as attributes."""
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def text_delexicalization(text):
    """delexicalization of each text string
    """
    regular_num = "#number "
    pattern_reg = re.compile('^(\d+(\.\d+)*(\.)?)|([a-z]+\.\s)', re.IGNORECASE)
    rep_text= pattern_reg.sub(regular_num,text)
    return rep_text

def generate_dataset(ann_file):

    target = []
    target_names = ['no','yes']
    feature_names = ["pos_nnp","without_verb_higher_line_space","font_weight","bold_italic","at_least_3_lines_upper","higher_line_space",'number_dot','text_len_group','seq_number','references_appendix','header_0','header_1','header_2',"title_case","all_upper"]
    #feature_names = ["font_weight","bold_italic",'number_dot','text_len_group','header_0','header_1','header_2',"title_case"]
    rawtext = []
    no_delex_rawtext =[]
    data =[]
    file_names=[]
    auxiliary_verb = ["is","was","were","am","are","may","might","be","will","shall","should","must","need","have","can","could","ought","would"]
    one_letter=list(string.ascii_uppercase)+list(string.ascii_lowercase)
    one_letter.append("I")
    one_letter.append("II")
    one_letter.append("III")
    one_letter.append("IV")
    one_letter.append("V")
    one_letter.append("VI")
    one_letter.append("VII")
    one_letter.append("VIII")
    one_letter.append("IX")
    one_letter.append("X")


    all_json_objs={}
    with open(ann_file, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #all_json_objs[row['file_name']] = row
            all_json_objs.setdefault(row['file_name'],[]).append(row)


    for reader in all_json_objs:
        each_file_json= all_json_objs[reader]
        print "processing file "+ reader
        #for row in each_file_json:
        all_font_weights=[]
        all_font_size=[]
        avg_font_weight =0.0
        avg_font_size =0.0
        avg_line_space =0.0
        minimum_line_space =100.0
        line_index=0
        counted_lines=0            

        for line in each_file_json:
            all_font_weights.append(float(line['font-weight']))
            all_font_size.append(float(line['font_size']))
            avg_font_weight += float(line['font-weight'])
            avg_font_size += float(line['font_size'])
            # line space 
            if line_index < len(each_file_json)-1:
                if each_file_json[line_index]["page-number"] == each_file_json[line_index+1]["page-number"]: 
                    if abs(float(each_file_json[line_index]["y-pos-l"]) - float(each_file_json[line_index+1]["y-pos-l"]))>50:
                        continue
                    avg_line_space += abs(float(each_file_json[line_index]["y-pos-l"]) - float(each_file_json[line_index+1]["y-pos-l"]))
                    if abs(float(each_file_json[line_index]["y-pos-l"]) - float(each_file_json[line_index+1]["y-pos-l"]))<minimum_line_space:
                        minimum_line_space = abs(float(each_file_json[line_index]["y-pos-l"]) - float(each_file_json[line_index+1]["y-pos-l"]))
                    counted_lines += 1
                    """
                    print each_file_json[line_index]["y-pos-l"]
                    print each_file_json[line_index+1]["y-pos-l"]
                    print "page ",each_file_json[line_index]["page-number"]
                    print "next page " ,each_file_json[line_index+1]["page-number"]
                    """

            line_index += 1
        if counted_lines !=0:
            avg_line_space = avg_line_space/counted_lines    
        avg_font_size = avg_font_size/len(each_file_json)
        avg_font_weight = avg_font_weight/len(each_file_json)
        #print avg_line_space
        #print minimum_line_space



        font_weight_counter = defaultdict(int)
        for word in all_font_weights:  
            font_weight_counter[word] += 1
        font_weight_counter = sorted(font_weight_counter, key = font_weight_counter.get, reverse = True)
        #print font_weight_counter[0]

        font_size_counter = defaultdict(int)
        for word in all_font_size:  
            font_size_counter[word] += 1            
        font_size_counter = sorted(font_size_counter, key = font_size_counter.get, reverse = True)
        #print font_size_counter[0]


        #print json.dumps(json_obj, sort_keys=True, indent=4, separators=(',', ': '))
        #exit()
        line_index=0
        for line in each_file_json:
            each_element={}
            each_element["text"]= line["text"].strip()

            # check line starts with a number or not
            if line["text"].decode('utf-8').split(" ")[0].replace(".","").isdigit() or line["text"].decode('utf-8').split(" ")[0] in one_letter:
                if len(line["text"].split())<5:
                    each_element["text_len_group"]=1
                elif len(line["text"].split())<7:
                    each_element["text_len_group"]=2
                else:
                    each_element["text_len_group"]=3
            else:
                if len(line["text"].split())<4:
                    each_element["text_len_group"]=1
                elif len(line["text"].split())<6:
                    each_element["text_len_group"]=2
                else:
                    each_element["text_len_group"]=3   

            #if ":" in line["text"].decode('utf-8'):
            if re.match("^(references|appendix)",line["text"],re.IGNORECASE):
                each_element["references_appendix"]=1
            else:    
                each_element["references_appendix"]=0

            #if re.match("((\d+|[a-z])\s?\.)",line["text"],re.IGNORECASE):
            if re.match("^\d+(\s|\.)+(\d+(\s|\.)+)*[a-z]+",line["text"],re.IGNORECASE) or re.match("^[a-z](\s\.\s)",line["text"],re.IGNORECASE):
                each_element["number_dot"]=1
            else:
                each_element["number_dot"]=0                    

            #if re.match("((\d+|(IX|IV|V?I{0,3}))\s?(\.|\))(\d*))",line["text"],re.IGNORECASE):
            #if re.match("(\d+|(([MDCLXVI])M*(C[MD]|D?C*)(X[CL]|L?X*)(I[XV]|V?I*)))(\s|\.|\))?\d*",line["text"],re.IGNORECASE):
            if re.match("^([a-z]|(IX|IV|V?I{0,3}))(\.|\s)",line["text"],re.IGNORECASE):
                each_element["seq_number"]=1
            else:
                each_element["seq_number"]=0    

            # case features
            each_element["at_least_3_lines_upper"] = 0
            if line["text"].isupper():
                each_element["all_upper"]=1
                if line_index > 0 and line_index< len(each_file_json)-1:
                    if each_file_json[line_index-1]["text"].isupper() and each_file_json[line_index+1]["text"].isupper():                     
                        each_element["at_least_3_lines_upper"]=1   
            else:
                each_element["all_upper"]=0    

            #line["text"]="2 Preliminaries and Main Results"
            count_title=0
            for word in line["text"].strip().decode('utf-8').split(" "):
                if word.istitle():
                    count_title+=1

            # checking the first word as number and then increase title word by one
            if line["text"].strip().decode('utf-8').split(" ")[0].replace(".","").isdigit():
                count_title+=1

            if count_title/float(len(line["text"].strip().decode('utf-8').split(" ")))>0.50:
                each_element["title_case"]=1
            else:
                each_element["title_case"]=0


            #import ipdb
            #ipdb.set_trace()

            #print line["text"]                
            #print "Count: ",count_title
            #print "Len: ",len(line["text"].split(" "))
            #print each_element["title_case"]

            #if re.sub("(\d+|(([MDCLXVI])M*(C[MD]|D?C*)(X[CL]|L?X*)(I[XV]|V?I*)))(\s|\.|\))?\d*","",line["text"].decode('utf-8')).istitle():
            #    each_element["title_case"]=1
            #else:
            #    each_element["title_case"]=0

            verb_flag =0 # no auxiliary verb
            for verb in auxiliary_verb:
                if verb in line["text"].decode('utf-8').split(" "):
                    verb_flag=1
                    break

            each_element["without_verb_higher_line_space"] = 0
            if verb_flag == 0:      
                if line_index < len(each_file_json)-1 and line_index > 0:              
                    if each_file_json[line_index-1]["page-number"] == each_file_json[line_index]["page-number"] and each_file_json[line_index]["page-number"] == each_file_json[line_index+1]["page-number"]:
                        if abs(float(each_file_json[line_index]["y-pos-l"]) - float(each_file_json[line_index+1]["y-pos-l"]))>avg_line_space and abs(float(each_file_json[line_index-1]["y-pos-l"]) - float(each_file_json[line_index]["y-pos-l"]))>minimum_line_space:
                            each_element["without_verb_higher_line_space"] =1
                elif line_index > 0:
                    if each_file_json[line_index-1]["page-number"] == each_file_json[line_index]["page-number"]:
                        if abs(float(each_file_json[line_index-1]["y-pos-l"]) - float(each_file_json[line_index]["y-pos-l"]))>avg_line_space:
                            each_element["without_verb_higher_line_space"] =1        
                elif line_index < len(each_file_json)-1:
                    if each_file_json[line_index]["page-number"] == each_file_json[line_index+1]["page-number"]:
                        if abs(float(each_file_json[line_index]["y-pos-l"]) - float(each_file_json[line_index+1]["y-pos-l"]))>avg_line_space:
                            each_element["without_verb_higher_line_space"] =1        

            # only line spaceing 
            each_element["higher_line_space"] = 0
            if line_index < len(each_file_json)-1 and line_index > 0:              
                if each_file_json[line_index-1]["page-number"] == each_file_json[line_index]["page-number"] and each_file_json[line_index]["page-number"] == each_file_json[line_index+1]["page-number"]:
                    if abs(float(each_file_json[line_index]["y-pos-l"]) - float(each_file_json[line_index+1]["y-pos-l"]))>avg_line_space and abs(float(each_file_json[line_index-1]["y-pos-l"]) - float(each_file_json[line_index]["y-pos-l"]))>minimum_line_space:
                        each_element["higher_line_space"] =1
            elif line_index > 0:
                if each_file_json[line_index-1]["page-number"] == each_file_json[line_index]["page-number"]:
                    if abs(float(each_file_json[line_index-1]["y-pos-l"]) - float(each_file_json[line_index]["y-pos-l"]))>avg_line_space:
                        each_element["higher_line_space"] =1        
            elif line_index < len(each_file_json)-1:
                if each_file_json[line_index]["page-number"] == each_file_json[line_index+1]["page-number"]:
                    if abs(float(each_file_json[line_index]["y-pos-l"]) - float(each_file_json[line_index+1]["y-pos-l"]))>avg_line_space:
                        each_element["higher_line_space"] =1        



            #if float(line["font_size"])>12.0:
            if float(line["font_size"])>font_size_counter[0]:
            #if float(line["font_size"])>avg_font_size:    
                each_element["header_0"] =1
            else:
                each_element["header_0"] =0

            #if float(line["font_size"])>=12.0 and float(line["font-weight"])>=300.0:
            if float(line["font_size"])>=font_size_counter[0] and float(line["font-weight"])>font_weight_counter[0]:
            #if float(line["font_size"])>=avg_font_size and float(line["font-weight"])>avg_font_weight:
                each_element["header_1"] =1
            else:
                each_element["header_1"] =0

            #if float(line["font_size"]) >=12.0 and "bold" in line["font-family"].lower():
            if float(line["font_size"]) >= font_size_counter[0] and "bold" in line["font-family"].lower():    
            #if float(line["font_size"]) >= avg_font_size and "bold" in line["font-family"].lower():
                each_element["header_2"] =1
            else:
                each_element["header_2"] =0

            if float(line["font-weight"])>font_weight_counter[0]:
                each_element["font_weight"] =1
            else:
                each_element["font_weight"] =0

            if "bold" in line["font-family"].lower() and "italic" in line["font-family"].lower():
                each_element["bold_italic"] =1
            else:
                each_element["bold_italic"] =0


            # POS tagging
            tokens = nltk.word_tokenize(line["text"].decode('utf-8'))
            text = nltk.Text(tokens)
            tags = nltk.pos_tag(text) 
            counts = Counter(tag for word,tag in tags)
            total_pos = sum(counts.values())
            pos = dict((word, float(count)/total_pos) for word,count in counts.items())

            if "NNP" in pos.keys() and "NN" in pos.keys():
                if pos["NNP"] + pos["NN"]  > 0.5:
                    each_element["pos_nnp"]=1
                else:
                    each_element["pos_nnp"]=0
            elif "NNP" in pos.keys():
                if pos["NNP"]  > 0.5:
                    each_element["pos_nnp"]=1
                else:
                    each_element["pos_nnp"]=0
            elif "NN" in pos.keys():
                if pos["NN"]  > 0.5:
                    each_element["pos_nnp"]=1
                else:
                    each_element["pos_nnp"]=0
            else:
                each_element["pos_nnp"]=0

            if line['class'] =="0":
                target.append(0)
            else:
                target.append(1)    

            data.append([each_element["pos_nnp"],each_element["without_verb_higher_line_space"],each_element["font_weight"],each_element["bold_italic"],each_element["at_least_3_lines_upper"],each_element["higher_line_space"],each_element['number_dot'],each_element['text_len_group'],each_element['seq_number'],each_element['references_appendix'],each_element['header_0'],each_element['header_1'],each_element['header_2'],each_element["title_case"],each_element["all_upper"]])

            #data.append([each_element["font_weight"],each_element["bold_italic"],each_element['number_dot'],each_element['text_len_group'],each_element['header_0'],each_element['header_1'],each_element['header_2'],each_element["title_case"]])
            #rawtext.append(self.text_delexicalization(each_element['text']))
            rawtext.append(each_element['text'])
            #no_delex_rawtext.append(each_element['text'])
            file_names.append(reader)
            line_index += 1

            #import ipdb
            #ipdb.set_trace()

            #print line["text"],count_title,len(line["text"].strip().decode('utf-8').split(" ")), data[-1]
    return Bunch(data=data, feature_names=feature_names,target_names=target_names,target=target,rawtext=rawtext,filenames=file_names)        
    


# We just need the text field, though it generate 15 features. because 15 features ar enot required for deep learning.
# This is just for the test on real data. 

# In[ ]:

test_dataset = generate_dataset("testset_acrobat_section_header.csv")


# In[ ]:

# for test data 
sh_test = defaultdict(lambda : None)
sh_test['target_names'] =['no','yes']
sh_test['feature_names'] = ["pos_nnp","without_verb_higher_line_space","font_weight","bold_italic","at_least_3_lines_upper","higher_line_space",'number_dot','text_len_group','seq_number','references_appendix','header_0','header_1','header_2',"title_case","all_upper"]
sh_test['target'] =[]
sh_test['data'] =[]


for row in test_dataset.rawtext:
    #sh_test['target'].append(int(row['class']))
    sh_test['data'].append(row)
    

print "Testing samples: ",len(sh_test['data'])    
# test data converting to numpy array 
#X_test= np.array(sh_test['data'],dtype='float64')
X_test = np.array(list(char_processor.transform(sh_test['data'])))

print X_test.shape
# Predicting based on acceptance dataset
print "\n\nTesting the classifier:"
predicted = classifier.predict(X_test)
out_file = codecs.open("result_section_header_rnn_more_epochs.txt", "w",encoding="utf-8")
unique_file_list=[]
for i in range(len(predicted)):
    if predicted[i] ==1:
        if test_dataset.filenames[i].split(".tetml")[0] not in unique_file_list:
            unique_file_list.append(test_dataset.filenames[i].split(".tetml")[0])
            out_file.write("\n\n")
            out_file.write(test_dataset.filenames[i].split(".tetml")[0])
            out_file.write("\n")
            out_file.write("======================================================")
            out_file.write("\n")
            out_file.write(test_dataset.no_delex_rawtext[i].decode('utf-8'))
            out_file.write("\n")
        else:
            out_file.write(test_dataset.no_delex_rawtext[i].decode('utf-8'))
            out_file.write("\n")

out_file.close()                
print "Done"


# In[ ]:




# In[ ]:



