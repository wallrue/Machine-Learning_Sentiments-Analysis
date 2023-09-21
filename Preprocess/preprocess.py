# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:13:03 2022

@author: mylocalaccount
"""

import os
import json
import pickle

from pathlib import Path
import numpy as np

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

def write_data(mydata,name,output_dir):
  json_file = os.path.join(output_dir,name+'.json')
  pkl_file = os.path.join(output_dir,name+'.pkl')

  with open(json_file, 'w') as outfile:
      for data in mydata:
          json.dump(mydata, outfile, indent = 4)
          outfile.write('\n')

  with open(pkl_file,'wb') as outfile:
      pickle.dump(mydata, outfile)
          #outfile.write('\n')
    

  #return

def tokenize_dataset(utterances):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(utterances) #create the indexes for text
    print(tokenizer.word_index)
    sequences = tokenizer.texts_to_sequences(utterances)

    
    text_sequences = []
    #print(sequences[0])
    for i in range(len(sequences)):
        if(len(sequences[i]) <= 25):
          L = [0 for i in range(len(sequences[i]),25)]
          #L.append(sequences[i][-1])
          sequences[i][len(sequences[i]):] = L
        else:
          #sequences[i][len(sequences[0])-1] = sequences[i][-1]
          del sequences[i][25:]
    
                        
    mysequences = np.array(sequences)
    mysequences.reshape(-1,25)
    return mysequences, len(tokenizer.word_counts)

def get_dataset_UCI(dataset):
    lines = []
    with open(dataset) as f:
        lines = f.readlines()
    
    value = []
    value_pos = 0
    end_pos = []
    utterances =[]
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            index = (len(lines[i]) - 1) - j
            #print(i,j,"-",lines[i][index])
            if(lines[i][index] != ' ' and len(value) > i):
                # line1 = lines[i][:index-1]
                # if(len(line1[i]) <= 25):
                #   L = [0 for i in range(len(line1[i]),25)]
                #   #L.append(sequences[i][-1])
                #   line1[i][len(line1[i]):] = L
                # else:
                #   #sequences[i][len(sequences[0])-1] = sequences[i][-1]
                #   del line1[i][25:]
                #print(index)
                utterances.append(str(lines[i][:index - 1]))
                #utterances.append("[CLS] " + str(line1[i][:index-1]).lower() + " [SEP]")#+str(lines[i][value_pos]))
                break
                
            elif(lines[i][index] != ' '):
                value_pos = index - 1
                value.append(int(lines[i][index-1]))
    
    return utterances, value
    # (utterances,my_word_count) = tokenize_dataset(utterances)
    # return utterances.tolist(), to_categorical(value, 2).reshape(-1,2).tolist(), my_word_count


def get_dataset(path, utterances, value, sub_folder):
    all_files = os.listdir(path)   # imagine you're one directory above test dir    
    txt_files = list(filter(lambda x: x[-4:] == '.txt', all_files))
    print(txt_files)
    if sub_folder == 'pos':
        myvalue = 1
    elif  sub_folder == 'neg':
        myvalue = 0 #For unknown = 0 in the first time, but we dont train this dataset
    else:
        myvalue = 0 #data is unknown
        
    for i in range(len(txt_files)):
        with open(path/txt_files[i], encoding="utf8") as f:
            lines = f.readlines() 
        for i in range(len(lines)):
            # if(len(lines[i]) <= 25):
            #   L = [0 for i in range(len(lines[i]),25)]
            #   #L.append(sequences[i][-1])
            #   lines[i][len(lines[i]):] = L
            # else:
            #   #sequences[i][len(sequences[0])-1] = sequences[i][-1]
            #   del lines[i][25:]
            #for j in range(len(lines[i])):
            utterances.append(str(lines[i]).lower()) 
            value.append(1 if sub_folder == 'pos' else 0)
            
    #return utterances, value
    
if __name__ == "__main__":
     
    dataset_Path = Path().absolute()/'dataset'
    dataset_UCI = [dataset_Path/'sentiment labelled sentences'/'amazon_cells_labelled.txt',
                dataset_Path/'sentiment labelled sentences'/'imdb_labelled.txt',
                dataset_Path/'sentiment labelled sentences'/'yelp_labelled.txt']
    
    dataset_Standford = [dataset_Path/'aclImdb'/'train',dataset_Path/'aclImdb'/'test']
    
    #For dataset_UCI
    mydata_raw = []
    for i in range(len(dataset_UCI)): 
        (my_input,my_output) = get_dataset_UCI(dataset_UCI[i])
        data = {"input": [], "output": []}   
        
        data["input"] = my_input
        data["output"] = my_output
        mydata_raw.append(data)
        
    print(np.shape(mydata_raw))

    utterances = []
    value = []
    for i in range(len(mydata_raw)):
        utterances += mydata_raw[i]['input']
        value += mydata_raw[i]['output']
        
        
    (my_input,my_word_count) = tokenize_dataset(utterances)
    my_input = my_input.tolist()
    my_output = to_categorical(value, 2).reshape(-1,2).tolist()
    
    mydata_combine = {"input": [], "output": [], "word_count": []} 
    mydata_combine["input"] = my_input
    mydata_combine["output"] = my_output
    mydata_combine["word_count"].append(my_word_count)

    


    write_data(mydata_combine, 'processed_sentiment_sentences', 'preprocessed_data/sentiment_sentences')

    
    
    #For Standford
    train_test_unknown = []
    for i in range(len(dataset_Standford)):
        value = []
        utterances =[]
        for subfolder in ['pos','neg']:
            sub_folder = subfolder
            get_dataset(dataset_Standford[i]/sub_folder, utterances, value, sub_folder)
        
        data = {"input": [], "output": []}   
        data["input"] = utterances
        data["output"] = value
        train_test_unknown.append(data)
        
    value = []
    utterances =[]
    sub_folder = 'unsup'
    get_dataset(dataset_Standford[0]/sub_folder, utterances, value, sub_folder)
    data = {"input": [], "output": []}   
    data["input"] = utterances
    data["output"] = value
    train_test_unknown.append(data)
    # mydata = {"input": [],"output": []}
    # mydata['input'] = train_test_unknown[0]['input'] + train_test_unknown[1]['input']
    # mydata['output'] = train_test_unknown[0]['output'] + train_test_unknown[1]['output'] 
    


    #Tokenize:
    lines = []
    with open(dataset_Path/'aclImdb'/'imdb.vocab', encoding = 'utf-8') as f:
        lines = f.readlines()
        
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines) #create the indexes for text
    print(tokenizer.word_index)
    
    

        
    mydata = []   
    for i in range(len(train_test_unknown)):
        #start
        sequences = tokenizer.texts_to_sequences(train_test_unknown[i]["input"])
        #print("my_sequence",sequences)
            
        text_sequences = []
        #print(sequences[0])
        for k in range(len(sequences)):
            if(len(sequences[k]) <= 100):
                L = [0 for k in range(len(sequences[k]),100)]
                #L.append(sequences[i][-1])
                sequences[k][len(sequences[k]):] = L
            else:
                #sequences[i][len(sequences[0])-1] = sequences[i][-1]
                del sequences[k][100:]
            
                                
        mysequences = np.array(sequences)
        mysequences.reshape(-1,100)
        #end
        
        
        my_input =  mysequences
        my_word_count = len(tokenizer.word_counts)
        my_input = my_input.tolist()
        my_output = to_categorical(train_test_unknown[i]["output"], 2).reshape(-1,2).tolist()
    
        mydata_combine = {"input": [], "output": [], "word_count": []} 
        mydata_combine["input"] = my_input
        mydata_combine["output"] = my_output
        mydata_combine["word_count"].append(my_word_count)
        mydata.append(mydata_combine)

    write_data(mydata, 'processed_standford', 'preprocessed_data/standford')

