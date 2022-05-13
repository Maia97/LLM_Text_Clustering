import torch

def encode_data(dataframe,tokenizer,max_seq_length=64):
    inputs = list(dataframe['text'])
    encoded = tokenizer(inputs,max_length=max_seq_length,truncation=True,padding="max_length",return_tensors="pt")
    return encoded

def extract_labels(dataframe):
    return list(dataframe['classid'])
