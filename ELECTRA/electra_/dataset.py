import data_utils
from data_utils import encode_data
from data_utils import extract_labels
import torch
from torch.utils.data import Dataset


class CreateDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_seq_length=64):
        self.encoded_data = encode_data(dataframe,tokenizer,max_seq_length)
        self.label_list = extract_labels(dataframe)

    def __len__(self): 
        return len(self.label_list)

    def __getitem__(self, i):
        item_i = {}
        item_i['input_ids'] = self.encoded_data['input_ids'][i]
        item_i['attention_mask'] = self.encoded_data['attention_mask'][i]
        item_i['labels'] = self.label_list[i]
        
        return item_i