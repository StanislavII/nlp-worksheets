#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
from torch.utils.data import Dataset

class Lstm_Dataset(Dataset):
    
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        
        return len(self.sequences)
    
    def __getitem__(self, item):
        
        sequence, label = self.sequences[item]
        
        return dict( sequence = torch.Tensor(sequence.values.tolist()),
                    label = torch.tensor(label).long()
        )

