import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import csv
import gzip
import pandas as pd


class NameDataset(Dataset):
    
    def remove_non_ascii_1(self,text):
        return ''.join(i for i in text if ord(i)<128)
    def __init__(self, is_train_set=False):
        
        text_list = []
        df = pd.read_csv("/home/ankan/project/try_demo.csv")
        for text in df.iterrows():
            text_list.append(self.remove_non_ascii_1(text[1][1]))
        
        self.stories = text_list
        self.authors = list(df['authors'])
        self.len = len(self.authors)

        self.author_list = list(sorted(set(self.authors)))

    def __getitem__(self, index):
        return self.names[index], self.authors[index]

    def __len__(self):
        return self.len

    def get_authors(self):
        return self.author_list

    def get_author(self, id):
        return self.author_list[id]

    def get_author_id(self, author):
        return self.author_list.index(author)

# Test the loader
if __name__ == "__main__":
    dataset = NameDataset(False)
    

    train_loader = DataLoader(dataset=dataset,
                              batch_size=10,
                              shuffle=True)

    for epoch in range(2):
        for i, (stories, authors) in enumerate(train_loader):
            # Run your training process
            print(epoch, i, "stories", names, "authors", authors)