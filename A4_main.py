import re
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch.optim as optim

# Set the device to enable GPU processing
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Amazon Reviews Dataset
reviews = pd.read_csv('Reviews.csv')

# Drop NaNs
reviews.dropna(subset=['Text', 'Summary'], inplace=True)

# Data Preprocessing
reviews['model_input'] = reviews['Text'] + " TL;DR " + reviews['Summary']

# Determining model input length
avg_length = sum([len(review.split()) for review in reviews.model_input.values])/len(reviews)

# Set max_length
max_length = 100

# Before training, sample a subset of reviews
reviews = reviews.sample(10000)
reviews = reviews.model_input.values.tolist()

# Define the dataset class
class ReviewDataset(Dataset):  
    def __init__(self, tokenizer, reviews, max_len):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.eos_id = self.tokenizer.eos_token_id
        self.reviews = reviews
        self.result = []
        
        for review in self.reviews:
            tokenized = self.tokenizer.encode(review + tokenizer.eos_token)
            padded = self.pad_truncate(tokenized)            
            self.result.append(torch.tensor(padded))
            
    def __len__(self):
        return len(self.result)

    def __getitem__(self, item):
        return self.result[item]

    def pad_truncate(self, name):
        name_length = len(name)
        if name_length < self.max_len:
            difference = self.max_len - name_length
            result = name + [self.eos_id] * difference
        elif name_length > self.max_len:
            result = name[:self.max_len] 
        else:
            result = name
        return result

# Load pre-trained GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2").to(device)

# Define training parameters
batch_size = 32
epochs = 3

# Load dataset
dataset = ReviewDataset(tokenizer, reviews, max_length)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Define training function
def train(model, optimizer, dl, epochs):
    model.train()
    for epoch in range(epochs):
        for idx, batch in enumerate(dl):
            optimizer.zero_grad()
            batch = batch.to(device)
            output = model(batch, labels=batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if idx % 100 == 0:
                print("Epoch {}, Batch {}, Loss: {:.4f}".format(epoch+1, idx, loss.item()))

# Train the model
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
train(model, optimizer, dataloader, epochs)

# Save the trained model
model_dir = "./ft_model"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
