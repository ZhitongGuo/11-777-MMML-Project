import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import torch.optim as optim
import json
import math

class combiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_param = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.bert_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, clip, bert):
        return clip * self.clip_param + bert * self.bert_param

class scoreDataset(Dataset):
    def __init__(self, clip, bert, score):
        self.clip = clip
        self.bert = bert
        self.score = score
        
    def __len__(self):
        return len(self.clip)
    
    def __getitem__(self, index):
        return (self.clip[index], self.bert[index], self.score[index])

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device='cuda', path = "./model.pth"):
    model.to(device)
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        min_val_loss = math.inf
        running_loss = 0.0

        for clip, bert, labels in train_loader:
            clip, bert, labels = clip.to(device), bert.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(clip, bert)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        model.eval()
        running_loss_val = 0.0

        for clip, bert, labels in val_loader:
            clip, bert, labels = clip.to(device), bert.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(clip, bert)
            loss = criterion(outputs, labels)

            running_loss_val += loss.item()

        # Print the average loss for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}, Val Loss: {running_loss_val / len(val_loader)}')
        if running_loss_val / len(val_loader) < min_val_loss:
            min_val_loss = running_loss_val / len(val_loader)
            torch.save(model.state_dict(), path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = combiner().to(device)
lr = 1e-5
num_epoch = 500
bert, clip_score, reward_score = json.load('bert.json'), json.load('clip.json'), json.load('reward.json')
dataset = scoreDataset(clip_score, bert, reward_score)
training, validation = random_split(dataset, [0.75, 0.25])
train_dataloader = DataLoader(training, batch_size=32, shuffle = True)
validation_dataloader = DataLoader(validation, batch_size=32, shuffle = True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = lr)
train_model(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs=num_epoch, device=device)