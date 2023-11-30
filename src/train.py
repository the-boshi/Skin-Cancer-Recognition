import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from utils import *
from data_loader import load_data
from model import GoogLeNet, LeNet, EfficientNet
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(train_loader, num_epochs, model, criterion, optimizer):
    
    model.train()
    loss_hist = []
    curr_epoch = 0
    '''
    model, optimizer = load_checkpoint(torch.load("D:\Coding\Skin-Cancer-Recognition\src\checkpoints\saved_model-epoch"+str(curr_epoch)+".pth.tar"), model, optimizer)
    np.load("D:/Coding/Skin-Cancer-Recognition/src/checkpoints/loss"+str(curr_epoch)+".npy")
    optimizer.param_groups[0]['lr'] = 3e-6
    '''


    step = 0
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

            data, targets = next(iter(train_loader))
        
            data = data.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device)
                
            scores = model(data)
            loss = criterion(scores, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.detach().cpu().numpy())

        plt.show(block=True)    
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint, filename="D:/Coding/Skin-Cancer-Recognition/src/checkpoints/saved_model-epoch"+str(epoch + curr_epoch + 1)+".pth.tar")
        np.save("D:/Coding/Skin-Cancer-Recognition/src/checkpoints/loss" + str(epoch + curr_epoch + 1) + ".npy", np.array(loss_hist))
    
    
    return model, loss_hist

def evaluate_model(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)
            
            scores = model(x)
            scores_max = torch.argmax(scores, dim=1)
            y_max = torch.argmax(y, dim=1)
            num_correct += (scores_max == y_max).sum()
            num_samples += scores.size(0)

    acc = float(num_correct)/float(num_samples)
    print(f'Got {num_correct} / {num_samples} with accuracy {acc:2f}.')
    
    
def main():
    batch_size, num_epochs = 32, 10
    num_classes, lr = 8, 3e-5

    train_loader, test_loader = load_data(batch_size)

    model = EfficientNet(version = "b0", num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    #model, optimizer = load_checkpoint(torch.load('D:\Coding\Skin-Cancer-Recognition\src\checkpoints\saved_model-epoch10.pth.tar'), model, optimizer)
    model, loss_hist = train_model(train_loader, num_epochs, model, criterion, optimizer)

    evaluate_model(model, train_loader)
    
    
if __name__ == '__main__':
    main()
    