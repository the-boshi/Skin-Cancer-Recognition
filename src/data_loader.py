import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.io import imread
from skimage.transform import resize


class MyDataset(torch.utils.data.Dataset):
    def __init__ (self, cvs_file, img_dir, transform):
        super(MyDataset, self).__init__()
        self.annotations = pd.read_csv(cvs_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        img = imread(img_path)
        img = resize(img, (224, 224))
        
        if self.transform:
            img = self.transform(img)
        
        y_label = torch.tensor([self.annotations.iloc[index, 1], self.annotations.iloc[index, 2],
                               self.annotations.iloc[index, 3], self.annotations.iloc[index, 4],
                               self.annotations.iloc[index, 5], self.annotations.iloc[index, 6],
                               self.annotations.iloc[index, 7], self.annotations.iloc[index, 8],
                               self.annotations.iloc[index, 9]])
        
        return (img, y_label)

def load_data(batch_size): 
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = MyDataset('D:/Coding/Skin-Cancer-Recognition/dataset/train.csv', 'D:\Coding\Skin-Cancer-Recognition\dataset\ISIC_2019_Training_Input\images', transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDataset('D:/Coding/Skin-Cancer-Recognition/dataset/test.csv', 'D:\Coding\Skin-Cancer-Recognition\dataset\ISIC_2019_Training_Input\images', transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader